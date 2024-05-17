use crate::model::*;
use egg::*;
use itertools::Itertools;
use std::{borrow::Borrow, collections::HashMap};
use cxx::{CxxVector};

const MAX_DIM: usize = 8;

#[cxx::bridge]
mod ffi {
  extern "Rust" {
    type CppGraphConverter;
    type TensorInfo;
    fn new_converter() -> Box<CppGraphConverter>;
    // Exposing the constructor functions with Box<TensorInfo>
    fn new_input(self: &mut CppGraphConverter, dims: &[i32]) -> Box<TensorInfo>;
    fn new_compare_op(self: &mut CppGraphConverter, inpt_1: Box<TensorInfo>, inpt_2: Box<TensorInfo>, comparison: i32, cost: i32) -> Box<TensorInfo>;
    fn new_broadcast_in_dim(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, dimensions: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_convert_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_reduce_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, dimensions: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_reshape_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, shape: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_gather_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, start_indices: Box<TensorInfo>, dimension_numbers: i32, cost: i32) -> Box<TensorInfo>;
    fn new_select_op(self: &mut CppGraphConverter, pred: Box<TensorInfo>, on_true: Box<TensorInfo>, on_false: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    // fn new_concatenate_op(self: &mut CppGraphConverter, inputs: &[Box<TensorInfo>], dimension: i32, cost: i32) -> Box<TensorInfo>;
    fn new_dot_general_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, dot_dimension_numbers: i32, cost: i32) -> Box<TensorInfo>;
    fn new_pad_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, padding_value: i32, padding_config: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_slice_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, start_indices: &[i32], limit_indices: &[i32], strides: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_transpose_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, permutation: &[i32], cost: i32) -> Box<TensorInfo>;
    fn new_mul_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_add_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_div_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_subtract_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_min_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_max_op(self: &mut CppGraphConverter, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_neg_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_tanh_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_exp_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_iota_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_constant_op(self: &mut CppGraphConverter, value: i32, cost: i32) -> Box<TensorInfo>;
    fn new_dynamic_update_slice_op(self: &mut CppGraphConverter, operand: Box<TensorInfo>, update: Box<TensorInfo>, start_indices: Box<TensorInfo>, cost: i32) -> Box<TensorInfo>;
    fn new_dynamic_slice_op(self: &mut CppGraphConverter, operand: Box<TensorInfo>, start_indices: Box<TensorInfo>, slice_sizes: i32, cost: i32) -> Box<TensorInfo>;
    fn new_scatter_op(self: &mut CppGraphConverter, inpt: Box<TensorInfo>, scatter_indices: Box<TensorInfo>, updates: Box<TensorInfo>, dimension_numbers: i32, cost: i32) -> Box<TensorInfo>;
    
    fn print_rec_expr(self: &CppGraphConverter);
    fn pretty_print_rec_expr(self: &CppGraphConverter, width: i32);
    }
}

/// Struct for storing information of a tensor. This is passed between functions
/// during graph creation.
#[derive(Copy, Clone, Default)]
pub struct TensorInfo {
    /// Id into the RecExpr constructed
    pub id: Id,
    /// Shape of the tensor. We deal with tensor up to MAX_DIM dimensions
    pub shape: [i32; 8],
    /// Number of dimensions of this tensor
    pub n_dim: usize,
}

/// Struct for converting a model specified using our Rust interface to RecExpr
///
/// The RecExpr is growed on the fly when member functions are called. Uses a
/// Hashmap to store the map of scalar nodes to their indices into the RexExpr to
/// avoid replication.
#[derive(Default)]
pub struct CppGraphConverter {
    rec_expr: RecExpr<Mdl>,
    scalar_map: HashMap<i32, Id>,
    name_gen: NameGen,
}

pub fn new_converter() -> Box<CppGraphConverter> {
  Box::new(CppGraphConverter::default())
}

/// The APIs of GraphConverter are (intended to) match TASO's so that we can easily
/// construct TASO graphs using this class
impl CppGraphConverter {
  pub fn rec_expr(self) -> RecExpr<Mdl> {
      self.rec_expr
  }

  pub fn input(&mut self, dims: &[i32]) -> TensorInfo {
      let name = self.name_gen.new_input_name() + "@" + &dims.iter().join("_");
      let node = Mdl::Var(Symbol::from(name));
      let name_id = self.rec_expr.add(node);

      let new_node = Mdl::Input([name_id]);
      let (shape, n_dim) = self.shape_from_dim(dims);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape,
          n_dim,
      }
  }

  pub fn compare_op(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo, comparison: i32, cost: i32) -> TensorInfo {
      let comparison_id = self.add_or_get_val(comparison);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::CompareOp([inpt_1.id, inpt_2.id, comparison_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt_1.n_dim,
      }
  }

  pub fn broadcast_in_dim(&mut self, inpt: TensorInfo, dimensions: &[i32], cost: i32) -> TensorInfo {
      let dim_name = &dimensions.iter().join("_");
      let node = Mdl::Var(Symbol::from(dim_name));
      let dimensions_id = self.rec_expr.add(node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::BroadcastInDimOp([inpt.id, dimensions_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn convert_op(&mut self, inpt: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ConvertOp([inpt.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn reduce_op(&mut self, inpt: TensorInfo, dimensions: &[i32], cost: i32) -> TensorInfo {
      let dim_name = &dimensions.iter().join("_");
      let node = Mdl::Var(Symbol::from(dim_name));
      let dimensions_id = self.rec_expr.add(node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ReduceOp([inpt.id, dimensions_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn reshape_op(&mut self, inpt: TensorInfo, shape: &[i32], cost: i32) -> TensorInfo {
      let shape_name = &shape.iter().join("_");
      let node = Mdl::Var(Symbol::from(shape_name));
      let shape_id = self.rec_expr.add(node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ReshapeOp([inpt.id, shape_id, cost_id]);
      let (shape_new, n_dim) = self.shape_from_dim(shape);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: shape_new,
          n_dim: n_dim,
      }
  }

  pub fn gather_op(&mut self, inpt: TensorInfo, start_indices: TensorInfo, dimension_numbers: i32, cost: i32) -> TensorInfo {
      let dimension_numbers_id = self.add_or_get_val(dimension_numbers);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::GatherOp([inpt.id, start_indices.id, dimension_numbers_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn select_op(&mut self, pred: TensorInfo, on_true: TensorInfo, on_false: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::SelectOp([pred.id, on_true.id, on_false.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: on_true.shape,
          n_dim: on_true.n_dim,
      }
  }

  // pub fn concatenate_op(&mut self, inputs: &[TensorInfo], dimension: i32, cost: i32) -> TensorInfo {
  //   let n_inputs = inputs.len();
  //   assert!(n_inputs > 0);
  //   let dimension_id = &[self.add_or_get_val(dimension)];
  //   let cost_id = &[self.add_or_get_val(cost)];
  //   let input_ids: Vec<Id> = inputs.iter().map(|x| x.id).collect();
  //   let new_node = Mdl::ConcatenateOp([input_ids, dimension_id, cost_id]);
  //   let mut shape = inputs[0].shape;
  //   shape[dimension as usize] = inputs.iter().map(|x| x.shape[dimension as usize]).sum();
  //   TensorInfo {
  //     id: self.rec_expr.add(new_node),
  //     shape,
  //     n_dim: inputs[0].n_dim,
  //   }
  // }

  pub fn dot_general_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, dot_dimension_numbers: i32, cost: i32) -> TensorInfo {
      let dot_dimension_numbers_id = self.add_or_get_val(dot_dimension_numbers);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::DotGeneralOp([lhs.id, rhs.id, dot_dimension_numbers_id, cost_id]);
      let mut shape = lhs.shape;
      shape[shape.len() - 1] = rhs.shape[rhs.shape.len() - 1];
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape,
          n_dim: lhs.n_dim,
      }
  }

  pub fn pad_op(&mut self, inpt: TensorInfo, padding_value: i32, padding_config: &[i32], cost: i32) -> TensorInfo {
      let padding_value_id = self.add_or_get_val(padding_value);
      let padding_config_name = &padding_config.iter().join("_");
      let node = Mdl::Var(Symbol::from(padding_config_name));
      let padding_config_id = self.rec_expr.add(node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::PadOp([inpt.id, padding_value_id, padding_config_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn slice_op(&mut self, inpt: TensorInfo, start_indices: &[i32], limit_indices: &[i32], strides: &[i32], cost: i32) -> TensorInfo {
      let start_indices_name = &start_indices.iter().join("_");
      let start_indices_node = Mdl::Var(Symbol::from(start_indices_name));
      let start_indices_id = self.rec_expr.add(start_indices_node);
      let limit_indices_name = &limit_indices.iter().join("_");
      let limit_indices_node = Mdl::Var(Symbol::from(limit_indices_name));
      let limit_indices_id = self.rec_expr.add(limit_indices_node);
      let strides_name = &strides.iter().join("_");
      let strides_node = Mdl::Var(Symbol::from(strides_name));
      let strides_id = self.rec_expr.add(strides_node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::SliceOp([inpt.id, start_indices_id, limit_indices_id, strides_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn transpose_op(&mut self, inpt: TensorInfo, permutation: &[i32], cost: i32) -> TensorInfo {
      let permutation_name = &permutation.iter().join("_");
      let node = Mdl::Var(Symbol::from(permutation_name));
      let permutation_id = self.rec_expr.add(node);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::TransposeOp([inpt.id, permutation_id, cost_id]);
      let mut shape = [0; MAX_DIM];
      let n_dim = inpt.n_dim;
      for (i, &perm_i) in permutation.iter().enumerate() {
          shape[i] = inpt.shape[perm_i as usize];
      }
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape,
          n_dim,
      }
  }

  pub fn mul_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::MulOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn add_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::AddOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn div_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::DivOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn subtract_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::SubtractOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn min_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::MinOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn max_op(&mut self, lhs: TensorInfo, rhs: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::MaxOp([lhs.id, rhs.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: lhs.shape, // This is an example, you might want to calculate actual shape
          n_dim: lhs.n_dim,
      }
  }

  pub fn neg_op(&mut self, inpt: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::NegOp([inpt.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn tanh_op(&mut self, inpt: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::TanhOp([inpt.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn exp_op(&mut self, inpt: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ExpOp([inpt.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn iota_op(&mut self, inpt: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::IotaOp([inpt.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  pub fn constant_op(&mut self, value: i32, cost: i32) -> TensorInfo {
      let value_id = self.add_or_get_val(value);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ConstantOp([value_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: [1; MAX_DIM], // Assuming constant has a shape of [1]
          n_dim: 1,
      }
  }

  pub fn dynamic_update_slice_op(&mut self, operand: TensorInfo, update: TensorInfo, start_indices: TensorInfo, cost: i32) -> TensorInfo {
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::DynamicUpdateSliceOp([operand.id, update.id, start_indices.id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: operand.shape, // This is an example, you might want to calculate actual shape
          n_dim: operand.n_dim,
      }
  }

  pub fn dynamic_slice_op(&mut self, operand: TensorInfo, start_indices: TensorInfo, slice_sizes: i32, cost: i32) -> TensorInfo {
      let slice_sizes_id = self.add_or_get_val(slice_sizes);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::DynamicSliceOp([operand.id, start_indices.id, slice_sizes_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: operand.shape, // This is an example, you might want to calculate actual shape
          n_dim: operand.n_dim,
      }
  }

  pub fn scatter_op(&mut self, inpt: TensorInfo, scatter_indices: TensorInfo, updates: TensorInfo, dimension_numbers: i32, cost: i32) -> TensorInfo {
      let dimension_numbers_id = self.add_or_get_val(dimension_numbers);
      let cost_id = self.add_or_get_val(cost);
      let new_node = Mdl::ScatterOp([inpt.id, scatter_indices.id, updates.id, dimension_numbers_id, cost_id]);
      TensorInfo {
          id: self.rec_expr.add(new_node),
          shape: inpt.shape, // This is an example, you might want to calculate actual shape
          n_dim: inpt.n_dim,
      }
  }

  fn add_or_get_val(&mut self, val: i32) -> Id {
      match self.scalar_map.get(&val) {
          Some(id) => *id,
          None => {
              let node = Mdl::Int(val);
              let id = self.rec_expr.add(node);
              self.scalar_map.insert(val, id);
              id
          }
      }
  }

  fn shape_from_dim(&self, dims: &[i32]) -> ([i32; MAX_DIM], usize) {
      let mut shape = [0; MAX_DIM];
      for (i, dim) in dims.iter().enumerate() {
          shape[i] = *dim;
      }
      (shape, dims.len())
  }

  fn get_conv_shape(
      &self,
      input_h: i32,
      input_w: i32,
      stride_h: i32,
      stride_w: i32,
      kernel_h: i32,
      kernel_w: i32,
      padding: i32,
  ) -> (i32, i32) {
      if padding == PSAME {
          let output_h = (input_h + stride_h - 1) / stride_h;
          let output_w = (input_w + stride_w - 1) / stride_w;
          (output_h, output_w)
      } else {
          let output_h = (input_h - kernel_h) / stride_h + 1;
          let output_w = (input_w - kernel_w) / stride_w + 1;
          (output_h, output_w)
      }
  }

  // Wrapper functions for C++ side
  pub fn new_input(&mut self, dims: &[i32]) -> Box<TensorInfo> {
      Box::new(self.input(dims))
  }

  pub fn new_compare_op(&mut self, inpt_1: Box<TensorInfo>, inpt_2: Box<TensorInfo>, comparison: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.compare_op(*inpt_1, *inpt_2, comparison, cost))
  }

  pub fn new_broadcast_in_dim(&mut self, inpt: Box<TensorInfo>, dimensions: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.broadcast_in_dim(*inpt, dimensions, cost))
  }

  pub fn new_convert_op(&mut self, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.convert_op(*inpt, cost))
  }

  pub fn new_reduce_op(&mut self, inpt: Box<TensorInfo>, dimensions: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.reduce_op(*inpt, dimensions, cost))
  }

  pub fn new_reshape_op(&mut self, inpt: Box<TensorInfo>, shape: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.reshape_op(*inpt, shape, cost))
  }

  pub fn new_gather_op(&mut self, inpt: Box<TensorInfo>, start_indices: Box<TensorInfo>, dimension_numbers: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.gather_op(*inpt, *start_indices, dimension_numbers, cost))
  }

  pub fn new_select_op(&mut self, pred: Box<TensorInfo>, on_true: Box<TensorInfo>, on_false: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.select_op(*pred, *on_true, *on_false, cost))
  }

  // pub fn new_concatenate_op(&mut self, inputs: &[Box<TensorInfo>], dimension: i32, cost: i32) -> Box<TensorInfo> {
  //     let unboxed_inputs: Vec<TensorInfo> = inputs.iter().map(|x| **x).collect();
  //     Box::new(self.concatenate_op(&unboxed_inputs, dimension, cost))
  // }

  pub fn new_dot_general_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, dot_dimension_numbers: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.dot_general_op(*lhs, *rhs, dot_dimension_numbers, cost))
  }

  pub fn new_pad_op(&mut self, inpt: Box<TensorInfo>, padding_value: i32, padding_config: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.pad_op(*inpt, padding_value, padding_config, cost))
  }

  pub fn new_slice_op(&mut self, inpt: Box<TensorInfo>, start_indices: &[i32], limit_indices: &[i32], strides: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.slice_op(*inpt, start_indices, limit_indices, strides, cost))
  }

  pub fn new_transpose_op(&mut self, inpt: Box<TensorInfo>, permutation: &[i32], cost: i32) -> Box<TensorInfo> {
      Box::new(self.transpose_op(*inpt, permutation, cost))
  }

  pub fn new_mul_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.mul_op(*lhs, *rhs, cost))
  }

  pub fn new_add_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.add_op(*lhs, *rhs, cost))
  }

  pub fn new_div_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.div_op(*lhs, *rhs, cost))
  }

  pub fn new_subtract_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.subtract_op(*lhs, *rhs, cost))
  }

  pub fn new_min_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.min_op(*lhs, *rhs, cost))
  }

  pub fn new_max_op(&mut self, lhs: Box<TensorInfo>, rhs: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.max_op(*lhs, *rhs, cost))
  }

  pub fn new_neg_op(&mut self, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.neg_op(*inpt, cost))
  }

  pub fn new_tanh_op(&mut self, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.tanh_op(*inpt, cost))
  }

  pub fn new_exp_op(&mut self, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.exp_op(*inpt, cost))
  }

  pub fn new_iota_op(&mut self, inpt: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.iota_op(*inpt, cost))
  }

  pub fn new_constant_op(&mut self, value: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.constant_op(value, cost))
  }

  pub fn new_dynamic_update_slice_op(&mut self, operand: Box<TensorInfo>, update: Box<TensorInfo>, start_indices: Box<TensorInfo>, cost: i32) -> Box<TensorInfo> {
      Box::new(self.dynamic_update_slice_op(*operand, *update, *start_indices, cost))
  }

  pub fn new_dynamic_slice_op(&mut self, operand: Box<TensorInfo>, start_indices: Box<TensorInfo>, slice_sizes: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.dynamic_slice_op(*operand, *start_indices, slice_sizes, cost))
  }

  pub fn new_scatter_op(&mut self, inpt: Box<TensorInfo>, scatter_indices: Box<TensorInfo>, updates: Box<TensorInfo>, dimension_numbers: i32, cost: i32) -> Box<TensorInfo> {
      Box::new(self.scatter_op(*inpt, *scatter_indices, *updates, dimension_numbers, cost))
  }

  pub fn print_rec_expr(&self) {
      println!("{:?}", self.rec_expr)
  }

  pub fn pretty_print_rec_expr(&self, width: i32) {
      println!("{}", self.rec_expr.pretty(width as usize))
  }
}


/// Struct for generating new names for weight tensors in the model
///
/// Generates names like w1, w2...
#[derive(Default)]
pub struct NameGen {
    count_input: i32,
    count_weight: i32,
}

impl NameGen {
    pub fn new_weight_name(&mut self) -> String {
        let name = format!("w_{}", self.count_weight);
        self.count_weight += 1;
        name
    }

    pub fn new_input_name(&mut self) -> String {
        let name = format!("input_{}", self.count_input);
        self.count_input += 1;
        name
    }
}
