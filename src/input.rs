use crate::model::*;
use crate::optimize::*;
use crate::rewrites::*;
use cxx::CxxVector;
use egg::*;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::*;
use std::process::{Command, Stdio};
use std::time::*;
use std::{borrow::Borrow, collections::HashMap};

const MAX_DIM: usize = 8;

#[cxx::bridge(namespace = "tensat")]
pub mod ffi {
    enum Type {
        i32,
        f32,
    }

    struct Node {
        name: String,
        operands: Vec<usize>,
    }

    // take floats from c++ and wrap them into f32s below
    extern "Rust" {
        type Mdl;
        type CppGraphConverter;
        type TensorInfo;
        fn new_converter() -> Box<CppGraphConverter>;
        // Exposing the constructor functions with Box<TensorInfo>
        fn new_input(self: &mut CppGraphConverter, dims: &[i32]) -> Box<TensorInfo>;
        fn new_compare_op(
            self: &mut CppGraphConverter,
            inpt_1: &TensorInfo,
            inpt_2: &TensorInfo,
            comparison_direction: i32,
            comparison_type: i32,
        ) -> Box<TensorInfo>;
        fn new_broadcast_in_dim(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: &[i32],
        ) -> Box<TensorInfo>;
        fn new_convert_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output_type: i32,
        ) -> Box<TensorInfo>;
        fn new_reduce_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: &[i32],
        ) -> Box<TensorInfo>;
        fn new_reshape_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_gather_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            start_indices: &TensorInfo,
            offset_dims: &[i32],
            collapsed_slice_dims: &[i32],
            operand_batching_dims: &[i32],
            start_indices_batching_dims: &[i32],
            start_index_map: &[i32],
            index_vector_dim: i32,
            slice_sizes: &[i32],
            indices_are_sorted: i32,
        ) -> Box<TensorInfo>;
        fn new_select_op(
            self: &mut CppGraphConverter,
            pred: &TensorInfo,
            on_true: &TensorInfo,
            on_false: &TensorInfo,
        ) -> Box<TensorInfo>;
        // fn new_concatenate_op(self: &mut CppGraphConverter, inputs: &[&TensorInfo], dimension: i32) -> Box<TensorInfo>;
        fn new_dot_general_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            lhs_batching_dimensions: &[i32],
            rhs_batching_dimensions: &[i32],
            lhs_contracting_dimensions: &[i32],
            rhs_contracting_dimensions: &[i32],
            precision_config: &[i32],
        ) -> Box<TensorInfo>;
        fn new_pad_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            padding_value: i32,
            edge_padding_low: &[i32],
            edge_padding_high: &[i32],
            interior_padding: &[i32],
        ) -> Box<TensorInfo>;
        fn new_slice_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            start_indices: &[i32],
            limit_indices: &[i32],
            strides: &[i32],
        ) -> Box<TensorInfo>;
        fn new_transpose_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            permutation: &[i32],
        ) -> Box<TensorInfo>;
        fn new_mul_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_add_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_div_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_subtract_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_min_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_max_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_neg_op(self: &mut CppGraphConverter, inpt: &TensorInfo) -> Box<TensorInfo>;
        fn new_tanh_op(self: &mut CppGraphConverter, inpt: &TensorInfo) -> Box<TensorInfo>;
        fn new_exp_op(self: &mut CppGraphConverter, inpt: &TensorInfo) -> Box<TensorInfo>;
        fn new_iota_op(
            self: &mut CppGraphConverter,
            iota_dimension: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_constant_op(self: &mut CppGraphConverter) -> Box<TensorInfo>;
        fn new_dynamic_update_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            update: &TensorInfo,
            start_indices: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_dynamic_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            start_indices: &TensorInfo,
            slice_sizes: i32,
        ) -> Box<TensorInfo>;
        fn new_scatter_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            scatter_indices: &TensorInfo,
            updates: &TensorInfo,
            dimension_numbers: i32,
        ) -> Box<TensorInfo>;

        fn optimize(self: &CppGraphConverter) -> Vec<Node>;
        fn print_rec_expr(self: &CppGraphConverter);
        fn pretty_print_rec_expr(self: &CppGraphConverter, width: i32);
        // fn get_rec_expr_as_ref(self: &CppGraphConverter) -> &[Mdl];
        // fn dfs_convert(root: &Mdl, rec_expr: &[Mdl]) -> Vec<i32>;
        // fn test_cost_model(op: String) -> u64;
    }

    unsafe extern "C++" {
        include!("EqualitySaturation.h");
        type CostModel;

        fn getAddOpCost(
            &self,
            lhsDims: &[i64],
            lhsType: Type,
            rhsDims: &[i64],
            rhsType: Type,
        ) -> u64;

        fn getMulOpCost(
            &self,
            lhsDims: &[i64],
            lhsType: Type,
            rhsDims: &[i64],
            rhsType: Type,
        ) -> u64;

        fn getDivOpCost(
            &self,
            lhsDims: &[i64],
            lhsType: Type,
            rhsDims: &[i64],
            rhsType: Type,
        ) -> u64;

        fn getSubtractOpCost(
            &self,
            lhsDims: &[i64],
            lhsType: Type,
            rhsDims: &[i64],
            rhsType: Type,
        ) -> u64;
        fn newCostModel() -> UniquePtr<CostModel>;
    }
}

// Struct for storing information of a tensor. This is passed between functions
// during graph creation.
#[derive(Copy, Clone, Default)]
pub struct TensorInfo {
    /// Id into the RecExpr constructed
    pub id: Id,
    /// Shape of the tensor. We deal with tensor up to MAX_DIM dimensions
    pub shape: [i32; MAX_DIM],
    /// Number of dimensions of this tensor
    pub n_dim: usize,
}

/// Struct for converting a model specified using our Rust interface to RecExpr
///
/// The RecExpr is growed on the fly when member functions are called. Uses a
/// Hashmap to store the map of scalar nodes to their indices into the RecExpr to
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

    pub fn blackbox_1(&mut self, inpt: TensorInfo, cpp_name: String) -> TensorInfo {
        let new_node = Mdl::BlackBox_1([inpt.id]);
        let cpp_name_node = Mdl::Var(Symbol::from(cpp_name));
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn blackbox_2(
        &mut self,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
        cpp_name: String,
    ) -> TensorInfo {
        let new_node = Mdl::BlackBox_2([inpt_1.id, inpt_2.id]);
        let cpp_name_node = Mdl::Var(Symbol::from(cpp_name));
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt_1.n_dim,
        }
    }

    pub fn blackbox_3(
        &mut self,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
        inpt_3: TensorInfo,
        cpp_name: String,
    ) -> TensorInfo {
        let new_node = Mdl::BlackBox_3([inpt_1.id, inpt_2.id, inpt_3.id]);
        let cpp_name_node = Mdl::Var(Symbol::from(cpp_name));
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt_1.n_dim,
        }
    }

    pub fn blackbox_4(
        &mut self,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
        inpt_3: TensorInfo,
        inpt_4: TensorInfo,
        cpp_name: String,
    ) -> TensorInfo {
        let new_node = Mdl::BlackBox_4([inpt_1.id, inpt_2.id, inpt_3.id, inpt_4.id]);
        let cpp_name_node = Mdl::Var(Symbol::from(cpp_name));
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt_1.n_dim,
        }
    }

    pub fn blackbox_5(
        &mut self,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
        inpt_3: TensorInfo,
        inpt_4: TensorInfo,
        inpt_5: TensorInfo,
        cpp_name: String,
    ) -> TensorInfo {
        let new_node = Mdl::BlackBox_5([inpt_1.id, inpt_2.id, inpt_3.id, inpt_4.id, inpt_5.id]);
        let cpp_name_node = Mdl::Var(Symbol::from(cpp_name));
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt_1.n_dim,
        }
    }

    pub fn compare_op(
        &mut self,
        inpt_1: TensorInfo,
        inpt_2: TensorInfo,
        comparison_direction: i32,
        comparison_type: i32,
    ) -> TensorInfo {
        let comparison_direction_node = self.add_or_get_val(comparison_direction);
        let comparison_type_node = self.add_or_get_val(comparison_type);
        let new_node = Mdl::CompareOp([
            inpt_1.id,
            inpt_2.id,
            comparison_direction_node,
            comparison_type_node,
        ]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt_1.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt_1.n_dim,
        }
    }

    pub fn broadcast_in_dim(&mut self, inpt: TensorInfo, dimensions: &[i32]) -> TensorInfo {
        let dim_name = &dimensions.iter().join("_");
        let node = Mdl::Var(Symbol::from(dim_name));
        let dimensions_id = self.rec_expr.add(node);
        let new_node = Mdl::BroadcastInDimOp([inpt.id, dimensions_id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    // Weird calling convention: the result type is specified with a type annotation, and is NOT a parameter
    pub fn convert_op(&mut self, inpt: TensorInfo, output_type: i32) -> TensorInfo {
        let output_type_node = self.add_or_get_val(output_type);
        let new_node = Mdl::ConvertOp([inpt.id, output_type_node]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape,
            n_dim: inpt.n_dim,
        }
    }

    // needs to take in a variadic number of input tensors
    pub fn reduce_op(&mut self, inpt: TensorInfo, dimensions: &[i32]) -> TensorInfo {
        let dim_name = &dimensions.iter().join("_");
        let node = Mdl::Var(Symbol::from(dim_name));
        let dimensions_id = self.rec_expr.add(node);
        let new_node = Mdl::ReduceOp([inpt.id, dimensions_id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn reshape_op(&mut self, inpt: TensorInfo, shape: &[i32]) -> TensorInfo {
        let shape_name = &shape.iter().join("_");
        let node = Mdl::Var(Symbol::from(shape_name));
        let shape_id = self.rec_expr.add(node);
        let new_node = Mdl::ReshapeOp([inpt.id, shape_id]);
        let (shape_new, n_dim) = self.shape_from_dim(shape);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape_new,
            n_dim: n_dim,
        }
    }

    // https://github.com/openxla/stablehlo/blob/main/docs/spec.md#inputs-44
    // Lots of inputs, we might want to investigate posisble rewrites and based on that decide how to implement this
    pub fn gather_op(
        &mut self,
        inpt: TensorInfo,
        start_indices: TensorInfo,
        offset_dims: &[i32],
        collapsed_slice_dims: &[i32],
        operand_batching_dims: &[i32],
        start_indices_batching_dims: &[i32],
        start_index_map: &[i32],
        index_vector_dim: i32,
        slice_sizes: &[i32],
        indices_are_sorted: i32,
    ) -> TensorInfo {
        let offset_dims_name = &offset_dims.iter().join("_");
        let collapsed_slice_dims_name = &collapsed_slice_dims.iter().join("_");
        let operand_batching_dims_name = &operand_batching_dims.iter().join("_");
        let start_indices_batching_dims_name = &start_indices_batching_dims.iter().join("_");
        let start_index_map_name = &start_index_map.iter().join("_");
        let slice_sizes_name = &slice_sizes.iter().join("_");

        let offset_dims_id = self.rec_expr.add(Mdl::Var(Symbol::from(offset_dims_name)));
        let collapsed_slice_dims_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(collapsed_slice_dims_name)));
        let operand_batching_dims_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(operand_batching_dims_name)));
        let start_indices_batching_dims_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(start_indices_batching_dims_name)));
        let start_index_map_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(start_index_map_name)));
        let slice_sizes_id = self.rec_expr.add(Mdl::Var(Symbol::from(slice_sizes_name)));
        let index_vector_dim_id = self.add_or_get_val(index_vector_dim);
        let indices_are_sorted_id = self.add_or_get_val(indices_are_sorted);

        let new_node = Mdl::GatherOp([
            inpt.id,
            start_indices.id,
            offset_dims_id,
            collapsed_slice_dims_id,
            operand_batching_dims_id,
            start_indices_batching_dims_id,
            start_index_map_id,
            index_vector_dim_id,
            slice_sizes_id,
            indices_are_sorted_id,
        ]);

        // This logic is incorrect
        // let mut batch_dim_sizes = start_indices.shape.clone();
        // // if index_vector_dim < batch_dim_sizes.len() as i32 {
        // //     batch_dim_sizes.remove(index_vector_dim);
        // // }
        //
        // let mut offset_dim_sizes = slice_sizes.iter().cloned().collect::<Vec<_>>();
        // for dim in collapsed_slice_dims
        //     .iter()
        //     .chain(operand_batching_dims.iter())
        // {
        //     offset_dim_sizes[*dim as usize] = 1;
        // }
        //
        // let mut shape = Vec::new();
        // shape.extend(batch_dim_sizes);
        // shape.extend(offset_dim_sizes);
        // let (shape, n_dim) = self.shape_from_dim(*(batch_dim_sizes as [i32]));

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape,
            n_dim: inpt.n_dim,
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

    pub fn select_op(
        &mut self,
        pred: TensorInfo,
        on_true: TensorInfo,
        on_false: TensorInfo,
    ) -> TensorInfo {
        let new_node = Mdl::SelectOp([pred.id, on_true.id, on_false.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: pred.shape,
            n_dim: pred.n_dim,
        }
    }

    pub fn dot_general_op(
        &mut self,
        lhs: TensorInfo,
        rhs: TensorInfo,
        lhs_batching_dimensions: &[i32],
        rhs_batching_dimensions: &[i32],
        lhs_contracting_dimensions: &[i32],
        rhs_contracting_dimensions: &[i32],
        precision_config: &[i32],
    ) -> TensorInfo {
        // This produces ugly empty nodes when there's no batch dimension
        let lhs_batch_dim_name = &lhs_batching_dimensions.iter().join("_");
        let rhs_batch_dim_name = &rhs_batching_dimensions.iter().join("_");
        let lhs_contract_dim_name = &lhs_contracting_dimensions.iter().join("_");
        let rhs_contract_dim_name = &rhs_contracting_dimensions.iter().join("_");
        let precision_config_name = &precision_config.iter().join("_");
        let lhs_batch_dim_name_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(lhs_batch_dim_name)));
        let rhs_batch_dim_name_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(lhs_batch_dim_name)));
        let lhs_contract_dim_name_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(lhs_contract_dim_name)));
        let rhs_contract_dim_name_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(rhs_contract_dim_name)));
        let precision_config_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(precision_config_name)));
        let new_node = Mdl::DotGeneralOp([
            lhs.id,
            rhs.id,
            lhs_batch_dim_name_id,
            rhs_batch_dim_name_id,
            lhs_contract_dim_name_id,
            rhs_contract_dim_name_id,
            precision_config_id,
        ]);
        let mut shape = lhs.shape;
        shape[shape.len() - 1] = rhs.shape[rhs.shape.len() - 1];
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape,
            n_dim: lhs.n_dim,
        }
    }

    fn pad_op(
        &mut self,
        inpt: TensorInfo,
        padding_value: i32,
        edge_padding_low: &[i32],
        edge_padding_high: &[i32],
        interior_padding: &[i32],
    ) -> TensorInfo {
        let edge_padding_low_name = &edge_padding_low.iter().join("_");
        let edge_padding_high_name = &edge_padding_high.iter().join("_");
        let interior_padding_name = &interior_padding.iter().join("_");

        let edge_padding_low_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(edge_padding_low_name)));
        let edge_padding_high_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(edge_padding_high_name)));
        let interior_padding_id = self
            .rec_expr
            .add(Mdl::Var(Symbol::from(interior_padding_name)));
        let padding_value_id = self.add_or_get_val(padding_value);

        let new_node = Mdl::PadOp([
            inpt.id,
            padding_value_id,
            edge_padding_low_id,
            edge_padding_high_id,
            interior_padding_id,
        ]);

        let mut new_shape = inpt.shape.clone();
        for (i, &dim) in inpt.shape.iter().enumerate() {
            new_shape[i] = dim
                + (edge_padding_low[i])
                + (edge_padding_high[i])
                + ((dim.max(1) - 1) * (interior_padding[i]));
        }

        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: new_shape,
            n_dim: inpt.n_dim,
        }
    }

    pub fn slice_op(
        &mut self,
        inpt: TensorInfo,
        start_indices: &[i32],
        limit_indices: &[i32],
        strides: &[i32],
    ) -> TensorInfo {
        let start_indices_name = &start_indices.iter().join("_");
        let start_indices_node = Mdl::Var(Symbol::from(start_indices_name));
        let start_indices_id = self.rec_expr.add(start_indices_node);
        let limit_indices_name = &limit_indices.iter().join("_");
        let limit_indices_node = Mdl::Var(Symbol::from(limit_indices_name));
        let limit_indices_id = self.rec_expr.add(limit_indices_node);
        let strides_name = &strides.iter().join("_");
        let strides_node = Mdl::Var(Symbol::from(strides_name));
        let strides_id = self.rec_expr.add(strides_node);
        let new_node = Mdl::SliceOp([inpt.id, start_indices_id, limit_indices_id, strides_id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn transpose_op(&mut self, inpt: TensorInfo, permutation: &[i32]) -> TensorInfo {
        let permutation_name = &permutation.iter().join("_");
        let node = Mdl::Var(Symbol::from(permutation_name));
        let permutation_id = self.rec_expr.add(node);
        let new_node = Mdl::TransposeOp([inpt.id, permutation_id]);
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

    pub fn mul_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::MulOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn add_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::AddOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn div_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::DivOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn subtract_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::SubtractOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn min_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::MinOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn max_op(&mut self, lhs: TensorInfo, rhs: TensorInfo) -> TensorInfo {
        let new_node = Mdl::MaxOp([lhs.id, rhs.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: lhs.shape, // This is an example, you might want to calculate actual shape
            n_dim: lhs.n_dim,
        }
    }

    pub fn neg_op(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::NegOp([inpt.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn tanh_op(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::TanhOp([inpt.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn exp_op(&mut self, inpt: TensorInfo) -> TensorInfo {
        let new_node = Mdl::ExpOp([inpt.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: inpt.shape, // This is an example, you might want to calculate actual shape
            n_dim: inpt.n_dim,
        }
    }

    pub fn iota_op(&mut self, iota_dimension: i32, shape: &[i32]) -> TensorInfo {
        let iota_dim_id = self.add_or_get_val(iota_dimension);
        let shape_name = &shape.iter().join("_");
        let shape_id = self.rec_expr.add(Mdl::Var(Symbol::from(shape_name)));
        let new_node = Mdl::IotaOp([iota_dim_id, shape_id]);
        let (shape_new, n_dim) = self.shape_from_dim(shape);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: shape_new,
            n_dim: n_dim,
        }
    }

    pub fn constant_op(&mut self) -> TensorInfo {
        let new_node = Mdl::ConstantOp([]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: [1; MAX_DIM], // Assuming constant has a shape of [1]
            n_dim: 1,
        }
    }

    pub fn dynamic_update_slice_op(
        &mut self,
        operand: TensorInfo,
        update: TensorInfo,
        start_indices: TensorInfo,
    ) -> TensorInfo {
        let new_node = Mdl::DynamicUpdateSliceOp([operand.id, update.id, start_indices.id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: operand.shape, // This is an example, you might want to calculate actual shape
            n_dim: operand.n_dim,
        }
    }

    pub fn dynamic_slice_op(
        &mut self,
        operand: TensorInfo,
        start_indices: TensorInfo,
        slice_sizes: i32,
    ) -> TensorInfo {
        let slice_sizes_id = self.add_or_get_val(slice_sizes);
        let new_node = Mdl::DynamicSliceOp([operand.id, start_indices.id, slice_sizes_id]);
        TensorInfo {
            id: self.rec_expr.add(new_node),
            shape: operand.shape, // This is an example, you might want to calculate actual shape
            n_dim: operand.n_dim,
        }
    }

    pub fn scatter_op(
        &mut self,
        inpt: TensorInfo,
        scatter_indices: TensorInfo,
        updates: TensorInfo,
        dimension_numbers: i32,
    ) -> TensorInfo {
        let dimension_numbers_id = self.add_or_get_val(dimension_numbers);
        let new_node = Mdl::ScatterOp([
            inpt.id,
            scatter_indices.id,
            updates.id,
            dimension_numbers_id,
        ]);
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
                let node = Mdl::Num(val);
                let id = self.rec_expr.add(node);
                self.scalar_map.insert(val, id);
                id
            }
        }
    }

    fn shape_from_dim(&self, dims: &[i32]) -> ([i32; MAX_DIM], usize) {
        if (dims.len() > MAX_DIM) {
            println!("ERROR: op shape exceeds MAX_DIM! e-graph no longer valid.");
        }
        let mut shape = [0; MAX_DIM];
        for (i, dim) in dims.iter().enumerate() {
            shape[i] = *dim;
        }
        (shape, dims.len())
    }

    // Wrapper functions for C++ side
    pub fn new_input(&mut self, dims: &[i32]) -> Box<TensorInfo> {
        Box::new(self.input(dims))
    }

    pub fn new_compare_op(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        comparison_direction: i32,
        comparison_type: i32,
    ) -> Box<TensorInfo> {
        Box::new(self.compare_op(*inpt_1, *inpt_2, comparison_direction, comparison_type))
    }

    pub fn new_broadcast_in_dim(
        &mut self,
        inpt: &TensorInfo,
        dimensions: &[i32],
    ) -> Box<TensorInfo> {
        Box::new(self.broadcast_in_dim(*inpt, dimensions))
    }

    pub fn new_convert_op(&mut self, inpt: &TensorInfo, output_type: i32) -> Box<TensorInfo> {
        Box::new(self.convert_op(*inpt, output_type))
    }

    pub fn new_reduce_op(&mut self, inpt: &TensorInfo, dimensions: &[i32]) -> Box<TensorInfo> {
        Box::new(self.reduce_op(*inpt, dimensions))
    }

    pub fn new_reshape_op(&mut self, inpt: &TensorInfo, shape: &[i32]) -> Box<TensorInfo> {
        Box::new(self.reshape_op(*inpt, shape))
    }

    fn new_gather_op(
        self: &mut CppGraphConverter,
        inpt: &TensorInfo,
        start_indices: &TensorInfo,
        offset_dims: &[i32],
        collapsed_slice_dims: &[i32],
        operand_batching_dims: &[i32],
        start_indices_batching_dims: &[i32],
        start_index_map: &[i32],
        index_vector_dim: i32,
        slice_sizes: &[i32],
        indices_are_sorted: i32,
    ) -> Box<TensorInfo> {
        Box::new(self.gather_op(
            *inpt,
            *start_indices,
            offset_dims,
            collapsed_slice_dims,
            operand_batching_dims,
            start_indices_batching_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
            indices_are_sorted,
        ))
    }

    pub fn new_select_op(
        &mut self,
        pred: &TensorInfo,
        on_true: &TensorInfo,
        on_false: &TensorInfo,
    ) -> Box<TensorInfo> {
        Box::new(self.select_op(*pred, *on_true, *on_false))
    }

    // pub fn new_concatenate_op(&mut self, inputs: &[&TensorInfo], dimension: i32, cost: i32) -> Box<TensorInfo> {
    //     let unboxed_inputs: Vec<TensorInfo> = inputs.iter().map(|x| **x).collect();
    //     Box::new(self.concatenate_op(&unboxed_inputs, dimension, cost))
    // }

    pub fn new_dot_general_op(
        self: &mut CppGraphConverter,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        lhs_batching_dimensions: &[i32],
        rhs_batching_dimensions: &[i32],
        lhs_contracting_dimensions: &[i32],
        rhs_contracting_dimensions: &[i32],
        precision_config: &[i32],
    ) -> Box<TensorInfo> {
        Box::new(self.dot_general_op(
            *lhs,
            *rhs,
            lhs_batching_dimensions,
            rhs_batching_dimensions,
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            precision_config,
        ))
    }

    pub fn new_pad_op(
        self: &mut CppGraphConverter,
        inpt: &TensorInfo,
        padding_value: i32,
        edge_padding_low: &[i32],
        edge_padding_high: &[i32],
        interior_padding: &[i32],
    ) -> Box<TensorInfo> {
        Box::new(self.pad_op(
            *inpt,
            padding_value,
            edge_padding_low,
            edge_padding_high,
            interior_padding,
        ))
    }

    pub fn new_slice_op(
        &mut self,
        inpt: &TensorInfo,
        start_indices: &[i32],
        limit_indices: &[i32],
        strides: &[i32],
    ) -> Box<TensorInfo> {
        Box::new(self.slice_op(*inpt, start_indices, limit_indices, strides))
    }

    pub fn new_transpose_op(&mut self, inpt: &TensorInfo, permutation: &[i32]) -> Box<TensorInfo> {
        Box::new(self.transpose_op(*inpt, permutation))
    }

    pub fn new_mul_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.mul_op(*lhs, *rhs))
    }

    pub fn new_add_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.add_op(*lhs, *rhs))
    }

    pub fn new_div_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.div_op(*lhs, *rhs))
    }

    pub fn new_subtract_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.subtract_op(*lhs, *rhs))
    }

    pub fn new_min_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.min_op(*lhs, *rhs))
    }

    pub fn new_max_op(&mut self, lhs: &TensorInfo, rhs: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.max_op(*lhs, *rhs))
    }

    pub fn new_neg_op(&mut self, inpt: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.neg_op(*inpt))
    }

    pub fn new_tanh_op(&mut self, inpt: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.tanh_op(*inpt))
    }

    pub fn new_exp_op(&mut self, inpt: &TensorInfo) -> Box<TensorInfo> {
        Box::new(self.exp_op(*inpt))
    }

    pub fn new_iota_op(&mut self, iota_dimension: i32, shape: &[i32]) -> Box<TensorInfo> {
        Box::new(self.iota_op(iota_dimension, shape))
    }

    pub fn new_constant_op(&mut self) -> Box<TensorInfo> {
        Box::new(self.constant_op())
    }

    pub fn new_dynamic_update_slice_op(
        &mut self,
        operand: &TensorInfo,
        update: &TensorInfo,
        start_indices: &TensorInfo,
    ) -> Box<TensorInfo> {
        Box::new(self.dynamic_update_slice_op(*operand, *update, *start_indices))
    }

    pub fn new_dynamic_slice_op(
        &mut self,
        operand: &TensorInfo,
        start_indices: &TensorInfo,
        slice_sizes: i32,
    ) -> Box<TensorInfo> {
        Box::new(self.dynamic_slice_op(*operand, *start_indices, slice_sizes))
    }

    pub fn new_scatter_op(
        &mut self,
        inpt: &TensorInfo,
        scatter_indices: &TensorInfo,
        updates: &TensorInfo,
        dimension_numbers: i32,
    ) -> Box<TensorInfo> {
        Box::new(self.scatter_op(*inpt, *scatter_indices, *updates, dimension_numbers))
    }

    pub fn print_rec_expr(&self) {
        println!("{:?}", self.rec_expr)
    }

    pub fn pretty_print_rec_expr(&self, width: i32) {
        println!("{}", self.rec_expr.pretty(width as usize))
    }

    pub fn get_rec_expr_as_ref(&self) -> &[Mdl] {
        return &self.rec_expr.as_ref();
    }

    fn convert_to_node(&self) -> Vec<ffi::Node> {
        let mut res: Vec<ffi::Node> = Vec::new();

        let index = |id: Id| usize::from(id);  // TODO: this is probably wrong
        let convert = |operands: &[Id]| operands.iter().map(|id: &Id| index(*id)).collect::<Vec<usize>>();
        let new_node = |name: &str, operands: &[Id]| 
            ffi::Node { 
                name: name.to_string(), 
                operands: convert(operands),
            };
        
        let rec_expr_ref = self.get_rec_expr_as_ref();

        for mdl in rec_expr_ref.iter() {
            let node = match mdl {
                Mdl::Var(label) => {
                    ffi::Node {
                        name: label.to_string(),
                        operands: vec![]
                    }
                }
                // TODO: More clever pattern matching
                Mdl::Input(ops) => new_node("Input", ops),
                Mdl::ConstantOp(ops) => new_node("ConstantOp", ops),
                Mdl::ReshapeOp(ops) => new_node("ReshapeOp", ops),
                Mdl::DotGeneralOp(ops) => new_node("DotGeneralOp", ops),
                Mdl::TransposeOp(ops) => new_node("TransposeOp", ops),
                Mdl::MulOp(ops) => new_node("MulOp", ops),
                Mdl::AddOp(ops) => new_node("AddOp", ops),
                Mdl::DivOp(ops) => new_node("DivOp", ops),
                Mdl::SubtractOp(ops) => new_node("SubtractOp", ops),
                Mdl::MinOp(ops) => new_node("MinOp", ops),
                Mdl::MaxOp(ops) => new_node("MaxOp", ops),
                Mdl::NegOp(ops) => new_node("NegOp", ops),
                Mdl::TanhOp(ops) => new_node("TanhOp", ops),
                Mdl::ExpOp(ops) => new_node("ExpOp", ops),
                Mdl::IotaOp(ops) => new_node("IotaOp", ops),
                _ => unimplemented!()
            };   

            res.push(node);
        }
        
        res
    }

    pub fn optimize(&self) -> Vec<ffi::Node>{
        let start = &self.rec_expr;

        // Configuration
        let n_sec = 10; // seconds for timeout
        let use_multi = false; // whether to use multi patterns
        let no_cycle = false; // is our graph by definition acyclic?
        let filter_after = false; // vanilla filtering or efficient filtering
        let rule_file =
            "/Users/vohraary/enz/Enzyme-JAX/src/enzyme_ad/jax/deps/tensat/converted.txt";
        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
        let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
        let do_filter_after = no_cycle && filter_after;
        let rules = rules_from_str(split_rules, do_filter_after);

        let iter_multi = 2;
        let node_multi = 30000;
        let multi_rules: Vec<(&str, bool)> = PRE_DEFINED_MULTI
            .iter()
            .map(|&x| (x, /*symmetric=*/ false))
            .collect();
        let mut multi_patterns = MultiPatterns::with_rules(
            multi_rules,
            no_cycle,
            iter_multi,
            filter_after,
            node_multi,
            n_sec,
        );

        let time_limit_sec = Duration::new(n_sec, 0);
        let iter_limit = 10000;
        let node_limit = 50000;
        let runner = Runner::<Mdl, TensorAnalysis, ()>::default()
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start)
            .with_hook(move |runner| multi_patterns.run_one(runner));
        let start_time = Instant::now();
        let mut runner = runner.run(&rules[..]);
        if do_filter_after {
            // Do cycle removal after the final iteration
            remove_cycle_by_order(&mut runner);
        }
        let sat_duration = start_time.elapsed();
        let num_iter_sat = runner.iterations.len() - 1;

        println!("Runner complete!");
        println!("  Nodes: {}", runner.egraph.total_size());
        println!("  Classes: {}", runner.egraph.number_of_classes());
        println!("  Stopped: {:?}", runner.stop_reason.unwrap());
        println!("  Time taken: {:?}", sat_duration);
        println!("  Number of iterations: {:?}", num_iter_sat);

        let (num_enodes, num_classes, avg_nodes_per_class, num_edges, num_programs) =
            get_stats(&runner.egraph);
        println!("  Average nodes per class: {}", avg_nodes_per_class);
        println!("  Number of edges: {}", num_edges);
        println!("  Number of programs: {}", num_programs);

        let (egraph, root) = (runner.egraph, runner.roots[0]);
        let cost_model = CostModel::new();
        let (best, ext_secs) = extract_by_ilp(&egraph, root, &cost_model);
        println!("{}", best);

        self.convert_to_node()
    }
}

/* 
fn dfs_convert(root: &Mdl, rec_expr: &[Mdl]) -> Vec<i32> {
    match root {
        Mdl::Var(label) => {
            let label_str = label.as_str();
            let mut res = if let Some(start_idx) = label_str.find("input_") {
                // This is an input variable. Note that this also has a shape
                // TODO: we need to make the input numbers correspond with the llvm::BlockArgument
                // identifier numbers
                let start = start_idx + 6; // Skip the "input_" part
                if let Some(end_idx) = label_str[start..].find('@') {
                    let end = start + end_idx;
                    let integer_str = &label_str[start..end];
                    vec![integer_str.parse().unwrap()]
                } else {
                    // Couldn't find '@' after "input_", handle this case
                    vec![]
                }
            } else {
                // This is a shape
                label_str
                    .split('_')
                    .map(|s| s.parse::<i32>().unwrap())
                    .collect()
            };
            res.push(1); // hacky enum
            res
        }
        Mdl::Input([id]) => {
            // This is the Id of a Var
            let size: usize = (*id).into();
            let mut res = dfs_convert(&rec_expr[size], rec_expr);
            res.push(2); // hacky enum
            res
        }
        // Mdl::ConstantOp([]) => 0.0,
        // Mdl::ReshapeOp([input, shape]) => 0.0,
        // Mdl::DotGeneralOp(
        //     [lhs, rhs, lhs_batch_dim, rhs_batch_dim, lhs_contract_dim, rhs_contract_dim, precision_config],
        // ) => 0.0,
        // Mdl::TransposeOp([input, permutation]) => 0.0,
        // Mdl::MulOp([lhs, rhs]) =>
        // Mdl::AddOp([lhs, rhs]) =>
        // Mdl::DivOp([lhs, rhs]) =>
        // Mdl::SubtractOp([lhs, rhs]) =>
        // Mdl::MinOp([lhs, rhs]) => 0.0,
        // Mdl::MaxOp([lhs, rhs]) => 0.0,
        // Mdl::NegOp([input]) => 0.0,
        // Mdl::TanhOp([input]) => 0.0,
        // Mdl::ExpOp([input]) => 0.0,
        // Mdl::IotaOp([iota_dimension, shape]) => 0.0,
        _ => vec![0],
    }
}

*/

fn extract_by_ilp(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32) {
    // Prepare data for ILP formulation, save to json
    let (m_id_map, e_m, h_i, cost_i, g_i, root_m, i_to_nodes, blacklist_i) =
        prep_ilp_data(egraph, root, cost_model);

    let data = json!({
        "e_m": e_m,
        "h_i": h_i,
        "cost_i": cost_i,
        "g_i": g_i,
        "root_m": root_m,
        "blacklist_i": blacklist_i,
    });
    let data_str = serde_json::to_string(&data).expect("Fail to convert json to string");
    create_dir_all("./tmp");
    write("./tmp/ilp_data.json", data_str).expect("Unable to write file");

    // Call python script to run ILP
    let order_var_int = false;
    let class_constraint = false;
    let no_order = true;
    let mut arg_vec = vec!["extractor/extract.py"];
    if order_var_int {
        arg_vec.push("--order_var_int");
    }
    if class_constraint {
        arg_vec.push("--eclass_constraint");
    }
    if no_order {
        arg_vec.push("--no_order");
    }
    let time_lim = "1000";
    let num_thread = "1";
    arg_vec.push("--time_lim_sec");
    arg_vec.push(time_lim);
    arg_vec.push("--num_thread");
    arg_vec.push(num_thread);
    let child = Command::new("python")
        .args(&arg_vec)
        .spawn()
        .expect("failed to execute child");
    let output = child.wait_with_output().expect("failed to get output");

    if output.status.success() {
        // Read back solved results, construct optimized graph
        let solved_str = read_to_string("./tmp/solved.json")
            .expect("Something went wrong reading the solved file");
        let solved_data: SolvedResults =
            serde_json::from_str(&solved_str).expect("JSON was not well-formatted");

        let mut node_picked: HashMap<Id, Mdl> = HashMap::new();
        for (i, x_i) in solved_data.solved_x.iter().enumerate() {
            if *x_i == 1 {
                let eclass_id = m_id_map[g_i[i]];
                if node_picked.contains_key(&eclass_id) {
                    println!("Duplicate node in eclass");
                    println!("{}", node_picked.get(&eclass_id).unwrap().display_op());
                    println!("{}", i_to_nodes[i].display_op());
                    continue;
                }
                //assert!(!node_picked.contains_key(&eclass_id));
                node_picked.insert(eclass_id, i_to_nodes[i].clone());
            }
        }

        let mut expr = RecExpr::default();
        let mut added_memo: HashMap<Id, Id> = Default::default();
        let _ = construct_best_rec(&node_picked, root, &mut added_memo, egraph, &mut expr);
        (expr, solved_data.time)
    } else {
        panic!("Python script failed");
    }
}

// this is copied from main.rs
fn get_stats(egraph: &EGraph<Mdl, TensorAnalysis>) -> (usize, usize, f32, usize, f32) {
    let num_enodes = egraph.total_size();
    let num_classes = egraph.number_of_classes();
    let avg_nodes_per_class = num_enodes as f32 / (num_classes as f32);
    let num_edges = egraph
        .classes()
        .fold(0, |acc, c| c.iter().fold(0, |sum, n| n.len() + sum) + acc);
    let num_programs = egraph
        .classes()
        .fold(0.0, |acc, c| acc + (c.len() as f32).log2());
    (
        num_enodes,
        num_classes,
        avg_nodes_per_class,
        num_edges,
        num_programs,
    )
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
