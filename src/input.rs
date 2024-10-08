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

#[cxx::bridge(namespace = "tensat")]
pub mod ffi {
    enum Type {
        i32,
        f32,
    }

    enum Ops {
        Input,
        CompareOp,
        BroadcastInDimOp,
        ConvertOp,
        ReduceOp,
        ReshapeOp,
        GatherOp,
        SelectOp,
        ConcatenateOp,
        DotGeneralOp,
        PadOp,
        SliceOp,
        TransposeOp,
        MulOp,
        AddOp,
        DivOp,
        SubtractOp,
        MinOp,
        MaxOp,
        NegOp,
        TanhOp,
        ExpOp,
        IotaOp,
        // ConstantOp,
        DynamicUpdateSliceOp,
        DynamicSliceOp,
        ScatterOp,
        BlackBoxOp,
        ReturnOp,
    }

    struct Node {
        name: String,
        label: String,
        operands: Vec<i32>,
    }

    // CXX won't let me construct a Vec<Vec<i32>>, so we use Vec<ffi::Shape> instead
    // TODO: We should replace all the &[i32]s we see in Rust ffi function arguments
    // to Vec<Shape> or similar. rust::Slice in CXX is quite error prone, because
    // a common pattern is to create a std::vector then create a slice out of it,
    // but the data is easily corrupted by the vector going out of scope.
    #[derive(Debug)]
    struct Shape {
        shape: Vec<i64>,
    }

    // take floats from c++ and wrap them into f32s below
    extern "Rust" {
        type Mdl;
        type CppGraphConverter;
        type TensorData;
        type TensorInfo;
        fn new_converter() -> Box<CppGraphConverter>;
        // Exposing the constructor functions with Box<TensorInfo>
        fn new_input(
            self: &mut CppGraphConverter,
            block_arg_number: i32,
            dims: &[i32],
        ) -> Box<TensorInfo>;
        fn new_index(
            self: &mut CppGraphConverter,
            index: i32,
            inpt: &TensorInfo,
        ) -> Box<TensorInfo>;
        fn new_compare_op(
            self: &mut CppGraphConverter,
            inpt_1: &TensorInfo,
            inpt_2: &TensorInfo,
            comparison_direction: i32,
            comparison_type: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_broadcast_in_dim(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: &[i32],
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_convert_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            output_type: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_reduce_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            dimensions: &[i32],
            shapes: &Vec<Shape>,
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
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_select_op(
            self: &mut CppGraphConverter,
            pred: &TensorInfo,
            on_true: &TensorInfo,
            on_false: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_concatenate_op(
            self: &mut CppGraphConverter,
            inputs: &[*mut TensorInfo],
            dimension: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_dot_general_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            lhs_batching_dimensions: &[i32],
            rhs_batching_dimensions: &[i32],
            lhs_contracting_dimensions: &[i32],
            rhs_contracting_dimensions: &[i32],
            precision_config: &[i32],
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_pad_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            padding_value: &TensorInfo,
            edge_padding_low: &[i32],
            edge_padding_high: &[i32],
            interior_padding: &[i32],
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_slice_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            start_indices: &[i32],
            limit_indices: &[i32],
            strides: &[i32],
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_transpose_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            permutation: &[i32],
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_mul_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_add_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_div_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_subtract_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_min_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_max_op(
            self: &mut CppGraphConverter,
            lhs: &TensorInfo,
            rhs: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_neg_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_tanh_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_exp_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_iota_op(
            self: &mut CppGraphConverter,
            iota_dimension: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        // fn new_constant_op(self: &mut CppGraphConverter, shape: &[i32]) -> Box<TensorInfo>;
        fn new_dynamic_update_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            update: &TensorInfo,
            start_indices: &TensorInfo,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_dynamic_slice_op(
            self: &mut CppGraphConverter,
            operand: &TensorInfo,
            start_indices: &TensorInfo,
            slice_sizes: i32,
            shape: &[i32],
        ) -> Box<TensorInfo>;
        fn new_scatter_op(
            self: &mut CppGraphConverter,
            inpt: &TensorInfo,
            scatter_indices: &TensorInfo,
            updates: &TensorInfo,
            dimension_numbers: i32,
            shapes: &Vec<Shape>,
        ) -> Box<TensorInfo>;
        fn new_blackbox_op(
            self: &mut CppGraphConverter,
            inpts: &[*mut TensorInfo],
            cpp_num: i32,
            shapes: &Vec<Shape>,
        ) -> Box<TensorInfo>;
        fn new_return_op(
            self: &mut CppGraphConverter,
            inpts: &[*mut TensorInfo],
        ) -> Box<TensorInfo>;
        fn optimize(self: &CppGraphConverter) -> Vec<Node>;
        fn print_rec_expr(self: &CppGraphConverter);
        fn pretty_print_rec_expr(self: &CppGraphConverter, width: i32);
    }

    unsafe extern "C++" {
        fn get_cost(
            op: Ops,
            operand_dims: Vec<Shape>,
            operands_types: Vec<Type>,
            other_vector_args: Vec<Shape>, // These are not shapes..
            int_args: Vec<i64>,
        ) -> u64;
    }

    unsafe extern "C++" {
        include!("EqualitySaturation.h");

        fn get_shape(
            op: Ops,
            operand_dims: Vec<Shape>,
            operands_types: Vec<Type>,
            other_vector_args: Vec<Shape>, // These are not shapes..
            int_args: Vec<i64>,
        ) -> Vec<Shape>;
    }
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
    blackbox_cpp_num_to_tensorinfo: HashMap<i32, TensorInfo>,
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

    fn vec_node(&mut self, seq: &[i32]) -> Id {
        let vec: Vec<Id> = seq.iter().map(|n| self.add_or_get_val(*n)).collect();
        let node = Mdl::Vec(vec);
        let id = self.rec_expr.add(node);
        id
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

    fn single_shape_vec(&self, vec: &[i32]) -> Vec<ffi::Shape> {
        vec![ffi::Shape {
            shape: vec.to_vec().iter().map(|x| *x as i64).collect(),
        }]
    }

    fn shape_from_dim(&self, dims: &Vec<ffi::Shape>) -> (Vec<[i32; MAX_DIM]>, Vec<usize>) {
        let mut shapes: Vec<[i32; 8]> = vec![];
        let mut n_dims: Vec<usize> = vec![];
        for dims in dims.iter() {
            let dims = &dims.shape;
            if (dims.len() > MAX_DIM) {
                println!("ERROR: op shape exceeds MAX_DIM! e-graph no longer valid.");
            }
            let mut shape = [0; MAX_DIM];
            for (i, dim) in dims.iter().enumerate() {
                shape[i] = *dim as i32;
            }
            shapes.push(shape);
            n_dims.push(dims.len())
        }
        (shapes, n_dims)
    }

    // Wrapper functions for C++ side
    pub fn new_input(&mut self, block_arg_number: i32, shape: &[i32]) -> Box<TensorInfo> {
        // Check if the shape array is empty and replace it with a shape of 0 if needed
        let shape = if shape.is_empty() { &[0] } else { shape };
        let name = format!("input_{}", block_arg_number) + "@" + &shape.iter().join("_");
        let node = Mdl::Var(Symbol::from(name));
        let name_id = self.rec_expr.add(node);
        let block_arg_node_id = self.add_or_get_val(block_arg_number);
        let new_node = Mdl::Input([name_id, block_arg_node_id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_index(&mut self, index: i32, inpt: &TensorInfo) -> Box<TensorInfo> {
        let index_num_node = self.add_or_get_val(index);
        let new_node = Mdl::Index([index_num_node, inpt.id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes: vec![inpt.tensor_data.shapes[index as usize]],
                n_dims: vec![inpt.tensor_data.n_dims[index as usize]],
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_compare_op(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        comparison_direction: i32,
        comparison_type: i32,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let comparison_direction_node = self.add_or_get_val(comparison_direction);
        let comparison_type_node = self.add_or_get_val(comparison_type);
        let new_node = Mdl::CompareOp([
            inpt_1.id,
            inpt_2.id,
            comparison_direction_node,
            comparison_type_node,
        ]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_broadcast_in_dim(
        &mut self,
        inpt: &TensorInfo,
        dimensions: &[i32],
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let dimensions_id = self.vec_node(dimensions);
        let new_node = Mdl::BroadcastInDimOp([inpt.id, dimensions_id]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_convert_op(
        &mut self,
        inpt: &TensorInfo,
        output_type: i32,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let output_type_node = self.add_or_get_val(output_type);
        let new_node = Mdl::ConvertOp([inpt.id, output_type_node]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_reduce_op(
        &mut self,
        inpt: &TensorInfo,
        dimensions: &[i32],
        shapes: &Vec<ffi::Shape>,
    ) -> Box<TensorInfo> {
        let dimensions_id = self.vec_node(dimensions);
        let new_node = Mdl::ReduceOp([inpt.id, dimensions_id]);
        let (shapes, n_dims) = self.shape_from_dim(shapes);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_reshape_op(&mut self, inpt: &TensorInfo, shape: &[i32]) -> Box<TensorInfo> {
        let shape_id = self.vec_node(shape);
        let new_node = Mdl::ReshapeOp([inpt.id, shape_id]);
        let (shapes_new, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes: shapes_new,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
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
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let offset_dims_id = self.vec_node(offset_dims);
        let collapsed_slice_dims_id = self.vec_node(collapsed_slice_dims);
        let operand_batching_dims_id = self.vec_node(operand_batching_dims);
        let start_indices_batching_dims_id = self.vec_node(start_indices_batching_dims);
        let start_index_map_id = self.vec_node(start_index_map);
        let slice_sizes_id = self.vec_node(slice_sizes);
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

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_select_op(
        &mut self,
        pred: &TensorInfo,
        on_true: &TensorInfo,
        on_false: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::SelectOp([pred.id, on_true.id, on_false.id]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_concatenate_op(
        &mut self,
        inputs: &[*mut TensorInfo],
        dimension: i32,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let tensor_infos: Vec<&TensorInfo> = inputs.iter().map(|&ptr| unsafe { &*ptr }).collect();
        let inputs_node = Mdl::Vec(tensor_infos.iter().map(|i| i.id).collect());
        let inputs_id = self.rec_expr.add(inputs_node);
        let dimension_id = self.add_or_get_val(dimension);
        let new_node = Mdl::ConcatenateOp([inputs_id, dimension_id]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_dot_general_op(
        self: &mut CppGraphConverter,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        lhs_batching_dimensions: &[i32],
        rhs_batching_dimensions: &[i32],
        lhs_contracting_dimensions: &[i32],
        rhs_contracting_dimensions: &[i32],
        precision_config: &[i32],
        shape: &[i32],
    ) -> Box<TensorInfo> {
        // This produces ugly empty nodes when there's no batch dimension
        let lhs_batch_dim_name_id = self.vec_node(lhs_batching_dimensions);
        let rhs_batch_dim_name_id = self.vec_node(rhs_batching_dimensions);
        let lhs_contract_dim_name_id = self.vec_node(lhs_contracting_dimensions);
        let rhs_contract_dim_name_id = self.vec_node(rhs_contracting_dimensions);
        let precision_config_id = self.vec_node(precision_config);

        let new_node = Mdl::DotGeneralOp([
            lhs.id,
            rhs.id,
            lhs_batch_dim_name_id,
            rhs_batch_dim_name_id,
            lhs_contract_dim_name_id,
            rhs_contract_dim_name_id,
            precision_config_id,
        ]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_pad_op(
        self: &mut CppGraphConverter,
        inpt: &TensorInfo,
        padding_value: &TensorInfo,
        edge_padding_low: &[i32],
        edge_padding_high: &[i32],
        interior_padding: &[i32],
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let edge_padding_low_id = self.vec_node(edge_padding_low);
        let edge_padding_high_id = self.vec_node(edge_padding_high);
        let interior_padding_id = self.vec_node(interior_padding);

        let new_node = Mdl::PadOp([
            inpt.id,
            padding_value.id,
            edge_padding_low_id,
            edge_padding_high_id,
            interior_padding_id,
        ]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));

        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_slice_op(
        &mut self,
        inpt: &TensorInfo,
        start_indices: &[i32],
        limit_indices: &[i32],
        strides: &[i32],
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let start_indices_id = self.vec_node(start_indices);
        let limit_indices_id = self.vec_node(limit_indices);
        let strides_id = self.vec_node(strides);
        let new_node = Mdl::SliceOp([inpt.id, start_indices_id, limit_indices_id, strides_id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_transpose_op(
        &mut self,
        inpt: &TensorInfo,
        permutation: &[i32],
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let permutation_id = self.vec_node(permutation);
        let new_node = Mdl::TransposeOp([inpt.id, permutation_id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_mul_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MulOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_add_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::AddOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_div_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::DivOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_subtract_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::SubtractOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_min_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MinOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_max_op(
        &mut self,
        lhs: &TensorInfo,
        rhs: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::MaxOp([lhs.id, rhs.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_neg_op(&mut self, inpt: &TensorInfo, shape: &[i32]) -> Box<TensorInfo> {
        let new_node = Mdl::NegOp([inpt.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_tanh_op(&mut self, inpt: &TensorInfo, shape: &[i32]) -> Box<TensorInfo> {
        let new_node = Mdl::TanhOp([inpt.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_exp_op(&mut self, inpt: &TensorInfo, shape: &[i32]) -> Box<TensorInfo> {
        let new_node = Mdl::ExpOp([inpt.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_iota_op(&mut self, iota_dimension: i32, shape: &[i32]) -> Box<TensorInfo> {
        let iota_dim_id = self.add_or_get_val(iota_dimension);
        let shape_id = self.vec_node(shape);
        let new_node = Mdl::IotaOp([iota_dim_id, shape_id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_dynamic_update_slice_op(
        &mut self,
        operand: &TensorInfo,
        update: &TensorInfo,
        start_indices: &TensorInfo,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let new_node = Mdl::DynamicUpdateSliceOp([operand.id, update.id, start_indices.id]);
        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_dynamic_slice_op(
        &mut self,
        operand: &TensorInfo,
        start_indices: &TensorInfo,
        slice_sizes: i32,
        shape: &[i32],
    ) -> Box<TensorInfo> {
        let slice_sizes_id = self.add_or_get_val(slice_sizes);
        let new_node = Mdl::DynamicSliceOp([operand.id, start_indices.id, slice_sizes_id]);

        let (shapes, n_dims) = self.shape_from_dim(&self.single_shape_vec(shape));
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_scatter_op(
        &mut self,
        inpt: &TensorInfo,
        scatter_indices: &TensorInfo,
        updates: &TensorInfo,
        dimension_numbers: i32,
        shapes: &Vec<ffi::Shape>,
    ) -> Box<TensorInfo> {
        let dimension_numbers_id = self.add_or_get_val(dimension_numbers);
        let new_node = Mdl::ScatterOp([
            inpt.id,
            scatter_indices.id,
            updates.id,
            dimension_numbers_id,
        ]);
        let (shapes, n_dims) = self.shape_from_dim(shapes);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn new_blackbox_op(
        &mut self,
        inpts: &[*mut TensorInfo],
        cpp_num: i32,
        shapes: &Vec<ffi::Shape>,
    ) -> Box<TensorInfo> {
        let tensor_infos: Vec<&TensorInfo> = inpts.iter().map(|&ptr| unsafe { &*ptr }).collect();
        let cpp_num_node = self.add_or_get_val(cpp_num);
        let mut ids: Vec<Id> = tensor_infos.iter().map(|inpt| inpt.id).collect();
        ids.push(cpp_num_node);

        // Convert the vector of Ids to a boxed slice and create the BlackBox node
        let new_node = Mdl::BlackBox(ids.into_boxed_slice());

        let (shapes, n_dims) = self.shape_from_dim(shapes);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes,
                n_dims,
                name: None,
            },
        };
        self.blackbox_cpp_num_to_tensorinfo
            .insert(cpp_num, res.clone());
        Box::new(res)
    }

    pub fn new_return_op(&mut self, inpts: &[*mut TensorInfo]) -> Box<TensorInfo> {
        let tensor_infos: Vec<&TensorInfo> = inpts.iter().map(|&ptr| unsafe { &*ptr }).collect();
        let inputs_node = Mdl::Vec(tensor_infos.iter().map(|i| i.id).collect());
        let inputs_id = self.rec_expr.add(inputs_node);
        let new_node = Mdl::ReturnOp([inputs_id]);
        let res = TensorInfo {
            id: self.rec_expr.add(new_node),
            tensor_data: TensorData {
                shapes: vec![],
                n_dims: vec![],
                name: None,
            },
        };
        Box::new(res)
    }

    pub fn print_rec_expr(&self) {
        println!("{:?}", self.rec_expr)
    }

    pub fn pretty_print_rec_expr(&self, width: i32) {
        println!("{}", self.rec_expr.pretty(width as usize))
    }

    fn convert_to_node(&self, rec_expr: RecExpr<Mdl>) -> Vec<ffi::Node> {
        let mut res: Vec<ffi::Node> = Vec::new();

        let index = |id: Id| (usize::from(id) as i32); // TODO: this is probably wrong
        let convert = |operands: &[Id]| {
            operands
                .iter()
                .map(|id: &Id| index(*id))
                .collect::<Vec<i32>>()
        };
        let new_node = |name: &str, operands: &[Id]| ffi::Node {
            name: name.to_string(),
            label: "".to_string(),
            operands: convert(operands),
        };

        let rec_expr_ref = rec_expr.as_ref();

        for mdl in rec_expr_ref.iter() {
            let node = match mdl {
                Mdl::Var(label) => ffi::Node {
                    name: "Var".to_string(),
                    label: label.to_string(),
                    operands: vec![],
                },
                Mdl::Num(num) => ffi::Node {
                    name: "Num".to_string(),
                    label: "".to_string(),
                    operands: vec![*num],
                },
                // TODO: More clever pattern matching
                Mdl::Vec(ops) => new_node("Vec", ops),
                Mdl::Input(ops) => new_node("Input", ops),
                Mdl::Index(ops) => new_node("Index", ops),
                // Mdl::ConstantOp(ops) => new_node("ConstantOp", ops),
                Mdl::ReshapeOp(ops) => new_node("ReshapeOp", ops),
                Mdl::ConcatenateOp(ops) => new_node("ConcatenateOp", ops),
                Mdl::DotGeneralOp(ops) => new_node("DotGeneralOp", ops),
                Mdl::SliceOp(ops) => new_node("SliceOp", ops),
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
                Mdl::PadOp(ops) => new_node("PadOp", ops),
                Mdl::ReturnOp(ops) => new_node("ReturnOp", ops),
                Mdl::BlackBox(ops) => new_node("blackbox", ops),
                _ => unimplemented!(),
            };

            res.push(node);
        }

        res
    }

    pub fn optimize<'a>(&'a self) -> Vec<ffi::Node> {
        let start = &self.rec_expr;

        // Configuration
        let n_sec = 30; // seconds for timeout
        let use_multi = false; // whether to use multi patterns
        let no_cycle = true; // is our graph by definition acyclic?
        let filter_after = false; // vanilla filtering or efficient filtering
        let iter_limit = 10000;
        let node_limit = 5000000; // max nodes in e-graph

        let path = std::env::current_dir().unwrap();
        println!("The current directory is {}", path.display());
        let rule_file = "src/enzyme_ad/jax/deps/tensat/converted.txt";

        let learned_rules =
            read_to_string(rule_file).expect("Something went wrong reading the rule file");
        let time_limit_sec = Duration::new(n_sec, 0);
        let pre_defined_rules = PRE_DEFINED_RULES.iter().map(|&x| x);
        let split_rules: Vec<&str> = learned_rules.split("\n").chain(pre_defined_rules).collect();
        let do_filter_after = no_cycle && filter_after;
        let analysis = TensorAnalysis::new(&self.blackbox_cpp_num_to_tensorinfo);
        let runner = Runner::<Mdl, TensorAnalysis, ()>::new(analysis)
            .with_node_limit(node_limit)
            .with_time_limit(time_limit_sec)
            .with_iter_limit(iter_limit)
            .with_expr(&start);
        // .with_hook(move |runner| multi_patterns.run_one(runner));
        let mut rules = rules_from_str(split_rules, do_filter_after);

        let mut custom_rules: Vec<Rewrite<Mdl, TensorAnalysis>> = vec![
            rewrite!("transpose-of-transpose";
                     "(TransposeOp (TransposeOp ?x ?p) ?p)" => "?x" if decreasing_perm("?p")),
            rewrite!("flatten-concat";
                     "(ConcatenateOp ?v ?d)" => { FlattenConcat {
                     vec: "?v".parse().unwrap(),
                     dim: "?d".parse().unwrap(),
            }}),
            rewrite!("merge-slices";
                     "(ConcatenateOp (Vec (SliceOp ?x ?s1 ?l1 ?s) (SliceOp ?x ?s2 ?l2 ?s)) ?d)" => { MergeSlices {
                     x: "?x".parse().unwrap(),
                     s1: "?s1".parse().unwrap(),
                     s2: "?s2".parse().unwrap(),
                     l1: "?l1".parse().unwrap(),
                     l2: "?l2".parse().unwrap(),
                     strides: "?s".parse().unwrap(),
                    dim: "?d".parse().unwrap()
            }}),
            rewrite!("concat-dot";
                     "(DotGeneralOp (ConcatenateOp (Vec ?a ?b) ?d1) (ConcatenateOp (Vec ?c ?d) ?d2) ?lb ?rb ?lc ?rc ?p)"
                     => "(AddOp (DotGeneralOp ?a ?c ?lb ?rb ?lc ?rc ?p) (DotGeneralOp ?b ?d ?lb ?rb ?lc ?rc ?p))"
                     if concat_dot_compatible("?lc", "?d1", "?rc", "?d2")),
        ];

        rules.append(&mut custom_rules);

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
        let cost_model: CostModel = CostModel::new();
        let (best, ext_secs) = extract_by_ilp(&egraph, root, &cost_model);
        // let (best, ext_secs) = extract_by_greedy(&egraph, root, &cost_model);

        // println!("{}", best);
        self.convert_to_node(best)
    }
}

fn extract_by_greedy(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32) {
    let tnsr_cost = TensorCost { egraph, cost_model };
    let start_time = Instant::now();
    let mut extractor = Extractor::new(egraph, tnsr_cost);
    let (best_cost, best) = extractor.find_best(root);
    let duration = start_time.elapsed();

    println!("Extractor complete!");
    println!("  Time taken: {:?}", duration);
    println!("  Best cost: {:?}", best_cost);
    let ext_secs = duration.as_secs_f32();

    (best, ext_secs)
}

fn extract_by_ilp(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    root: Id,
    cost_model: &CostModel,
) -> (RecExpr<Mdl>, f32) {
    // Prepare data for ILP formulation, save to json
    let (m_id_map, e_m, h_i, cost_i, g_i, root_m, i_to_nodes, blacklist_i) =
        prep_ilp_data(egraph, root, cost_model);

    println!("prepped ilp data");
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
    let mut arg_vec = vec!["src/enzyme_ad/jax/deps/tensat/extractor/extract.py"];
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
    let num_thread = "8";
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
