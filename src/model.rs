#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//use rand::prelude::*;
use crate::input::ffi::{self, Shape};
use rand;
use std::convert::TryInto;
use std::time::{Duration, Instant};
use std::{collections::HashMap, collections::HashSet};

use egg::*;

// Operator parameters, value matches the TASO side
pub const PSAME: i32 = 0;
pub const PVALID: i32 = 1;

pub const ACTNONE: i32 = 0;
pub const ACTSIGMOID: i32 = 1;
pub const ACTRELU: i32 = 2;
pub const ACTTANH: i32 = 3;

pub const NOSHUFFLE: i32 = 0;
pub const SHUFFLE: i32 = 1;

define_language! {
  pub enum Mdl {
      "input"              = Input([Id; 2]),  // takes Var: name@dim1_dim2, block_arg_number
      "CompareOp"          = CompareOp([Id; 4]), // input1, input2, comparison_direction,
                                                           // comparsion_type
      "BroadcastInDimOp"   = BroadcastInDimOp([Id; 2]), // input, broadcast_dimensions
      // TODO: we might need the input type as well.
      "ConvertOp"          = ConvertOp([Id; 2]), // input, output_tyoe.
      // TODO: we probably won't have any rewrites for reduces. Maybe function pointers for the
      // body
      "ReduceOp"           = ReduceOp([Id; 2]), // input, init_values, dimensions, body
      "ReshapeOp"          = ReshapeOp([Id; 2]), // input, shape
      "GatherOp"           = GatherOp([Id; 10]),
      "SelectOp"           = SelectOp([Id; 3]), // pred, on_true, on_false
      "ConcatenateOp"      = ConcatenateOp([Id; 2]), // inputs, dimension
      "DotGeneralOp"       = DotGeneralOp([Id; 8]), // lhs, rhs, ..., shape
      "PadOp"              = PadOp([Id; 5]), // input, padding_value, edge_padding_low,
                                                       // edge_padding_high, interior_padding
      "SliceOp"            = SliceOp([Id; 4]), // input, start_indices, limit_indices, strides
      "TransposeOp"        = TransposeOp([Id; 2]), // input, permutation
      // BINARY OPS
      "MulOp"              = MulOp([Id; 2]),
      "AddOp"              = AddOp([Id; 2]),
      "DivOp"              = DivOp([Id; 2]),
      "SubtractOp"         = SubtractOp([Id; 2]),
      "MinOp"              = MinOp([Id; 2]),
      "MaxOp"              = MaxOp([Id; 2]),
      // UNARY OPS
      "NegOp"              = NegOp([Id; 1]), // input
      "TanhOp"             = TanhOp([Id; 1]), // input
      "ExpOp"              = ExpOp([Id; 1]), // input
      // MISC OPS
      "IotaOp"             = IotaOp([Id; 2]), // iota_dimension, output_shape
      "ConstantOp"         = ConstantOp([Id; 0]),
      "DynamicUpdateSliceOp" = DynamicUpdateSliceOp([Id; 3]), // operand, update, start_indices
      "DynamicSliceOp"     = DynamicSliceOp([Id; 3]), // operand, start_indices, slice_sizes
      // Complete pain, has arity 12
      "ScatterOp"          = ScatterOp([Id; 4]), // input, scatter_indices, updates, dimension_numbers
       "BlackBox"           = BlackBox(Box<[Id]>),
       "Vec"                = Vec(Vec<Id>),
       Var(Symbol),
       Num(i32),
  }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name,
    Scalar,
    Tnsr,
    TnsrTuple,
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

// /// Metadata struct for TensorAnalysis
// #[derive(Debug, Clone)]
// pub struct ValTnsr {
//     // This is the cost of the op
//     pub val: i32,
//     pub cost: i32,
// }

pub const MAX_DIM: usize = 8;

// Struct for storing shape and value-related metadata for tensors. This
// is the base metadata struct that is used by Analysis as well.
#[derive(Clone, Debug)]
pub struct TensorData {
    // In StableHLO, each operation can have multiple results (for 
    // example, see ScatterOp).
    // 
    // To handle this, we assume that all operations have multiple 
    // results, hence the Vecs below. We can access the i-th element of
    // an operation x by using an Index node: (Index (Num i) x). 
    // For an Index node, we simply store singleton Vecs for the two
    // fields below. Then, we always take the 0-th result of any 
    // operation that appears as an operand. This allows us to omit
    // (Index 0) for the common case of using the only element in
    // the operation.

    /// Shapes of the tensor. We deal with tensor up to MAX_DIM dimensions.
    pub shapes: Vec<[i32; MAX_DIM]>,
    /// Number of dimensions of each result of this tensor
    pub n_dims: Vec<usize>,
    /// The name string of this eclass if it is a Name type
    pub name: Option<&'static str>,
}

// impl Default for TensorData {
//     fn default() -> Self {
//         TensorData {
//             shape: [50; 8],
//             n_dim: 8,
//             name: None,
//         }
//     }
// }

// Struct for storing information of a tensor. This is passed between functions
// during graph creation.
#[derive(Clone)]
pub struct TensorInfo {
    /// Id into the RecExpr constructed
    pub id: Id,
    pub tensor_data: TensorData,
}
/// Struct for metadata analysis
///
/// In this analysis, it calls functions on the TASO side (e.g. graph.matmul())
/// to create (or get) new ops/nodes and stores pointers to the output tensors.
/// TASO will measure and store the runtime cost when creating a new op/node.
pub struct TensorAnalysis {
    /// Record blacklisted nodes for filtering cycles
    pub blacklist_nodes: HashSet<Mdl>,
    /// Newly added nodes by order
    pub newly_added: Vec<Mdl>,
    /// Tracking TensorInfo for C++-originating ops
    pub tensorinfo_map: HashMap<Id, TensorInfo>,
    /// C++ FFI for shape inference using stablehlo
    pub cpp_shape_inference: cxx::UniquePtr<ffi::ShapeInference>, // Holding the C++ cost model
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        TensorAnalysis {
            blacklist_nodes: HashSet::<Mdl>::new(),
            newly_added: Vec::<Mdl>::new(),
            tensorinfo_map: HashMap::new(),
            cpp_shape_inference: ffi::newShapeInference(),
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = TensorData;

    /// Merges two metadata when two eclasses are merged.
    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        to.shapes == from.shapes
    }

    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        fn convert_i32_slice_to_i64_slice(input: &[i32; 8]) -> &[i64] {
            let converted_slice: Box<[i64]> = input
                .iter()
                .map(|x| *x as i64)
                .collect::<Vec<i64>>()
                .into_boxed_slice();

            Box::leak(converted_slice)
        }

        fn shape_from_dim(dims: Vec<Shape>) -> (Vec<[i32; MAX_DIM]>, Vec<usize>) {
            let mut shapes: Vec<[i32; 8]> = vec![];
            let mut n_dims: Vec<usize> = vec![];
            for dims in dims.iter() {
                let dims = &dims.shape;
                if (dims.len() > MAX_DIM) {
                    println!("ERROR: op shape exceeds MAX_DIM! e-graph no longer valid.");
                }
                let mut shape = [0; MAX_DIM];
                for (i, dim) in dims.iter().enumerate() {
                    shape[i] = *dim;
                }
                shapes.push(shape);
                n_dims.push(dims.len())
            }
            (shapes, n_dims)
        }

        fn dim_from_name_string(name: &str) -> (Vec<[i32; MAX_DIM]>, Vec<usize>) {
            let name_vec: Vec<&str> = name.split("@").collect();
            assert!(name_vec.len() == 2);
            let dims: Vec<i32> = name_vec[1]
                .split("_")
                .map(|x| x.parse::<i32>().unwrap())
                .collect();
            shape_from_dim(vec![Shape {shape: dims}])
        };

        fn print_joined_with_underscore(numbers: &Vec<i32>) {
            let joined_numbers = numbers
                .iter()
                .map(|&num| num.to_string())
                .collect::<Vec<_>>()
                .join("_");
            println!("{}", joined_numbers);
        }

        match enode {
            Mdl::Num(_) => TensorData {
                shapes: vec![[0; MAX_DIM]],
                n_dims: vec![0],
                name: Some(&"Num"),
            },
            Mdl::Var(name) => {
                let (shapes, n_dims) = dim_from_name_string(name.as_str());
                let name = Some(name.as_str());
                TensorData { shapes, n_dims, name }
            }
            Mdl::Input([node, block_arg_number]) => x(node).clone(),
            Mdl::MulOp([lhs, rhs]) => {
                let lhs_dims = x(lhs);
                let rhs_dims = x(rhs);
                let arg_dims = [
                    convert_i32_slice_to_i64_slice(&lhs_dims.shapes[0]),
                    convert_i32_slice_to_i64_slice(&rhs_dims.shapes[0]),
                ];
                let arg_types = [ffi::Type::f32, ffi::Type::f32];
                let shape_vec = egraph.analysis.cpp_shape_inference.get_shape(
                    ffi::Ops::MulOp,
                    &arg_dims,
                    &arg_types,
                    &[],
                    &[],
                );
                // print_joined_with_underscore(&shape_vec);
                let (shapes, n_dims) = shape_from_dim(shape_vec);
                TensorData {
                    shapes,
                    n_dims,
                    name: None,
                }
            }
            Mdl::AddOp([lhs, rhs]) => {
                let lhs_dims = x(lhs);
                let rhs_dims = x(rhs);
                let arg_dims = [
                    convert_i32_slice_to_i64_slice(&lhs_dims.shapes[0]),
                    convert_i32_slice_to_i64_slice(&rhs_dims.shapes[0]),
                ];
                let arg_types = [ffi::Type::f32, ffi::Type::f32];
                let shape_vec = egraph.analysis.cpp_shape_inference.get_shape(
                    ffi::Ops::AddOp,
                    &arg_dims,
                    &arg_types,
                    &[],
                    &[],
                );
                // print_joined_with_underscore(&shape_vec);
                let (shapes, n_dims) = shape_from_dim(shape_vec);
                TensorData {
                    shapes,
                    n_dims,
                    name: None,
                }
            }
            Mdl::DivOp([lhs, rhs]) => {
                let lhs_dims = x(lhs);
                let rhs_dims = x(rhs);
                let arg_dims = [
                    convert_i32_slice_to_i64_slice(&lhs_dims.shapes[0]),
                    convert_i32_slice_to_i64_slice(&rhs_dims.shapes[0]),
                ];
                let arg_types = [ffi::Type::f32, ffi::Type::f32];
                let shape_vec = egraph.analysis.cpp_shape_inference.get_shape(
                    ffi::Ops::DivOp,
                    &arg_dims,
                    &arg_types,
                    &[],
                    &[],
                );
                // print_joined_with_underscore(&shape_vec);
                let (shapes, n_dims) = shape_from_dim(shape_vec);
                TensorData {
                    shapes,
                    n_dims,
                    name: None,
                }
            }
            Mdl::SubtractOp([lhs, rhs]) => {
                let lhs_dims = x(lhs);
                let rhs_dims = x(rhs);
                let arg_dims = [
                    convert_i32_slice_to_i64_slice(&lhs_dims.shapes[0]),
                    convert_i32_slice_to_i64_slice(&rhs_dims.shapes[0]),
                ];
                let arg_types = [ffi::Type::f32, ffi::Type::f32];
                let shape_vec = egraph.analysis.cpp_shape_inference.get_shape(
                    ffi::Ops::SubtractOp,
                    &arg_dims,
                    &arg_types,
                    &[],
                    &[],
                );
                // print_joined_with_underscore(&shape_vec);
                let (shapes, n_dims) = shape_from_dim(shape_vec);
                TensorData {
                    shapes,
                    n_dims,
                    name: None,
                }
            }
            Mdl::NegOp([operand]) => {
                let operand_dims = x(operand);
                let arg_dims = [convert_i32_slice_to_i64_slice(&operand_dims.shapes[0])];
                let arg_types = [ffi::Type::f32];
                let shape_vec = egraph.analysis.cpp_shape_inference.get_shape(
                    ffi::Ops::NegOp,
                    &arg_dims,
                    &arg_types,
                    &[],
                    &[],
                );
                // print_joined_with_underscore(&shape_vec);
                let (shapes, n_dims) = shape_from_dim(shape_vec);
                TensorData {
                    shapes,
                    n_dims,
                    name: None,
                }
            }
            // Mdl::MinOp(_) => Self::Data { val: 0, cost: 0 },
            // Mdl::MaxOp([lhs, rhs]) => Self::Data { val: 0, cost: 0 },
            // Mdl::NegOp([input]) => Self::Data { val: 0, cost: 0 },
            // Mdl::TanhOp([input]) => Self::Data { val: 0, cost: 0 },
            // Mdl::ExpOp([input]) => Self::Data { val: 0, cost: 0 },
            // Mdl::ReshapeOp(_) => Self::Data { val: 0, cost: 10 },
            // Mdl::DotGeneralOp(_) => Self::Data { val: 0, cost: 0 },
            // Mdl::TransposeOp(_) => Self::Data { val: 0, cost: 0 },
            // Mdl::ConstantOp([]) => Self::Data { val: 0, cost: 0 },
            // Mdl::BlackBox(_) => Self::Data { val: 0, cost: 0 },
            x => {
                println!("{:?}", x);
                unimplemented!("Op unimplemented")
            }
        }
    }

    // Not needed to modify anything
    fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {}
}
