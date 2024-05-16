#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

//use rand::prelude::*;
use rand;
use std::collections::HashSet;
use std::convert::TryInto;
use std::time::{Duration, Instant};

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

// TODO: match stableHLO opset
// TODO: add cost as parameter for each op

define_language! {
    pub enum Mdl {
        "input"     = Input([Id; 1]), // takes a Var, format: name@dim1_dim2...
        "weight"    = Weight([Id; 1]), // takes a Var, format : name@dim1_dim2...
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "smul"      = Smul([Id; 2]),
        "transpose" = Transpose([Id; 3]), // input, perm_name (format: dim1_dim2...), shuffle
        "matmul"    = Matmul([Id; 3]), // activation, input1, input2
        "conv2d"    = Conv2d([Id; 6]), // conv2d's weight tensor kernel size can not be even, it seems that TASO's output shape computation is incorrect for even kernal size (like 4x4)
        "enlarge"   = Enlarge([Id; 2]), // input_to_enlarge, ref_input
        "dropout"   = Dropout(Id),
        "relu"      = Relu(Id),
        "tanh"      = Tanh(Id),
        "sigmoid"   = Sigmoid(Id),
        "poolmax"   = Poolmax([Id; 7]), // input, kernel_h, kernel_w, stride_h, stride_w, padding, activation
        "poolavg"   = Poolavg([Id; 7]), // input, kernel_h, kernel_w, stride_h, stride_w, padding, activation
        "concat"    = Concat([Id; 4]), // axis, ndim, input1, input2. ndim is for using in CheckApply only
        "concat3"    = Concat3([Id; 5]), // axis, ndim, input1, input2. input3, ndim is for using in CheckApply only
        "concat4"    = Concat4([Id; 6]), // axis, ndim, input1, input2. input3, input4, ndim is for using in CheckApply only
        "concat5"    = Concat5([Id; 7]), // axis, ndim, input1, input2, input3, input4, input5. ndim is for using in CheckApply only
        // Add a concat for each number of inputs if needed
        "split_0"   = Split0(Id), // must take a split node as input
        "split_1"   = Split1(Id), // must take a split node as input
        "split"     = Split([Id; 2]), // axis, input
        "Cpool"     = Cpool([Id; 2]),
        "Iconv"     = Iconv([Id; 2]),
        "Imatmul"   = Imatmul,
        "Iewmul"    = Iewmul,
        "merge"     = Merge([Id; 2]), // merge_gconv, takes [weight, count]
        "reshape"   = Reshape([Id; 2]), // input, shape_name (format: dim1_dim2...)
        "noop"      = Noop([Id; 2]), // No op, use to combine the outputs of a graph in case there are multiple, since egg works with single root graph
        "batchnorm" = BatchNorm([Id; 5]), // input, scale, bias, mean, var
        Num(i32),
        Var(Symbol),
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

/// Metadata struct for TensorAnalysis
#[derive(Debug, Clone)]
pub struct ValTnsr {
    /// The data type of this eclass, can be a name/scalar/tensor
    pub dtype: DataKind,
    /// The value of this eclass if it is a Scalar type
    pub val: i32,
    /// The name string of this eclass if it is a Name type
    pub name: String,
    /// The cost of this eclass
    pub cost: f32,
    /// If the tensor results from all weights computations
    pub all_weights: bool,
}

impl Default for ValTnsr {
  fn default() -> Self {
      ValTnsr {
          dtype: DataKind::Name,
          val: 0,
          name: String::new(),
          cost: 0.0,
          all_weights: false,
      }
  }
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
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        TensorAnalysis {
            blacklist_nodes: HashSet::<Mdl>::new(),
            newly_added: Vec::<Mdl>::new(),
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = ValTnsr;

    /// Merges two metadata when two eclasses are merged.
    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        if from.all_weights && (!to.all_weights) {
            to.all_weights = from.all_weights;
            true
        } else {
            false
        }
    }

    // Constructs metadata for a new enode, using TASO side functions for tensors.
    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        let dim_from_name = |name: &Id| {
            let name_vec: Vec<&str> = x(name).name.split("@").collect();
            assert!(name_vec.len() == 2);
            let dims: Vec<i32> = name_vec[1]
                .split("_")
                .map(|x| x.parse::<i32>().unwrap())
                .collect();
            dims
        };

        match enode {
            Mdl::Num(n) => Self::Data {
                dtype: DataKind::Scalar,
                val: *n,
                name: String::new(),
                cost: 0.0,
                all_weights: false,
            },

            Mdl::Var(s) => Self::Data {
                dtype: DataKind::Name,
                val: 0,
                name: s.as_str().to_string(),
                cost: 0.0,
                all_weights: false,
            },

            Mdl::Reshape([input, shape_name]) => {
                let cost = 0.0;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    cost,
                    all_weights: false,
                }
            },

            Mdl::Transpose([input, perm_name, shuffle]) => {
                let cost = 0.0;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    cost,
                    all_weights: false,
                }
            },

            Mdl::Tanh(input) => {
                let cost = 0.0;
                Self::Data {
                    dtype: DataKind::Tnsr,
                    val: 0,
                    name: String::new(),
                    cost,
                    all_weights: false,
                }
            },  

            // Handle other cases similarly...
            _ => unimplemented!(),
            }
    }

    // Not needed to modify anything
    fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {}
}
