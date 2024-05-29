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
      Var(Symbol),
      Int(i32),
      "input"                        = Input([Id; 1]),  // takes a Var, format: name@dim1_dim2...
      "stablehlo.CompareOp"          = CompareOp([Id; 4]), // input1, input2, comparison, cost
      "stablehlo.BroadcastInDimOp"   = BroadcastInDimOp([Id; 3]), // input, dimensions, cost
      "stablehlo.ConvertOp"          = ConvertOp([Id; 2]), // input, cost
      "stablehlo.ReduceOp"           = ReduceOp([Id; 3]), // input, dimensions, cost
      "stablehlo.ReshapeOp"          = ReshapeOp([Id; 3]), // input, shape, cost
      "stablehlo.GatherOp"           = GatherOp([Id; 4]), // input, start_indices, dimension_numbers, cost
      "stablehlo.SelectOp"           = SelectOp([Id; 4]), // pred, on_true, on_false, cost
      "stablehlo.ConcatenateOp"      = ConcatenateOp([Id; 3]), // inputs, dimension, cost
      "stablehlo.DotGeneralOp"       = DotGeneralOp([Id; 4]), // lhs, rhs, dot_dimension_numbers, cost
      "stablehlo.PadOp"              = PadOp([Id; 4]), // input, padding_value, padding_config, cost
      "stablehlo.SliceOp"            = SliceOp([Id; 5]), // input, start_indices, limit_indices, strides, cost
      "stablehlo.TransposeOp"        = TransposeOp([Id; 3]), // input, permutation, cost
      "stablehlo.MulOp"              = MulOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.AddOp"              = AddOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.DivOp"              = DivOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.SubtractOp"         = SubtractOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.MinOp"              = MinOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.MaxOp"              = MaxOp([Id; 3]), // lhs, rhs, cost
      "stablehlo.NegOp"              = NegOp([Id; 2]), // input, cost
      "stablehlo.TanhOp"             = TanhOp([Id; 2]), // input, cost
      "stablehlo.ExpOp"              = ExpOp([Id; 2]), // input, cost
      "stablehlo.IotaOp"             = IotaOp([Id; 2]), // input, cost
      "stablehlo.ConstantOp"         = ConstantOp([Id; 2]), // value, cost
      "stablehlo.DynamicUpdateSliceOp" = DynamicUpdateSliceOp([Id; 4]), // operand, update, start_indices, cost
      "stablehlo.DynamicSliceOp"     = DynamicSliceOp([Id; 4]), // operand, start_indices, slice_sizes, cost
      "stablehlo.ScatterOp"          = ScatterOp([Id; 5]), // input, scatter_indices, updates, dimension_numbers, cost
      "blackbox_1"                   = BlackBox_1([Id; 1]), 
      "blackbox_2"                   = BlackBox_2([Id; 2]),
      "blackbox_3"                   = BlackBox_3([Id; 3]),
      "blackbox_4"                   = BlackBox_4([Id; 4]),
      "blackbox_5"                   = BlackBox_5([Id; 5]),
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
    // This is the cost of the op
    pub val: i32,
    pub cost: i32,
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
        true
    }

    // Constructs metadata for a new enode, using TASO side functions for tensors.
    fn make(egraph: &EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;

        // let dim_from_name = |name: &Id| {
        //     let name_vec: Vec<&str> = x(name).name.split("@").collect();
        //     assert!(name_vec.len() == 2);
        //     let dims: Vec<i32> = name_vec[1]
        //         .split("_")
        //         .map(|x| x.parse::<i32>().unwrap())
        //         .collect();
        //     dims
        // };

        match enode {
          Mdl::Var(_) => Self::Data { val: 0, cost: 0 }, /* we might need a name field... */
          Mdl::Int(i) => Self::Data { val: *i, cost: 0 },
          // Mdl::CompareOp([input1, input2, comparison, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::BroadcastInDimOp([input, dimensions, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ConvertOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ReduceOp([input, dimensions, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ReshapeOp([input, shape, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::GatherOp([input, start_indices, dimension_numbers, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::SelectOp([pred, on_true, on_false, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ConcatenateOp([inputs, dimension, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::DotGeneralOp([lhs, rhs, dot_dimension_numbers, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::PadOp([input, padding_value, padding_config, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::SliceOp([input, start_indices, limit_indices, strides, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::TransposeOp([input, permutation, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::MulOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::AddOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::DivOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::SubtractOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::MinOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::MaxOp([lhs, rhs, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::NegOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::TanhOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ExpOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::IotaOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ConstantOp([value, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::DynamicUpdateSliceOp([operand, update, start_indices, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::DynamicSliceOp([operand, start_indices, slice_sizes, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::ScatterOp([input, scatter_indices, updates, dimension_numbers, cost]) => Self::Data { val: 0, cost: x(cost).val },
          // Mdl::BlackBox_1([input]) => Self::Data { val: 0, cost: 0 },
          // Mdl::BlackBox_2([input1, input2]) => Self::Data { val: 0, cost: 0 },
          // Mdl::BlackBox_3([input1, input2, input3]) => Self::Data { val: 0, cost: 0 },
          // Mdl::BlackBox_4([input1, input2, input3, input4]) => Self::Data { val: 0, cost: 0 },
          // Mdl::BlackBox_5([input1, input2, input3, input4, input5]) => Self::Data { val: 0, cost: 0 },
          _ => unimplemented!(),
      }
    }

    // Not needed to modify anything
    fn modify(egraph: &mut EGraph<Mdl, Self>, id: Id) {}
}
