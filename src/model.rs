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

define_language! {
  pub enum Mdl {
      Var(Symbol),
      Int(i32),
      // note that the integer arg after the Id is the arity
      "input"                        = Input([Id; 1]),  // takes a Var, format: name@dim1_dim2...
      "stablehlo.CompareOp"          = CompareOp([Id; 4]), // input1, input2, comparison_direction,
                                                           // comparsion_type
      "stablehlo.BroadcastInDimOp"   = BroadcastInDimOp([Id; 3]), // input, broadcast_dimensions
      // TODO: we might need the input type as well.
      "stablehlo.ConvertOp"          = ConvertOp([Id; 2]), // input, output_tyoe.
      // TODO: we probably won't have any rewrites for reduces. Maybe function pointers for the
      // body
      "stablehlo.ReduceOp"           = ReduceOp([Id; 3]), // input, init_values, dimensions, body
      "stablehlo.ReshapeOp"          = ReshapeOp([Id; 3]), // input, shape
      "stablehlo.GatherOp"           = GatherOp([Id; 10]), 
      "stablehlo.SelectOp"           = SelectOp([Id; 3]), // pred, on_true, on_false
      "stablehlo.ConcatenateOp"      = ConcatenateOp([Id; 2]), // inputs, dimension
      "stablehlo.DotGeneralOp"       = DotGeneralOp([Id; 8]), // lhs, rhs... 
      "stablehlo.PadOp"              = PadOp([Id; 5]), // input, padding_value, edge_padding_low,
                                                       // edge_padding_high, interior_padding
      "stablehlo.SliceOp"            = SliceOp([Id; 4]), // input, start_indices, limit_indices, strides
      "stablehlo.TransposeOp"        = TransposeOp([Id; 3]), // input, permutation
      // BINARY OPS
      "stablehlo.MulOp"              = MulOp([Id; 3]), 
      "stablehlo.AddOp"              = AddOp([Id; 3]),
      "stablehlo.DivOp"              = DivOp([Id; 3]),
      "stablehlo.SubtractOp"         = SubtractOp([Id; 3]),
      "stablehlo.MinOp"              = MinOp([Id; 3]),
      "stablehlo.MaxOp"              = MaxOp([Id; 3]),
      // UNARY OPS
      "stablehlo.NegOp"              = NegOp([Id; 2]), // input
      "stablehlo.TanhOp"             = TanhOp([Id; 2]), // input
      "stablehlo.ExpOp"              = ExpOp([Id; 2]), // input
      "stablehlo.IotaOp"             = IotaOp([Id; 3]), // iota_dimension, output_shape
      "stablehlo.ConstantOp"         = ConstantOp([Id; 0]), 
      "stablehlo.DynamicUpdateSliceOp" = DynamicUpdateSliceOp([Id; 4]), // operand, update, start_indices
      "stablehlo.DynamicSliceOp"     = DynamicSliceOp([Id; 4]), // operand, start_indices, slice_sizes
      "stablehlo.ScatterOp"          = ScatterOp([Id; 5]), // input, scatter_indices, updates, dimension_numbers

      // Maybe we can have a single enode with variable arity
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

        // TODO: what do we need to be storing as node metadata? 
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
