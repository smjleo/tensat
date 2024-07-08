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

      // Maybe we can have a single enode with variable arity
      "blackbox_1"         = BlackBox_1([Id; 2]),
      "blackbox_2"         = BlackBox_2([Id; 2]),
      "blackbox_3"         = BlackBox_3([Id; 3]),
      "blackbox_4"         = BlackBox_4([Id; 4]),
      "blackbox_5"         = BlackBox_5([Id; 5]),
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
            Mdl::Num(i) => Self::Data { val: *i, cost: 0 },

            // TODO: Here for testing. Remove later and call cost function with appropriate arguments
            Mdl::Input([node, block_arg_number]) => Self::Data { val: 0, cost: 0 },
            Mdl::MulOp([lhs, rhs]) => Self::Data { val: 0, cost: 10 },
            Mdl::AddOp([lhs, rhs]) => Self::Data { val: 0, cost: 10 },
            Mdl::DivOp([lhs, rhs]) => Self::Data { val: 0, cost: 10 },
            Mdl::SubtractOp([lhs, rhs]) => Self::Data { val: 0, cost: 10 },

            // Mdl::CompareOp([input1, input2, comparison, cost]) => Self::Data { val: 0, cost: x(cost).val },
            // Mdl::BroadcastInDimOp([input, dimensions, cost]) => Self::Data { val: 0, cost: x(cost).val },
            // Mdl::ConvertOp([input, cost]) => Self::Data { val: 0, cost: x(cost).val },
            // Mdl::ReduceOp([input, dimensions, cost]) => Self::Data { val: 0, cost: x(cost).val },
            Mdl::ReshapeOp([operand, shape]) => Self::Data { val: 0, cost: 10 },
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
            Mdl::BlackBox_1([input, cpp_num]) => Self::Data { val: 0, cost: 0 },
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
