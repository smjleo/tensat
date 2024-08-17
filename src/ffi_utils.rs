use crate::{
    input::ffi,
    model::*,
    rewrites::{get_num_option, get_vec_of_nums_option, get_vec_option},
};
use egg::*;

fn dim_to_i64_vec(input: &[i32; MAX_DIM]) -> ffi::Shape {
    ffi::Shape {
        shape: input
            .iter()
            .filter(|&x| *x != 0)
            .map(|x| *x as i64)
            .collect::<Vec<i64>>(),
    }
}

fn map_to_i64(vec: Vec<i32>) -> Vec<i64> {
    vec.into_iter().map(|x| x as i64).collect()
}

fn process_enode_args(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
) -> (Vec<ffi::Shape>, Vec<ffi::Type>, Vec<ffi::Shape>, Vec<i64>) {
    let mut args: Vec<ffi::Shape> = vec![];
    let mut other_vecs: Vec<ffi::Shape> = vec![];
    let mut int_args: Vec<i64> = vec![];

    for child in enode.children().iter() {
        if let Some(other_vec) = get_vec_of_nums_option(egraph, &egraph[*child]) {
            other_vecs.push(ffi::Shape {
                shape: map_to_i64(other_vec),
            })
        } else if let Some(vec) = get_vec_option(&egraph[*child]) {
            vec.iter()
                .for_each(|&id| args.push(dim_to_i64_vec(&egraph[id].data.shapes[0])))
        } else if let Some(num) = get_num_option(&egraph[*child]) {
            int_args.push(num as i64)
        } else {
            args.push(dim_to_i64_vec(&egraph[*child].data.shapes[0]))
        }
    }

    // TODO: need to handle types
    let arg_types: Vec<ffi::Type> = args.iter().map(|_| ffi::Type::f32).collect();

    (args, arg_types, other_vecs, int_args)
}

pub fn convert_mdl_to_ffi_op(enode: &Mdl) -> ffi::Ops {
    match enode {
        Mdl::Input(_) => ffi::Ops::Input,
        Mdl::CompareOp(_) => ffi::Ops::CompareOp,
        Mdl::BroadcastInDimOp(_) => ffi::Ops::BroadcastInDimOp,
        Mdl::ConvertOp(_) => ffi::Ops::ConvertOp,
        Mdl::ReduceOp(_) => ffi::Ops::ReduceOp,
        Mdl::ReshapeOp(_) => ffi::Ops::ReshapeOp,
        Mdl::GatherOp(_) => ffi::Ops::GatherOp,
        Mdl::SelectOp(_) => ffi::Ops::SelectOp,
        Mdl::ConcatenateOp(_) => ffi::Ops::ConcatenateOp,
        Mdl::DotGeneralOp(_) => ffi::Ops::DotGeneralOp,
        Mdl::PadOp(_) => ffi::Ops::PadOp,
        Mdl::SliceOp(_) => ffi::Ops::SliceOp,
        Mdl::TransposeOp(_) => ffi::Ops::TransposeOp,
        Mdl::MulOp(_) => ffi::Ops::MulOp,
        Mdl::AddOp(_) => ffi::Ops::AddOp,
        Mdl::DivOp(_) => ffi::Ops::DivOp,
        Mdl::SubtractOp(_) => ffi::Ops::SubtractOp,
        Mdl::MinOp(_) => ffi::Ops::MinOp,
        Mdl::MaxOp(_) => ffi::Ops::MaxOp,
        Mdl::NegOp(_) => ffi::Ops::NegOp,
        Mdl::TanhOp(_) => ffi::Ops::TanhOp,
        Mdl::ExpOp(_) => ffi::Ops::ExpOp,
        Mdl::IotaOp(_) => ffi::Ops::IotaOp,
        Mdl::DynamicUpdateSliceOp(_) => ffi::Ops::DynamicUpdateSliceOp,
        Mdl::DynamicSliceOp(_) => ffi::Ops::DynamicSliceOp,
        Mdl::ScatterOp(_) => ffi::Ops::ScatterOp,
        _ => panic!("Unsupported op for creating StableHLO op"),
    }
}

pub fn create_stablehlo_op<F, R>(
    egraph: &EGraph<Mdl, TensorAnalysis>,
    enode: &Mdl,
    process_output: F,
) -> R
where
    F: Fn(ffi::Ops, Vec<ffi::Shape>, Vec<ffi::Type>, Vec<ffi::Shape>, Vec<i64>) -> R,
{
    let op = convert_mdl_to_ffi_op(enode);
    let (args, arg_types, other_vecs, int_args) = process_enode_args(egraph, enode);
    let res = process_output(op, args, arg_types, other_vecs, int_args);
    res
}
