use crate::input::*;

#[cxx::bridge]
pub mod ffi {
    extern "Rust" {
        type CppTensorInfo;
        type CppGraphConverter;

        fn new_converter() -> &CppGraphConverter;
        fn new_input(graph: &mut CppGraphConverter, dims: &[i32]) -> CppTensorInfo;
        fn debug(graph: &CppGraphConverter);
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct CppTensorInfo {
    ti: TensorInfo
}

#[derive(Debug)]
pub struct CppGraphConverter {
    gc: &mut GraphConverter,
}

pub fn new_converter() -> &CppGraphConverter {
    &CppGraphConverter { gc: &mut GraphConverter::default() }
}

pub fn new_input(graph: &mut CppGraphConverter, dims: &[i32]) -> CppTensorInfo {
    CppTensorInfo { ti: graph.gc.new_input(dims) }
}

pub fn relu(graph: &mut CppGraphConverter, inpt: CppTensorInfo) -> CppTensorInfo {
    graph.gc.relu(inpt.ti)
}

pub fn debug(graph: &CppGraphConverter) {
    println!("{:?}", graph)
}