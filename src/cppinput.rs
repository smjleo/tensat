use crate::input::GraphConverter;

#[cxx::bridge]
pub mod ffi {
    extern "Rust" {
        type TensorInfo = crate::input::TensorInfo;
        type CppGraphConverter;

        fn new_converter() -> &CppGraphConverter;
        fn new_input(graph: &mut CppGraphConverter, dims: &[i32]) -> TensorInfo;
        fn debug(graph: &CppGraphConverter);
    }
}

#[derive(Debug)]
pub struct CppGraphConverter {
    gc: &mut GraphConverter,
}

pub fn new_converter() -> &CppGraphConverter {
    &CppGraphConverter { gc: &mut GraphConverter::default() }
}

pub fn new_input(graph: &mut CppGraphConverter, dims: &[i32]) -> TensorInfo {
    graph.gc.new_input(dims)
}

pub fn relu(graph: &mut CppGraphConverter, inpt: TensorInfo) -> TensorInfo {
    graph.gc.relu(inpt)
}

pub fn debug(graph: &CppGraphConverter) {
    println!("{:?}", graph)
}