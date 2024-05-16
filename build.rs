extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/input.rs");
    println!("cargo:rerun-if-changed=src/graph.cc");
    println!("cargo:rerun-if-changed=include/tensat.h");

    // C++ graph input bindings
    // cxx_build::bridge("src/input.rs")
    //     .compile("tensatcpp");
}
