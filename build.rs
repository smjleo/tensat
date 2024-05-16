extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/input.rs");
    println!("cargo:rerun-if-changed=src/graph.cc");
    println!("cargo:rerun-if-changed=include/tensat.h");

    // C++ graph input bindings
    cxx_build::bridge("src/input.rs")
        .flag_if_supported("-std=c++20")
        .flag_if_supported("-lc++")   // Link libc++ explicitly
        .flag_if_supported("-lc++abi") // Link libc++abi if needed
        .compile("tensatcpp");
}
