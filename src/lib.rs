//! kasumi compiles short math formulas into CUDA kernels and draws them as
//! tens of millions of points per frame.
//!
//! text -> `parser` -> `ast` -> `ir` -> `codegen` -> `cuda` -> `render`

pub mod archive;
pub mod args;
pub mod ast;
pub mod bench;
pub mod cli;
pub mod codegen;
pub mod cuda;
pub mod evolve;
pub mod explore;
pub mod export;
pub mod features;
pub mod grid;
pub mod interp;
pub mod ir;
pub mod live;
pub mod look;
pub mod mutate;
pub mod parser;
pub mod print;
pub mod render;
pub mod rng;
pub mod ui;
