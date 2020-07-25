//! vulkano_blas is a library of GLSL Vulkano shaders for BLAS operations.
//! 
//! ### Crate status
//! Currently due to the limitations of Vulkano very few BLAS operations can be implemented ([my issue on GitHub concerning this](https://github.com/vulkano-rs/vulkano/issues/1395)).
//! 
//! All shaders I am certain I will implement but cannot implement at the moment due to this limitation are marked as deprecated.
//! 
//! I hope perhaps this crate while at the moment extremely limited might spike some interest and motiviation in the topic.
//! 
//! ### A note on testing
//! None of the rustdoc code is tested due to the extreme awkwardness of this. A test is run for every (implemented) shader through, just not in the rustdoc comments.

/// Level 1 BLAS operations (vector operations).
pub use crate::level_1::*;
mod level_1;

/// Level 2 BLAS operations (matrix-vector operations).
pub use crate::level_2::*;
mod level_2;

/// Level 3 BLAS operations (matrix-matrix operations).
pub use crate::level_3::*;
mod level_3;