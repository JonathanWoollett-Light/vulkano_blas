/// Matrix-vector multiply.
/// 
/// $ y = α * A * x + β * y $
/// 
/// Where y and x are column vectors (of length m), α and β are scalars, and A is a matrix (of m × n).
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html)
#[deprecated(note="All scan and reduce operations are impossible since Vulkano does not support enabling extensions in GLSL. This will be added when Vulkano supports this.")]
pub mod sgemv { }