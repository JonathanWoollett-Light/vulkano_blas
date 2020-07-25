/// Matrix-matrix multiply
/// 
/// $ y = α * A * B + β * C $
/// 
/// Where A,B and C are matricies (m × k, k × n and m × n respectively), and α and β are scalars.
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html)
#[deprecated(note="All scan and reduce operations are impossible since Vulkano does not support enabling extensions in GLSL. This will be added when Vulkano supports this.")]
pub mod sgemm { }