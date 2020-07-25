/// Scales a vector by a constant.
/// 
/// $ x = a * x $
/// 
/// Where x is vector and a is a scalar.
/// 
/// http://www.netlib.org/lapack/explore-html/d9/d04/sscal_8f.html
/// 
/// ```ignore
/// // Sets the data buffer (x vector).
/// // CLoning x here is unneccessary for functionality, it is only convenient such that we can use it later for checking the result.
/// let x = 0..5u32;
/// let data_buffer_x = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,x.clone()).expect("failed to create buffer");
///
/// // Initiates a scalar.
/// let a = 2u32;
///
/// // Gets the layout of the pipeline (x,y,z)
/// let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
/// // Descriptors push buffers to the compute pipeline.
/// // Intiates descriptor set.
/// let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
///     .add_buffer(data_buffer_x.clone()).unwrap()
///     .build().unwrap()
/// );
///
/// // Intiates command buffer for compute pipeline (1 work groups)
/// let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
/// // Dispatches command buffer builder passing a scalar as push constant.
/// builder.dispatch([1024,1,1],compute_pipeline.clone(),set.clone(),a).unwrap();
///
/// let command_buffer = builder.build().unwrap();
///
/// // Submit command buffer.
/// let finished = command_buffer.execute(queue.clone()).unwrap();
///
/// // Halts thread until command buffer executed.
/// finished.then_signal_fence_and_flush().unwrap()
/// .wait(None).unwrap();
///
/// // Checks content of data buffer.
/// let x_content = data_buffer_x.read().unwrap();
/// for (new_x, x) in x_content.iter().zip(x) {
///     assert_eq!(*new_x, x * a);
/// }
/// ```
pub mod sscal {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0,binding = 0) buffer Buffer {
            uint x[];
        };
        layout(push_constant) uniform PushConstantData {
            uint a;
        };

        void main() {
            uint indx = gl_GlobalInvocationID.x;
            x[indx] *= a;
        }"
    }
}
/// Constant times a vector plus a vector.
/// 
/// $ y = a * x + y $
/// 
/// Where x and y are vectors and a is a scalar.
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/d8/daf/saxpy_8f.html)
/// 
/// ```ignore
/// // Sets data buffers (y and x vectors).
/// // CLoning y and x here is unneccessary for functionality, it is only convenient such that we can use it later for checking the result.
/// let y = 0..5u32;
/// let data_buffer_y = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,y.clone()).expect("failed to create buffer");
/// let x = 5..10u32;
/// let (data_buffer_x,x_future) = ImmutableBuffer::from_iter(x.clone(), BufferUsage::all(), queue.clone()).expect("failed to create buffer");
///
/// // Halts thread until data_buffer_x intialised.
/// x_future.then_signal_fence_and_flush().unwrap()
/// .wait(None).unwrap();
///
/// // Initiates a scalar.
/// let a = 2u32;
///
/// // Gets the layout of the pipeline (x,y,z)
/// let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
/// // Intiates descriptor set.
/// let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
///     .add_buffer(data_buffer_x.clone()).unwrap()
///     .add_buffer(data_buffer_y.clone()).unwrap()
///     .build().unwrap()
/// );
///
/// // Intiates command buffer builder.
/// let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
/// // Dispatches command buffer builder passing a scalar as push constant.
/// builder.dispatch([1024,1,1],compute_pipeline.clone(),set.clone(),a).unwrap();
/// // Intiates command buffer.
/// let command_buffer = builder.build().unwrap();
///
/// // Submits command buffer.
/// let finished = command_buffer.execute(queue.clone()).unwrap();
///
/// // Halts thread until command buffer executed.
/// finished.then_signal_fence_and_flush().unwrap()
/// .wait(None).unwrap();
///
/// // Checks content of data buffer.
/// let y_content = data_buffer_y.read().unwrap();
/// for (new_y,old_y,old_x) in izip!(y_content.iter(),y,x) {
///     assert_eq!(*new_y, a * old_x + old_y);
/// }
/// ```
pub mod saxpy {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
        #version 450

        #extension GL_KHR_shader_subgroup_basic: enable

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0,binding = 0) buffer Buffer0 {
            uint x[];
        };
        layout(set = 0,binding = 1) buffer Buffer1 {
            uint y[];
        };
        layout(push_constant) uniform PushConstantData {
            uint a;
        };

        void main() {
            uint indx = gl_GlobalInvocationID.x;
            y[indx] += x[indx] * a;
        }"
    }
}
/// Vector dot product.
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/d0/d16/sdot_8f.html)
#[deprecated(note="All scan and reduce operations are impossible since Vulkano does not support enabling extensions in GLSL. This will be added when Vulkano supports this.")]
pub mod sdot {}

/// Gets sum a vector.
/// 
/// $ sigma_x $ 
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/df/d1f/sasum_8f.html)
#[deprecated(note="All scan and reduce operations are impossible since Vulkano does not support enabling extensions in GLSL. This will be added when Vulkano supports this.")]
pub mod sasum { }

/// Gets index of max value in vector.
/// 
/// $ max(x) $
/// 
/// [Official documentation](http://www.netlib.org/lapack/explore-html/d6/d44/isamax_8f.html)
#[deprecated(note="All scan and reduce operations are impossible since Vulkano does not support enabling extensions in GLSL. This will be added when Vulkano supports this.")]
pub mod isamax { }