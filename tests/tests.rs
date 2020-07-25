#[cfg(test)]
mod tests {
    use vulkano_blas::*;
    use itertools::izip;

    use vulkano::{
        instance::{
            Instance,
            InstanceExtensions,
            PhysicalDevice,
        },
        device::{
            Device,
            DeviceExtensions,
            Features,
            Queue
        },
        buffer::{
            BufferUsage,
            CpuAccessibleBuffer,
            ImmutableBuffer
        },
        command_buffer::{
            AutoCommandBufferBuilder,
            CommandBuffer
        },
        sync::GpuFuture,
        descriptor::{
            descriptor_set::PersistentDescriptorSet,
            PipelineLayoutAbstract
        },
        pipeline::ComputePipeline
    };
    
    use std::sync::Arc;

    fn get_compute_queue_and_device() -> (Arc<Device>,Arc<Queue>){
        // Intiates an instance of Vulkan.
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .expect("failed to create instance");

        // Sets the physical device to use (e.g. the graphics card).
        let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

        // Selects the 1st queue family which supports compute
        let queue_family = physical.queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a compute queue family");

        // `device` is the channel through which we communicate with our physical device `physical`.
        // Initates a device connecting to `queue_family`.
        let (device, mut queues) = {
            Device::new(
                physical, 
                &Features::none(), 
                &DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    .. DeviceExtensions::none()
                },
                [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
        };

        // Gets 1st queue of device.
        let queue = queues.next().unwrap();

        return (device,queue)
    }
    #[test]
    fn sscal() {
        let (device,queue) = get_compute_queue_and_device();

        // Sets the shader (/kernel/function).
        let shader = sscal::Shader::load(device.clone()).expect("failed to create shader module");

        // Compute pipeline holder list of shaders to run seqeuntially.
        // Sets the compute pipeline (list of shaders).
        let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

        // Sets the data buffer (x vector).
        // CLoning x here is unneccessary for functionality, it is only convenient such that we can use it later for checking the result.
        let x = 0..5u32;
        let data_buffer_x = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,x.clone()).expect("failed to create buffer");
        
        // Initiates a scalar.
        let a = 2u32;

        // Gets the layout of the pipeline (x,y,z)
        let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
        // Descriptors push buffers to the compute pipeline.
        // Intiates descriptor set.
        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer_x.clone()).unwrap()
            .build().unwrap()
        );

        // Intiates command buffer for compute pipeline (1 work groups)
        let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        // Dispatches command buffer builder passing a scalar as push constant.
        builder.dispatch([1024,1,1],compute_pipeline.clone(),set.clone(),a).unwrap();

        let command_buffer = builder.build().unwrap();

        // Submit command buffer.
        let finished = command_buffer.execute(queue.clone()).unwrap();

        // Halts thread until command buffer executed.
        finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

        // Checks content of data buffer.
        let x_content = data_buffer_x.read().unwrap();
        for (new_x, x) in x_content.iter().zip(x) {
            assert_eq!(*new_x, x * a);
        }
    }
    #[test]
    fn saxpy() {
        let (device,queue) = get_compute_queue_and_device();

        // Sets the shader (/kernel/function).
        let shader = saxpy::Shader::load(device.clone()).expect("failed to create shader module");

        // Compute pipeline holder list of shaders to run seqeuntially.
        // Sets the compute pipeline (list of shaders).
        let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

        // Sets data buffers (y and x vectors).
        // CLoning y and x here is unneccessary for functionality, it is only convenient such that we can use it later for checking the result.
        let y = 0..5u32;
        let data_buffer_y = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,y.clone()).expect("failed to create buffer");
        let x = 5..10u32;
        let (data_buffer_x,x_future) = ImmutableBuffer::from_iter(x.clone(), BufferUsage::all(), queue.clone()).expect("failed to create buffer");
        
        // Halts thread until data_buffer_x intialised.
        x_future.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

        // Initiates a scalar.
        let a = 2u32;

        // Gets the layout of the pipeline (x,y,z)
        let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
        // Intiates descriptor set.
        let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer_x.clone()).unwrap()
            .add_buffer(data_buffer_y.clone()).unwrap()
            .build().unwrap()
        );

        // Intiates command buffer builder.
        let mut builder = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap();
        // Dispatches command buffer builder passing a scalar as push constant.
        builder.dispatch([1024,1,1],compute_pipeline.clone(),set.clone(),a).unwrap();
        // Intiates command buffer.
        let command_buffer = builder.build().unwrap();

        // Submits command buffer.
        let finished = command_buffer.execute(queue.clone()).unwrap();

        // Halts thread until command buffer executed.
        finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

        // Checks content of data buffer.
        let y_content = data_buffer_y.read().unwrap();
        for (new_y,old_y,old_x) in izip!(y_content.iter(),y,x) {
            assert_eq!(*new_y, a * old_x + old_y);
        }
    }
}