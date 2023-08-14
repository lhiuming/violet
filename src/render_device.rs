use std::borrow::Cow;
use std::ffi::CStr;
use std::io::Write;
use std::os::raw::c_void;

use ash::extensions::{ext, khr, nv};
use ash::vk;

pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl PhysicalDevice {
    pub fn pick_memory_type_index(
        &self,
        memory_type_bits: u32,
        property_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let memory_types = &self.memory_properties.memory_types;
        for i in 0..self.memory_properties.memory_type_count {
            if ((memory_type_bits & (1 << i)) != 0)
                && ((memory_types[i as usize].property_flags & property_flags) == property_flags)
            {
                return Some(i);
            }
        }

        let mut support_properties = Vec::new();
        for i in 0..self.memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0 {
                support_properties.push(memory_types[i as usize].property_flags);
            }
        }

        println!(
            "Vulkan: No compatible device memory type with required properties {:?}. Support types are: {:?}",
            property_flags, support_properties
        );
        return None;
    }
}

// Convinient accessors
impl PhysicalDevice {
    #[inline]
    pub fn shader_group_handle_size(&self) -> u32 {
        self.ray_tracing_pipeline_properties
            .shader_group_handle_size
    }

    #[inline]
    pub fn shader_group_base_alignment(&self) -> u32 {
        self.ray_tracing_pipeline_properties
            .shader_group_base_alignment
    }
}

pub struct RenderDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: PhysicalDevice,

    // Entry wrappers
    pub device_entry: ash::Device,
    pub swapchain_entry: SwapchainEntry, // TODO remove this wrapping
    pub surface_entry: khr::Surface,
    pub raytracing_pipeline_entry: khr::RayTracingPipeline,
    pub acceleration_structure_entry: khr::AccelerationStructure,
    pub nv_diagnostic_checkpoints_entry: nv::DeviceDiagnosticCheckpoints,

    pub gfx_queue: vk::Queue,

    pub surface: Surface,
    pub swapchain: Swapchain,
}

impl RenderDevice {
    pub fn create(app_handle: u64, window_handle: u64) -> RenderDevice {
        // Load functions
        //let entry = unsafe { ash::Entry::new().expect("Ash entry creation failed") };
        let entry = unsafe {
            ash::Entry::load()
                .expect("Ash: can not load Vulkan. Maybe vulkan runtime is not installed.")
        };

        // Create instance
        let instance = {
            let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);

            let validation_layer_name =
                CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
            let layer_names = [
                validation_layer_name.as_ptr(), // Vulkan validation (debug layer)
            ];

            let ext_names_raw = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(), // Platform: Windows
                ext::DebugUtils::name().as_ptr(),   // Debug
            ];

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&ext_names_raw);

            print!("Vulkan: creating instance ... ");
            std::io::stdout().flush().unwrap();
            let instance = unsafe { entry.create_instance(&create_info, None) }
                .expect("Vulkan instance creation failed");
            println!("done.");
            instance
        };

        // Debug callback
        let debug_report = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        unsafe {
            use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
            use vk::DebugUtilsMessageTypeFlagsEXT as Type;
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(Severity::ERROR | Severity::WARNING)
                .message_type(Type::PERFORMANCE | Type::VALIDATION)
                .pfn_user_callback(Some(vulkan_debug_report_callback));
            debug_report
                .create_debug_utils_messenger(&create_info, None)
                .expect("Failed to register debug callback");
            println!("Vulkan Debug report callback registered.");
        }

        // Pick physical device
        let physical_device = {
            let phy_devs = unsafe { instance.enumerate_physical_devices() }.unwrap();
            assert!(phy_devs.len() > 0);
            let picked = phy_devs
                .iter()
                .find(|phy_dev| {
                    let prop = unsafe { instance.get_physical_device_properties(**phy_dev) };
                    prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .or(phy_devs.last());
            // print some info
            if picked.is_some() {
                let phy_dev = *picked.unwrap();
                let prop = unsafe { instance.get_physical_device_properties(phy_dev) };
                let name = unsafe { CStr::from_ptr(prop.device_name.as_ptr()) };
                println!("Vulkan: using physical device {:?}", name);
            }
            *picked.expect("Vulkan: None physical device?!")
        };

        // Get Physical Device properties (core and extensions)
        let mut physical_device_ray_tracing_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::builder().build();
        let mut physical_device_properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut physical_device_ray_tracing_pipeline_properties);
        unsafe {
            instance
                .get_physical_device_properties2(physical_device, &mut physical_device_properties2)
        }

        // Get supported device extensions (for debug info)
        /*
        let supported_device_extensions: HashSet<_> = unsafe {
            let extension_properties = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();
            extension_properties
                .iter()
                .map(|ext| CStr::from_ptr(ext.extension_name.as_ptr()).to_owned())
                .collect()
        };
        */

        // Get memory properties
        let physical_device_mem_prop =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Record in a wrapping struct
        let physical_device = PhysicalDevice {
            handle: physical_device,
            properties: physical_device_properties2.properties,
            ray_tracing_pipeline_properties: physical_device_ray_tracing_pipeline_properties,
            memory_properties: physical_device_mem_prop,
        };

        // Create device
        let gfx_queue_family_index;
        let device = {
            // Enumerate and pick queue families to create with the device
            let queue_fams = unsafe {
                instance.get_physical_device_queue_family_properties(physical_device.handle)
            };
            let mut found_gfx_queue_family_index = 0;
            for i in 0..queue_fams.len() {
                let queue_fam = &queue_fams[i];
                if queue_fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    found_gfx_queue_family_index = i as u32;
                    println!(
                        "Vulkan: found graphics queue family index: index {}, count {}",
                        i, queue_fam.queue_count
                    );
                    continue;
                }
                if queue_fam.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    println!(
                        "Vulkan: found compute queue family index: index {}, count {}",
                        i, queue_fam.queue_count
                    );
                    continue;
                }
            }
            gfx_queue_family_index = found_gfx_queue_family_index;
            let queue_create_infos = [
                // Just create a graphics queue for everything ATM...
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(gfx_queue_family_index)
                    .queue_priorities(&[1.0])
                    .build(),
            ];
            // Specify device extensions
            let enabled_extension_names = [
                khr::Swapchain::name().as_ptr(),
                // Raytracing Extensions (and dependencies under vulkan 1.2)
                vk::KhrRayTracingPipelineFn::name().as_ptr(),
                vk::KhrAccelerationStructureFn::name().as_ptr(),
                vk::KhrDeferredHostOperationsFn::name().as_ptr(),
                // DEVICE_LOST debug tools
                vk::NvDeviceDiagnosticCheckpointsFn::name().as_ptr(),
                // Workaround a DXC bug (causing all raytracing shader requiring ray query extensions when it is not used at all)
                // ref: https://github.com/microsoft/DirectXShaderCompiler/commit/ce31e10902732c8cd8f6f3b5b78699110afddb2b#diff-44e37c9720575ff94b7842b9ceb70a87fe72486d2b5da2e3828512dc64a352e6R217-R222
                vk::KhrRayQueryFn::name().as_ptr(),
            ];
            // Get physical device supported features
            let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
            let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default();
            let mut ray_tracing_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
            let mut acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
            let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
            let mut supported_features = vk::PhysicalDeviceFeatures2::builder()
                .push_next(&mut vulkan12_features)
                .push_next(&mut vulkan13_features)
                .push_next(&mut ray_tracing_pipeline_features)
                .push_next(&mut acceleration_structure_features)
                .push_next(&mut ray_query_features)
                .build();
            unsafe {
                instance
                    .get_physical_device_features2(physical_device.handle, &mut supported_features);
            };
            // Check features
            // Dynamic Rendering
            assert!(vulkan13_features.dynamic_rendering == vk::TRUE);
            // Bindless
            assert!(vulkan12_features.descriptor_binding_partially_bound == vk::TRUE);
            assert!(vulkan12_features.runtime_descriptor_array == vk::TRUE);
            // Ray Tracing
            assert!(ray_tracing_pipeline_features.ray_tracing_pipeline == vk::TRUE);
            assert!(acceleration_structure_features.acceleration_structure == vk::TRUE);
            // Buffer Device Address (required by ray tracing, to retrive buffer address for shader binding table)
            assert!(vulkan12_features.buffer_device_address == vk::TRUE);
            // Finally, create the device, with all supproted feature enabled (for simplicity)
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&enabled_extension_names)
                .push_next(&mut supported_features);
            unsafe {
                let ret = instance.create_device(physical_device.handle, &create_info, None);

                // Return extension not present
                if let Err(err) = ret {
                    match err {
                        vk::Result::ERROR_EXTENSION_NOT_PRESENT => {
                            for name in enabled_extension_names.iter() {
                                let name = CStr::from_ptr(*name);
                                println!("Error[Vulkan]: extension not present: {:?}", name);
                            }
                            println!("NOTE: extensions supported may be limited if device is create under debugging layers, e.g. VK_LAYER_RENDERDOC_Capture");
                        }
                        _ => {}
                    }
                }

                ret.unwrap()
            }
        };

        // Get quques
        let gfx_queue = unsafe { device.get_device_queue(gfx_queue_family_index, 0) };

        // Create surface
        let surface_entry = khr::Surface::new(&entry, &&instance);
        let win_surface_entry = khr::Win32Surface::new(&entry, &instance);
        let surface: Surface = {
            // Create platform surface
            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(app_handle as vk::HINSTANCE)
                .hwnd(window_handle as vk::HWND);
            let vk_surface = unsafe { win_surface_entry.create_win32_surface(&create_info, None) }
                .expect("Vulkan: failed to crate win32 surface");
            // Query format
            let formats = unsafe {
                surface_entry
                    .get_physical_device_surface_formats(physical_device.handle, vk_surface)
            }
            .unwrap();
            assert!(formats.len() > 0);
            // Debug
            println!("Vulkan surface supported formats:");
            for format in formats.iter() {
                println!("\t{:?}: {:?}", format.format, format.color_space);
            }
            Surface {
                handle: vk_surface,
                format: formats[0],
            }
        };

        // Create swapchain
        let swapchain_entry = SwapchainEntry::new(&instance, &device);
        let swapchain = {
            let surface_size = surface.query_size(&surface_entry, &physical_device.handle);
            swapchain_entry.create(&device, &surface, &surface_size)
        };

        let raytracing_pipeline_entry = khr::RayTracingPipeline::new(&instance, &device);
        let acceleration_structure_entry = khr::AccelerationStructure::new(&instance, &device);
        let nv_diagnostic_checkpoints_entry =
            nv::DeviceDiagnosticCheckpoints::new(&instance, &device);

        RenderDevice {
            entry,
            instance,
            physical_device,
            device_entry: device,
            surface_entry,
            swapchain_entry,
            raytracing_pipeline_entry,
            acceleration_structure_entry,
            nv_diagnostic_checkpoints_entry,

            gfx_queue,

            surface,
            swapchain,
        }
    }

    pub fn create_fence(&self, signaled: bool) -> vk::Fence {
        unsafe {
            let create_info = vk::FenceCreateInfo::builder().flags(if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            });
            self.device_entry.create_fence(&create_info, None)
        }
        .expect("Vulakn: failed to create fence")
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        unsafe {
            let create_info = vk::SemaphoreCreateInfo::builder();
            self.device_entry.create_semaphore(&create_info, None)
        }
        .expect("Vulkan: failed to create semaphore")
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        unsafe {
            self.device_entry
                .create_command_pool(&create_info, None)
                .expect("Vulkan: failed to create command pool?!")
        }
    }

    pub fn create_command_buffer(&self, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();
        unsafe {
            self.device_entry
                .allocate_command_buffers(&create_info)
                .expect("Vulkan: failed to allocated command buffer?!")[0]
        }
    }

    pub fn create_descriptor_pool(
        &self,
        flags: vk::DescriptorPoolCreateFlags,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> vk::DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);
        unsafe { self.device_entry.create_descriptor_pool(&create_info, None) }
            .expect("Vulkan: failed to create descriptor pool?!")
    }
}

// Debug
unsafe extern "system" fn vulkan_debug_report_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    // Allow break on error
    // TODO insert command line option
    if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        unsafe {
            std::intrinsics::breakpoint();
        }
    }

    vk::FALSE
}

// Surface and Swapchain stuffs

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
}

impl Surface {
    pub fn query_size(
        &self,
        entry: &khr::Surface,
        physical_device: &vk::PhysicalDevice,
    ) -> vk::Extent2D {
        let cap = unsafe {
            entry.get_physical_device_surface_capabilities(*physical_device, self.handle)
        }
        .unwrap();
        cap.current_extent
    }
}

pub struct Swapchain {
    pub extent: vk::Extent2D,
    pub handle: vk::SwapchainKHR,
    pub num_image: u32,
    //pub image: [vk::Image; 8],
    //pub image_view: [vk::ImageView; 8],
    // TODO use array vec
    pub image: Vec<Texture>,
    pub image_view: Vec<TextureView>,
}

impl Swapchain {
    fn default() -> Swapchain {
        Swapchain {
            extent: vk::Extent2D {
                width: 0,
                height: 0,
            },
            handle: vk::SwapchainKHR::default(),
            num_image: 0,
            //image: [vk::Image::default(); 8],
            //image_view: [vk::ImageView::default(); 8],
            image: Vec::new(),
            image_view: Vec::new(),
        }
    }
}

pub struct SwapchainEntry {
    pub entry: khr::Swapchain,
}

impl SwapchainEntry {
    fn new(instance: &ash::Instance, device: &ash::Device) -> SwapchainEntry {
        SwapchainEntry {
            entry: khr::Swapchain::new(instance, device),
        }
    }

    fn create(&self, device: &ash::Device, surface: &Surface, extent: &vk::Extent2D) -> Swapchain {
        let mut ret = Swapchain::default();
        ret.extent = *extent;
        //let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE // compute post processing

        let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE;

        // Create swapchain object
        let create_info = {
            vk::SwapchainCreateInfoKHR::builder()
                .flags(vk::SwapchainCreateFlagsKHR::empty())
                .surface(surface.handle)
                .min_image_count(2)
                .image_format(surface.format.format)
                .image_color_space(surface.format.color_space)
                .image_extent(*extent)
                .image_array_layers(1)
                .image_usage(image_usage)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
        };
        ret.handle = unsafe { self.entry.create_swapchain(&create_info, None) }
            .expect("Vulkan: Swapchain creatino failed???");

        // Get images
        {
            let images = unsafe { self.entry.get_swapchain_images(ret.handle) }.unwrap_or(vec![]);
            ret.num_image = images.len() as u32;

            // Add ret
            for image in images {
                let desc = TextureDesc::new_2d(
                    extent.width,
                    extent.height,
                    surface.format.format,
                    image_usage,
                );
                ret.image.push(Texture {
                    desc,
                    image,
                    memory: vk::DeviceMemory::null(), // TODO is this ok?
                });
            }
        }

        // Create image views
        let sub_res_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);
        for img_index in 0..ret.num_image as usize {
            let texture = ret.image[img_index];
            let image_view = unsafe {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(texture.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface.format.format)
                    .subresource_range(*sub_res_range);
                device
                    .create_image_view(&create_info, None)
                    .expect("Vulkan: failed to create image view for swapchain")
            };
            let desc = TextureViewDesc::auto(&texture.desc);
            ret.image_view.push(TextureView {
                texture,
                desc,
                image_view,
            });
        }

        ret
    }

    /*
    fn detroy(&self, device: &ash::Device, swapchain: &mut Swapchain) {
        for image_index in 0..swapchain.num_image as usize {
            let image_view = swapchain.image_view[image_index];
            unsafe {
                device.destroy_image_view(image_view, None);
            }
        }
        unsafe {
            self.entry.destroy_swapchain(swapchain.handle, None);
        }
        swapchain.handle = vk::SwapchainKHR::default();
        swapchain.num_image = 0;
    }
    */
}

impl RenderDevice {
    // Wait undefinitely for next swapchain image.
    // Use a semaphore or fence to wait until the image is ok to be modified.
    #[inline]
    pub fn acquire_next_swapchain_image(
        &self,
        semaphore_to_signal: vk::Semaphore,
        fence_to_signal: vk::Fence,
    ) -> u32 {
        // Validate "semaphore and fence must not both be equal to VK_NULL_HANDLE"
        assert!(
            (semaphore_to_signal != vk::Semaphore::null())
                || (fence_to_signal != vk::Fence::null())
        );

        let wait = std::time::Instant::now();
        let (index, is_suboptimal) = unsafe {
            self.swapchain_entry
                .entry
                .acquire_next_image(
                    self.swapchain.handle,
                    std::u64::MAX,
                    semaphore_to_signal,
                    fence_to_signal,
                )
                .expect("RenderDevice: failed to acquire next swapchain image")
        };

        // Warn if acquire next image takes too long
        let elapsed = wait.elapsed().as_micros();
        if elapsed > 500 {
            println!("RenderDevice: acquire next image takes {} ms!", elapsed);
        }

        if is_suboptimal {
            panic!("RenderDevice: acquired surface image has unexpected properties");
        }

        index
    }
}

/*
 * Common Resource Types
 */

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: vk::BufferUsageFlags,
    pub memory_property: vk::MemoryPropertyFlags,
}

impl BufferDesc {
    // Read/Write in GPU
    pub fn compute(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    // Write (typically once) in CPU, read in GPU
    pub fn shader_binding_table(size: u64) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO no need to coherent?
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Buffer {
    pub desc: BufferDesc,
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub data: *mut u8, // TODO make optional
    pub device_address: Option<vk::DeviceAddress>,
}

impl RenderDevice {
    pub fn create_buffer(&self, desc: BufferDesc) -> Option<Buffer> {
        let device = &self.device_entry;

        // Create the vk buffer object
        // TODO drop buffer if later stage failed
        let buffer = {
            let create_info = vk::BufferCreateInfo::builder()
                .size(desc.size)
                .usage(desc.usage);
            match unsafe { device.create_buffer(&create_info, None) } {
                Ok(buffer) => buffer,
                Err(err) => {
                    println!("RenderDevice: (Vulkan) failed to create buffer: {:?}", err);
                    return None;
                }
            }
        };

        let has_device_address = desc
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        // Allocate memory for ths buffer
        // TODO drop device_memory if later stage failed
        // TODO use a allocator like VMA to do sub allocate
        let memory: vk::DeviceMemory = {
            let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

            // Pick memory type
            /*
            let mem_property_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            */
            let memory_type_index = self
                .physical_device
                .pick_memory_type_index(mem_req.memory_type_bits, desc.memory_property)
                .unwrap();

            let mut flags = vk::MemoryAllocateFlags::default();
            if has_device_address {
                // allocation requirement (03339)
                flags |= vk::MemoryAllocateFlags::DEVICE_ADDRESS;
            }
            let mut flag_info = vk::MemoryAllocateFlagsInfo::builder().flags(flags).build();

            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_req.size)
                .memory_type_index(memory_type_index)
                .push_next(&mut flag_info);
            match unsafe { device.allocate_memory(&create_info, None) } {
                Ok(mem) => mem,
                Err(err) => {
                    println!("RenderDevice: (Vulkan) failed to bind buffer: {:?}", err);
                    return None;
                }
            }
        };

        // Bind
        let offset: vk::DeviceSize = 0;
        match unsafe { device.bind_buffer_memory(buffer, memory, offset) } {
            Ok(_) => {}
            Err(err) => {
                println!("RenderDevice: (Vulkan) failed to bind buffer: {:?}", err);
                return None;
            }
        }

        // Get address (for later use, e.g. ray tracing)
        let device_address = if has_device_address {
            unsafe {
                let info = vk::BufferDeviceAddressInfo::builder().buffer(buffer);
                Some(device.get_buffer_device_address(&info))
            }
        } else {
            None
        };

        // Map (staging buffer) persistently
        // TODO unmap if later stage failed
        let is_mappable = desc
            .memory_property
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        let data = if is_mappable {
            let map_flags = vk::MemoryMapFlags::default(); // dummy parameter
            unsafe { device.map_memory(memory, offset, desc.size, map_flags) }.unwrap() as *mut u8
        } else {
            std::ptr::null_mut::<u8>()
        };

        Some(Buffer {
            desc,
            handle: buffer,
            memory,
            data,
            device_address,
        })
    }

    pub fn create_buffer_view(
        &self,
        buffer: vk::Buffer,
        format: vk::Format,
    ) -> Option<vk::BufferView> {
        // Create SRV
        let create_info = vk::BufferViewCreateInfo::builder()
            .buffer(buffer)
            .format(format)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let srv = unsafe { self.device_entry.create_buffer_view(&create_info, None) }.ok()?;
        Some(srv)
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub layer_count: u32,
    pub mip_level_count: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            layer_count: 1,
            mip_level_count: 1,
            format: vk::Format::UNDEFINED,
            usage: vk::ImageUsageFlags::default(),
            flags: vk::ImageCreateFlags::default(),
        }
    }
}

impl TextureDesc {
    pub fn new_2d(
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> TextureDesc {
        TextureDesc {
            width,
            height,
            layer_count: 1,
            mip_level_count: 1,
            format,
            usage,
            flags: vk::ImageCreateFlags::default(),
        }
    }

    pub fn new_2d_array(
        width: u32,
        height: u32,
        array_len: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> TextureDesc {
        TextureDesc {
            width,
            height,
            layer_count: array_len,
            mip_level_count: 1,
            format,
            usage,
            flags: vk::ImageCreateFlags::default(),
        }
    }

    pub fn with_flags(mut self, flag: vk::ImageCreateFlags) -> Self {
        self.flags = flag;
        self
    }

    pub fn size_2d(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width,
            height: self.height,
        }
    }

    pub fn size_3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.layer_count,
        }
    }
}

// Mini struct for a texture
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Texture {
    pub desc: TextureDesc,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
}

/*
impl PartialEq for Texture {
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image
    }
}

impl Eq for Texture {}
 */

// Mini struct for a texture view
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct TextureViewDesc {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl Default for TextureViewDesc {
    fn default() -> Self {
        TextureViewDesc {
            view_type: vk::ImageViewType::TYPE_2D,
            format: vk::Format::UNDEFINED,
            aspect: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        }
    }
}

pub fn format_has_depth(format: vk::Format) -> bool {
    (format == vk::Format::D16_UNORM)
        || (format == vk::Format::D16_UNORM_S8_UINT)
        || (format == vk::Format::D24_UNORM_S8_UINT)
        || (format == vk::Format::D32_SFLOAT)
        || (format == vk::Format::D32_SFLOAT_S8_UINT)
}

impl TextureViewDesc {
    pub fn auto(texture_desc: &TextureDesc) -> TextureViewDesc {
        let view_type = if texture_desc.layer_count > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };
        let format = texture_desc.format;
        let has_depth = format_has_depth(format);
        let aspect = if has_depth {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        TextureViewDesc {
            view_type,
            format: texture_desc.format,
            aspect,
            base_mip_level: 0,
            level_count: texture_desc.mip_level_count,
            base_array_layer: 0,
            layer_count: texture_desc.layer_count,
        }
    }

    pub fn with_format(texture_desc: &TextureDesc, format: vk::Format) -> TextureViewDesc {
        let mut desc = Self::auto(texture_desc);
        desc.format = format;
        desc
    }

    pub fn make_subresource_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: self.base_mip_level,
            level_count: self.level_count,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }

    pub fn make_subresrouce_layer(&self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers {
            aspect_mask: self.aspect,
            mip_level: self.base_mip_level,
            base_array_layer: self.base_array_layer,
            layer_count: self.layer_count,
        }
    }

    pub fn extent3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: 0,
            height: 0,
            depth: 1,
        }
    }
}

#[derive(Clone, Copy)]
pub struct TextureView {
    pub texture: Texture,
    pub desc: TextureViewDesc,
    pub image_view: vk::ImageView,
}

impl PartialEq for TextureView {
    fn eq(&self, other: &Self) -> bool {
        self.image_view == other.image_view
    }
}

impl RenderDevice {
    pub fn create_texture(&self, desc: TextureDesc) -> Option<Texture> {
        let device = &self.device_entry;

        let format_prop = unsafe {
            self.instance.get_physical_device_image_format_properties(
                self.physical_device.handle,
                desc.format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::default(),
                desc.usage,
                vk::ImageCreateFlags::default(),
            )
        };
        if let Err(e) = format_prop {
            println!(
                "Error: texture creation for {:?} failed: {:?}. Try something else.",
                desc.format, e
            );
            return None;
        }

        // Create image object
        let initial_layout = vk::ImageLayout::UNDEFINED;
        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .array_layers(desc.layer_count)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .initial_layout(initial_layout)
            .usage(desc.usage)
            .flags(desc.flags);
        let image = unsafe { device.create_image(&create_info, None) }.unwrap();

        // Bind memory
        let device_memory = {
            let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
            let momory_type_index = self.physical_device.pick_memory_type_index(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL, // That's basically what texture can have
            )?;
            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(momory_type_index);
            unsafe { device.allocate_memory(&create_info, None) }.unwrap()
        };
        unsafe { device.bind_image_memory(image, device_memory, 0) }.unwrap();

        Some(Texture {
            desc,
            image,
            memory: device_memory,
        })
    }

    pub fn create_texture_view(
        &self,
        texture: Texture,
        desc: TextureViewDesc,
    ) -> Option<TextureView> {
        let device = &self.device_entry;
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image)
            .view_type(desc.view_type)
            .format(desc.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: desc.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: texture.desc.layer_count,
            });
        let image_view = unsafe { device.create_image_view(&create_info, None) }.ok()?;

        Some(TextureView {
            texture,
            desc,
            image_view,
        })
    }
}

#[derive(Clone, Copy)]
pub struct AccelerationStructure {
    pub buffer: Buffer,
    pub ty: vk::AccelerationStructureTypeKHR,
    pub handle: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress, // used to fill vk::AccelerationStructureInstanceKHR::accelerationStructureReference
}

impl PartialEq for AccelerationStructure {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl RenderDevice {
    pub fn create_accel_struct(
        &self,
        buffer: Buffer,
        offset: u64,
        size: u64,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> Option<AccelerationStructure> {
        assert!(buffer.desc.size >= (offset + size));
        assert!(offset & 0xff == 0); // required by spec
        let create_info: vk::AccelerationStructureCreateInfoKHRBuilder<'_> =
            vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(buffer.handle)
                .offset(offset)
                .size(size)
                .ty(ty);
        let handle = unsafe {
            self.acceleration_structure_entry
                .create_acceleration_structure(&create_info, None)
                .unwrap()
        };

        let device_address = unsafe {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(handle);
            self.acceleration_structure_entry
                .get_acceleration_structure_device_address(&info)
        };

        Some(AccelerationStructure {
            buffer,
            ty,
            handle,
            device_address,
        })
    }

    pub fn get_ray_tracing_shader_group_handles(
        &self,
        pipeline: vk::Pipeline,
        first_group: u32,
        group_count: u32,
    ) -> Vec<u8> {
        let handle_size = self
            .physical_device
            .ray_tracing_pipeline_properties
            .shader_group_handle_size;
        let data_size = group_count * handle_size;
        unsafe {
            self.raytracing_pipeline_entry
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    first_group,
                    group_count,
                    data_size as usize,
                )
                .unwrap()
        }
    }
}

/*
 * Helper and Utiliites
 */

// Helper to fill Shader Binding Table
pub struct ShaderBindingTableFiller {
    dst: *mut u8,
    pos: isize,
    handle_size: u32,
    base_alignment: u32,
}

impl ShaderBindingTableFiller {
    pub fn new(physical_device: &PhysicalDevice, dst: *mut u8) -> ShaderBindingTableFiller {
        ShaderBindingTableFiller {
            dst,
            pos: 0,
            handle_size: physical_device
                .ray_tracing_pipeline_properties
                .shader_group_handle_size,
            base_alignment: physical_device
                .ray_tracing_pipeline_properties
                .shader_group_base_alignment,
        }
    }

    pub fn start_group(&mut self) -> u32 {
        let mask = (self.base_alignment - 1) as isize;
        self.pos = (self.pos + mask) & !mask;
        self.pos as u32
    }

    pub fn write_handles(&mut self, data: &[u8], first_handle: u32, handle_count: u32) -> u32 {
        assert!(data.len() % self.handle_size as usize == 0);
        assert!(handle_count + first_handle <= (data.len() / self.handle_size as usize) as u32);
        let size = (handle_count * self.handle_size) as usize;

        unsafe {
            let src = std::slice::from_raw_parts(
                data.as_ptr()
                    .offset(first_handle as isize * self.handle_size as isize),
                size,
            );
            let dst = std::slice::from_raw_parts_mut(self.dst.offset(self.pos as isize), size);
            dst.copy_from_slice(src);
        }

        self.pos += size as isize;
        size as u32
    }
}
