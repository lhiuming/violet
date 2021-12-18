use ash::extensions::{ext, khr};
use ash::vk;
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::c_void;

mod window;
use window::Window;

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

    vk::FALSE
}

struct Surface {
    pub handle: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
}

impl Surface {
    fn query_size(
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

struct Swapchain {
    pub width: u32,
    pub height: u32,
    pub handle: vk::SwapchainKHR,
    pub num_image: u32,
    pub image: [vk::Image; 8],
    pub image_view: [vk::ImageView; 8],
}

impl Swapchain {
    fn default() -> Swapchain {
        Swapchain {
            width: 0,
            height: 0,
            handle: vk::SwapchainKHR::default(),
            num_image: 0,
            image: [vk::Image::default(); 8],
            image_view: [vk::ImageView::default(); 8],
        }
    }
}

struct SwapchainEntry {
    entry: khr::Swapchain,
}

impl SwapchainEntry {
    fn new(instance: &ash::Instance, device: &ash::Device) -> SwapchainEntry {
        SwapchainEntry {
            entry: khr::Swapchain::new(instance, device),
        }
    }

    fn create(&self, device: &ash::Device, surface: &Surface, extent: &vk::Extent2D) -> Swapchain {
        let mut ret = Swapchain::default();
        ret.width = extent.width;
        ret.height = extent.height;

        // Create swapchain object
        let create_info = {
            use vk::ImageUsageFlags as Usage;
            vk::SwapchainCreateInfoKHR::builder()
                .flags(vk::SwapchainCreateFlagsKHR::empty())
                .surface(surface.handle)
                .min_image_count(2)
                .image_format(surface.format.format)
                .image_color_space(surface.format.color_space)
                .image_extent(*extent)
                .image_array_layers(1)
                .image_usage(Usage::COLOR_ATTACHMENT | Usage::TRANSFER_DST | Usage::STORAGE)
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
            ret.image[0..images.len()].copy_from_slice(&images);
        }

        // Create image views
        let sub_res_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1)
            .level_count(1);
        for img_index in 0..ret.num_image as usize {
            ret.image_view[img_index] = unsafe {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(ret.image[img_index])
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface.format.format)
                    .subresource_range(*sub_res_range);
                device
                    .create_image_view(&create_info, None)
                    .expect("Vulkan: failed to create image view for swapchain")
            }
        }

        ret
    }

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
}

pub struct PipelineDevice {
    device: ash::Device,
    pipeline_cache: vk::PipelineCache,
    descriptor_pool: vk::DescriptorPool,
}

use spirv_reflect;

// AKA shader
pub struct PipelineProgram {
    pub shader_module: vk::ShaderModule,
    pub set_layout: vk::DescriptorSetLayout,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

fn create_pipeline_program(
    device: &PipelineDevice,
    binary: &[u32],
    entry_point: &str,
) -> Option<PipelineProgram> {
    let pipeline_cache = device.pipeline_cache;
    let descriptor_pool = device.descriptor_pool;
    let device = &device.device;

    // Create shader module
    let shader_module = {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(binary);
        unsafe { device.create_shader_module(&create_info, None) }.ok()?
    };

    // Get reflect info
    let bin_char =
        unsafe { std::slice::from_raw_parts(binary.as_ptr() as *const u8, binary.len() * 4) };
    let reflect_module = spirv_reflect::create_shader_module(bin_char).ok()?;

    // Create all used set layouts
    let set_layout = {
        assert!(
            reflect_module
                .enumerate_descriptor_sets(Some(entry_point))
                .unwrap()
                .len()
                <= 1
        );
        let bindings = reflect_module
            .enumerate_descriptor_bindings(Some(entry_point))
            .unwrap()
            .iter()
            .map(|b| {
                assert!(
                    b.descriptor_type != spirv_reflect::types::ReflectDescriptorType::Undefined
                );
                assert!(
                    b.descriptor_type
                        != spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV
                );
                vk::DescriptorSetLayoutBinding {
                    binding: b.binding,
                    // TODO ReflectDescriptorType does not match vk::DescriptorType in bit value, also AccelerationStructureNV is in spirv_reflect but not in ash
                    descriptor_type: vk::DescriptorType::from_raw(b.descriptor_type as i32 - 1),
                    descriptor_count: b.count,
                    stage_flags: vk::ShaderStageFlags::from_raw(
                        reflect_module.get_shader_stage().bits(),
                    ),
                    p_immutable_samplers: std::ptr::null(),
                }
            })
            .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
        let create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings.as_slice());
        unsafe { device.create_descriptor_set_layout(&create_info, None) }.ok()?
    };

    let set_layouts = [set_layout];

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Create pipeline object
    let pipeline = {
        let entry_point_c =
            CString::new(entry_point).expect(&format!("Bad entry point name: {}", entry_point));
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point_c);
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(layout);
        let create_infos = [create_info.build()];
        match unsafe { device.create_compute_pipelines(pipeline_cache, &create_infos, None) } {
            Ok(pipeline) => pipeline[0],
            Err(_) => return None,
        }
    };

    // Create all used descriptor sets
    let descriptor_sets = {
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe { device.allocate_descriptor_sets(&create_info) }.ok()?
    };

    Some(PipelineProgram {
        shader_module,
        set_layout,
        layout,
        pipeline,
        descriptor_sets,
    })
}

mod shader {
    use shaderc;

    use std::fs::File;
    use std::io::Read;
    use std::path::{Path, PathBuf};

    use std::collections::HashMap;

    use crate::PipelineDevice;
    use crate::PipelineProgram;

    pub struct ShaderDefinition {
        pub virutal_path: String,
        pub entry_point: String,
    }

    pub struct CompiledShader {
        pub artifact: shaderc::CompilationArtifact,
        pub program: PipelineProgram,
    }

    impl CompiledShader {}

    type CompiledShaderSet = HashMap<String, CompiledShader>;

    pub struct ShaderLibrary {
        shader_types: Vec<ShaderDefinition>,
        compiled_shaders: CompiledShaderSet,
    }

    impl ShaderLibrary {
        pub fn new() -> ShaderLibrary {
            ShaderLibrary {
                shader_types: Vec::new(),
                compiled_shaders: HashMap::new(),
            }
        }

        pub fn add_shader(&mut self, shader_type: ShaderDefinition) {
            self.shader_types.push(shader_type);
        }

        pub fn find_shader(&self, virtual_path: &str) -> Option<&CompiledShader> {
            self.compiled_shaders.get(virtual_path)
        }

        fn update_shader(
            set: &mut CompiledShaderSet,
            device: &PipelineDevice,
            v_path: &str,
            entry_point: &str,
        ) {
            match load_shader(device, v_path, entry_point) {
                Some(compiled) => {
                    set.insert(v_path.to_string(), compiled);
                }
                _ => (),
            };
        }

        pub fn update(&mut self, device: &PipelineDevice) {
            for shader_type in self.shader_types.iter() {
                ShaderLibrary::update_shader(
                    &mut self.compiled_shaders,
                    device,
                    &shader_type.virutal_path,
                    &shader_type.entry_point,
                );
            }
        }
    }

    fn load_shader(
        device: &PipelineDevice,
        v_path: &str,
        entry_point: &str,
    ) -> Option<CompiledShader> {
        // todo map v_path to actuall pathes
        let mut path = PathBuf::new();
        path.push("./shader/");
        path.push(v_path);
        let display = path.display();

        // Read file content
        let mut file = match File::open(&path) {
            Err(why) => panic!("Coundn't open shader path {}: {}", display, why),
            Ok(file) => file,
        };
        let mut text = String::new();
        match file.read_to_string(&mut text) {
            Err(why) => panic!("Couldn't read file {}: {}", display, why),
            Ok(_) => (),
        };

        // Compile the shader
        let file_name_os = path.file_name().unwrap();
        let file_name = file_name_os.to_str().unwrap();
        let mut compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_source_language(shaderc::SourceLanguage::HLSL);
        let compile_result = compiler.compile_into_spirv(
            &text,
            shaderc::ShaderKind::Compute,
            file_name,
            entry_point,
            Some(&options),
        );

        let artifact = match compile_result {
            Err(why) => {
                println!("Shaer compiled binay is not valid: {}", why);
                return None;
            }
            Ok(artifact) => artifact,
        };

        let program =
            match crate::create_pipeline_program(device, artifact.as_binary(), entry_point) {
                Some(program) => program,
                None => return None,
            };

        Some(CompiledShader {
            artifact: artifact,
            program: program,
        })
    }
}

fn main() {
    println!("Hello, rusty world!");

    // Create a system window
    // TODO implement Drop for Window
    let window = Window::new(1280, 720, "Rusty Violet");

    // Load functions
    //let entry = unsafe { ash::Entry::new().expect("Ash entry creation failed") };
    let entry = ash::Entry::new();

    // Create instance
    let instance = {
        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_2);

        let layer_names = [
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(), // Debug
        ];
        let layer_names_raw: Vec<_> = layer_names.iter().map(|name| name.as_ptr()).collect();

        let ext_names_raw = [
            khr::Surface::name().as_ptr(),
            khr::GetPhysicalDeviceProperties2::name().as_ptr(), // Required by dynamic_rendering
            khr::Win32Surface::name().as_ptr(),                 // Platform: Windows
            ext::DebugUtils::name().as_ptr(),                   // Debug
        ];

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_raw)
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
            .message_severity(Severity::ERROR | Severity::WARNING | Severity::INFO)
            .message_type(Type::PERFORMANCE | Type::VALIDATION)
            .pfn_user_callback(Some(vulkan_debug_report_callback));
        debug_report
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to register debug callback");
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

    // Create device
    let b_support_dynamic_rendering;
    let gfx_queue_family_index;
    let device = {
        // Enumerate and pick queue families to create with the device
        let queue_fams =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
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
            khr::DynamicRendering::name().as_ptr(),
        ];
        // Query supported features
        let mut dynamic_rendering_ft = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
        let mut supported_features =
            vk::PhysicalDeviceFeatures2::builder().push_next(&mut dynamic_rendering_ft);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut supported_features);
        };
        // Enable all supported features
        let enabled_features = supported_features.features;
        b_support_dynamic_rendering = dynamic_rendering_ft.dynamic_rendering == vk::TRUE;
        // Finally, create the device
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enabled_extension_names)
            .enabled_features(&enabled_features)
            .push_next(&mut dynamic_rendering_ft);
        unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .expect("Failed to create Vulkan device")
        }
    };

    // Load dynamic_rendering
    assert!(b_support_dynamic_rendering);
    let dynamic_render = khr::DynamicRendering::new(&instance, &device);

    // Get quques
    let gfx_queue = unsafe { device.get_device_queue(gfx_queue_family_index, 0) };

    // Create surface
    let surface_entry = khr::Surface::new(&entry, &&instance);
    let win_surface_entry = khr::Win32Surface::new(&entry, &instance);
    let surface: Surface = {
        // Create platform surface
        let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
            .hinstance(Window::system_handle_for_module() as vk::HINSTANCE)
            .hwnd(window.system_handle() as vk::HWND);
        let vk_surface = unsafe { win_surface_entry.create_win32_surface(&create_info, None) }
            .expect("Vulkan: failed to crate win32 surface");
        // Query format
        let formats = unsafe {
            surface_entry.get_physical_device_surface_formats(physical_device, vk_surface)
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
    let mut swapchain = {
        let surface_size = surface.query_size(&surface_entry, &physical_device);
        swapchain_entry.create(&device, &surface, &surface_size)
    };

    let present_semaphore = unsafe {
        let create_info = vk::SemaphoreCreateInfo::builder();
        device.create_semaphore(&create_info, None)
    }
    .expect("Vulkan: failed to create semaphore");

    // Command buffer (and pool)
    let cmd_pool = {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        unsafe {
            device
                .create_command_pool(&create_info, None)
                .expect("Vulkan: failed to create command pool?!")
        }
    };
    let cmd_buf = {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(cmd_pool)
            .command_buffer_count(1)
            .build();
        unsafe {
            device
                .allocate_command_buffers(&create_info)
                .expect("Vulkan: failed to allocated command buffer?!")[0]
        }
    };

    // Descriptor pool
    let descriptor_pool = {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1024, // TODO big enough?
        }];
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(256) // TODO big enough?
            .pool_sizes(&pool_sizes);
        unsafe { device.create_descriptor_pool(&create_info, None) }
            .expect("Vulkan: failed to create descriptor pool?!")
    };

    // Initialize shaders
    let pipeline_device = PipelineDevice {
        device: device.clone(),
        pipeline_cache: vk::PipelineCache::null(),
        descriptor_pool: descriptor_pool,
    };
    let mut shader_lib = shader::ShaderLibrary::new();
    shader_lib.add_shader(shader::ShaderDefinition {
        virutal_path: String::from("MeshCS.hlsl"),
        entry_point: String::from("main"),
    });
    shader_lib.update(&pipeline_device);

    while !window.should_close() {
        window.poll_events();

        // wait idle (for now)
        unsafe {
            device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())
                .expect("Vulkan: Reset command buffer failed???");
        };

        // Resize swapchain
        let surface_size = surface.query_size(&surface_entry, &physical_device);
        let b_invalid_size = (surface_size.width == 0) || (surface_size.height == 0);
        if b_invalid_size {
            continue;
        }
        let b_resize =
            (surface_size.width != swapchain.width) || (surface_size.height != swapchain.height);
        if b_resize {
            swapchain_entry.detroy(&device, &mut swapchain);
            swapchain = swapchain_entry.create(&device, &surface, &surface_size);
        }

        // Acquire target image
        let (image_index, b_image_suboptimal) = unsafe {
            swapchain_entry.entry.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                present_semaphore,
                vk::Fence::default(),
            )
        }
        .expect("Vulkan: failed to acquire swapchain image");
        if b_image_suboptimal {
            println!("Vulkan: suboptimal image is get (?)");
        }

        // SIMPLE CONFIGURATION
        let b_clear_using_render_pass = true;
        let clear_color = vk::ClearColorValue {
            float32: [
                0x5A as f32 / 255.0,
                0x44 as f32 / 255.0,
                0x94 as f32 / 255.0,
                0xFF as f32 / 255.0,
            ],
        };

        // Being command recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device.begin_command_buffer(cmd_buf, &begin_info).unwrap();
            }
        }

        let mut swapchain_image_layout = vk::ImageLayout::UNDEFINED;

        // Transition for render
        if b_clear_using_render_pass {
            let sub_res_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(swapchain_image_layout)
                .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL_KHR)
                .subresource_range(*sub_res_range)
                .image(swapchain.image[image_index as usize]);
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier],
                );
            }

            swapchain_image_layout = image_barrier.new_layout;
        }

        // Begin render pass
        if b_support_dynamic_rendering {
            let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(swapchain.image_view[image_index as usize])
                .image_layout(swapchain_image_layout)
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color });
            let color_attachments = [*color_attachment];
            let rendering_info = vk::RenderingInfoKHR::builder()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_size,
                })
                .layer_count(1)
                .view_mask(0)
                .color_attachments(&color_attachments);
            unsafe {
                dynamic_render.cmd_begin_rendering(cmd_buf, &rendering_info);
            }
        }

        // TODO draw some mesh here

        // End render pass
        unsafe {
            dynamic_render.cmd_end_rendering(cmd_buf);
        }

        // Draw mesh with compute
        let mesh_cs = shader_lib.find_shader("MeshCS.hlsl");
        if let Some(mesh_cs) = mesh_cs {
            // Transition for compute
            {
                let sub_res_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1);
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(swapchain_image_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(*sub_res_range)
                    .image(swapchain.image[image_index as usize]);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd_buf,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[*image_barrier],
                    );
                }
                swapchain_image_layout = image_barrier.new_layout;
            }

            // Set and dispatch compute
            {
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        mesh_cs.program.pipeline,
                    )
                }

                // Bind the swapchain image (the only descriptor)
                unsafe {
                    let image_info = vk::DescriptorImageInfo::builder()
                        .image_view(swapchain.image_view[image_index as usize])
                        .image_layout(swapchain_image_layout);
                    let image_infos = [image_info.build()];
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(mesh_cs.program.descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_infos);
                    let writes = [write.build()];
                    device.update_descriptor_sets(&writes, &[]);
                }

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        mesh_cs.program.layout,
                        0,
                        &mesh_cs.program.descriptor_sets,
                        &[],
                    )
                }

                let dispatch_x = (swapchain.width + 7) / 8;
                let dispatch_y = (swapchain.height + 3) / 4;
                unsafe {
                    device.cmd_dispatch(cmd_buf, dispatch_x, dispatch_y, 1);
                }
            }
        }

        // Transition for present
        if swapchain_image_layout != vk::ImageLayout::PRESENT_SRC_KHR {
            let sub_res_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(swapchain_image_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(*sub_res_range)
                .image(swapchain.image[image_index as usize]);
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier],
                );
            }
            //swapchain_image_layout = image_barrier.new_layout;
        }

        // End command recoding
        unsafe {
            device.end_command_buffer(cmd_buf).unwrap();
        }

        // Submit
        {
            let command_buffers = [cmd_buf];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            unsafe {
                device
                    .queue_submit(gfx_queue, &[*submit_info], vk::Fence::null())
                    .unwrap();
            }
        }

        // Present
        {
            let mut present_info = vk::PresentInfoKHR::default();
            present_info.wait_semaphore_count = 1;
            present_info.p_wait_semaphores = &present_semaphore;
            present_info.swapchain_count = 1;
            present_info.p_swapchains = &swapchain.handle;
            present_info.p_image_indices = &image_index;
            unsafe {
                swapchain_entry
                    .entry
                    .queue_present(gfx_queue, &present_info)
                    .unwrap();
            }
        }

        // Wait brute-force (ATM)
        unsafe {
            device.device_wait_idle().unwrap();
        }
    }
}
