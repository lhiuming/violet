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
    pub extent: vk::Extent2D,
    pub handle: vk::SwapchainKHR,
    pub num_image: u32,
    pub image: [vk::Image; 8],
    pub image_view: [vk::ImageView; 8],
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
        ret.extent = *extent;

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
    descriptor_pool: vk::DescriptorPool,
    pipeline_cache: vk::PipelineCache,
}

// AKA shader
pub struct PipelineProgram {
    pub shader_module: vk::ShaderModule,
    pub reflect_module: spirv_reflect::ShaderModule,
    pub entry_point_c: CString,
}

fn create_pipeline_program(
    device: &PipelineDevice,
    binary: &[u32],
    shader_def: &ShaderDefinition,
) -> Option<PipelineProgram> {
    let descriptor_pool = &device.descriptor_pool;
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

    let entry_point_c = reflect_module.get_entry_point_name().to_cstring();

    Some(PipelineProgram {
        shader_module,
        reflect_module,
        entry_point_c,
    })
}

use shaderc;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

trait ToCString {
    fn to_cstring(&self) -> CString;
}

impl ToCString for String {
    fn to_cstring(&self) -> CString {
        CString::new(self.clone()).unwrap()
    }
}

enum ShaderStage {
    Compute,
    Vert,
    Frag,
}

struct ShaderDefinition {
    pub virtual_path: String,
    pub entry_point: String,
    pub stage: ShaderStage,
}

impl ShaderDefinition {
    pub fn new(virtual_path: &str, entry_point: &str, stage: ShaderStage) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path: virtual_path.to_string(),
            entry_point: entry_point.to_string(),
            stage,
        }
    }
}

pub struct CompiledShader {
    pub artifact: shaderc::CompilationArtifact,
    pub program: PipelineProgram,
}

fn load_shader(device: &PipelineDevice, shader_def: &ShaderDefinition) -> Option<CompiledShader> {
    // todo map v_path to actuall pathes
    let mut path = PathBuf::new();
    path.push("./shader/");
    path.push(&shader_def.virtual_path);
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

    let shader_kind = match shader_def.stage {
        ShaderStage::Compute => shaderc::ShaderKind::Compute,
        ShaderStage::Vert => shaderc::ShaderKind::Vertex,
        ShaderStage::Frag => shaderc::ShaderKind::Fragment,
    };

    // Compile the shader
    let file_name_os = path.file_name().unwrap();
    let file_name = file_name_os.to_str().unwrap();
    let mut compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_source_language(shaderc::SourceLanguage::HLSL);
    options.set_auto_bind_uniforms(true);
    let compile_result = compiler.compile_into_spirv(
        &text,
        shader_kind,
        file_name,
        &shader_def.entry_point,
        Some(&options),
    );

    let artifact = match compile_result {
        Err(why) => {
            println!("Shaer compiled binay is not valid: {}", why);
            return None;
        }
        Ok(artifact) => artifact,
    };

    let program = match create_pipeline_program(device, artifact.as_binary(), &shader_def) {
        Some(program) => program,
        None => return None,
    };

    Some(CompiledShader {
        artifact: artifact,
        program: program,
    })
}

struct Pipeline {
    pub handle: vk::Pipeline,
    //pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

fn create_compute_pipeline(
    device: &PipelineDevice,
    shader_def: &ShaderDefinition,
    compiled: &CompiledShader,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let descriptor_pool = device.descriptor_pool;
    let device = &device.device;

    let program = &compiled.program;
    let reflect_module = &program.reflect_module;

    // Create all set layouts used in compute
    let set_layout = {
        assert!(
            reflect_module
                .enumerate_descriptor_sets(Some(&shader_def.entry_point))
                .unwrap()
                .len()
                <= 1
        );
        let bindings = reflect_module
            .enumerate_descriptor_bindings(Some(&shader_def.entry_point))
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

    let set_layouts = vec![set_layout];

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Create all used descriptor sets
    let descriptor_sets = {
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe { device.allocate_descriptor_sets(&create_info) }.ok()?
    };

    // Create pipeline object
    let pipeline = {
        let entry_point_c = CString::new(shader_def.entry_point.clone())
            .expect(&format!("Bad entry point name: {}", shader_def.entry_point));
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(program.shader_module)
            .name(&entry_point_c);
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(layout);
        let create_infos = [create_info.build()];
        let result =
            unsafe { device.create_compute_pipelines(pipeline_cache, &create_infos, None) }.ok()?;
        result[0]
    };

    Some(Pipeline {
        handle: pipeline,
        //set_layouts,
        layout,
        descriptor_sets,
    })
}

fn create_graphics_pipeline(
    device: &PipelineDevice,
    vs: &CompiledShader,
    ps: &CompiledShader,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let descriptor_pool = device.descriptor_pool;
    let device = &device.device;

    let shaders = [vs, ps];

    // Collect and merge descriptor set from all stages
    use std::collections::hash_map::{Entry, HashMap};
    type MergedSet = HashMap<u32, vk::DescriptorSetLayoutBinding>;
    type MergedLayout = HashMap<u32, MergedSet>;
    let mut merged_layout = MergedLayout::new();
    for shader in shaders {
        let reflect = &shader.program.reflect_module;
        let stage = reflect.get_shader_stage().bits();
        let stage = vk::ShaderStageFlags::from_raw(stage);
        for s in reflect.enumerate_descriptor_sets(None).unwrap() {
            let set = match merged_layout.entry(s.set) {
                Entry::Occupied(o) => o.into_mut(),
                Entry::Vacant(v) => v.insert(MergedSet::new()),
            };
            for b in s.bindings {
                // TODO ReflectDescriptorType does not match vk::DescriptorType in bit value, also AccelerationStructureNV is in spirv_reflect but not in ash
                let descriptor_type = vk::DescriptorType::from_raw(b.descriptor_type as i32 - 1);
                assert!(b.count > 0);

                if let Some(binding) = set.get_mut(&b.binding) {
                    assert!(descriptor_type == binding.descriptor_type);
                    assert!(b.count == binding.descriptor_count); // really?
                    binding.stage_flags |= stage;
                } else {
                    let binding = vk::DescriptorSetLayoutBinding::builder()
                        .binding(b.binding)
                        .descriptor_type(descriptor_type)
                        .descriptor_count(b.count)
                        .stage_flags(stage);
                    set.insert(b.binding, binding.build());
                }
            }
        }
    }

    let set_layouts: Vec<vk::DescriptorSetLayout> = merged_layout
        .into_iter()
        .map(|(set_index, set_bindings)| {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> =
                set_bindings.into_values().collect();
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap()
        })
        .into_iter()
        .collect();

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Create all used descriptor sets
    let descriptor_sets = {
        if set_layouts.len() == 0 {
            Vec::new()
        } else {
            let create_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe { device.allocate_descriptor_sets(&create_info) }.ok()?
        }
    };

    // Stages
    let vs_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vs.program.shader_module)
        .name(&vs.program.entry_point_c);
    let ps_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(ps.program.shader_module)
        .name(&ps.program.entry_point_c);
    let stage_infos = [vs_info.build(), ps_info.build()];

    // States
    let vertex_info = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    //let tess = ();
    let viewport = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1) // actual state is dynamic
        .scissor_count(1); // actual dynamic
    let raster = vk::PipelineRasterizationStateCreateInfo::builder();
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder();
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::from_raw(0xFFFFFFFF));
    let attachments = [attachment.build()];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stage_infos)
        .vertex_input_state(&vertex_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .dynamic_state(&dynamic_state);
    let create_infos = [create_info.build()];
    let result =
        unsafe { device.create_graphics_pipelines(pipeline_cache, &create_infos, None) }.ok()?;
    let pipeline = result[0];

    Some(Pipeline {
        handle: pipeline,
        //set_layouts,
        layout,
        descriptor_sets,
    })
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
    let mesh_cs_def = ShaderDefinition::new("MeshCS.hlsl", "main", ShaderStage::Compute);
    let (mesh_cs, mesh_cs_pipeline) = {
        let shader = load_shader(&pipeline_device, &mesh_cs_def).unwrap();
        let pipeline = create_compute_pipeline(&pipeline_device, &mesh_cs_def, &shader).unwrap();
        (shader, Some(pipeline))
    };
    let mesh_vs_def = ShaderDefinition::new("MeshVSPS.hlsl", "vs_main", ShaderStage::Vert);
    let mesh_ps_def = ShaderDefinition::new("MeshVSPS.hlsl", "ps_main", ShaderStage::Frag);
    let mesh_gfx_pipeline = {
        let vs = load_shader(&pipeline_device, &mesh_vs_def).unwrap();
        let ps = load_shader(&pipeline_device, &mesh_ps_def).unwrap();
        create_graphics_pipeline(&pipeline_device, &vs, &ps)
    };

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
        let b_resize = surface_size != swapchain.extent;
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

        // Draw mesh
        if let Some(pipeline) = &mesh_gfx_pipeline {
            // Set viewport and scissor
            {
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: swapchain.extent.height as f32,
                    width: swapchain.extent.width as f32,
                    height: -(swapchain.extent.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let viewports = [viewport];
                unsafe {
                    device.cmd_set_viewport(cmd_buf, 0, &viewports);
                }

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain.extent,
                };
                let scissors = [scissor];
                unsafe {
                    device.cmd_set_scissor(cmd_buf, 0, &scissors);
                }
            }

            // Set pipeline and Draw
            unsafe {
                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            }
            unsafe {
                device.cmd_draw(cmd_buf, 3, 1, 0, 0);
            }
        }

        // End render pass
        unsafe {
            dynamic_render.cmd_end_rendering(cmd_buf);
        }

        // Draw something with compute
        if let Some(pipeline) = &mesh_cs_pipeline {
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
                        pipeline.handle,
                    )
                }

                // Bind the swapchain image (the only descriptor)
                unsafe {
                    let image_info = vk::DescriptorImageInfo::builder()
                        .image_view(swapchain.image_view[image_index as usize])
                        .image_layout(swapchain_image_layout);
                    let image_infos = [image_info.build()];
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(pipeline.descriptor_sets[0])
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
                        pipeline.layout,
                        0,
                        &pipeline.descriptor_sets,
                        &[],
                    )
                }

                let dispatch_x = (swapchain.extent.width + 7) / 8;
                let dispatch_y = (swapchain.extent.height + 3) / 4;
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
