use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::c_void;

use ash::extensions::{ext, khr};
use ash::vk::{self, MemoryPropertyFlags, CommandPool};

pub struct RenderDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub physical_device_mem_prop: vk::PhysicalDeviceMemoryProperties,
    pub device: ash::Device,
    pub swapchain_entry: SwapchainEntry,
    pub surface_entry: khr::Surface,
    pub dynamic_rendering_entry: khr::DynamicRendering,

    pub b_support_dynamic_rendering: bool,

    pub descriptor_pool: vk::DescriptorPool,
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
        let dynamic_rendering_entry = khr::DynamicRendering::new(&instance, &device);

        // Get memory properties
        let physical_device_mem_prop =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

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
        let swapchain = {
            let surface_size = surface.query_size(&surface_entry, &physical_device);
            swapchain_entry.create(&device, &surface, &surface_size)
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

        RenderDevice {
            entry,
            instance,
            physical_device,
            physical_device_mem_prop,
            device,
            surface_entry,
            swapchain_entry,
            dynamic_rendering_entry,

            b_support_dynamic_rendering,

            descriptor_pool,
            gfx_queue,

            surface,
            swapchain,
        }
    }

    pub fn create_fence(&self, signaled: bool) -> vk::Fence {
        unsafe { 
            let create_info = vk::FenceCreateInfo::builder()
            .flags(if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() });
            self.device.create_fence(&create_info, None)
        }
        .expect("Vulakn: failed to create fence")
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        unsafe {
            let create_info = vk::SemaphoreCreateInfo::builder();
            self.device.create_semaphore(&create_info, None)
        }
        .expect("Vulkan: failed to create semaphore")
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        unsafe {
            self.device
                .create_command_pool(&create_info, None)
                .expect("Vulkan: failed to create command pool?!")
        }
    }

    pub fn create_command_buffer(&self, command_pool: CommandPool) -> vk::CommandBuffer {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();
        unsafe {
            self.device
                .allocate_command_buffers(&create_info)
                .expect("Vulkan: failed to allocated command buffer?!")[0]
        }
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

    vk::FALSE
}

// Resrouces

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

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
    pub data: *mut c_void,
    pub srv: Option<vk::BufferView>,
}

impl RenderDevice {
    pub fn pick_memory_type_index(
        &self,
        memory_type_bits: u32,
        property_flags: MemoryPropertyFlags,
    ) -> Option<u32> {
        let memory_types = &self.physical_device_mem_prop.memory_types;
        for i in 0..self.physical_device_mem_prop.memory_type_count {
            if ((memory_type_bits & (1 << i)) != 0)
                && ((memory_types[i as usize].property_flags & property_flags) == property_flags)
            {
                return Some(i);
            }
        }

        let mut support_properties= Vec::new();
        for i in 0..self.physical_device_mem_prop.memory_type_count {
            if memory_type_bits & (1 << i) != 0
            {
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

impl RenderDevice {
    pub fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        srv_format: vk::Format,
    ) -> Option<Buffer> {
        let device = &self.device;

        // Create the vk buffer object
        // TODO drop buffer if later stage failed
        let buffer = {
            let create_info = vk::BufferCreateInfo::builder().size(size).usage(usage);
            unsafe { device.create_buffer(&create_info, None) }.ok()?
        };

        // Allocate memory for ths buffer
        // TODO drop device_memory if later stage failed
        let memory = {
            let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

            // Pick memory type
            // TODO currently treating all buffer like a staging buffer
            let mem_property_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory_type_index = self
                .pick_memory_type_index(mem_req.memory_type_bits, mem_property_flags)
                .unwrap();

            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_req.size)
                .memory_type_index(memory_type_index);
            unsafe { device.allocate_memory(&create_info, None) }.ok()?
        };

        // Bind
        let offset: vk::DeviceSize = 0;
        unsafe { device.bind_buffer_memory(buffer, memory, offset) }.ok()?;

        // Map (staging buffer) persistently
        // TODO unmap if later stage failed
        let map_flags = vk::MemoryMapFlags::default(); // dummy parameter
        let data = unsafe { device.map_memory(memory, offset, size, map_flags) }.ok()?;

        // Create SRV
        let srv = if srv_format != vk::Format::UNDEFINED {
            let create_info = vk::BufferViewCreateInfo::builder()
                .buffer(buffer)
                .format(srv_format)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let srv = unsafe { device.create_buffer_view(&create_info, None) }.ok()?;
            Some(srv)
        } else {
            None
        };

        Some(Buffer {
            buffer,
            memory,
            size,
            data,
            srv,
        })
    }
}

pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub array_len: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
}

impl TextureDesc {
    pub fn new_2d(width: u32, height: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> TextureDesc {
        TextureDesc { width, height, array_len: 1, format, usage }
    }

    pub fn new_2d_array(width: u32, height: u32, array_len: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> TextureDesc {
        TextureDesc { width, height, array_len, format, usage }
    }
}

pub struct Texture {
    pub desc: TextureDesc,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,

}

pub struct TextureView {
    pub image_view: vk::ImageView,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
}

impl RenderDevice {
    pub fn create_texture(&self, desc: TextureDesc) -> Option<Texture> {
        let device = &self.device;

        let format_prop = unsafe { 
            self.instance.get_physical_device_image_format_properties(self.physical_device, desc.format, vk::ImageType::TYPE_2D, vk::ImageTiling::default(), desc.usage, vk::ImageCreateFlags::default())
        };
        if let Err(e) = format_prop {
            println!("Error: texture creation for {:?} failed: {:?}. Try something else.", desc.format, e);
            return None
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
            .array_layers(desc.array_len)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .initial_layout(initial_layout)
            .usage(desc.usage)
            ;
        let image = unsafe { device.create_image(&create_info, None) }.unwrap();

        // Bind memory
        let device_memory = {
            let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
            let momory_type_index = self.pick_memory_type_index(
                    mem_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL // That's basically what texture can have
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

    pub fn create_texture_view(&self, texture: &Texture, format: vk::Format, aspect: vk::ImageAspectFlags) -> Option<TextureView> {
        let device = &self.device;
        let create_info = vk::ImageViewCreateInfo::builder()
                .image(texture.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
        let image_view = unsafe { device.create_image_view(&create_info, None) }.ok()?;

        Some(TextureView {
            image_view,
            format,
            aspect,
        })
    }
}
