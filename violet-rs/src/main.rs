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

        print!("Vulkan: creating instance ... "); std::io::stdout().flush().unwrap();
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
    let mut b_support_dynamic_rendering = false;
    let device = {
        // Enumerate and pick queue families to create with the device
        let queue_fams =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut gfx_queue_family_index = 0;
        for i in 0..queue_fams.len() {
            let queue_fam = &queue_fams[i];
            if queue_fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                gfx_queue_family_index = i as u32;
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
        let b_resize =
            (surface_size.width != swapchain.width) || (surface_size.height != swapchain.height);
        if b_resize {
            swapchain_entry.detroy(&device, &mut swapchain);
            swapchain = swapchain_entry.create(&device, &surface, &surface_size);
        }

        // todo render things
    }
}
