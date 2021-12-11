use ash::extensions::{ext, khr};
use ash::vk;
use std::borrow::Cow;
use std::ffi::{CStr, CString};
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

fn main() {
    println!("Hello, rusty world!");

    // Create a system window
    // TODO implement Drop for Window
    let window = Window::new(1280, 720, "Rusty Violet");

    // Load functions
    //let entry = unsafe { ash::Entry::new().expect("Ash entry creation failed") };
    let entry = ash::Entry::new();

    // Create instance
    let instance = unsafe {
        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_2);

        let layer_names = [
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(), // Debug
        ];
        let layer_names_raw: Vec<_> = layer_names.iter().map(|name| name.as_ptr()).collect();

        let ext_names_raw = [
            khr::Surface::name().as_ptr(),
            khr::GetPhysicalDeviceProperties2::name().as_ptr(), // Required by dynamic_rendering
            ext::DebugUtils::name().as_ptr(),                   // Debug
        ];

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names_raw)
            .enabled_extension_names(&ext_names_raw)
            .build();
        entry
            .create_instance(&create_info, None)
            .expect("Vulkan instance creation failed")
    };

    // Debug callback
    let debug_report = ash::extensions::ext::DebugUtils::new(&entry, &instance);
    unsafe {
        use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
        use vk::DebugUtilsMessageTypeFlagsEXT as Type;
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(Severity::ERROR | Severity::WARNING | Severity::INFO)
            .message_type(Type::PERFORMANCE | Type::VALIDATION)
            .pfn_user_callback(Some(vulkan_debug_report_callback))
            .build();
        debug_report
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to register debug callback");
    }

    // Pick physical device
    let phy_dev = {
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
    let mut bSupportDynamicRendering = false;
    let device = {
        // Enumerate and pick queue families to create with the device
        let queue_fams = unsafe { instance.get_physical_device_queue_family_properties(phy_dev) };
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
        let mut supported_features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut dynamic_rendering_ft)
            .build();
        unsafe {
            instance.get_physical_device_features2(phy_dev, &mut supported_features);
        };
        bSupportDynamicRendering = dynamic_rendering_ft.dynamic_rendering == vk::TRUE;
        // Finally, create the device
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&enabled_extension_names)
            .enabled_features(&supported_features.features)
            .push_next(&mut dynamic_rendering_ft)
            .build();
        unsafe {
            instance
                .create_device(phy_dev, &create_info, None)
                .expect("Failed to create Vulkan device")
        }
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
                .expect("Vulkan: failed to allocated command buffer?!")[0];
        }
    };

    while !window.should_close() {
        window.poll_events();

        // todo render things
    }
}
