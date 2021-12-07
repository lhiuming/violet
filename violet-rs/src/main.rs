mod window;
use ash::vk;
use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
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
    let entry = unsafe { ash::Entry::new().expect("Ash entry creation failed") };

    // Create instance
    let instance = unsafe {
        let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_2);

        let layer_names = [
            CString::new("VK_LAYER_KHRONOS_validation").unwrap(), // Debug
        ];
        let layer_names_raw: Vec<_> = layer_names.iter().map(|name| name.as_ptr()).collect();

        let ext_names = [
            CString::new("VK_KHR_surface").unwrap(),
            CString::new("VK_KHR_get_physical_device_properties2").unwrap(),
            CString::new("VK_KHR_win32_surface").unwrap(),
            CString::new("VK_EXT_debug_utils").unwrap(), // Debug
        ];
        let ext_names_raw: Vec<_> = ext_names.iter().map(|name| name.as_ptr()).collect();

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
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(vulkan_debug_report_callback))
            .build();
        debug_report
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to register debug callback");
    }

    loop {
        if window.should_close() {
            break;
        }
        window.poll_events();

        // todo render things
    }
}
