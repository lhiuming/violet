use std::borrow::Cow;
use std::ffi::CStr;
use std::os::raw::c_void;

use ash::extensions::ext;
use ash::vk;

pub fn set_up_message_callback(ext_debug_utils: &ext::DebugUtils) {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    use vk::DebugUtilsMessageTypeFlagsEXT as Type;
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(Severity::ERROR | Severity::WARNING)
        .message_type(Type::PERFORMANCE | Type::VALIDATION)
        .pfn_user_callback(Some(self::vulkan_debug_report_callback));
    unsafe {
        ext_debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .expect("Vulkan: Failed to register DebugUtils message callback");
    }
}

pub(super) unsafe extern "system" fn vulkan_debug_report_callback(
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
