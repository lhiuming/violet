use std::ffi::c_void;
use std::{mem, ptr};
use windows_sys::Win32::System::LibraryLoader::*;

const API_VERSION: u32 = 10401; // 1.4.1

pub struct RenderDoc {
    api: *mut renderdoc_sys::RENDERDOC_API_1_4_1,
}

impl RenderDoc {
    pub fn new() -> Option<Self> {
        // Load library (assume not loaded)
        let hinstance = unsafe { LoadLibraryA(b"renderdoc.dll\0".as_ptr() as *mut _) };

        // Fetch the library entry point
        // TODO handle error
        let proc_name = b"RENDERDOC_GetAPI\0";
        let get_api: unsafe extern "system" fn(u32, *mut *mut c_void) -> i32 = unsafe {
            let addr = GetProcAddress(hinstance, proc_name.as_ptr() as *mut _);
            mem::transmute(addr)
        };

        // Get API
        let mut obj = ptr::null_mut();
        let ret = unsafe { get_api(API_VERSION, ptr::addr_of_mut!(obj)) };
        if ret != 1 {
            return None;
        }
        assert!(obj != ptr::null_mut());

        let rdoc = RenderDoc {
            api: unsafe { mem::transmute(obj) },
        };

        // Some initialize setup
        rdoc.set_capture_file_path_template("./capture/violet.rdc");
        let enable_validation_layer = match std::env::var_os("VIOLET_VALIDATION_LAYER") {
            Some(val) => val != "0",
            None => false,
        };
        if enable_validation_layer {
            rdoc.set_enable_api_validation(true);
            rdoc.set_debug_output_mute(false);
        }

        Some(rdoc)
    }

    /// Default: false
    pub fn set_enable_api_validation(&self, value: bool) {
        unsafe {
            (*self.api).SetCaptureOptionU32.unwrap()(
                renderdoc_sys::eRENDERDOC_Option_APIValidation,
                value as u32,
            );
        }
    }

    /// Default: true
    pub fn set_debug_output_mute(&self, value: bool) {
        unsafe {
            (*self.api).SetCaptureOptionU32.unwrap()(
                renderdoc_sys::eRENDERDOC_Option_DebugOutputMute,
                value as u32,
            );
        }
    }

    pub fn set_capture_file_path_template(&self, template: &str) {
        unsafe {
            let path = std::ffi::CString::new(template).unwrap();
            (*self.api)
                .__bindgen_anon_2
                .SetCaptureFilePathTemplate
                .unwrap()(path.as_ptr());
        }
    }

    pub fn trigger_capture(&self) {
        unsafe {
            (*self.api).TriggerCapture.unwrap()();
        }
    }
}
