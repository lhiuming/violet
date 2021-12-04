use std::mem::{size_of, size_of_val};
use windows_sys::{
    Win32::Foundation::*,
    Win32::System::LibraryLoader::*,
    Win32::UI::Input::KeyboardAndMouse::SetFocus,
    Win32::UI::WindowsAndMessaging::*,
};

fn report_last_error() {
    // TODO
}

unsafe extern "system" fn wnd_callback(
    hwnd: HWND,
    uMsg: u32,
    wParam: WPARAM,
    lParam: LPARAM,
) -> LRESULT {
    0
}

static K_WINDOW_CLASS_NAME: &str = "Violet";
static K_WINDOW_INSTANCE_PROP_NAME: &str = "VioletWindowInstance";

fn register_window_class() -> bool {
    let hmodule: HINSTANCE = unsafe { GetModuleHandleW(std::ptr::null_mut()) as HINSTANCE };

    let mut class_name: Vec<u16> = K_WINDOW_CLASS_NAME.encode_utf16().collect();

    let window_class = WNDCLASSEXW {
        cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
        style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
        lpfnWndProc: wnd_callback,
        cbClsExtra: 0i32,
        cbWndExtra: 0i32,
        hInstance: hmodule,
        hIcon: 0,
        hCursor: unsafe { LoadCursorW(0, IDC_ARROW) },
        hbrBackground: 0,
        lpszMenuName: std::ptr::null_mut(),
        lpszClassName: class_name.as_mut_ptr(),
        hIconSm: 0,
    };

    unsafe {
        if RegisterClassExW(&window_class) == 0u16 {
            report_last_error();
            return false;
        }
    }

    return true;
}

fn ensure_register_window_class() {
    unsafe {
        static mut REGISTERED: bool = false;
        if !REGISTERED {
            REGISTERED = register_window_class();
        }
    }
}

pub struct Window {
    system_handle: u64,
    should_close: bool,
}

impl Window {
    // General constructor
    pub fn new(init_width: u32, init_height: u32, title: &str) -> std::boxed::Box<Window> {
        // Check if we have reigster window class (wnd_callback)
        ensure_register_window_class();

        // Get some global properties
        let mut class_name: Vec<u16> = K_WINDOW_CLASS_NAME.encode_utf16().collect();
        let hmodule = unsafe { GetModuleHandleW(std::ptr::null_mut()) };

        // Create window
        let mut window_name: Vec<u16> = title.encode_utf16().collect();
        let mut style: u32 = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
        style |= WS_SYSMENU | WS_MINIMIZEBOX; // ?
        style |= WS_CAPTION | WS_MAXIMIZEBOX | WS_THICKFRAME; // Title and resizable frame
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_APPWINDOW,
                class_name.as_mut_ptr(),
                window_name.as_mut_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                init_width as i32,
                init_height as i32,
                0,
                0,
                hmodule,
                std::ptr::null(),
            )
        };

        if hwnd == 0 {
            report_last_error();
        }

        let result = {
            std::assert!(size_of_val(&hwnd) >= size_of::<HWND>());
            let temp = Window {
                system_handle: hwnd as u64,
                should_close: false,
            };
            Box::new(temp)
        };

        // Bind the instance to system window
        {
            let mut inst_prop_name: Vec<u16> = K_WINDOW_INSTANCE_PROP_NAME.encode_utf16().collect();
            let succeed = unsafe {
                let inst_ptr: *const Window = &*result;
                SetPropW(hwnd, inst_prop_name.as_mut_ptr(), inst_ptr as HANDLE)
            };
            if succeed == 0 {
                report_last_error();

                // Relase the system window
                let destroyed = unsafe { DestroyWindow(hwnd) };
                if destroyed == 0 {
                    report_last_error();
                }
            }
        }

        // Show and focus the window
        unsafe {
            ShowWindow(hwnd, SW_SHOWNA);
            if BringWindowToTop(hwnd) == 0 {
                report_last_error();
            }
            if SetForegroundWindow(hwnd) == 0 {
                report_last_error();
            }
            if SetFocus(hwnd) == 0 {
                report_last_error();
            }
        }

        result
    }
}
