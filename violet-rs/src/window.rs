use std::os::windows::prelude::OsStrExt;
use std::{ffi, mem, ptr};
use windows_sys::Win32::{
    Foundation::*, System::Diagnostics::Debug::*, System::LibraryLoader::*, System::Memory::*,
    UI::Input::KeyboardAndMouse::*, UI::WindowsAndMessaging::*,
};

// TODO this process shoule be run in compile time, and return a const u16 array; but currently this is hard to write in Rust (might need to use proc_macro)
fn to_cstr_wide(s: &str) -> Vec<u16> {
    ffi::OsStr::new(s)
        .encode_wide()
        .chain(Some(0).into_iter())
        .collect()
}

static K_WINDOW_CLASS_NAME: &str = "Violet";
static K_WINDOW_INSTANCE_PROP_NAME: &str = "VioletWindowInstance";
static K_MSGBOX_ERROR_CAPTION: &str = "Error";

fn report_last_error() {
    unsafe {
        // Get message
        let last_err = GetLastError();
        if last_err == 0 {
            println!("Window api call failed, but no error code is reported for last error.");
            return;
        }
        assert!(last_err != 0);

        let mut lp_msg_buf = ptr::null_mut();
        let fmt_result = FormatMessageW(
            FORMAT_MESSAGE_ALLOCATE_BUFFER
                | FORMAT_MESSAGE_FROM_SYSTEM
                | FORMAT_MESSAGE_IGNORE_INSERTS,
            ptr::null(),
            last_err,
            0,
            ptr::addr_of_mut!(lp_msg_buf) as _,
            0,
            ptr::null(),
        );

        if fmt_result != 0 {
            println!(
                "Window pai FormatMessageW might be not correct! (error_code:{0})",
                fmt_result
            );
        }

        // Display message (and exit?)
        MessageBoxW(
            0,
            lp_msg_buf,
            to_cstr_wide(K_MSGBOX_ERROR_CAPTION).as_mut_ptr(),
            MB_OK,
        );

        LocalFree(lp_msg_buf as isize);
    }
}

unsafe extern "system" fn wnd_callback(
    hwnd: HWND,
    msg: u32,
    w_param: WPARAM,
    l_param: LPARAM,
) -> LRESULT {
    match msg {
        WM_CLOSE => {
            let mut cstr_prop_name = to_cstr_wide(K_WINDOW_INSTANCE_PROP_NAME);
            let window: *mut Window = GetPropW(hwnd, cstr_prop_name.as_mut_ptr()) as _;
            (&mut *window).should_close = true;
            0
        }
        _ => DefWindowProcW(hwnd, msg, w_param, l_param),
    }
}

fn register_window_class() -> bool {
    // NOTE: lifetime must be longer than the RegisterClassExW call
    let mut c_class_name = to_cstr_wide(K_WINDOW_CLASS_NAME);

    let class_id = unsafe {
        let hmodule: HINSTANCE = GetModuleHandleW(std::ptr::null_mut()) as HINSTANCE;
        let window_class = WNDCLASSEXW {
            cbSize: mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW | CS_OWNDC,
            lpfnWndProc: wnd_callback,
            cbClsExtra: 0i32,
            cbWndExtra: 0i32,
            hInstance: hmodule,
            hIcon: 0,
            hCursor: LoadCursorW(0, IDC_ARROW),
            hbrBackground: 0,
            lpszMenuName: std::ptr::null_mut(),
            lpszClassName: c_class_name.as_mut_ptr(),
            hIconSm: 0,
        };
        RegisterClassExW(&window_class)
    };

    if class_id == 0 {
        report_last_error();
        return false;
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

        // NOTE: lifetime musu be longer than the CreateWindowExW call
        let mut cstr_class_name = to_cstr_wide(K_WINDOW_CLASS_NAME);
        let mut cstr_title_name = to_cstr_wide(title);

        // Create window
        let mut style: u32 = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
        style |= WS_SYSMENU | WS_MINIMIZEBOX; // ?
        style |= WS_CAPTION | WS_MAXIMIZEBOX | WS_THICKFRAME; // Title and resizable frame
        let hwnd = unsafe {
            let hmodule = GetModuleHandleW(std::ptr::null_mut());
            CreateWindowExW(
                WS_EX_APPWINDOW,
                cstr_class_name.as_mut_ptr(),
                cstr_title_name.as_mut_ptr(),
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

        // Allocate the window object first, because this pointer is passed to windows system via SetPropW
        let result = {
            std::assert!(mem::size_of_val(&hwnd) >= mem::size_of::<HWND>());
            let temp = Window {
                system_handle: hwnd as u64,
                should_close: false,
            };
            Box::new(temp)
        };

        // Bind the instance to system window
        {
            let succeed = unsafe {
                let inst_ptr: *const Window = &*result;
                SetPropW(
                    hwnd,
                    to_cstr_wide(K_WINDOW_INSTANCE_PROP_NAME).as_mut_ptr(),
                    inst_ptr as HANDLE,
                )
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

    pub fn poll_events(&self) {
        unsafe {
            let mut msg: MSG = mem::zeroed();
            while PeekMessageW(
                ptr::addr_of_mut!(msg),
                self.system_handle as isize,
                0,
                0,
                PM_REMOVE,
            ) != 0
            {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }

    pub fn system_handle(&self) -> u64 {
        self.system_handle
    }

    pub fn system_handle_for_module() -> u64 {
        let hmodule = unsafe { GetModuleHandleW(std::ptr::null_mut()) };
        hmodule as u64
    }

    pub fn should_close(&self) -> bool {
        self.should_close
    }
}
