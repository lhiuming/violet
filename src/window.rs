use std::cell::RefCell;
use std::os::windows::prelude::OsStrExt;
use std::rc::Rc;
use std::{ffi, mem, ptr};
use windows_sys::Win32::{
    Foundation::*, System::LibraryLoader::*, System::Memory::*, UI::Input::KeyboardAndMouse::*,
    UI::WindowsAndMessaging::*,
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

static mut S_MESSAGE_HANDLER: Option<Rc<RefCell<MessageHandler>>> = None;

fn report_last_error() {
    unsafe {
        // Get message
        let last_err = GetLastError();
        if last_err == 0 {
            println!("Window api call failed, but no error code is reported for last error.");
            return;
        }
        assert!(last_err != 0);

        use windows_sys::Win32::System::Diagnostics::Debug::*;
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

struct MessageHandler {
    pub hwnd: HWND, // Copied this from Window, for validation porpose
    // TODO move these kind of logic to app side
    pub should_close: bool,
    pub mouse_pos: (i16, i16),    // most up-to-dated mouse pos
    pub mouse_right_button: bool, // most up-to-dated right-button down
    pub curr_drag_beg_mouse_pos: Option<(i16, i16)>, // current frame init mouse pos with right-button down
    pub curr_drag_end_mouse_pos: Option<(i16, i16)>, // current frame last mouse pos with right-button down
    pub push_W: bool,
    pub push_A: bool,
    pub push_S: bool,
    pub push_D: bool,
    pub push_Q: bool,
    pub push_E: bool,
    pub click_R: bool,
}

impl MessageHandler {
    pub fn new(hwnd: HWND) -> MessageHandler {
        MessageHandler {
            hwnd,
            should_close: false,
            mouse_pos: (0, 0),
            mouse_right_button: false,
            curr_drag_beg_mouse_pos: None,
            curr_drag_end_mouse_pos: None,
            push_W: false,
            push_A: false,
            push_S: false,
            push_D: false,
            push_E: false,
            push_Q: false,
            click_R: false,
        }
    }

    pub fn new_frame(&mut self) {
        // Continue last frame unrelease drag
        self.curr_drag_beg_mouse_pos = if self.mouse_right_button {
            Some(self.mouse_pos)
        } else {
            None
        };
        self.curr_drag_end_mouse_pos = None;
        self.click_R = false;
    }
}

unsafe extern "system" fn wnd_callback(
    hwnd: HWND,
    msg: u32,
    w_param: WPARAM,
    l_param: LPARAM,
) -> LRESULT {
    let handler = match &S_MESSAGE_HANDLER {
        Some(handler) => handler,
        None => {
            //println!("Win32 message: no handler");
            return DefWindowProcW(hwnd, msg, w_param, l_param);
        }
    };

    assert!(hwnd == handler.borrow().hwnd);

    let decode_cursor_pos = || -> (i16, i16) {
        assert!(mem::size_of_val(&l_param) >= 4); // at least contain 2 short integers
        let x = (l_param & 0xFFFF) as i16;
        let y = ((l_param >> 16) & 0xFFFF) as i16;
        (x, y)
    };

    match msg {
        WM_CLOSE => {
            handler.borrow_mut().should_close = true;
            0
        }
        WM_RBUTTONDOWN => {
            SetCapture(hwnd);
            let (x, y) = decode_cursor_pos();
            {
                let mut handler = handler.borrow_mut();
                handler.mouse_pos = (x, y);
                handler.mouse_right_button = true;
                if handler.curr_drag_beg_mouse_pos.is_none() {
                    handler.curr_drag_beg_mouse_pos = Some((x, y));
                }
            }
            //println!("Win32 message: right button down");
            0
        }
        WM_RBUTTONUP => {
            ReleaseCapture();
            let (x, y) = decode_cursor_pos();
            {
                let mut handler = handler.borrow_mut();
                handler.mouse_pos = (x, y);
                handler.mouse_right_button = false;
                handler.curr_drag_end_mouse_pos = Some((x, y));
            }
            //println!("Win32 message: right button up");
            0
        }
        WM_MOUSEMOVE => {
            let rb_down = (w_param as u32) == MK_RBUTTON;
            let (x, y) = decode_cursor_pos();
            {
                let mut handler = handler.borrow_mut();
                handler.mouse_pos = (x, y);
                if rb_down {
                    handler.curr_drag_end_mouse_pos = Some((x, y));
                }
            }
            /*
            println!(
                "Win32 message: mouse move x {}, y {}, right button down {}",
                x, y, rb_down
            );
            */
            0
        }
        WM_KEYDOWN => {
            let vk_code = w_param as u16;
            match vk_code {
                VK_W => handler.borrow_mut().push_W = true,
                VK_A => handler.borrow_mut().push_A = true,
                VK_S => handler.borrow_mut().push_S = true,
                VK_D => handler.borrow_mut().push_D = true,
                VK_E => handler.borrow_mut().push_E = true,
                VK_Q => handler.borrow_mut().push_Q = true,
                _ => {
                    //println!("Win32 message: unhandle key down {}", vk_code);
                    return DefWindowProcW(hwnd, msg, w_param, l_param);
                }
            }
            /*
            let repeat_count = (l_param & 0xFFFF) as u16;
            let is_prev_down = ((l_param >> 30) & 0x1) > 0;
            println!(
                "Win32 message: key W is down {}, count {}, holding {}",
                vk_code, repeat_count, is_prev_down
            );
            */
            0
        }
        WM_KEYUP => {
            let vk_code = w_param as u16;
            match vk_code {
                VK_W => handler.borrow_mut().push_W = false,
                VK_A => handler.borrow_mut().push_A = false,
                VK_S => handler.borrow_mut().push_S = false,
                VK_D => handler.borrow_mut().push_D = false,
                VK_E => handler.borrow_mut().push_E = false,
                VK_Q => handler.borrow_mut().push_Q = false,
                VK_R => handler.borrow_mut().click_R = true,
                _ => {
                    //println!("Win32 message: unhandle key up {}", vk_code);
                    return DefWindowProcW(hwnd, msg, w_param, l_param);
                }
            }
            /*
            let repeat_count = (l_param & 0xFFFF) as u16;
            println!(
                "Win32 message: key W is up {}, count {}",
                vk_code, repeat_count
            );
            */
            0
        }
        _ => {
            //println!("Win32 message: unhandled");
            DefWindowProcW(hwnd, msg, w_param, l_param)
        }
    }
}

pub struct Window {
    system_handle: u64,
    message_handler: Rc<RefCell<MessageHandler>>,
}

impl Window {
    // General constructor
    pub fn new(init_width: u32, init_height: u32, title: &str) -> Box<Window> {
        // Check if we have reigster window class (wnd_callback)
        ensure_register_window_class();

        // NOTE: lifetime musu be longer than the CreateWindowExW call
        let mut cstr_class_name = to_cstr_wide(K_WINDOW_CLASS_NAME);
        let mut cstr_title_name = to_cstr_wide(title);

        // Create window
        let mut style: u32 = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
        style |= WS_CAPTION // Title
            | WS_SYSMENU // Required by WS_MINIMIZEBOX
            | WS_MINIMIZEBOX // Minimize button
            ;
        let _resizable = false;
        if _resizable {
            style |= WS_THICKFRAME // resizable frame
            | WS_MAXIMIZEBOX // maximize button
            ;
        }
        let ex_style = WS_EX_APPWINDOW;
        let hwnd = unsafe {
            // Found proper windows size to produce desired surface size
            let mut rect = RECT {
                left: 0,
                top: 0,
                right: init_width as i32,
                bottom: init_height as i32,
            };
            let cal_size = AdjustWindowRectEx(std::ptr::addr_of_mut!(rect), style, 0, ex_style);
            if cal_size == 0 {
                report_last_error();
            }

            let win_width = rect.right - rect.left;
            let win_height = rect.bottom - rect.top;

            let hmodule = GetModuleHandleW(std::ptr::null_mut());
            CreateWindowExW(
                WS_EX_APPWINDOW,
                cstr_class_name.as_mut_ptr(),
                cstr_title_name.as_mut_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                win_width,
                win_height,
                0,
                0,
                hmodule,
                std::ptr::null(),
            )
        };

        if hwnd == 0 {
            report_last_error();
        }

        // A ref-count message handler, to pass message from global call back to this window instance
        let message_handler = Rc::new(RefCell::new(MessageHandler::new(hwnd)));

        // Allocate the window object first, because this pointer is passed to windows system via SetPropW
        let result = {
            std::assert!(mem::size_of_val(&hwnd) >= mem::size_of::<HWND>());
            let temp = Window {
                system_handle: hwnd as u64,
                message_handler: message_handler.clone(),
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

        // Register message handler to global state
        unsafe {
            S_MESSAGE_HANDLER = Some(message_handler);
        }

        result
    }

    pub fn poll_events(&self) {
        // Cache last frame info
        self.message_handler.borrow_mut().new_frame();

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
        self.message_handler.borrow().should_close
    }

    // Forward, Right, Up
    pub fn nav_dir(&self) -> (f32, f32, f32) {
        let handler = self.message_handler.borrow();
        let make_dir = |positive, negative| -> f32 {
            let mut ret = 0.0;
            if positive {
                ret += 1.0;
            }
            if negative {
                ret -= 1.0;
            }
            ret
        };
        (
            make_dir(handler.push_W, handler.push_S),
            make_dir(handler.push_D, handler.push_A),
            make_dir(handler.push_E, handler.push_Q),
        )
    }

    // Drag start pos, darg end pos
    pub fn effective_darg(&self) -> Option<(i16, i16, i16, i16)> {
        let handler = self.message_handler.borrow();
        if let Some((end_x, end_y)) = handler.curr_drag_end_mouse_pos {
            if let Some((beg_x, beg_y)) = handler.curr_drag_beg_mouse_pos {
                Some((beg_x, beg_y, end_x, end_y))
            } else {
                println!("Window: has end drag pos, but no beg drag pos ?!");
                Some((end_x, end_y, end_x, end_y))
            }
        } else {
            None
        }
    }

    pub fn click_R(&self) -> bool {
        self.message_handler.borrow().click_R
    }
}
