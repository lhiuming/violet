#include "Window.h"

#include <cassert>

// Win32 Platform
//#include <wctype.h>
#define NOMINMAX
#include <windows.h>
#include <strsafe.h>
#undef NOMINMAX

#define WINDOW_INVALID_HANDLE 0

namespace WindowPrivate
{

// Names for win32 api
static const wchar_t *kClassName = TEXT("VioletWindow");
static const wchar_t *kInstancePropName = TEXT("VioletWindowInstance");

// Win32 stuffs
void ReportLastError(LPCTSTR functionName)
{
	// see https://docs.microsoft.com/en-us/windows/win32/debug/retrieving-the-last-error-code

	LPVOID lpMsgBuf;
	LPVOID lpDisplayBuf;
	DWORD dw = GetLastError();

	FormatMessageW(
		FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL,
		dw,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPTSTR)&lpMsgBuf,
		0,
		NULL);

	// Display the error message and exit the process

	lpDisplayBuf = (LPVOID)LocalAlloc(
		LMEM_ZEROINIT, (lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)functionName) + 40) * sizeof(TCHAR));
	StringCchPrintf(
		(LPTSTR)lpDisplayBuf,
		LocalSize(lpDisplayBuf) / sizeof(TCHAR),
		TEXT("%s failed with error %d: %s"),
		functionName,
		dw,
		lpMsgBuf);
	MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);

	LocalFree(lpMsgBuf);
	LocalFree(lpDisplayBuf);
}

// Static callback handler (friend of Window)
class SystemCallbackHandler
{
public:
	static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		Window *window = (Window *)GetPropW(hWnd, kInstancePropName);

		switch (uMsg)
		{
			case WM_CLOSE:
			{
				assert(window);
				window->SetShouldClose();
				return 0;
			}
			//case WM_SIZE:
			//{
			//	UINT width = LOWORD(lParam);
			//	UINT height = HIWORD(lParam);
			//	break;
			//}
		};

		return DefWindowProcW(hWnd, uMsg, wParam, lParam);
	}
};

bool RegisterWindowClass()
{
	WNDCLASSEXW wc;

	ZeroMemory(&wc, sizeof(wc));
	wc.cbSize = sizeof(wc);
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = (WNDPROC)SystemCallbackHandler::WindowProc;
	wc.hInstance = GetModuleHandleW(NULL);
	wc.hCursor = LoadCursorW(NULL, IDC_ARROW);
	wc.lpszClassName = WindowPrivate::kClassName;

	if (!RegisterClassExW(&wc))
	{
		ReportLastError(TEXT("RegisterClassExW"));
		return false;
	}

	return true;
}

void EnsureRegisterWindowClass()
{
	static bool WindowClassRegistered = false;
	if (!WindowClassRegistered)
	{
		WindowClassRegistered = RegisterWindowClass();
	}
}

} // namespace WindowPrivate

Window::Window(const WindowCreateDesc &desc)
	: systemHandle(WINDOW_INVALID_HANDLE)
	, bShouldClose(false)
{
	using namespace WindowPrivate;

	EnsureRegisterWindowClass();

	// todo may be we want WS_SIZEBOX
	DWORD style = WS_CLIPSIBLINGS | WS_CLIPCHILDREN;
	style |= WS_SYSMENU | WS_MINIMIZEBOX; // ?
	style |= WS_CAPTION | WS_MAXIMIZEBOX | WS_THICKFRAME; // Title and resizable frame
    DWORD exStyle = WS_EX_APPWINDOW;
    const WCHAR *className = kClassName;
    const WCHAR* windowName = desc.initTitle;
    // todo better init pos ?
    int32_t posX = 0;
    int32_t posY = 0;
	HWND hwnd = CreateWindowExW(
		exStyle,
		className,
		windowName,
		style,
		CW_USEDEFAULT, // Default pos x
		CW_USEDEFAULT, // Default pos y
		desc.initWidth,
		desc.initHeight,
		NULL, // No parent window
		NULL, // No window menu
		GetModuleHandleW(NULL),
		NULL // No passing any data to created window
	);

	if (hwnd == 0)
	{
		ReportLastError(TEXT("CreateWindowExW"));
		return;
	}

	// Bind this instance to system window
	const bool bSucceed = SetPropW(hwnd, kInstancePropName, this);
	if (!bSucceed)
	{
		ReportLastError(TEXT("SetPropW"));

		// Release the system window
		bool bDestroyed = DestroyWindow(hwnd);
		if (!bDestroyed)
		{
			ReportLastError(TEXT("DestroyWindow"));
		}

		return;
	}

	static_assert(sizeof(systemHandle) >= sizeof(HWND), "Casting HWND to a smaller int type!");
	systemHandle = (uint64_t)hwnd;
	assert(systemHandle != WINDOW_INVALID_HANDLE);

	// Show and focus the windows
	ShowWindow(hwnd, SW_SHOWNA);
    if (!BringWindowToTop(hwnd))
	{
		ReportLastError(TEXT("BringWindowToTop"));
	}
    if (!SetForegroundWindow(hwnd))
	{
		ReportLastError(TEXT("SetForgroundWindow"));
	}
	if (!SetFocus(hwnd))
	{
		ReportLastError(TEXT("SetFocus"));
	}
}

Window::~Window()
{
    if (systemHandle != WINDOW_INVALID_HANDLE)
    {
        HWND hwnd = (HWND)systemHandle;
        bool bSucceed = DestroyWindow(hwnd);
		if (!bSucceed)
		{
			WindowPrivate::ReportLastError(TEXT("DestroyWindow"));
		}
        systemHandle = WINDOW_INVALID_HANDLE;
    }
}

bool Window::IsValid() const
{
    return systemHandle != WINDOW_INVALID_HANDLE;
}

uint64_t Window::GetSystemHandle() const
{
    assert(IsValid());
    return systemHandle;
}

bool Window::ShouldClose() const
{
    return bShouldClose; 
}

void Window::PollEvents()
{
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
	{
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}
}

void Window::GetSize(uint32_t& width, uint32_t& height)
{
    assert(IsValid());

    HWND hwnd = (HWND)systemHandle;

    RECT rect;
    bool bSucceed = GetClientRect(hwnd, &rect);
    if (bSucceed)
    {
        // .left and .top are zero
		assert(rect.right < 0xFFFFFFFF);
		assert(rect.bottom < 0xFFFFFFFF);
        width = rect.right;
        height = rect.bottom;
    }
    else
    {
        width = height = 1;
        WindowPrivate::ReportLastError(TEXT("GetClientRect"));
    }
}

void Window::SetShouldClose()
{
    bShouldClose = true;
}
