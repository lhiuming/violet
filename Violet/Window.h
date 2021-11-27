#pragma once

#include <cstdint>

namespace WindowPrivate
{
class SystemCallbackHandler;
};

struct WindowCreateDesc
{
	int32_t initWidth;
	int32_t initHeight;
	const wchar_t* initTitle;
};

class Window
{
public:
	Window(const WindowCreateDesc& desc);
	~Window();

	bool IsValid() const;

	// Platform specifics
	uint64_t GetSystemHandle() const;
    bool ShouldClose() const;
	void PollEvents();

	// Window properties
	void GetSize(uint32_t& width, uint32_t& height);

private:
	friend class WindowPrivate::SystemCallbackHandler;
	void SetShouldClose(); 

	uint64_t systemHandle;
	bool bShouldClose;
};
