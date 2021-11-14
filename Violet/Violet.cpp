#include <stdio.h>
#include <assert.h>
#include <cstring>

// Window management
#include "Window.h"

// Renderer
#include <vulkan/vulkan.h> 

#define VK_CHECK(call) \
	{\
		 VkResult vkRes = call;\
		 assert(vkRes == VK_SUCCESS);\
	}

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

VkSurfaceFormatKHR GSurfaceDefaultFormat{};

#if _DEBUG
VkBool32 VulkanDebugReportCallback(
	VkDebugReportFlagsEXT                       flags,
	VkDebugReportObjectTypeEXT                  objectType,
	uint64_t                                    object,
	size_t                                      location,
	int32_t                                     messageCode,
	const char* pLayerPrefix,
	const char* pMessage,
	void* pUserData)
{
	printf("%s\n", pMessage);
	return VK_FALSE;
}
#endif

VkInstance CreateVkInstance(VkDebugReportCallbackEXT& outDebugReportCallback)
{
	VkApplicationInfo vkAppInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	vkAppInfo.apiVersion = VK_API_VERSION_1_2;

#if _DEBUG
	const char* layerNames[] = {
		"VK_LAYER_KHRONOS_validation"
	};
	const uint32_t layerCount = ARRAY_LENGTH(layerNames);
	uint32_t vkLayerPropCount = 0;
	VK_CHECK(vkEnumerateInstanceLayerProperties(&vkLayerPropCount, nullptr));
	VkLayerProperties* vkLayerPropPtr = new VkLayerProperties[vkLayerPropCount];
	VK_CHECK(vkEnumerateInstanceLayerProperties(&vkLayerPropCount, vkLayerPropPtr));
	bool bIsAllLayersSupported = true;
	printf("Supported Vulkan layers:\n");
	for (uint32_t i = 0; i < layerCount; ++i)
	{
		bool bIsLayersSupported = false;
		for (uint32_t j = 0; j < vkLayerPropCount; j++)
		{
			VkLayerProperties& prop = vkLayerPropPtr[j];
			if (i == 0)
			{
				printf("\t%d: %s | %s\n", j, prop.layerName, prop.description);
			}
			if (!strcmp(prop.layerName, layerNames[i]))
			{
				bIsLayersSupported = true;
				break;
			}
		}
		bIsAllLayersSupported &= bIsLayersSupported;
		if (!bIsLayersSupported)
		{
			printf("Vulkan layer %s is not supported.\n", layerNames[i]);
		}
	}
	assert(bIsAllLayersSupported);
	const char* const* layerNamesPtr = layerNames;
#else
	const char* const* layerNamesPtr = nullptr;
	const uint32_t layerCount = 0;
#endif

	const char* instanceExtentions[] = {
		VK_KHR_SURFACE_EXTENSION_NAME
#if VK_USE_PLATFORM_WIN32_KHR
		, VK_KHR_WIN32_SURFACE_EXTENSION_NAME
#endif
#if _DEBUG
		, VK_EXT_DEBUG_REPORT_EXTENSION_NAME
#endif
	};
	const uint32_t instanceExtentionCount = ARRAY_LENGTH(instanceExtentions);

	VkInstanceCreateInfo vkInstCInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	vkInstCInfo.pApplicationInfo = &vkAppInfo;
	vkInstCInfo.enabledLayerCount = layerCount;
	vkInstCInfo.ppEnabledLayerNames = layerNamesPtr;
	vkInstCInfo.enabledExtensionCount = instanceExtentionCount;
	vkInstCInfo.ppEnabledExtensionNames = instanceExtentions;
	VkInstance vkInst;
	VK_CHECK(vkCreateInstance(&vkInstCInfo, nullptr, &vkInst));

#if _DEBUG
	// Add debug report callback to found out a Swapchain creation bug.
	PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback = VK_NULL_HANDLE;
	CreateDebugReportCallback = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInst, "vkCreateDebugReportCallbackEXT");

	VkDebugReportCallbackCreateInfoEXT vkDebugReportCInfo{ VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT };
	vkDebugReportCInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
	vkDebugReportCInfo.pfnCallback = VulkanDebugReportCallback;
	VK_CHECK(CreateDebugReportCallback(vkInst, &vkDebugReportCInfo, nullptr, &outDebugReportCallback));
#else
	outDebugReportCallback = nullptr;
#endif

#if _DEBUG
	delete[] vkLayerPropPtr;
#endif

	return vkInst;
}

VkDevice CreateVkDevice(VkInstance instance, VkPhysicalDevice& outPhysicalDevice, uint32_t& outDefaultQueueFamily)
{
	VkPhysicalDevice physicalDevicePtr[8];
	uint32_t physicalDeviceCount;
	vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevicePtr);
	assert(physicalDeviceCount);

	// Pick a device: descrete GPU, or the first one
	outPhysicalDevice = physicalDevicePtr[0];
	for (uint32_t i = 0; i < physicalDeviceCount; ++i)
	{
		VkPhysicalDevice& device = physicalDevicePtr[i];
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(device, &props);
		const bool bIsDiscrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
		if (bIsDiscrete)
		{
			outPhysicalDevice = device;
			printf("Pick discrete GPU: %s\n", props.deviceName);
			break;
		}
		else if (i == physicalDeviceCount - 1)
		{
			printf("Pick fallback GPU: %s\n", props.deviceName);
		}
	}


	VkDevice vkDevice;
	outDefaultQueueFamily = -1;
	{
		VkQueueFamilyProperties vkQueueFamilyPtr[8];
		uint32_t vkQueueFamilyCount = 8;
		vkGetPhysicalDeviceQueueFamilyProperties(outPhysicalDevice, &vkQueueFamilyCount, vkQueueFamilyPtr);
		uint32_t vkGraphicQueueFamilyIndex = -1;
		uint32_t vkComputeQueueFamilyIndex = -1;
		printf("Supported Queue Families:\n");
		for (uint32_t i = 0; i < vkQueueFamilyCount; ++i)
		{
			VkQueueFamilyProperties& prop = vkQueueFamilyPtr[i];
			printf("\tQueue Family %d: %d\n", i, prop.queueFlags);
			if (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				vkGraphicQueueFamilyIndex = i;
				printf("\tfound graphics queue family: index %d, count %d\n", i, prop.queueCount);
				continue;
			}
			if (prop.queueFlags & VK_QUEUE_COMPUTE_BIT)
			{
				vkComputeQueueFamilyIndex = i;
				printf("\tfound compute queue family: index %d, count %d\n", i, prop.queueCount);
				continue;
			}
		}
		outDefaultQueueFamily = vkGraphicQueueFamilyIndex;

		VkDeviceQueueCreateInfo vkQueueCInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
		vkQueueCInfo.queueFamilyIndex = vkGraphicQueueFamilyIndex;
		vkQueueCInfo.queueCount = 1;
		float ququePriotiries[] = { 1.0f };
		vkQueueCInfo.pQueuePriorities = ququePriotiries;

		// NOTE: just enable all supported features for simplicity
		VkPhysicalDeviceFeatures supportedDeviceFeatures;
		vkGetPhysicalDeviceFeatures(outPhysicalDevice, &supportedDeviceFeatures);

		const char* deviceExtensionNames[] = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};
		const uint32_t deviceExtensionCount = ARRAY_LENGTH(deviceExtensionNames);

		VkDeviceCreateInfo vkDeviceCInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
		vkDeviceCInfo.queueCreateInfoCount = 1;
		vkDeviceCInfo.pQueueCreateInfos = &vkQueueCInfo;
		vkDeviceCInfo.enabledExtensionCount = deviceExtensionCount;
		vkDeviceCInfo.ppEnabledExtensionNames = deviceExtensionNames;
		vkDeviceCInfo.pEnabledFeatures = &supportedDeviceFeatures;
		VK_CHECK(vkCreateDevice(outPhysicalDevice, &vkDeviceCInfo, nullptr, &vkDevice));
	}

	return vkDevice;
}

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, uint32_t queueFamilyIndex, uint32_t width, uint32_t height)
{
	VkSwapchainCreateInfoKHR vkSwapchainCInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	vkSwapchainCInfo.flags = 0;
	vkSwapchainCInfo.surface = surface;
	vkSwapchainCInfo.minImageCount = 2; // double buffer as minimum
	vkSwapchainCInfo.imageFormat = GSurfaceDefaultFormat.format;
	vkSwapchainCInfo.imageColorSpace = GSurfaceDefaultFormat.colorSpace;
	vkSwapchainCInfo.imageExtent.width = width;
	vkSwapchainCInfo.imageExtent.height = height;
	vkSwapchainCInfo.imageArrayLayers = 1;
	vkSwapchainCInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	vkSwapchainCInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT; // for clear
	vkSwapchainCInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	vkSwapchainCInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	vkSwapchainCInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	VkSwapchainKHR vkSwapchain;
	VK_CHECK(vkCreateSwapchainKHR(device, &vkSwapchainCInfo, nullptr, &vkSwapchain));

	return vkSwapchain;
}

VkRenderPass CreateRenderPass(VkDevice device)
{
	VkAttachmentReference colorAttachment{ };
	colorAttachment.attachment = 0;
	colorAttachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass{ };
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachment;

	VkAttachmentDescription attachment{};
	attachment.format = GSurfaceDefaultFormat.format;
	attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkRenderPassCreateInfo cInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
	cInfo.attachmentCount = 1;
	cInfo.pAttachments = &attachment;
	cInfo.subpassCount = 1;
	cInfo.pSubpasses = &subpass;
	VkRenderPass renderPass;
	VK_CHECK(vkCreateRenderPass(device, &cInfo, nullptr, &renderPass));

	return renderPass;
}

VkImageView CreateImageView(VkDevice device, VkImage image)
{
	VkImageViewCreateInfo cInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	cInfo.image = image;
	cInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	cInfo.format = GSurfaceDefaultFormat.format;
	cInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	cInfo.subresourceRange.layerCount = 1;
	cInfo.subresourceRange.levelCount = 1;
	VkImageView imageView;
	vkCreateImageView(device, &cInfo, nullptr, &imageView);
	return imageView;
}

VkFramebuffer CreateFrameBuffer(VkDevice device, VkRenderPass renderPass, VkImageView imageView, uint32_t width, uint32_t height)
{
	VkFramebufferCreateInfo cInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
	cInfo.renderPass = renderPass;
	cInfo.attachmentCount = 1;
	cInfo.pAttachments = &imageView;
	cInfo.width = width;
	cInfo.height = height;
	cInfo.layers = 1;
	VkFramebuffer frameBuffer;
	VK_CHECK(vkCreateFramebuffer(device, &cInfo, nullptr, &frameBuffer));

	return frameBuffer;
}

int main()
{
	printf("Hello, Violet on Vulkan.\n");

	WindowCreateDesc windowCreateDesc {
		1280,
		720,
		TEXT("Violet")
	};
	Window window { windowCreateDesc };

	assert(window.IsValid());

	//int initRes = glfwInit();
	//assert(initRes == GLFW_TRUE);

	// NOTE: must hint GLFW to dont create openGL context, avoiding a bug in swapchain creation.
	//glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	//GLFWwindow* glfwWindow = glfwCreateWindow(1024, 720, "REI2", NULL, NULL);
	//assert(glfwWindow);

	VkDebugReportCallbackEXT vkDebugReport = nullptr;
	VkInstance vkInstance = CreateVkInstance(vkDebugReport);

	VkPhysicalDevice vkPhysicalDevice;
	uint32_t vkDefaultQueueFamilyIndex;
	VkDevice vkDevice = CreateVkDevice(vkInstance, vkPhysicalDevice, vkDefaultQueueFamilyIndex);

	// should be under VK_USE_PLATFORM_WIN32_KHR
	VkWin32SurfaceCreateInfoKHR vkSurfaceCInfo{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
	vkSurfaceCInfo.hinstance = GetModuleHandle(NULL);
	vkSurfaceCInfo.hwnd = (HWND)window.GetSystemHandle();//glfwGetWin32Window(glfwWindow);
	VkSurfaceKHR vkSurface;
	VK_CHECK(vkCreateWin32SurfaceKHR(vkInstance, &vkSurfaceCInfo, nullptr, &vkSurface));

	VkBool32 bIsSurfaceSupported;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice, vkDefaultQueueFamilyIndex, vkSurface, &bIsSurfaceSupported));
	assert(bIsSurfaceSupported == VK_TRUE);

	{
		uint32_t vkSurfaceFormatCount = 0;
		VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice, vkSurface, &vkSurfaceFormatCount, nullptr));
		assert(vkSurfaceFormatCount);
		VkSurfaceFormatKHR* vkSurfaceFormat = new VkSurfaceFormatKHR[vkSurfaceFormatCount];
		VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice, vkSurface, &vkSurfaceFormatCount, vkSurfaceFormat));
#if _DEBUG
		printf("Vulkan surface supported format:\n");
		for (uint32_t i = 0; i < vkSurfaceFormatCount; ++i)
		{
			VkSurfaceFormatKHR& format = vkSurfaceFormat[i];
			printf("\t%d:\t%d - %d\n", i, format.format, format.colorSpace);
		}
#endif
		GSurfaceDefaultFormat = vkSurfaceFormat[0];
		delete[] vkSurfaceFormat;
	}


	//int glfwWidth, glfwHeight;
	//glfwGetWindowSize(glfwWindow, &glfwWidth, &glfwHeight);
	int32_t width, height;
	window.GetSize(width, height);
	VkSwapchainKHR vkSwapchain = CreateSwapchain(vkDevice, vkSurface, vkDefaultQueueFamilyIndex, width, height);
	assert(vkSwapchain);

	VkSemaphoreCreateInfo vkSemaphoreCInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkSemaphore vkPresentSemaphore = nullptr;
	VK_CHECK(vkCreateSemaphore(vkDevice, &vkSemaphoreCInfo, nullptr, &vkPresentSemaphore));

	VkQueue vkDefaultQueue = nullptr;
	vkGetDeviceQueue(vkDevice, vkDefaultQueueFamilyIndex, 0, &vkDefaultQueue);
	assert(vkDefaultQueue);

	// Trick to initial swapchain image for testing
	VkImage vkSwapchainImages[8];
	uint32_t vkSwapchainImageCount = ARRAY_LENGTH(vkSwapchainImages);
	VK_CHECK(vkGetSwapchainImagesKHR(vkDevice, vkSwapchain, &vkSwapchainImageCount, vkSwapchainImages));
	assert(vkSwapchainImageCount);

	VkImageView vkImageView[8];
	for (uint32_t i = 0; i < vkSwapchainImageCount; ++i)
	{
		vkImageView[i] = CreateImageView(vkDevice, vkSwapchainImages[i]);
		assert(vkImageView[i]);
	}

	VkRenderPass vkRenderPass = CreateRenderPass(vkDevice);
	assert(vkRenderPass);

	VkFramebuffer vkFrameBuffer[8] = { nullptr, };
	for (uint32_t i = 0; i < vkSwapchainImageCount; ++i)
	{
		vkFrameBuffer[i] = CreateFrameBuffer(vkDevice, vkRenderPass, vkImageView[i], width, height);
		assert(vkFrameBuffer[i]);
	}

	VkCommandPoolCreateInfo vkCmdPoolCInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	vkCmdPoolCInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	vkCmdPoolCInfo.queueFamilyIndex = vkDefaultQueueFamilyIndex;
	VkCommandPool vkCmdPool;
	VK_CHECK(vkCreateCommandPool(vkDevice, &vkCmdPoolCInfo, nullptr, &vkCmdPool));

	VkCommandBufferAllocateInfo vkCmdBufferAInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	vkCmdBufferAInfo.commandPool = vkCmdPool;
	vkCmdBufferAInfo.commandBufferCount = 1;
	VkCommandBuffer vkCmdBuffer;
	VK_CHECK(vkAllocateCommandBuffers(vkDevice, &vkCmdBufferAInfo, &vkCmdBuffer));

	//while (!glfwWindowShouldClose(glfwWindow))
	while (!window.ShouldClose())
	{
		//glfwPollEvents();
		window.PollEvents();

		// wait idle at previous frame
		VK_CHECK(vkResetCommandBuffer(vkCmdBuffer, 0));

		uint32_t imageIndex = -1;
		VK_CHECK(vkAcquireNextImageKHR(vkDevice, vkSwapchain, UINT64_MAX, vkPresentSemaphore, VK_NULL_HANDLE, &imageIndex));

		{
			VkCommandBufferBeginInfo begInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			VK_CHECK(vkBeginCommandBuffer(vkCmdBuffer, &begInfo));
		}

		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			//imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = vkSwapchainImages[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);
		}

		// NOTE: violet #5a4498
		VkClearColorValue clearColorValue{};
		clearColorValue.float32[0] = 0x5A / 255.0f;
		clearColorValue.float32[1] = 0x44 / 255.0f;
		clearColorValue.float32[2] = 0x94 / 255.0f;

#if 0
		VkImageSubresourceRange range{};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.levelCount = 1;
		range.layerCount = 1;
		vkCmdClearColorImage(vkCmdBuffer, vkSwapchainImages[imageIndex], VK_IMAGE_LAYOUT_GENERAL, &clearColorValue, 1, &range);
#endif

		{
			VkRenderPassBeginInfo beginInfo {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
			beginInfo.renderPass = vkRenderPass;
			beginInfo.framebuffer = vkFrameBuffer[imageIndex];
			beginInfo.renderArea.extent.width = width;
			beginInfo.renderArea.extent.height = height;
			beginInfo.clearValueCount = 1;
			VkClearValue clearValue;
			clearValue.color = clearColorValue;
			beginInfo.pClearValues = &clearValue;
			vkCmdBeginRenderPass(vkCmdBuffer, &beginInfo, VkSubpassContents::VK_SUBPASS_CONTENTS_INLINE);
		}


		vkCmdEndRenderPass(vkCmdBuffer);

		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			//imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageBarrier.oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = vkSwapchainImages[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);
		}

		VK_CHECK(vkEndCommandBuffer(vkCmdBuffer));

		VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &vkCmdBuffer;
		VK_CHECK(vkQueueSubmit(vkDefaultQueue, 1, &submitInfo, VK_NULL_HANDLE));

		VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &vkPresentSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &vkSwapchain;
		presentInfo.pImageIndices = &imageIndex;
		VK_CHECK(vkQueuePresentKHR(vkDefaultQueue, &presentInfo));

		// brute force for now
		VK_CHECK(vkDeviceWaitIdle(vkDevice));
	}

	vkDestroyCommandPool(vkDevice, vkCmdPool, nullptr);
	vkDestroySemaphore(vkDevice, vkPresentSemaphore, nullptr);
	for (uint32_t i = 0; i < vkSwapchainImageCount; ++i)
	{
		vkDestroyImageView(vkDevice, vkImageView[i], nullptr);
		vkDestroyFramebuffer(vkDevice, vkFrameBuffer[i], nullptr);
	}
	vkDestroyRenderPass(vkDevice, vkRenderPass, nullptr);
	vkDestroySwapchainKHR(vkDevice, vkSwapchain, nullptr);
	vkDestroySurfaceKHR(vkInstance, vkSurface, nullptr);
	vkDestroyDevice(vkDevice, nullptr);
#if _DEBUG
	{
		PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = VK_NULL_HANDLE;
		DestroyDebugReportCallback = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugReportCallbackEXT");
		DestroyDebugReportCallback (vkInstance, vkDebugReport, nullptr);
	}
#endif
	vkDestroyInstance(vkInstance, nullptr);

	//glfwDestroyWindow(glfwWindow);
}