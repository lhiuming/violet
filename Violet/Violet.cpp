#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <fstream>

// Win32 Platform
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX

// Window management
#include "Window.h"

// Renderer
//#include <vulkan/vulkan.h> 
// use volk for vulkan function loading
#include <volk.h>
				
#define VK_CHECK(call) \
	{\
		 VkResult vkRes = call;\
		 assert(vkRes == VK_SUCCESS);\
	}

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

VkDevice GVkDevice;
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
	printf("%s\n\n", pMessage);
	return VK_FALSE;
}
#endif

VkInstance CreateVkInstance()
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
		VK_KHR_SURFACE_EXTENSION_NAME // required by VK_KHR_swapchain
		, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME // required by VK_KHR_dynamic_rendering
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
		VkPhysicalDeviceFeatures2 supportedDeviceFeatures {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
		VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRendering {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
		supportedDeviceFeatures.pNext = &dynamicRendering;
		vkGetPhysicalDeviceFeatures2KHR(outPhysicalDevice, &supportedDeviceFeatures);

		// NOTE: requring dynamic rendering
		assert(dynamicRendering.dynamicRendering);

		const char* deviceExtensionNames[] = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
			, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME
		};
		const uint32_t deviceExtensionCount = ARRAY_LENGTH(deviceExtensionNames);

		VkDeviceCreateInfo vkDeviceCInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
		vkDeviceCInfo.queueCreateInfoCount = 1;
		vkDeviceCInfo.pQueueCreateInfos = &vkQueueCInfo;
		vkDeviceCInfo.enabledExtensionCount = deviceExtensionCount;
		vkDeviceCInfo.ppEnabledExtensionNames = deviceExtensionNames;
		vkDeviceCInfo.pEnabledFeatures = &supportedDeviceFeatures.features;
		vkDeviceCInfo.pNext = &dynamicRendering;
		VK_CHECK(vkCreateDevice(outPhysicalDevice, &vkDeviceCInfo, nullptr, &vkDevice));
	}

	return vkDevice;
}

VkImageView CreateImageView(VkDevice device, VkImage image);

struct Swapchain
{
	uint32_t width;
	uint32_t heigth;
	VkSwapchainKHR vkSwapchain;
	uint32_t imageCount;
	VkImage images[8];
	VkImageView imageViews[8];
};

Swapchain
CreateSwapchain(VkDevice device, VkSurfaceKHR surface, uint32_t queueFamilyIndex, uint32_t width, uint32_t height)
{
	Swapchain swapchain {width, height};

	VkSwapchainCreateInfoKHR vkSwapchainCInfo {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
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
	vkSwapchainCInfo.imageUsage |= VK_IMAGE_USAGE_STORAGE_BIT;      // for compute
	vkSwapchainCInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	vkSwapchainCInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	vkSwapchainCInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	VK_CHECK(vkCreateSwapchainKHR(device, &vkSwapchainCInfo, nullptr, &swapchain.vkSwapchain));

	swapchain.imageCount = ARRAY_LENGTH(swapchain.images);
	VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain.vkSwapchain, &swapchain.imageCount, swapchain.images));
	assert(swapchain.imageCount);

	for (uint32_t i = 0; i < swapchain.imageCount; ++i)
	{
		VkImageView imageView = CreateImageView(device, swapchain.images[i]);
		assert(imageView);
		swapchain.imageViews[i] = imageView;
	}

	return swapchain;
}

void DestroySwapchain(VkDevice device, Swapchain& swapchain) 
{
	for (uint32_t imageIndex = 0; imageIndex < swapchain.imageCount; ++imageIndex)
	{
		vkDestroyImageView(device, swapchain.imageViews[imageIndex], 0);
	}

	// NOTE: this also destoy images
	vkDestroySwapchainKHR(device, swapchain.vkSwapchain, 0);
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

std::string GetShaderBinFilePath(const char* shaderShortPath) 
{ 
	static std::string ShaderDir = "Shader\\";
	std::string fullPath = ShaderDir + shaderShortPath + ".spv";
	return fullPath;
}


#include "spirv_reflect.h"
namespace shader
{

struct ShaderReflect
{
	// This is too much, but should be okay because we dont have much shaders.
	SpvReflectShaderModule spvModule;

	bool IsCompute() const;
	uint32_t FindBinding(const char *name) const;
};

bool ShaderReflect::IsCompute() const { return spvModule.spirv_execution_model == SpvExecutionModelGLCompute; }

uint32_t ShaderReflect::FindBinding(const char *name) const
{
	for (uint32_t i = 0; i < spvModule.descriptor_binding_count; i++)
	{
		auto &binding = spvModule.descriptor_bindings[i];
		if (strcmp(name, binding.name) == 0)
		{
			//assert(binding.resource_type == SPV_REFLECT_RESOURCE_FLAG_UAV);
			return i;
		}
	}
	return ~0u;
}

#define SPV_CHECK(call) \
	{ \
		SpvReflectResult spvResult = call; \
		assert(spvResult == SPV_REFLECT_RESULT_SUCCESS); \
	}

ShaderReflect CreateShaderReflection(const char *shaderCode, size_t codeSize)
{
	ShaderReflect reflect;
	SPV_CHECK(spvReflectCreateShaderModule(codeSize, shaderCode, &reflect.spvModule));
	return reflect;
};

VkShaderModule LoadShader(const char *filePath, ShaderReflect *reflect)
{
	std::ifstream is(filePath, std::ios::binary | std::ios::in | std::ios::ate);

	char *shaderCode = nullptr;
	size_t shaderSize = 0;
	if (is.is_open())
	{
		shaderSize = is.tellg();
		is.seekg(0, std::ios::beg);
		shaderCode = new char[shaderSize];
		is.read(shaderCode, shaderSize);
		is.close();
		assert(shaderSize > 0);
	}

	if (shaderCode)
	{
		assert(shaderSize > 0);

		VkShaderModuleCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = shaderSize;
		createInfo.pCode = (uint32_t *)shaderCode;

		VkShaderModule shaderModule;
		VK_CHECK(vkCreateShaderModule(GVkDevice, &createInfo, nullptr, &shaderModule));

		if (reflect)
		{
			*reflect = CreateShaderReflection(shaderCode, shaderSize);
		}

		delete[] shaderCode;

		return shaderModule;
	}

	return nullptr;
}

} // namespace shader

VkPipelineCache GPipelineCache = VK_NULL_HANDLE;

VkPipeline CreatePipeline(VkShaderModule shaderModule, VkPipelineLayout layout) 
{
	VkComputePipelineCreateInfo createInfo {};
	createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	createInfo.flags;
	{
		VkPipelineShaderStageCreateInfo &stage = createInfo.stage;
		stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.flags;
		stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = shaderModule;
		stage.pName = "main";
		stage.pSpecializationInfo = NULL;
	}
	createInfo.layout = layout;
	createInfo.basePipelineHandle = VK_NULL_HANDLE;

	VkPipeline pipeline;
	VK_CHECK(vkCreateComputePipelines(GVkDevice, GPipelineCache, 1, &createInfo, nullptr, &pipeline));

	return pipeline;
}

VkDescriptorPool GDescriptorPool = VK_NULL_HANDLE;

void CreateDescriptorSet(shader::ShaderReflect reflect, VkDescriptorSetLayout& outSetLayout, VkPipelineLayout &outPipelineLayout, VkDescriptorSet &outDescriptorSets)
{
	// NOTE: single set layrout atm
	{
		VkDescriptorSetLayoutBinding bindings[64];
		uint32_t bindingCount = 0;
		assert(reflect.spvModule.descriptor_binding_count < ARRAY_LENGTH(bindings));
		for (uint32_t i = 0; i < reflect.spvModule.descriptor_binding_count; i++)
		{
			auto &spvBinding = reflect.spvModule.descriptor_bindings[i];
			if (spvBinding.count == 0)
			{
				// reserved binding slot
				continue;
			}

			auto &binding = bindings[bindingCount];
			binding.binding = reflect.spvModule.descriptor_bindings[i].binding;
			// NOTE: they seems to have same value; might want to do safer converion by table/map
			binding.descriptorType = (VkDescriptorType)spvBinding.descriptor_type;
			binding.descriptorCount = spvBinding.count;
			binding.stageFlags = reflect.spvModule.shader_stage;

			bindingCount++;
		}

		VkDescriptorSetLayoutCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		createInfo.pBindings = bindings;
		createInfo.bindingCount = bindingCount;
		VK_CHECK(vkCreateDescriptorSetLayout(GVkDevice, &createInfo, 0, &outSetLayout));
	}

	// Layout object
	{
		VkPipelineLayoutCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		createInfo.setLayoutCount = 1;
		createInfo.pSetLayouts = &outSetLayout;
		createInfo.pushConstantRangeCount = 0;
		VK_CHECK(vkCreatePipelineLayout(GVkDevice, &createInfo, 0, &outPipelineLayout));
	}

	// Descriptor sets object (from pool)
	{
		VkDescriptorSetAllocateInfo info {};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		info.descriptorPool = GDescriptorPool;
		info.descriptorSetCount = 1;
		info.pSetLayouts = &outSetLayout;
		VK_CHECK(vkAllocateDescriptorSets(GVkDevice, &info, &outDescriptorSets));
	}
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

	// Init vulkan loader
	VK_CHECK(volkInitialize());

	VkInstance vkInstance = CreateVkInstance();

	// Load vulkan functions for instance (other functions are loaded later via volkLoadDevice)
	volkLoadInstanceOnly(vkInstance);

#if _DEBUG
	// Add debug report callback to found out a Swapchain creation bug.
	VkDebugReportCallbackEXT vkDebugReport = nullptr;
	{
		VkDebugReportCallbackCreateInfoEXT vkDebugReportCInfo {VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT};
		vkDebugReportCInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		vkDebugReportCInfo.pfnCallback = VulkanDebugReportCallback;
		VK_CHECK(vkCreateDebugReportCallbackEXT(vkInstance, &vkDebugReportCInfo, nullptr, &vkDebugReport));
	}
#else
	outDebugReportCallback = nullptr;
#endif


	VkPhysicalDevice vkPhysicalDevice;
	uint32_t vkDefaultQueueFamilyIndex;
	VkDevice vkDevice = CreateVkDevice(vkInstance, vkPhysicalDevice, vkDefaultQueueFamilyIndex);

	// Load vulkan functions (for the only vulkan device)
	volkLoadDevice(vkDevice);

	assert(vkDevice);
	GVkDevice = vkDevice;

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
	Swapchain swapchain;
	{
		uint32_t width, height;
		window.GetSize(width, height);
		swapchain = CreateSwapchain(vkDevice, vkSurface, vkDefaultQueueFamilyIndex, width, height);
	}

	VkSemaphoreCreateInfo vkSemaphoreCInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VkSemaphore vkPresentSemaphore = nullptr;
	VK_CHECK(vkCreateSemaphore(vkDevice, &vkSemaphoreCInfo, nullptr, &vkPresentSemaphore));

	VkQueue vkDefaultQueue = nullptr;
	vkGetDeviceQueue(vkDevice, vkDefaultQueueFamilyIndex, 0, &vkDefaultQueue);
	assert(vkDefaultQueue);

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

	// Shader Pipeline cache
	{
		VkPipelineCacheCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		createInfo.initialDataSize = 0;
		createInfo.pInitialData = nullptr;
		VK_CHECK(vkCreatePipelineCache(vkDevice, &createInfo, 0, &GPipelineCache));
	}

	// Descriotor pool
	{
		VkDescriptorPoolSize poolSizes[] = {
			{VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
		};

		VkDescriptorPoolCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		createInfo.maxSets = 1;
		createInfo.poolSizeCount = ARRAY_LENGTH(poolSizes);
		createInfo.pPoolSizes = poolSizes;
		VK_CHECK(vkCreateDescriptorPool(vkDevice, &createInfo, 0, &GDescriptorPool));
	}

	// Create shader
	struct
	{
		VkPipeline pipeline;
		VkDescriptorSetLayout setLayout;
		VkPipelineLayout pipelineLayout;
		VkDescriptorSet descriptorSets[1];

		shader::ShaderReflect relfect;
	} fullscreenCS;
	{
		std::string shaderFileName = GetShaderBinFilePath("FullScreenCS.s");
		VkShaderModule shaderModule = LoadShader(shaderFileName.data(), &fullscreenCS.relfect);
		assert(shaderModule);

		// layout and descriptor stuffs
		CreateDescriptorSet(fullscreenCS.relfect, fullscreenCS.setLayout, fullscreenCS.pipelineLayout, fullscreenCS.descriptorSets[0]);

		fullscreenCS.pipeline = CreatePipeline(shaderModule, fullscreenCS.pipelineLayout);

		// TODO it is safe to detroy here?
		vkDestroyShaderModule(vkDevice, shaderModule, 0);
	}

	//while (!glfwWindowShouldClose(glfwWindow))
	while (!window.ShouldClose())
	{
		//glfwPollEvents();
		window.PollEvents();

		// check new size
		uint32_t newWidth, newHeight;
		window.GetSize(newWidth, newHeight);

		// wait idle at previous frame
		VK_CHECK(vkResetCommandBuffer(vkCmdBuffer, 0));
		
		// Create swapchain if window resized
		const bool bResize = (newWidth != swapchain.width) || (newHeight != swapchain.heigth);
		if (bResize)
		{
			DestroySwapchain(vkDevice, swapchain);
			swapchain = CreateSwapchain(vkDevice, vkSurface, vkDefaultQueueFamilyIndex, newWidth, newHeight);
		}


		uint32_t imageIndex = -1;
		VK_CHECK(vkAcquireNextImageKHR(vkDevice, swapchain.vkSwapchain, UINT64_MAX, vkPresentSemaphore, VK_NULL_HANDLE, &imageIndex));

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
			imageBarrier.image = swapchain.images[imageIndex];
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

		bool bFillWithCompute = true;
		bool bClearUsingRenderPass = !bFillWithCompute;

		// Using dynamic render pass
		if (bClearUsingRenderPass)
		{
			VkRenderingAttachmentInfoKHR attachmentInfo {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
			attachmentInfo.imageView = swapchain.imageViews[imageIndex];
			attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			attachmentInfo.resolveMode = VK_RESOLVE_MODE_NONE;
			attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear 
			attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentInfo.clearValue.color = clearColorValue;
			attachmentInfo.clearValue.depthStencil;

			VkRenderingInfoKHR renderingInfo {VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
			renderingInfo.flags;
			renderingInfo.renderArea.offset = {0, 0};
			renderingInfo.renderArea.extent = {swapchain.width, swapchain.heigth};
			renderingInfo.layerCount = 1;
			renderingInfo.viewMask;
			renderingInfo.colorAttachmentCount = 1;
			renderingInfo.pColorAttachments = &attachmentInfo;
			vkCmdBeginRenderingKHR(vkCmdBuffer, &renderingInfo);

			vkCmdEndRenderingKHR(vkCmdBuffer);
		}

		// Transition for compute
		if (bFillWithCompute)
		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = swapchain.images[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);
		}

		// Fill the buffer with compute
		if (bFillWithCompute)
		{
			vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, fullscreenCS.pipeline);

			VkDescriptorImageInfo dstImageInfo {};
			dstImageInfo.sampler = VK_NULL_HANDLE;
			dstImageInfo.imageView = swapchain.imageViews[imageIndex];
			dstImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

			VkWriteDescriptorSet writeDescriptorSet {};
			writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeDescriptorSet.dstSet = fullscreenCS.descriptorSets[0];
			writeDescriptorSet.dstBinding = 0;
			writeDescriptorSet.dstArrayElement = 0;
			writeDescriptorSet.descriptorCount = 1;
			writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			writeDescriptorSet.pImageInfo = &dstImageInfo;
			vkUpdateDescriptorSets(vkDevice, 1, &writeDescriptorSet, 0, nullptr);

			vkCmdBindDescriptorSets(
				vkCmdBuffer,
				VK_PIPELINE_BIND_POINT_COMPUTE,
				fullscreenCS.pipelineLayout,
				0,
				ARRAY_LENGTH(fullscreenCS.descriptorSets),
				fullscreenCS.descriptorSets,
				0,
				0);

			uint32_t dispatchX = (swapchain.width + 7) / 8;
			uint32_t dispatchY = (swapchain.heigth + 3) / 4;
			vkCmdDispatch(vkCmdBuffer, dispatchX, dispatchY, 1);
		}

		// Transition for present
		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = bFillWithCompute ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = swapchain.images[imageIndex];
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
		presentInfo.pSwapchains = &swapchain.vkSwapchain;
		presentInfo.pImageIndices = &imageIndex;
		VK_CHECK(vkQueuePresentKHR(vkDefaultQueue, &presentInfo));

		// brute force for now
		VK_CHECK(vkDeviceWaitIdle(vkDevice));
	}

	vkDestroyPipeline(vkDevice, fullscreenCS.pipeline, 0);
	vkDestroyDescriptorSetLayout(vkDevice, fullscreenCS.setLayout, 0);
	vkDestroyPipelineLayout(vkDevice, fullscreenCS.pipelineLayout, 0);
	vkFreeDescriptorSets(vkDevice, GDescriptorPool, ARRAY_LENGTH(fullscreenCS.descriptorSets), fullscreenCS.descriptorSets);

	vkDestroyPipelineCache(vkDevice, GPipelineCache, 0);
	vkDestroyDescriptorPool(vkDevice, GDescriptorPool, 0);

	vkDestroyCommandPool(vkDevice, vkCmdPool, nullptr);
	vkDestroySemaphore(vkDevice, vkPresentSemaphore, nullptr);

	DestroySwapchain(vkDevice, swapchain);

	vkDestroySurfaceKHR(vkInstance, vkSurface, nullptr);
	vkDestroyDevice(vkDevice, nullptr);
#if _DEBUG
	vkDestroyDebugReportCallbackEXT(vkInstance, vkDebugReport, nullptr);
#endif
	vkDestroyInstance(vkInstance, nullptr);

	//glfwDestroyWindow(glfwWindow);
}