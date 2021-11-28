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
#include "Extern/volk/volk.h"
				
#define VK_CHECK(call) \
	{\
		 VkResult vkRes = call;\
		 assert(vkRes == VK_SUCCESS);\
	}

#define ARRAY_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

VkDevice GVkDevice;
VkPhysicalDeviceMemoryProperties GMemoryProperties;
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

std::string GetModelFilePath(const char *modelShortPath) {
	static std::string ModelDir = "Data\\";
	std::string fullPath = ModelDir + modelShortPath;
	return fullPath;
}

#include "Extern/SPIRV-Reflect/spirv_reflect.h"
namespace shader
{

struct ShaderReflect
{
	// This is too much, but should be okay because we dont have much shaders.
	SpvReflectShaderModule spvModule;

	bool IsCompute() const;
	uint32_t FindBinding(const char *name) const;
	bool FindBinding(const char* name, VkDescriptorType& outDescriptorType, uint32_t& outSetIndex, uint32_t& outBindingInSet) const;
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

bool ShaderReflect::FindBinding(const char* name, VkDescriptorType& outDescriptorType, uint32_t& outSetIndex, uint32_t& outBindingInSet) const
{
	spvModule.descriptor_binding_count;
	for (uint32_t bindingIndex = 0; bindingIndex < spvModule.descriptor_binding_count; ++bindingIndex)
	{
		SpvReflectDescriptorBinding& binding = spvModule.descriptor_bindings[bindingIndex];
		if (strcmp(name, binding.name) == 0)
		{
			outDescriptorType = (VkDescriptorType)binding.descriptor_type;
			outSetIndex = binding.set;
			outBindingInSet = binding.binding;
			return true;
		}
	}
	return false;
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

void DestroyShaderReflection(ShaderReflect& reflect)
{
	spvReflectDestroyShaderModule(&reflect.spvModule);
}

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

VkPipeline CreateComputePipeline(VkShaderModule shaderModule, VkPipelineLayout layout, const char* entryPoint)
{
	VkComputePipelineCreateInfo createInfo {};
	createInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	createInfo.flags;
	{
		VkPipelineShaderStageCreateInfo& stage = createInfo.stage;
		stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage.flags;
		stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		stage.module = shaderModule;
		stage.pName = entryPoint;
		stage.pSpecializationInfo = NULL;
	}
	createInfo.layout = layout;
	createInfo.basePipelineHandle = VK_NULL_HANDLE;

	VkPipeline pipeline;
	VK_CHECK(vkCreateComputePipelines(GVkDevice, GPipelineCache, 1, &createInfo, nullptr, &pipeline));

	return pipeline;
}

VkPipeline CreateGraphicPipeline(VkShaderModule vertModule, const char* vertEntryPoint, 
	VkShaderModule fragModule, const char* fragEntryPoint, VkPipelineLayout layout)
{
	VkGraphicsPipelineCreateInfo createInfo {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	createInfo.flags;

	VkPipelineShaderStageCreateInfo stages[2] = {};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertModule;
	stages[0].pName = vertEntryPoint;
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragModule;
	stages[1].pName = fragEntryPoint;
	createInfo.stageCount = ARRAY_LENGTH(stages);
	createInfo.pStages = stages;

	// Not really using any vertex input
	VkPipelineVertexInputStateCreateInfo vertexInputState {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	createInfo.pVertexInputState = &vertexInputState;

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	createInfo.pInputAssemblyState = &inputAssemblyState;

    const VkPipelineTessellationStateCreateInfo*     pTessellationState;

	// dummy; it is dynamic
	VkPipelineViewportStateCreateInfo viewportState {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	VkViewport viewport {};
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	VkRect2D scissor {};
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;
	createInfo.pViewportState = &viewportState;

    VkPipelineRasterizationStateCreateInfo rasterizationState {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	//rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	//rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	//rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	createInfo.pRasterizationState = &rasterizationState;

	VkPipelineMultisampleStateCreateInfo multisampleState {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	createInfo.pMultisampleState = &multisampleState;

    VkPipelineDepthStencilStateCreateInfo depthStencilState {
		VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	createInfo.pDepthStencilState = &depthStencilState;

    VkPipelineColorBlendStateCreateInfo colorBlendState {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	VkPipelineColorBlendAttachmentState attachmentState {};
	attachmentState.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendState.attachmentCount = 1;
	colorBlendState.pAttachments = &attachmentState; 
	createInfo.pColorBlendState = &colorBlendState;

	VkPipelineDynamicStateCreateInfo dynamicState {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT
		,VK_DYNAMIC_STATE_SCISSOR
	};
	dynamicState.dynamicStateCount = ARRAY_LENGTH(dynamicStates);
	dynamicState.pDynamicStates = dynamicStates;
	createInfo.pDynamicState = &dynamicState;

	createInfo.layout = layout;

	// using dynamic_rendering
	createInfo.renderPass = VK_NULL_HANDLE;
	createInfo.subpass = 0;

    VkPipeline                                       basePipelineHandle;
    int32_t                                          basePipelineIndex;

	VkPipeline pipeline;
	VK_CHECK(vkCreateGraphicsPipelines(GVkDevice, GPipelineCache, 1, &createInfo, nullptr, &pipeline));

	return pipeline;
}


VkDescriptorPool GDescriptorPool = VK_NULL_HANDLE;

void CreateDescriptorSet(shader::ShaderReflect* reflects, uint32_t reflectCount, VkDescriptorSetLayout& outSetLayout, VkPipelineLayout &outPipelineLayout, VkDescriptorSet &outDescriptorSets)
{
	// NOTE: single set layrout atm
	{
		VkDescriptorSetLayoutBinding bindings[64];
		uint32_t bindingCount = 0;

		for (uint32_t reflectIndex = 0; reflectIndex < reflectCount; ++reflectIndex)
		{
			shader::ShaderReflect& reflect = reflects[reflectIndex];

			assert(reflect.spvModule.descriptor_binding_count <= ARRAY_LENGTH(bindings));
			for (uint32_t i = 0; i < reflect.spvModule.descriptor_binding_count; i++)
			{
				auto& spvBinding = reflect.spvModule.descriptor_bindings[i];
				if (spvBinding.count == 0)
				{
					// reserved binding slot
					continue;
				}

				// only one set atm
				assert(spvBinding.set == 0);

				auto& binding = bindings[bindingCount];
				binding = {};
				binding.binding = spvBinding.binding;
				binding.descriptorType = (VkDescriptorType)spvBinding.descriptor_type;
				// NOTE: this is for array
				binding.descriptorCount = spvBinding.count;
				binding.stageFlags = (VkShaderStageFlags)reflect.spvModule.shader_stage;

				bindingCount++;
			}
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

uint32_t SelectMemoryTypeIndex(uint32_t memoryTypeBits, VkMemoryPropertyFlags requiredProperties)
{
	for (uint32_t typeIndex = 0; typeIndex < GMemoryProperties.memoryTypeCount; ++typeIndex)
	{
		const VkMemoryType &memoryType = GMemoryProperties.memoryTypes[typeIndex];
		if ((memoryTypeBits & (1 << typeIndex)) &&
		    ((memoryType.propertyFlags & requiredProperties) == requiredProperties))
		{
			return typeIndex;
		}
	}
	assert(!"No compatible memory type with required properties");
	return ~0u;
}


struct Buffer
{
	VkBuffer vkBuffer;
	VkDeviceMemory vkMemory;
	size_t size;
	void *data; // persistenly mapped

	VkBufferView SRV;
};

Buffer CreateBuffer(VkDevice device, size_t size, VkBufferUsageFlags usage, VkFormat srvFormat)
{
	// Create the vk buffer object (no memory)
	VkBuffer vkBuffer;
	{
		VkBufferCreateInfo createInfo {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		createInfo.flags;
		createInfo.size = size;
		createInfo.usage = usage;
		createInfo.sharingMode;
		createInfo.queueFamilyIndexCount;
		createInfo.pQueueFamilyIndices;
		vkCreateBuffer(device, &createInfo, nullptr, &vkBuffer);
	}

	// Decide memory type
	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(device, vkBuffer, &memoryRequirements);
	// NOTE treat evey buffer like a staging buffer, atm
	VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	uint32_t memoryTypeIndex = SelectMemoryTypeIndex(memoryRequirements.memoryTypeBits, memoryProperties);
	assert(memoryTypeIndex != ~0u);

	// Allocate memory (just) for the buffer
	VkDeviceMemory vkMemory;
	{
		VkMemoryAllocateInfo allocateInfo {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		allocateInfo.allocationSize = memoryRequirements.size;
		allocateInfo.memoryTypeIndex = memoryTypeIndex;
		VK_CHECK(vkAllocateMemory(device, &allocateInfo, nullptr, &vkMemory));
	}

	// Bind the two 
	const VkDeviceSize offset = 0;
	VK_CHECK(vkBindBufferMemory(device, vkBuffer, vkMemory, offset));

	// Map the buffer, persistenly
	void* mappedPtr;
	vkMapMemory(device, vkMemory, offset, size, 0 /*api dummy*/, &mappedPtr);

	// Create SRV
	VkBufferView SRV = VK_NULL_HANDLE;
	if (srvFormat != VkFormat::VK_FORMAT_UNDEFINED)
	{
		VkBufferViewCreateInfo viewInfo {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
		viewInfo.buffer = vkBuffer;
		viewInfo.format = srvFormat;
		viewInfo.offset = 0;
		viewInfo.range = VK_WHOLE_SIZE;
		VK_CHECK(vkCreateBufferView(device, &viewInfo, nullptr, &SRV));
	}

	Buffer result;
	result.vkBuffer = vkBuffer;
	result.vkMemory = vkMemory;
	result.size = size;
	result.data = mappedPtr;
	result.SRV = SRV;
	return result;
}

void DestroyBuffer(VkDevice device, Buffer &buffer)
{
	if (buffer.SRV != VK_NULL_HANDLE)
	{
		vkDestroyBufferView(device, buffer.SRV, nullptr);
	}

	vkFreeMemory(device, buffer.vkMemory, nullptr);
	vkDestroyBuffer(device, buffer.vkBuffer, nullptr);
}

// Mesh 
#include "Extern/tinyobjloader/tiny_obj_loader.h"
namespace Mesh
{

struct Vertex
{
	float posx;
	float posy;
	float posz;
	float padding0;
	float normalx;
	float normaly;
	float normalz;
	float padding1;
};

struct MeshData
{
	uint32_t* indices;
	size_t indexCount;
	Vertex* vertices;
	size_t vertexCount;
};

MeshData LoadMesh(const char *modelPath)
{
	std::ifstream is(modelPath, std::ios::in | std::ios::ate);
	assert(is.is_open());
	if (!is.is_open())
	{
		printf("LoadMesh [error]: file not exist (%s)", modelPath);
	}

	is.seekg(0, std::ios::beg);

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::string warn;
	std::string err;
	const bool bSucceed = tinyobj::LoadObj(&attrib, &shapes, nullptr /* material */, &warn, &err, &is);
	if (warn.length() > 0)
	{
		printf("LoadMesh [warning]: %s\n", warn.data());
	}
	if (err.length() > 0)
	{
		printf("LoadMesh [error]: %s\n", err.data());
	}

	is.close();

	assert(bSucceed);
	assert(shapes.size() >= 1);

	tinyobj::mesh_t &mesh = shapes[0].mesh;
	std::vector<float> &normalsRaw = attrib.normals;
	std::vector<float> &verticesRaw = attrib.vertices;

	// Index
	// NOTE: just assume identical index for vertex (position) and normal
	size_t indexCount = mesh.indices.size();
	uint32_t* indices = new uint32_t[indexCount];
	for (size_t i = 0; i < indexCount; ++i)
	{
		indices[i] = mesh.indices[i].vertex_index;
	}

	// Vertex
	size_t vertexCount = verticesRaw.size() / 3;
	Vertex* vertices = new Vertex[vertexCount];
	for (size_t i = 0; i < vertexCount; ++i)
	{
		Vertex& v = vertices[i];
		size_t offset = i * 3;
		v.posx = verticesRaw[offset + 0];
		v.posy = verticesRaw[offset + 1];
		v.posz = verticesRaw[offset + 2];
		if (normalsRaw.size() > 0)
		{
			v.normalx = normalsRaw[offset + 0];
			v.normaly = normalsRaw[offset + 1];
			v.normalz = normalsRaw[offset + 2];
		}
	}

	MeshData result;
	result.indexCount = indexCount;
	result.indices = indices;
	result.vertexCount = vertexCount;
	result.vertices = vertices;
	return result;
}

void DestroyMesh(MeshData &mesh) 
{
	delete[] mesh.indices;
	delete[] mesh.vertices;
}
} // Mesh


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

	vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice, &GMemoryProperties);

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
			{VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32},
			{VkDescriptorType::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32},
		};

		VkDescriptorPoolCreateInfo createInfo {};
		createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		createInfo.maxSets = 128; // just arbitualy large
		createInfo.poolSizeCount = ARRAY_LENGTH(poolSizes);
		createInfo.pPoolSizes = poolSizes;
		VK_CHECK(vkCreateDescriptorPool(vkDevice, &createInfo, 0, &GDescriptorPool));
	}

	// Create shader
	struct Shader
	{
		VkPipeline pipeline;
		VkDescriptorSetLayout setLayout;
		VkPipelineLayout pipelineLayout;
		VkDescriptorSet descriptorSets[1];
	};
	struct ComputeShader : public Shader
	{
		shader::ShaderReflect relfect;
	};
	auto createComputeShader = [&](const char* filename)
	{
		ComputeShader shader;
		std::string shaderFileName = GetShaderBinFilePath(filename);
		VkShaderModule shaderModule = LoadShader(shaderFileName.data(), &shader.relfect);
		assert(shaderModule);

		// layout and descriptor stuffs
		CreateDescriptorSet(&shader.relfect, 1, shader.setLayout, shader.pipelineLayout, shader.descriptorSets[0]);

		shader.pipeline = CreateComputePipeline(shaderModule, shader.pipelineLayout, shader.relfect.spvModule.entry_point_name);

		// TODO it is safe to detroy here?
		vkDestroyShaderModule(vkDevice, shaderModule, 0);
		return shader;
	};
	ComputeShader fullscreenCS = createComputeShader("FullScreenCS.s");
	struct GraphicShader : public Shader
	{
		shader::ShaderReflect vertReflect;
		shader::ShaderReflect fragReflect;
	};
	auto createGraphicShader = [&](const char* filename)
	{
		GraphicShader shader;

		std::string vertFilename = GetShaderBinFilePath((std::string(filename) + ".vert").data());
		VkShaderModule vertModule = LoadShader(vertFilename.data(), &shader.vertReflect);
		assert(vertModule);

		std::string fragFilename = GetShaderBinFilePath((std::string(filename) + ".frag").data());
		VkShaderModule fragModule = LoadShader(fragFilename.data(), &shader.fragReflect);
		assert(fragModule);

		shader::ShaderReflect reflects[] {shader.vertReflect, shader.fragReflect};
		CreateDescriptorSet(
			reflects, ARRAY_LENGTH(reflects), shader.setLayout, shader.pipelineLayout, shader.descriptorSets[0]);

		shader.pipeline = CreateGraphicPipeline(
			vertModule,
			shader.vertReflect.spvModule.entry_point_name,
			fragModule,
			shader.fragReflect.spvModule.entry_point_name,
			shader.pipelineLayout
		);

		vkDestroyShaderModule(vkDevice, vertModule, nullptr);
		vkDestroyShaderModule(vkDevice, fragModule, nullptr);

		return shader;
	};
	GraphicShader meshGraphic = createGraphicShader("Mesh.s");

	// Create Mesh
	Mesh::MeshData mesh;
	{
		std::string modelFileName = GetModelFilePath("test\\teapot.obj");
		mesh = Mesh::LoadMesh(modelFileName.data());
	}
	Buffer indexBuffer = CreateBuffer(GVkDevice, mesh.indexCount * sizeof(mesh.indices[0]), VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT, VkFormat::VK_FORMAT_R32_UINT);
	Buffer vertexBuffer = CreateBuffer(GVkDevice, mesh.vertexCount * sizeof(mesh.vertices[0]), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VkFormat::VK_FORMAT_UNDEFINED);

	memcpy(indexBuffer.data, mesh.indices, indexBuffer.size);
	memcpy(vertexBuffer.data, mesh.vertices, vertexBuffer.size);

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
		VkSurfaceCapabilitiesKHR surfaceCap;
		VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice, vkSurface, &surfaceCap));
		const bool bResize = (surfaceCap.currentExtent.width != swapchain.width) || (surfaceCap.currentExtent.height!= swapchain.heigth);
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


		// NOTE: violet #5a4498
		VkClearColorValue clearColorValue{};
		clearColorValue.float32[0] = 0x5A / 255.0f;
		clearColorValue.float32[1] = 0x44 / 255.0f;
		clearColorValue.float32[2] = 0x94 / 255.0f;

		bool bFillWithCompute = true;
		bool bClearUsingRenderPass = !bFillWithCompute;
		const bool bDrawMesh = true;

		VkImageLayout swapchainImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		// Transition for compute
		if (bFillWithCompute)
		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = swapchainImageLayout;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = swapchain.images[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);

			swapchainImageLayout = VK_IMAGE_LAYOUT_GENERAL;
		}

		// Fill the buffer with compute
		if (bFillWithCompute)
		{
			vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, fullscreenCS.pipeline);

			VkDescriptorImageInfo dstImageInfo {};
			dstImageInfo.sampler = VK_NULL_HANDLE;
			dstImageInfo.imageView = swapchain.imageViews[imageIndex];
			dstImageInfo.imageLayout = swapchainImageLayout;

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

		// Transition for render
		if (bDrawMesh || bClearUsingRenderPass)
		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = swapchainImageLayout;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = swapchain.images[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);

			swapchainImageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR;
		}

		// Using dynamic render pass
		if (bDrawMesh || bClearUsingRenderPass)
		{
			VkRenderingAttachmentInfoKHR attachmentInfo {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
			attachmentInfo.imageView = swapchain.imageViews[imageIndex];
			attachmentInfo.imageLayout = swapchainImageLayout;
			attachmentInfo.resolveMode = VK_RESOLVE_MODE_NONE;
			attachmentInfo.loadOp = bClearUsingRenderPass ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
			attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachmentInfo.clearValue.color = clearColorValue;
			attachmentInfo.clearValue.depthStencil;

			VkRenderingInfoKHR renderingInfo {VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
			renderingInfo.flags;
			renderingInfo.renderArea.offset = {0, 0};
			renderingInfo.renderArea.extent = {swapchain.width, swapchain.heigth};
			renderingInfo.layerCount = 1;
			renderingInfo.viewMask = 0; // all view
			renderingInfo.colorAttachmentCount = 1;
			renderingInfo.pColorAttachments = &attachmentInfo;
			vkCmdBeginRenderingKHR(vkCmdBuffer, &renderingInfo);

			// Draw mesh
			if (bDrawMesh)
			{
				// set shader (pipeline)
				vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshGraphic.pipeline);

				// set pipeline dynamic states
				{
					VkViewport viewport {};
					viewport.x = 0;
					viewport.y = 0;
					viewport.width = swapchain.width;
					viewport.height = swapchain.heigth;
					viewport.minDepth = 0;
					viewport.maxDepth = 1;
					vkCmdSetViewport(vkCmdBuffer, 0, 1, &viewport);

					VkRect2D scissor {};
					scissor.offset = {0, 0};
					scissor.extent = {swapchain.width, swapchain.heigth};
					vkCmdSetScissor(vkCmdBuffer, 0, 1, &scissor);
				}

				// set buffers
				{
					VkDescriptorType descriptorType;
					uint32_t setIndex, binding;
					if (meshGraphic.vertReflect.FindBinding("indexBuffer", descriptorType, setIndex, binding))
					{
						assert(setIndex == 0);
						VkWriteDescriptorSet writeDescriptorSet {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
						writeDescriptorSet.dstSet = meshGraphic.descriptorSets[setIndex];
						writeDescriptorSet.dstBinding = binding;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.descriptorType = descriptorType;
						writeDescriptorSet.pTexelBufferView = &indexBuffer.SRV;
						vkUpdateDescriptorSets(vkDevice, 1, &writeDescriptorSet, 0, nullptr);
					}
				}
				{
					VkDescriptorType descriptorType;
					uint32_t setIndex, binding;
					if (meshGraphic.vertReflect.FindBinding("vertexBuffer", descriptorType, setIndex, binding))
					{
						VkDescriptorBufferInfo bufferInfo {};
						bufferInfo.buffer = vertexBuffer.vkBuffer;
						bufferInfo.offset = 0;
						bufferInfo.range = VK_WHOLE_SIZE;

						assert(setIndex == 0);
						VkWriteDescriptorSet writeDescriptorSet {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
						writeDescriptorSet.dstSet = meshGraphic.descriptorSets[setIndex];
						writeDescriptorSet.dstBinding = binding;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.descriptorType = descriptorType;
						writeDescriptorSet.pBufferInfo = &bufferInfo;
						vkUpdateDescriptorSets(vkDevice, 1, &writeDescriptorSet, 0, nullptr);
					}	
				}

				vkCmdBindDescriptorSets(
					vkCmdBuffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					meshGraphic.pipelineLayout,
					0,
					ARRAY_LENGTH(meshGraphic.descriptorSets),
					meshGraphic.descriptorSets,
					0,
					0);

				vkCmdDraw(vkCmdBuffer, mesh.indexCount, 1, 0, 0);
			}

			vkCmdEndRenderingKHR(vkCmdBuffer);
		}

		// Transition for present
		{
			VkImageMemoryBarrier imageBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			imageBarrier.oldLayout = swapchainImageLayout;
			imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
			imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageBarrier.subresourceRange.layerCount = 1;
			imageBarrier.subresourceRange.levelCount = 1;
			imageBarrier.image = swapchain.images[imageIndex];
			vkCmdPipelineBarrier(vkCmdBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &imageBarrier);

			swapchainImageLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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

	Mesh::DestroyMesh(mesh);
	DestroyBuffer(vkDevice, indexBuffer);
	DestroyBuffer(vkDevice, vertexBuffer);

	auto destroyShader = [&](Shader& shader)
	{
		vkDestroyPipeline(vkDevice, shader.pipeline, 0);
		vkDestroyDescriptorSetLayout(vkDevice, shader.setLayout, 0);
		vkDestroyPipelineLayout(vkDevice, shader.pipelineLayout, 0);
		vkFreeDescriptorSets(
			vkDevice, GDescriptorPool, ARRAY_LENGTH(shader.descriptorSets), shader.descriptorSets);
	};
	// destroy compute
	destroyShader(fullscreenCS);
	shader::DestroyShaderReflection(fullscreenCS.relfect);
	// destroy graphcs
	destroyShader(meshGraphic);
	shader::DestroyShaderReflection(meshGraphic.vertReflect);
	shader::DestroyShaderReflection(meshGraphic.fragReflect);

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