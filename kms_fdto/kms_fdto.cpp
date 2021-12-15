#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>
#include <limits>
#include <array>
#include <cmath>
#include <vulkan/vulkan.h>
#include <alsa/asoundlib.h>
#include <pthread.h>

#define DEFAULT_FENCE_TIMEOUT 100000000000

#define VK_CHECK_RESULT(f)			\
{									\
	VkResult res = (f);				\
	assert(res == VK_SUCCESS);		\
}

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR		m_capabilities;
	std::vector<VkSurfaceFormatKHR>	m_formats;
	std::vector<VkPresentModeKHR>	m_presentModes;
};

struct Vertex 
{
	float m_position[3];
	float m_color[3];
};

struct VertexBuffer
{
	VkDeviceMemory	m_memory;
	VkBuffer		m_buffer;
};

struct IndexBuffer
{
	VkDeviceMemory	m_memory;
	VkBuffer		m_buffer;
	uint32_t		m_count;
};

char const* g_instanceExtensions[] =
{
	"VK_KHR_surface",
	"VK_KHR_display"
};

char const* g_deviceExtensions[] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

uint32_t							g_renderWidth = 1080;
uint32_t							g_renderHeight = 1920;
uint32_t							g_currentFrameIndex = 0;

VkInstance							g_instance;
VkSurfaceKHR						g_surface = VK_NULL_HANDLE;
VkPhysicalDevice					g_physicalDevice = VK_NULL_HANDLE;
VkPhysicalDeviceProperties			g_physicalDeviceProperties;
VkPhysicalDeviceMemoryProperties	g_physicalDeviceMemProperties;
int									g_queueFamilyIndex = -1;
VkDevice							g_device = VK_NULL_HANDLE;
VkQueue								g_queue;
VkSwapchainKHR						g_swapChain;
std::vector<VkImage>				g_swapChainImages;					
VkFormat							g_swapChainImageFormat;
VkExtent2D							g_swapChainExtent;
std::vector<VkImageView>			g_swapChainImageViews;
VkSemaphore							g_presentCompleteSemaphore;			
VkSemaphore							g_renderCompleteSemaphore;
std::vector<VkFence>				g_commandBufferCompleteFences;		
VkCommandPool						g_commandPool;
VkPipelineCache						g_pipelineCache;
VkFormat							g_depthFormat;
VkImage								g_depthImage;
VkDeviceMemory						g_depthMemory;
VkImageView							g_depthImageView;
VkRenderPass						g_renderPass;
std::vector<VkFramebuffer>			g_frameBuffers;
VertexBuffer						g_vertices;
IndexBuffer							g_indices;
VkPipelineLayout					g_pipelineLayout;
VkPipeline							g_pipeline;
std::vector<VkCommandBuffer>		g_drawCmdBuffers;

uint32_t g_vertexShader[] =
{
	#include "../kms_ftdo.vert.spv"
};

uint32_t g_fragmentShader[] =
{
	#include "../kms_ftdo.frag.spv"
};

char const*				g_audioDevice = "default";
snd_pcm_t*				g_audioInstance = nullptr;
snd_pcm_hw_params_t*	g_hwparams = nullptr;
snd_pcm_sw_params_t*	g_swparams = nullptr;
snd_output_t*			g_output = nullptr;
snd_pcm_format_t		g_format = SND_PCM_FORMAT_S16;
snd_pcm_access_t		g_access = SND_PCM_ACCESS_RW_INTERLEAVED;
uint32_t				g_rate = 48000;
uint32_t				g_channels = 2;
int						g_resample = 1;
int						g_period_event = 0;
uint32_t				g_period_samples = 480;
uint32_t				g_period_time = (g_period_samples * 1'000'000) / g_rate;
uint32_t				g_buffer_time = 3 * g_period_time;
snd_pcm_sframes_t		g_period_size;
snd_pcm_sframes_t		g_buffer_size;
pthread_t				g_audioThread;
uint32_t				g_numPollDescriptors;
struct pollfd*			g_pollDescriptors;
uint8_t*				g_audioBuffer;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
	VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* /*pUserData*/)
{
	printf("DEBUG[%d]: %s\n", g_currentFrameIndex, pCallbackData->pMessage);
	//assert((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) == 0);
	return VK_FALSE;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
}

void createInstance()
{
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "kms_ftdo";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "None";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledExtensionCount = sizeof(g_instanceExtensions) / sizeof(g_instanceExtensions[0]);
	createInfo.ppEnabledExtensionNames = g_instanceExtensions;
	createInfo.enabledLayerCount = 0;
	createInfo.ppEnabledLayerNames = nullptr;

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
	populateDebugMessengerCreateInfo(debugCreateInfo);
	createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

	VkResult result = vkCreateInstance(&createInfo, nullptr, &g_instance);
	assert(result == VK_SUCCESS);
}

void createDirect2DisplaySurface(VkPhysicalDevice physicalDevice, uint32_t width, uint32_t height)
{
	uint32_t displayPropertyCount = 0;
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, NULL);
	VkDisplayPropertiesKHR* pDisplayProperties = new VkDisplayPropertiesKHR[displayPropertyCount];
	vkGetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, &displayPropertyCount, pDisplayProperties);

	uint32_t planePropertyCount = 0;
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, NULL);
	VkDisplayPlanePropertiesKHR* pPlaneProperties = new VkDisplayPlanePropertiesKHR[planePropertyCount];
	vkGetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, &planePropertyCount, pPlaneProperties);

	VkDisplayKHR display = VK_NULL_HANDLE;
	VkDisplayModeKHR displayMode;
	VkDisplayModePropertiesKHR* pModeProperties;
	VkSurfaceTransformFlagBitsKHR surfaceTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	bool foundMode = false;

	for (uint32_t i = 0; i < displayPropertyCount; ++i)
	{
		display = pDisplayProperties[i].display;

		uint32_t modeCount;
		vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, NULL);

		pModeProperties = new VkDisplayModePropertiesKHR[modeCount];
		vkGetDisplayModePropertiesKHR(physicalDevice, display, &modeCount, pModeProperties);

		for (uint32_t j = 0; j < modeCount; ++j)
		{
			const VkDisplayModePropertiesKHR* mode = &pModeProperties[j];

			if (mode->parameters.visibleRegion.width == width &&
				mode->parameters.visibleRegion.height == height)
			{
				displayMode = mode->displayMode;
				foundMode = true;
				break;
			}

			if (mode->parameters.visibleRegion.width == height &&
				mode->parameters.visibleRegion.height == width)
			{
				displayMode = mode->displayMode;
				foundMode = true;
				break;
			}
		}
		if (foundMode)
		{
			break;
		}
		delete[] pModeProperties;
	}

	assert(foundMode);

	uint32_t bestPlaneIndex = UINT32_MAX;
	VkDisplayKHR* pDisplays = NULL;
	for (uint32_t i = 0; i < planePropertyCount; i++)
	{
		uint32_t planeIndex = i;
		uint32_t displayCount;
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, NULL);
		if (pDisplays)
		{
			delete[] pDisplays;
		}
		pDisplays = new VkDisplayKHR[displayCount];
		vkGetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex, &displayCount, pDisplays);

		bestPlaneIndex = UINT32_MAX;
		for (uint32_t j = 0; j < displayCount; j++)
		{
			if (display == pDisplays[j])
			{
				bestPlaneIndex = i;
				break;
			}
		}
		if (bestPlaneIndex != UINT32_MAX)
		{
			break;
		}
	}

	assert(bestPlaneIndex != UINT32_MAX);

	VkDisplayPlaneCapabilitiesKHR planeCap;
	vkGetDisplayPlaneCapabilitiesKHR(physicalDevice, displayMode, bestPlaneIndex, &planeCap);
	VkDisplayPlaneAlphaFlagBitsKHR alphaMode = (VkDisplayPlaneAlphaFlagBitsKHR)0;

	if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_PREMULTIPLIED_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_PER_PIXEL_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_GLOBAL_BIT_KHR;
	}
	else if (planeCap.supportedAlpha & VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR)
	{
		alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
	}

	VkDisplaySurfaceCreateInfoKHR surfaceInfo{};
	surfaceInfo.sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR;
	surfaceInfo.pNext = NULL;
	surfaceInfo.flags = 0;
	surfaceInfo.displayMode = displayMode;
	surfaceInfo.planeIndex = bestPlaneIndex;
	surfaceInfo.planeStackIndex = pPlaneProperties[bestPlaneIndex].currentStackIndex;
	surfaceInfo.transform = surfaceTransform;
	surfaceInfo.globalAlpha = 1.0f;
	surfaceInfo.alphaMode = alphaMode;
	surfaceInfo.imageExtent.width = width;
	surfaceInfo.imageExtent.height = height;

	VkResult result = vkCreateDisplayPlaneSurfaceKHR(g_instance, &surfaceInfo, NULL, &g_surface);
	assert(result == VK_SUCCESS);

	delete[] pDisplays;
	delete[] pModeProperties;
	delete[] pDisplayProperties;
	delete[] pPlaneProperties;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
	int const numRequired = sizeof(g_deviceExtensions) / sizeof(g_deviceExtensions[0]);
	int numFound = 0;
	for (auto const& extension : availableExtensions)
	{
		for (int i = 0; i < numRequired; i++)
		{
			if (!strcmp(extension.extensionName, g_deviceExtensions[i]))
			{
				numFound++;
			}
		}
	}
	return numFound == numRequired;
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface)
{
	assert(surface != VK_NULL_HANDLE);

	SwapChainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.m_capabilities);

	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

	if (formatCount != 0) {
		details.m_formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.m_formats.data());
	}

	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

	if (presentModeCount != 0) {
		details.m_presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.m_presentModes.data());
	}

	return details;
}

int isPhysicalDeviceSuitable(VkPhysicalDevice device)
{
	bool extensionsSupported = checkDeviceExtensionSupport(device);

	bool swapChainAdequate = true;
	if (extensionsSupported)
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, g_surface);
		swapChainAdequate = !swapChainSupport.m_formats.empty() && !swapChainSupport.m_presentModes.empty();
	}

	VkPhysicalDeviceFeatures supportedFeatures;
	vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

	int queueFamilyIndex = 0;
	for (auto const& queueFamily : queueFamilies)
	{
		VkBool32 presentSupport = true;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, queueFamilyIndex, g_surface, &presentSupport);

		if (queueFamily.queueCount > 0 &&
			queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT &&
			presentSupport &&
			extensionsSupported &&
			swapChainAdequate &&
			supportedFeatures.samplerAnisotropy)
		{
			return queueFamilyIndex;
		}

		queueFamilyIndex++;
	}

	return -1;
}

void findPhysicalDevice()
{
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(g_instance, &deviceCount, nullptr);

	assert(deviceCount > 0);

	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(g_instance, &deviceCount, devices.data());

	for (auto const& device : devices)
	{
		createDirect2DisplaySurface(device, g_renderWidth, g_renderHeight);

		int queueFamilyIndex = isPhysicalDeviceSuitable(device);
		if (queueFamilyIndex != -1)
		{
			g_physicalDevice = device;
			g_queueFamilyIndex = queueFamilyIndex;
			break;
		}

		vkDestroySurfaceKHR(g_instance, g_surface, nullptr);
	}

	assert(g_physicalDevice != VK_NULL_HANDLE);

	vkGetPhysicalDeviceProperties(g_physicalDevice, &g_physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(g_physicalDevice, &g_physicalDeviceMemProperties);
}

void createLogicalDevice()
{
	assert(g_queueFamilyIndex != -1);

	float queuePriority = 1.0f;

	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = g_queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkPhysicalDeviceFeatures deviceFeatures = {};

	VkDeviceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = &queueCreateInfo;
	createInfo.queueCreateInfoCount = 1;
	createInfo.pEnabledFeatures = &deviceFeatures;
	createInfo.enabledExtensionCount = sizeof(g_deviceExtensions) / sizeof(g_deviceExtensions[0]);
	createInfo.ppEnabledExtensionNames = g_deviceExtensions;
	createInfo.enabledLayerCount = 0;

	VkResult result = vkCreateDevice(g_physicalDevice, &createInfo, nullptr, &g_device);
	assert(result == VK_SUCCESS);

	vkGetDeviceQueue(g_device, g_queueFamilyIndex, 0, &g_queue);
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats)
{
	if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
	{
		return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR };
	}

	for (auto const& availableFormat : availableFormats)
	{
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
		{
			return availableFormat;
		}
	}
	assert(false);
	return availableFormats[0];
}

VkPresentModeKHR chooseSwapPresentMode(std::vector<VkPresentModeKHR> const& availablePresentModes)
{
	VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;
	for (auto const& availablePresentMode : availablePresentModes)
	{
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return availablePresentMode;
		}
		else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
		{
			bestMode = availablePresentMode;
		}
	}
	return bestMode;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		VkExtent2D actualExtent = { g_renderWidth, g_renderHeight };
		actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
		return actualExtent;
	}
}

VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels)
{
	VkImageViewCreateInfo viewInfo = {};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	VkResult result = vkCreateImageView(g_device, &viewInfo, nullptr, &imageView);
	assert(result == VK_SUCCESS);
	return imageView;
}

VkSampler createSampler(bool linearFilter, uint32_t numMipLevels, VkSamplerAddressMode addressMode)
{
	VkSamplerCreateInfo samplerInfo = {};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = linearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	samplerInfo.minFilter = linearFilter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	samplerInfo.addressModeU = addressMode;
	samplerInfo.addressModeV = addressMode;
	samplerInfo.addressModeW = addressMode;
	samplerInfo.anisotropyEnable = VK_FALSE;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = linearFilter ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.maxAnisotropy = 1.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = static_cast<float>(numMipLevels);

	VkSampler sampler;
	VkResult result = vkCreateSampler(g_device, &samplerInfo, nullptr, &sampler);
	assert(result == VK_SUCCESS);
	return sampler;
}

void createSwapChain()
{
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(g_physicalDevice, g_surface);
	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.m_formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.m_presentModes);
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.m_capabilities);

	uint32_t imageCount = swapChainSupport.m_capabilities.minImageCount + 1;
	if (swapChainSupport.m_capabilities.maxImageCount > 0 &&
		imageCount > swapChainSupport.m_capabilities.maxImageCount)
	{
		imageCount = swapChainSupport.m_capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = g_surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	createInfo.queueFamilyIndexCount = 0;
	createInfo.pQueueFamilyIndices = nullptr;
	createInfo.preTransform = swapChainSupport.m_capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	VkResult result = vkCreateSwapchainKHR(g_device, &createInfo, nullptr, &g_swapChain);
	assert(result == VK_SUCCESS);

	uint32_t actualImageCount = 0;
	vkGetSwapchainImagesKHR(g_device, g_swapChain, &actualImageCount, nullptr);
	g_swapChainImages.resize(actualImageCount);
	vkGetSwapchainImagesKHR(g_device, g_swapChain, &actualImageCount, g_swapChainImages.data());

	g_swapChainImageFormat = surfaceFormat.format;
	g_swapChainExtent = extent;

	g_swapChainImageViews.resize(g_swapChainImages.size());
	for (size_t i = 0; i < g_swapChainImages.size(); i++)
	{
		g_swapChainImageViews[i] = createImageView(g_swapChainImages[i], g_swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}
}

void createSyncObjects()
{
	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	VkResult result = vkCreateSemaphore(g_device, &semaphoreInfo, nullptr, &g_presentCompleteSemaphore);
	assert(result == VK_SUCCESS);
	result = vkCreateSemaphore(g_device, &semaphoreInfo, nullptr, &g_renderCompleteSemaphore);
	assert(result == VK_SUCCESS);

	size_t const maxFramesInFlight = g_swapChainImages.size();
	g_commandBufferCompleteFences.resize(maxFramesInFlight);

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < maxFramesInFlight; i++)
	{
		result = vkCreateFence(g_device, &fenceInfo, nullptr, &g_commandBufferCompleteFences[i]);
		assert(result == VK_SUCCESS);
	}
}

void createCommandPool()
{
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = g_queueFamilyIndex;
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VkResult result = vkCreateCommandPool(g_device, &poolInfo, nullptr, &g_commandPool);
	assert(result == VK_SUCCESS);
}

void createPipelineCache()
{
	VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
	pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	VkResult result = vkCreatePipelineCache(g_device, &pipelineCacheCreateInfo, nullptr, &g_pipelineCache);
	assert(result == VK_SUCCESS);
}

VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, VkFormat* depthFormat)
{
	std::vector<VkFormat> depthFormats = {
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM_S8_UINT,
		VK_FORMAT_D16_UNORM
	};

	for (auto& format : depthFormats)
	{
		VkFormatProperties formatProps;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);
		if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
		{
			*depthFormat = format;
			return true;
		}
	}

	return false;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	for (uint32_t i = 0; i < g_physicalDeviceMemProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) &&
			(g_physicalDeviceMemProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}
	assert(false);
	return (uint32_t)-1;
}

void createDepthStencil()
{
	VkBool32 validDepthFormat = getSupportedDepthFormat(g_physicalDevice, &g_depthFormat);
	assert(validDepthFormat);

	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = g_depthFormat;
	imageCI.extent = { g_renderWidth, g_renderHeight, 1 };
	imageCI.mipLevels = 1;
	imageCI.arrayLayers = 1;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

	VkResult result = vkCreateImage(g_device, &imageCI, nullptr, &g_depthImage);
	assert(result == VK_SUCCESS);

	VkMemoryRequirements memReqs{};
	vkGetImageMemoryRequirements(g_device, g_depthImage, &memReqs);

	VkMemoryAllocateInfo memAllloc{};
	memAllloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllloc.allocationSize = memReqs.size;
	memAllloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result = vkAllocateMemory(g_device, &memAllloc, nullptr, &g_depthMemory);
	assert(result == VK_SUCCESS);
	result = vkBindImageMemory(g_device, g_depthImage, g_depthMemory, 0);
	assert(result == VK_SUCCESS);

	VkImageViewCreateInfo imageViewCI{};
	imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCI.image = g_depthImage;
	imageViewCI.format = g_depthFormat;
	imageViewCI.subresourceRange.baseMipLevel = 0;
	imageViewCI.subresourceRange.levelCount = 1;
	imageViewCI.subresourceRange.baseArrayLayer = 0;
	imageViewCI.subresourceRange.layerCount = 1;
	imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	if (g_depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT) {
		imageViewCI.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
	}
	result = vkCreateImageView(g_device, &imageViewCI, nullptr, &g_depthImageView);
	assert(result == VK_SUCCESS);
}

void createRenderPass()
{
	std::array<VkAttachmentDescription, 2> attachments = {};

	attachments[0].format = g_swapChainImageFormat;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	attachments[1].format = g_depthFormat;
	attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference colorReference = {};
	colorReference.attachment = 0;
	colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthReference = {};
	depthReference.attachment = 1;
	depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpassDescription = {};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = 1;
	subpassDescription.pColorAttachments = &colorReference;
	subpassDescription.pDepthStencilAttachment = &depthReference;
	subpassDescription.inputAttachmentCount = 0;
	subpassDescription.pInputAttachments = nullptr;
	subpassDescription.preserveAttachmentCount = 0;
	subpassDescription.pPreserveAttachments = nullptr;
	subpassDescription.pResolveAttachments = nullptr;

	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpassDescription;
	renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
	renderPassInfo.pDependencies = dependencies.data();

	VkResult result = vkCreateRenderPass(g_device, &renderPassInfo, nullptr, &g_renderPass);
	assert(result == VK_SUCCESS);
}

void createFrameBuffers()
{
	VkImageView attachments[2];

	attachments[1] = g_depthImageView;

	VkFramebufferCreateInfo frameBufferCreateInfo = {};
	frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	frameBufferCreateInfo.pNext = NULL;
	frameBufferCreateInfo.renderPass = g_renderPass;
	frameBufferCreateInfo.attachmentCount = 2;
	frameBufferCreateInfo.pAttachments = attachments;
	frameBufferCreateInfo.width = g_renderWidth;
	frameBufferCreateInfo.height = g_renderHeight;
	frameBufferCreateInfo.layers = 1;

	g_frameBuffers.resize(g_swapChainImages.size());
	for (uint32_t i = 0; i < g_frameBuffers.size(); i++)
	{
		attachments[0] = g_swapChainImageViews[i];
		VkResult result = vkCreateFramebuffer(g_device, &frameBufferCreateInfo, nullptr, &g_frameBuffers[i]);
		assert(result == VK_SUCCESS);
	}
}

VkCommandBuffer getCommandBuffer(bool begin)
{
	VkCommandBuffer cmdBuffer;

	VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
	cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocateInfo.commandPool = g_commandPool;
	cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocateInfo.commandBufferCount = 1;

	VkResult result = vkAllocateCommandBuffers(g_device, &cmdBufAllocateInfo, &cmdBuffer);
	assert(result == VK_SUCCESS);

	if (begin)
	{
		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		result = vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);
		assert(result == VK_SUCCESS);
	}

	return cmdBuffer;
}

void flushCommandBuffer(VkCommandBuffer commandBuffer)
{
	assert(commandBuffer != VK_NULL_HANDLE);

	VkResult result = vkEndCommandBuffer(commandBuffer);
	assert(result == VK_SUCCESS);

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	VkFenceCreateInfo fenceCreateInfo = {};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceCreateInfo.flags = 0;
	VkFence fence;
	result = vkCreateFence(g_device, &fenceCreateInfo, nullptr, &fence);
	assert(result == VK_SUCCESS);

	result = vkQueueSubmit(g_queue, 1, &submitInfo, fence);
	assert(result == VK_SUCCESS);

	result = vkWaitForFences(g_device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
	assert(result == VK_SUCCESS);

	vkDestroyFence(g_device, fence, nullptr);
	vkFreeCommandBuffers(g_device, g_commandPool, 1, &commandBuffer);
}

void createVertices()
{
	std::vector<Vertex> vertexBuffer =
	{
		{ {  1.0f,  1.0f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
		{ { -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
		{ {  0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
	};
	uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);

	std::vector<uint32_t> indexBuffer = { 0, 1, 2 };
	g_indices.m_count = static_cast<uint32_t>(indexBuffer.size());
	uint32_t indexBufferSize = g_indices.m_count * sizeof(uint32_t);

	VkMemoryAllocateInfo memAlloc = {};
	memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	VkMemoryRequirements memReqs;

	void* data;

	struct StagingBuffer 
	{
		VkDeviceMemory memory;
		VkBuffer buffer;
	};

	struct 
	{
		StagingBuffer vertices;
		StagingBuffer indices;
	} stagingBuffers;

	VkBufferCreateInfo vertexBufferInfo = {};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.size = vertexBufferSize;
	vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	VkResult result = vkCreateBuffer(g_device, &vertexBufferInfo, nullptr, &stagingBuffers.vertices.buffer);
	assert(result == VK_SUCCESS);

	vkGetBufferMemoryRequirements(g_device, stagingBuffers.vertices.buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	result = vkAllocateMemory(g_device, &memAlloc, nullptr, &stagingBuffers.vertices.memory);
	assert(result == VK_SUCCESS);

	result = vkMapMemory(g_device, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0, &data);
	assert(result == VK_SUCCESS);
	memcpy(data, vertexBuffer.data(), vertexBufferSize);
	vkUnmapMemory(g_device, stagingBuffers.vertices.memory);
	result = vkBindBufferMemory(g_device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0);
	assert(result == VK_SUCCESS);

	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	result = vkCreateBuffer(g_device, &vertexBufferInfo, nullptr, &g_vertices.m_buffer);
	assert(result == VK_SUCCESS);
	vkGetBufferMemoryRequirements(g_device, g_vertices.m_buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result = vkAllocateMemory(g_device, &memAlloc, nullptr, &g_vertices.m_memory);
	result = vkBindBufferMemory(g_device, g_vertices.m_buffer, g_vertices.m_memory, 0);

	VkBufferCreateInfo indexbufferInfo = {};
	indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexbufferInfo.size = indexBufferSize;
	indexbufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	result = vkCreateBuffer(g_device, &indexbufferInfo, nullptr, &stagingBuffers.indices.buffer);
	assert(result == VK_SUCCESS);
	vkGetBufferMemoryRequirements(g_device, stagingBuffers.indices.buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	result = vkAllocateMemory(g_device, &memAlloc, nullptr, &stagingBuffers.indices.memory);
	assert(result == VK_SUCCESS);
	result = vkMapMemory(g_device, stagingBuffers.indices.memory, 0, indexBufferSize, 0, &data);
	assert(result == VK_SUCCESS);
	memcpy(data, indexBuffer.data(), indexBufferSize);
	vkUnmapMemory(g_device, stagingBuffers.indices.memory);
	result = vkBindBufferMemory(g_device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0);
	assert(result == VK_SUCCESS);

	indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	result = vkCreateBuffer(g_device, &indexbufferInfo, nullptr, &g_indices.m_buffer);
	assert(result == VK_SUCCESS);
	vkGetBufferMemoryRequirements(g_device, g_indices.m_buffer, &memReqs);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result = vkAllocateMemory(g_device, &memAlloc, nullptr, &g_indices.m_memory);
	assert(result == VK_SUCCESS); 
	result = vkBindBufferMemory(g_device, g_indices.m_buffer, g_indices.m_memory, 0);
	assert(result == VK_SUCCESS);

	VkCommandBuffer copyCmd = getCommandBuffer(true);
	VkBufferCopy copyRegion = {};
	copyRegion.size = vertexBufferSize;
	vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, g_vertices.m_buffer, 1, &copyRegion);
	copyRegion.size = indexBufferSize;
	vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, g_indices.m_buffer, 1, &copyRegion);
	flushCommandBuffer(copyCmd);

	vkDestroyBuffer(g_device, stagingBuffers.vertices.buffer, nullptr);
	vkFreeMemory(g_device, stagingBuffers.vertices.memory, nullptr);
	vkDestroyBuffer(g_device, stagingBuffers.indices.buffer, nullptr);
	vkFreeMemory(g_device, stagingBuffers.indices.memory, nullptr);
}

VkShaderModule createShaderModule(uint32_t const* code, size_t size)
{
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = size;
	createInfo.pCode = code;

	VkShaderModule shaderModule;
	VkResult result = vkCreateShaderModule(g_device, &createInfo, nullptr, &shaderModule);
	assert(result == VK_SUCCESS);

	return shaderModule;
}

void createPipeline()
{
	VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
	pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pPipelineLayoutCreateInfo.pNext = nullptr;
	pPipelineLayoutCreateInfo.setLayoutCount = 0;
	pPipelineLayoutCreateInfo.pSetLayouts = nullptr;
	VK_CHECK_RESULT(vkCreatePipelineLayout(g_device, &pPipelineLayoutCreateInfo, nullptr, &g_pipelineLayout));

	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.layout = g_pipelineLayout;
	pipelineCreateInfo.renderPass = g_renderPass;

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
	inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineRasterizationStateCreateInfo rasterizationState = {};
	rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.cullMode = VK_CULL_MODE_NONE;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.depthBiasEnable = VK_FALSE;
	rasterizationState.lineWidth = 1.0f;

	VkPipelineColorBlendAttachmentState blendAttachmentState[1] = {};
	blendAttachmentState[0].colorWriteMask = 0xf;
	blendAttachmentState[0].blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo colorBlendState = {};
	colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendState.attachmentCount = 1;
	colorBlendState.pAttachments = blendAttachmentState;

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.scissorCount = 1;

	std::vector<VkDynamicState> dynamicStateEnables;
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
	dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
	VkPipelineDynamicStateCreateInfo dynamicState = {};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.pDynamicStates = dynamicStateEnables.data();
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

	VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
	depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.back.failOp = VK_STENCIL_OP_KEEP;
	depthStencilState.back.passOp = VK_STENCIL_OP_KEEP;
	depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;
	depthStencilState.stencilTestEnable = VK_FALSE;
	depthStencilState.front = depthStencilState.back;

	VkPipelineMultisampleStateCreateInfo multisampleState = {};
	multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampleState.pSampleMask = nullptr;

	VkVertexInputBindingDescription vertexInputBinding = {};
	vertexInputBinding.binding = 0;
	vertexInputBinding.stride = sizeof(Vertex);
	vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	std::array<VkVertexInputAttributeDescription, 2> vertexInputAttributs;
	vertexInputAttributs[0].binding = 0;
	vertexInputAttributs[0].location = 0;
	vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	vertexInputAttributs[0].offset = offsetof(Vertex, m_position);
	vertexInputAttributs[1].binding = 0;
	vertexInputAttributs[1].location = 1;
	vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	vertexInputAttributs[1].offset = offsetof(Vertex, m_color);

	VkPipelineVertexInputStateCreateInfo vertexInputState = {};
	vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputState.vertexBindingDescriptionCount = 1;
	vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
	vertexInputState.vertexAttributeDescriptionCount = 2;
	vertexInputState.pVertexAttributeDescriptions = vertexInputAttributs.data();

	std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

	shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderStages[0].module = createShaderModule(g_vertexShader, sizeof(g_vertexShader));
	shaderStages[0].pName = "main";
	assert(shaderStages[0].module != VK_NULL_HANDLE);

	shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderStages[1].module = createShaderModule(g_fragmentShader, sizeof(g_fragmentShader));
	shaderStages[1].pName = "main";
	assert(shaderStages[1].module != VK_NULL_HANDLE);

	pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineCreateInfo.pStages = shaderStages.data();
	pipelineCreateInfo.pVertexInputState = &vertexInputState;
	pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
	pipelineCreateInfo.pRasterizationState = &rasterizationState;
	pipelineCreateInfo.pColorBlendState = &colorBlendState;
	pipelineCreateInfo.pMultisampleState = &multisampleState;
	pipelineCreateInfo.pViewportState = &viewportState;
	pipelineCreateInfo.pDepthStencilState = &depthStencilState;
	pipelineCreateInfo.renderPass = g_renderPass;
	pipelineCreateInfo.pDynamicState = &dynamicState;

	VK_CHECK_RESULT(vkCreateGraphicsPipelines(g_device, g_pipelineCache, 1, &pipelineCreateInfo, nullptr, &g_pipeline));

	vkDestroyShaderModule(g_device, shaderStages[0].module, nullptr);
	vkDestroyShaderModule(g_device, shaderStages[1].module, nullptr);
}

void createCommandBuffers()
{
	g_drawCmdBuffers.resize(g_swapChainImages.size());

	VkCommandBufferAllocateInfo cmdBufAllocInfo{};
	cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmdBufAllocInfo.commandPool = g_commandPool;
	cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdBufAllocInfo.commandBufferCount = g_drawCmdBuffers.size();
	VK_CHECK_RESULT(vkAllocateCommandBuffers(g_device, &cmdBufAllocInfo, g_drawCmdBuffers.data()));

	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufInfo.pNext = nullptr;

	VkClearValue clearValues[2];
	clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 1.0f } };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.pNext = nullptr;
	renderPassBeginInfo.renderPass = g_renderPass;
	renderPassBeginInfo.renderArea.offset.x = 0;
	renderPassBeginInfo.renderArea.offset.y = 0;
	renderPassBeginInfo.renderArea.extent.width = g_renderWidth;
	renderPassBeginInfo.renderArea.extent.height = g_renderHeight;
	renderPassBeginInfo.clearValueCount = 2;
	renderPassBeginInfo.pClearValues = clearValues;

	for (size_t i = 0; i < g_drawCmdBuffers.size(); ++i)
	{
		renderPassBeginInfo.framebuffer = g_frameBuffers[i];

		VK_CHECK_RESULT(vkBeginCommandBuffer(g_drawCmdBuffers[i], &cmdBufInfo));

		vkCmdBeginRenderPass(g_drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkViewport viewport = {};
		viewport.height = (float)g_renderHeight;
		viewport.width = (float)g_renderWidth;
		viewport.minDepth = (float)0.0f;
		viewport.maxDepth = (float)1.0f;
		vkCmdSetViewport(g_drawCmdBuffers[i], 0, 1, &viewport);

		VkRect2D scissor = {};
		scissor.extent.width = g_renderWidth;
		scissor.extent.height = g_renderHeight;
		scissor.offset.x = 0;
		scissor.offset.y = 0;
		vkCmdSetScissor(g_drawCmdBuffers[i], 0, 1, &scissor);

		vkCmdBindPipeline(g_drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);

		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(g_drawCmdBuffers[i], 0, 1, &g_vertices.m_buffer, offsets);

		vkCmdBindIndexBuffer(g_drawCmdBuffers[i], g_indices.m_buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdDrawIndexed(g_drawCmdBuffers[i], g_indices.m_count, 1, 0, 0, 1);

		vkCmdEndRenderPass(g_drawCmdBuffers[i]);

		VK_CHECK_RESULT(vkEndCommandBuffer(g_drawCmdBuffers[i]));
	}
}

void initVulkan()
{
	createInstance();
	findPhysicalDevice();
	createLogicalDevice();
	createSwapChain();
	createSyncObjects();
	createCommandPool();
	createPipelineCache();
	createDepthStencil();
	createRenderPass();
	createFrameBuffers();
	createVertices();
	createPipeline();
	createCommandBuffers();
}

int waitForPoll()
{
	unsigned short revents;
	while (1)
	{
		poll(g_pollDescriptors, g_numPollDescriptors, -1);
		snd_pcm_poll_descriptors_revents(g_audioInstance, g_pollDescriptors, g_numPollDescriptors, &revents);
		if (revents & POLLERR) return -EIO;
		if (revents & POLLOUT) return 0;
	}
}

int underrunRecovery(int err)
{
	if (err == -EPIPE) {
		err = snd_pcm_prepare(g_audioInstance);
		if (err < 0) {
			printf("Can't recover from under run, prepare failed: %s\n", snd_strerror(err));
		}
		return 0;
	}
	else if (err == -ESTRPIPE) {
		while ((err = snd_pcm_resume(g_audioInstance)) == -EAGAIN) {
			sleep(1);
		}
		if (err < 0) {
			err = snd_pcm_prepare(g_audioInstance);
			if (err < 0) {
				printf("Can't recover from suspend, prepare failed: %s\n", snd_strerror(err));
			}
		}
		return 0;
	}
	return err;
}

void waitAndSubmit()
{
	int err = waitForPoll();
	if (err < 0)
	{
		if (snd_pcm_state(g_audioInstance) == SND_PCM_STATE_XRUN ||
			snd_pcm_state(g_audioInstance) == SND_PCM_STATE_SUSPENDED)
		{
			err = snd_pcm_state(g_audioInstance) == SND_PCM_STATE_XRUN ? -EPIPE : -ESTRPIPE;
			if (underrunRecovery(err) < 0) {
				printf("Audio recovery error: %s\n", snd_strerror(err));
				assert(false);
			}
		}
		else {
			printf("Wait for poll failed\n");
			assert(false);
		}
	}

#if 1	// this function call causes the flip done timeout
	err = snd_pcm_writei(g_audioInstance, g_audioBuffer, g_period_size);
	if (err < 0) {
		printf("Audio write error: %s\n", snd_strerror(err));
	}
#endif
}

void* audioThreadProc(void*)
{
	g_audioBuffer = (uint8_t*)malloc(g_buffer_size);
	assert(g_audioBuffer != nullptr);

	memset(g_audioBuffer, 0, g_buffer_size);

	g_numPollDescriptors = snd_pcm_poll_descriptors_count(g_audioInstance);
	assert(g_numPollDescriptors > 0);
	g_pollDescriptors = (struct pollfd*)malloc(g_numPollDescriptors * sizeof(struct pollfd));
	assert(g_pollDescriptors != nullptr);
	int retCode = snd_pcm_poll_descriptors(g_audioInstance, g_pollDescriptors, g_numPollDescriptors);
	assert(retCode >= 0);

	while(true)	{
		waitAndSubmit();
	}

	return nullptr;
}

int setHardwareParams(snd_pcm_t* handle, snd_pcm_hw_params_t* params)
{
	int err = snd_pcm_hw_params_any(handle, params);
	if (err < 0)
	{
		printf("Broken configuration for playback: no configurations available: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_hw_params_set_rate_resample(handle, params, g_resample);
	if (err < 0)
	{
		printf("Resampling setup failed for playback: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_hw_params_set_access(handle, params, g_access);
	if (err < 0)
	{
		printf("Access type not available for playback: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_hw_params_set_format(handle, params, g_format);
	if (err < 0)
	{
		printf("Sample format not available for playback: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_hw_params_set_channels(handle, params, g_channels);
	if (err < 0)
	{
		printf("Channels count (%u) not available for playbacks: %s\n", g_channels, snd_strerror(err));
		return err;
	}

	unsigned int rrate = g_rate;
	err = snd_pcm_hw_params_set_rate_near(handle, params, &rrate, 0);
	if (err < 0)
	{
		printf("Rate %uHz not available for playback: %s\n", g_rate, snd_strerror(err));
		return err;
	}
	if (rrate != g_rate)
	{
		printf("Rate doesn't match (requested %uHz, get %iHz)\n", g_rate, err);
		return -EINVAL;
	}

	int dir;
	err = snd_pcm_hw_params_set_buffer_time_near(handle, params, &g_buffer_time, &dir);
	if (err < 0)
	{
		printf("Unable to set buffer time %u for playback: %s\n", g_buffer_time, snd_strerror(err));
		return err;
	}
	snd_pcm_uframes_t size;
	err = snd_pcm_hw_params_get_buffer_size(params, &size);
	if (err < 0)
	{
		printf("Unable to get buffer size for playback: %s\n", snd_strerror(err));
		return err;
	}
	g_buffer_size = size;

	err = snd_pcm_hw_params_set_period_time_near(handle, params, &g_period_time, &dir);
	if (err < 0)
	{
		printf("Unable to set period time %u for playback: %s\n", g_period_time, snd_strerror(err));
		return err;
	}
	err = snd_pcm_hw_params_get_period_size(params, &size, &dir);
	if (err < 0)
	{
		printf("Unable to get period size for playback: %s\n", snd_strerror(err));
		return err;
	}
	g_period_size = size;

	err = snd_pcm_hw_params(handle, params);
	if (err < 0)
	{
		printf("Unable to set hw params for playback: %s\n", snd_strerror(err));
		return err;
	}

	return 0;
}

int setSoftwareParams(snd_pcm_t* handle, snd_pcm_sw_params_t* swparams)
{
	int err = snd_pcm_sw_params_current(handle, swparams);
	if (err < 0)
	{
		printf("Unable to determine current swparams for playback: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_sw_params_set_start_threshold(handle, swparams, (g_buffer_size / g_period_size) * g_period_size);
	if (err < 0)
	{
		printf("Unable to set start threshold mode for playback: %s\n", snd_strerror(err));
		return err;
	}

	err = snd_pcm_sw_params_set_avail_min(handle, swparams, g_period_event ? g_buffer_size : g_period_size);
	if (err < 0)
	{
		printf("Unable to set avail min for playback: %s\n", snd_strerror(err));
		return err;
	}

	if (g_period_event)
	{
		err = snd_pcm_sw_params_set_period_event(handle, swparams, 1);
		if (err < 0)
		{
			printf("Unable to set period event: %s\n", snd_strerror(err));
			return err;
		}
	}

	err = snd_pcm_sw_params(handle, swparams);
	if (err < 0)
	{
		printf("Unable to set sw params for playback: %s\n", snd_strerror(err));
		return err;
	}

	return 0;
}

void initAudio()
{
	int err = snd_output_stdio_attach(&g_output, stdout, 0);
	if (err < 0) 
	{
		printf("Output attach failed: %s\n", snd_strerror(err));
	}

	if ((err = snd_pcm_open(&g_audioInstance, g_audioDevice, SND_PCM_STREAM_PLAYBACK, 0)) < 0) 
	{
		printf("Cannot open audio device %s (%s)\n", g_audioDevice, snd_strerror(err));
		return;
	}

	if ((err = snd_pcm_hw_params_malloc(&g_hwparams)) < 0)
	{
		printf("Cannot allocate hardware parameter structure (%s)", snd_strerror(err));
		return;
	}

	if ((err = snd_pcm_sw_params_malloc(&g_swparams)) < 0)
	{
		printf("Cannot allocate software parameter structure (%s)", snd_strerror(err));
		return;
	}

	if ((err = setHardwareParams(g_audioInstance, g_hwparams)) < 0)
	{
		printf("Setting hardware parameters failed: %s\n", snd_strerror(err));
		return;
	}

	if ((err = setSoftwareParams(g_audioInstance, g_swparams)) < 0)
	{
		printf("Setting software parameters failed: %s\n", snd_strerror(err));
		return;
	}

	snd_pcm_dump(g_audioInstance, g_output);

	int retCode = pthread_create(&g_audioThread, nullptr, audioThreadProc, nullptr);
	assert(retCode == 0);

	retCode = pthread_setname_np(g_audioThread, "AudioThread");
	assert(retCode == 0);
}

void render()
{
	uint64_t const timeOut = std::numeric_limits<uint64_t>::max();
	VK_CHECK_RESULT(vkAcquireNextImageKHR(g_device, g_swapChain, timeOut, g_presentCompleteSemaphore, VK_NULL_HANDLE, &g_currentFrameIndex));

	VK_CHECK_RESULT(vkWaitForFences(g_device, 1, &g_commandBufferCompleteFences[g_currentFrameIndex], VK_TRUE, UINT64_MAX));
	VK_CHECK_RESULT(vkResetFences(g_device, 1, &g_commandBufferCompleteFences[g_currentFrameIndex]));

	VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.pWaitDstStageMask = &waitStageMask;
	submitInfo.pWaitSemaphores = &g_presentCompleteSemaphore;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &g_renderCompleteSemaphore;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pCommandBuffers = &g_drawCmdBuffers[g_currentFrameIndex];
	submitInfo.commandBufferCount = 1;

	VK_CHECK_RESULT(vkQueueSubmit(g_queue, 1, &submitInfo, g_commandBufferCompleteFences[g_currentFrameIndex]));

	VkSemaphore waitSemaphores[] =
	{
		g_renderCompleteSemaphore
	};

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = waitSemaphores;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &g_swapChain;
	presentInfo.pImageIndices = &g_currentFrameIndex;
	presentInfo.pResults = nullptr; // Optional

	VkResult present = vkQueuePresentKHR(g_queue, &presentInfo);
	if (!((present == VK_SUCCESS) || (present == VK_SUBOPTIMAL_KHR))) {
		VK_CHECK_RESULT(present);
	}
}

int main()
{
	initVulkan();
	initAudio();
	while(true) {
		render();
	}
	return 0;
}
