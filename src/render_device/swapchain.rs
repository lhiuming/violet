/*
 * Surface and Swapchain
 */

use ash::{extensions::khr, vk};

use super::{PhysicalDevice, Texture, TextureDesc, TextureView, TextureViewDesc};

pub struct Surface {
    pub handle: vk::SurfaceKHR,
}

pub fn create_surface(
    win32_surface: &khr::Win32Surface,
    app_handle: u64,
    window_handle: u64,
) -> Surface {
    // Create platform surface
    let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
        .hinstance(app_handle as vk::HINSTANCE)
        .hwnd(window_handle as vk::HWND);
    let vk_surface = unsafe {
        win32_surface
            .create_win32_surface(&create_info, None)
            .expect("Vulkan: failed to crate win32 surface")
    };

    Surface { handle: vk_surface }
}

pub struct Swapchain {
    pub extent: vk::Extent2D,
    pub handle: vk::SwapchainKHR,
    pub textures: Vec<Texture>,
    pub texture_views: Vec<TextureView>,
}

pub fn create_swapchain(
    khr_surface: &khr::Surface,
    khr_swapchain: &khr::Swapchain,
    device: &ash::Device,
    pd: &PhysicalDevice,
    surface: &Surface,
) -> Swapchain {
    let cap = unsafe {
        khr_surface
            .get_physical_device_surface_capabilities(pd.handle, surface.handle)
            .unwrap()
    };
    let supported_present_modes = unsafe {
        khr_surface
            .get_physical_device_surface_present_modes(pd.handle, surface.handle)
            .unwrap()
    };

    /*
    // Verbose info
    println!("Vulkan: surface capabilities: {:?}", cap);
    println!(
        "Vulkan: surface present modes: {:?}",
        supported_present_modes
    );
    */

    // UI Overlay (color_attachment) and Compute PostProcessing (storage)
    let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE;

    // TODO handle failure?
    assert!(cap.supported_usage_flags.contains(image_usage));

    // Query surface format
    let surface_formats = unsafe {
        khr_surface
            .get_physical_device_surface_formats(pd.handle, surface.handle)
            .unwrap()
    };
    assert!(surface_formats.len() > 0);

    /*
    // Verbose info
    println!("Vulkan surface supported formats:");
    for format in surface_formats.iter() {
        let props = pd.get_format_properties(format.format);
        println!(
            "\t({:?}, {:?}), features: {:?}",
            format.format, format.color_space, props.optimal_tiling_features
        );
    }
    */

    // TODO pick a proper image format
    let surface_format = surface_formats[0];

    let vsync = true;
    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        vk::PresentModeKHR::IMMEDIATE
    };
    assert!(supported_present_modes.contains(&present_mode));

    let min_image_count = cap.min_image_count.max(2); // 2 should be enough
    let surface_extent = cap.current_extent;

    // Create swapchain object
    let create_info = {
        vk::SwapchainCreateInfoKHR::builder()
            .flags(vk::SwapchainCreateFlagsKHR::empty())
            .surface(surface.handle)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(surface_extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
    };
    let swapchain_handle = unsafe {
        khr_swapchain
            .create_swapchain(&create_info, None)
            .expect("Vulkan: Swapchain creatino failed???")
    };

    // Get images/textures
    // NOTE: swapchain image has (fixed) equivalent image properties.
    // see: https://registry.khronos.org/vulkan/specs/1.3-khr-extensions/html/chap30.html#_wsi_swapchain
    let textures = {
        let mut images = unsafe {
            khr_swapchain
                .get_swapchain_images(swapchain_handle)
                .unwrap()
        };
        images
            .drain(0..)
            .map(|image| {
                let desc = TextureDesc::new_2d(
                    surface_extent.width,
                    surface_extent.height,
                    surface_format.format,
                    image_usage,
                );
                Texture {
                    desc,
                    image,
                    memory: vk::DeviceMemory::null(), // TODO really?
                }
            })
            .collect::<Vec<_>>()
    };

    // Create image views / texture views
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(1);
    let texture_views = textures
        .iter()
        .map(|texture| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(texture.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .subresource_range(*subresource_range);
            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .expect("Vulkan: failed to create image view for swapchain")
            };
            let desc = TextureViewDesc::auto(&texture.desc);
            TextureView {
                texture: *texture,
                desc,
                image_view,
            }
        })
        .collect::<Vec<_>>();

    Swapchain {
        extent: surface_extent,
        handle: swapchain_handle,
        textures,
        texture_views,
    }
}

impl super::RenderDevice {
    // Wait undefinitely for next swapchain image.
    // Use a semaphore or fence to wait until the image is ok to be modified.
    #[inline]
    pub fn acquire_next_swapchain_image(
        &self,
        semaphore_to_signal: vk::Semaphore,
        fence_to_signal: vk::Fence,
    ) -> u32 {
        // Validate "semaphore and fence must not both be equal to VK_NULL_HANDLE"
        assert!(
            (semaphore_to_signal != vk::Semaphore::null())
                || (fence_to_signal != vk::Fence::null())
        );

        let (index, is_suboptimal) = unsafe {
            self.khr_swapchain
                .acquire_next_image(
                    self.swapchain.handle,
                    std::u64::MAX,
                    semaphore_to_signal,
                    fence_to_signal,
                )
                .expect("RenderDevice: failed to acquire next swapchain image")
        };

        if is_suboptimal {
            panic!("RenderDevice: acquired surface image has unexpected properties");
        }

        index
    }

    pub fn queue_present(&self, present_info: &vk::PresentInfoKHR) {
        unsafe {
            self.khr_swapchain
                .queue_present(self.gfx_queue, &present_info)
                .unwrap();
        }
    }
}
