/*
 * Surface and Swapchain
 */

use ash::{extensions::khr, vk};

use super::{Texture, TextureDesc, TextureView, TextureViewDesc};

pub struct Surface {
    pub handle: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
}

pub fn create_surface(
    win32_surface: &khr::Win32Surface,
    surface: &khr::Surface,
    physical_device: vk::PhysicalDevice,
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
    // Query format
    let formats =
        unsafe { surface.get_physical_device_surface_formats(physical_device, vk_surface) }
            .unwrap();
    assert!(formats.len() > 0);
    // Debug
    println!("Vulkan surface supported formats:");
    for format in formats.iter() {
        println!("\t{:?}: {:?}", format.format, format.color_space);
    }
    Surface {
        handle: vk_surface,
        format: formats[0],
    }
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
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> Swapchain {
    let surface_extent = {
        let cap = unsafe {
            khr_surface.get_physical_device_surface_capabilities(physical_device, surface.handle)
        }
        .unwrap();
        cap.current_extent
    };

    // UI Overlay (color_attachment) and Compute PostProcessing (storage)
    let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE;

    let vsync = true;
    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        vk::PresentModeKHR::IMMEDIATE
    };

    // Create swapchain object
    let create_info = {
        vk::SwapchainCreateInfoKHR::builder()
            .flags(vk::SwapchainCreateFlagsKHR::empty())
            .surface(surface.handle)
            .min_image_count(2)
            .image_format(surface.format.format)
            .image_color_space(surface.format.color_space)
            .image_extent(surface_extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
    };
    let swapchain_handle = unsafe { khr_swapchain.create_swapchain(&create_info, None) }
        .expect("Vulkan: Swapchain creatino failed???");

    // Get images/textures
    let textures = {
        let mut images = unsafe { khr_swapchain.get_swapchain_images(swapchain_handle) }.unwrap();
        images
            .drain(0..)
            .map(|image| {
                let desc = TextureDesc::new_2d(
                    surface_extent.width,
                    surface_extent.height,
                    surface.format.format,
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
    let sub_res_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1)
        .level_count(1);
    let texture_views = textures
        .iter()
        .map(|texture| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(texture.image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface.format.format)
                .subresource_range(*sub_res_range);
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
