use std::ffi::{c_char, CStr};
use std::io::Write;

use ash::extensions::{ext, khr};
use ash::vk;

use crate::render_device::raytracing;

use super::swapchain::{self, Surface, Swapchain};

use super::debug_utils;
use super::physical::{PhysicalDevice, PhysicalDeviceFeatures};

pub struct DeviceConfig {
    pub app_handle: u64,
    pub window_handle: u64,
    pub vsync: bool,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            app_handle: 0,    // not valid
            window_handle: 0, // not valid
            vsync: true,
        }
    }
}

pub struct RenderDevice {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: PhysicalDevice,
    pub device: ash::Device,

    // Instance extentions
    pub ext_debug_utils: ext::DebugUtils,
    pub khr_surface: khr::Surface,
    pub khr_win32_surface: khr::Win32Surface,

    // Device extensions
    pub khr_swapchain: khr::Swapchain,
    pub khr_ray_tracing_pipeline: Option<khr::RayTracingPipeline>,
    pub khr_accel_struct: Option<khr::AccelerationStructure>,

    // Feature flags
    pub support_raytracing: bool,

    // Convinient resources
    pub gfx_queue_family_index: u32,
    pub gfx_queue: vk::Queue,
    pub surface: Surface,
    pub swapchain: Swapchain,
}

impl super::RenderDevice {
    pub fn create(config: DeviceConfig) -> Option<Self> {
        // Load functions
        let entry = unsafe {
            ash::Entry::load()
                .expect("Ash can not load Vulkan. Maybe vulkan runtime is not installed.")
        };

        // Create instance
        let instance = {
            // Log version
            match entry
                .try_enumerate_instance_version()
                .expect("Vulkan: failed to enumerate instance version")
            {
                Some(version) => {
                    let major = vk::api_version_major(version);
                    let minor = vk::api_version_minor(version);
                    let patch = vk::api_version_patch(version);
                    let variant = vk::api_version_variant(version);
                    println!(
                        "Vulkan API Version: {}.{} (patch {}, variant {})",
                        major, minor, patch, variant
                    );
                }
                None => {
                    println!("Vulkan API Version: 1.0");
                }
            }

            let app_info = vk::ApplicationInfo::builder().api_version(vk::API_VERSION_1_3);

            let enable_validation_layer = match std::env::var_os("VIOLET_VALIDATION_LAYER") {
                Some(val) => val != "0",
                None => false,
            };

            let validation_layer_name =
                CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0").unwrap();
            let layer_names = [
                validation_layer_name.as_ptr(), // Vulkan validation (debug layer)
            ];

            let ext_names_raw = [
                khr::Surface::name().as_ptr(),
                khr::Win32Surface::name().as_ptr(), // Platform: Windows
                ext::DebugUtils::name().as_ptr(),   // Debug
            ];

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names[..enable_validation_layer as usize])
                .enabled_extension_names(&ext_names_raw);

            print!("Vulkan: creating instance ... ");
            std::io::stdout().flush().unwrap();
            let instance = unsafe {
                entry
                    .create_instance(&create_info, None)
                    .expect("Vulkan instance creation failed")
            };
            println!("done.");

            instance
        };

        // Create Instance Extensions
        let khr_surface = khr::Surface::new(&entry, &instance);
        let khr_win32_surface = khr::Win32Surface::new(&entry, &instance);
        let ext_debug_utils = ext::DebugUtils::new(&entry, &instance);

        // Setup DebugUtils message callback (for validation, etc.)
        debug_utils::set_up_message_callback(&ext_debug_utils);
        println!("Vulkan Debug message callback registered.");

        // Physical Device
        let physical = PhysicalDevice::new(&instance);

        let supported_features = physical.get_supported_features();
        let supported_extentions = physical.get_supported_device_extensions();

        // Check raytracing supports
        let support_raytracing = raytracing::DEVICE_EXTENSIONS.iter().all(|name| {
            supported_extentions.contains(unsafe { CStr::from_ptr(*name as *mut c_char) })
        }) && raytracing::check_features(&supported_features);

        // Eanble required features
        let mut features = PhysicalDeviceFeatures::default();
        features.core_mut().sampler_anisotropy = vk::TRUE;
        // TODO make this optional?
        features.core_mut().fragment_stores_and_atomics = vk::TRUE;
        features.core_mut().dual_src_blend = vk::TRUE;
        // dynamic rendering
        features.vulkan13.dynamic_rendering = vk::TRUE;
        // bindless
        features
            .vulkan12
            .descriptor_binding_sampled_image_update_after_bind = vk::TRUE;
        features.vulkan12.descriptor_binding_partially_bound = vk::TRUE;
        features.vulkan12.runtime_descriptor_array = vk::TRUE;
        // host query reset (for internal GPU profiling)
        features.vulkan12.host_query_reset = vk::TRUE;
        if support_raytracing {
            raytracing::enable_features(&mut features);
        }

        // Enumerate and pick queue families to create with the device
        let queue_family_props =
            unsafe { instance.get_physical_device_queue_family_properties(physical.handle) };
        let mut found_gfx_queue_family_index = None;
        for i in 0..queue_family_props.len() {
            let prop = &queue_family_props[i];
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                found_gfx_queue_family_index = Some(i as u32);
                /* Verbose info
                println!(
                        "Vulkan: found graphics queue family index: index {}, prop: flags {:?}, count {}",
                        i, prop.queue_flags, prop.queue_count
                    );
                */
                continue;
            }
            /* Verbose info
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                println!(
                    "Vulkan: found compute queue family index: index {}, flags {:?}, count {}",
                    i, prop.queue_flags, prop.queue_count
                );
                continue;
            }
            */
        }
        let gfx_queue_family_index =
            found_gfx_queue_family_index.expect("Vulkan: didn't found any graphics queue family?!");
        let queue_create_infos = [
            // Just create a graphics queue for everything ATM...
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(gfx_queue_family_index)
                .queue_priorities(&[1.0])
                .build(),
        ];

        // Create device
        let device = {
            // Specify device extensions
            let mut enabled_extension_names = vec![khr::Swapchain::name().as_ptr()];
            if support_raytracing {
                enabled_extension_names.extend_from_slice(&raytracing::DEVICE_EXTENSIONS);
            }

            // Finally, create the device
            let create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&enabled_extension_names)
                .push_next(features.chain());
            unsafe {
                let ret = instance.create_device(physical.handle, &create_info, None);

                // Return extension not present
                if let Err(err) = ret {
                    match err {
                        vk::Result::ERROR_EXTENSION_NOT_PRESENT => {
                            for name in enabled_extension_names.iter() {
                                let name = CStr::from_ptr(*name);
                                println!(
                                    "Error[Vulkan]: extension maybe is not presented: {:?}",
                                    name
                                );
                            }
                            println!("NOTE: extensions supported may be limited if device is create under debugging layers, e.g. VK_LAYER_RENDERDOC_Capture");
                        }
                        _ => {}
                    }
                }

                ret.unwrap()
            }
        };

        let gfx_queue = unsafe { device.get_device_queue(gfx_queue_family_index, 0) };

        // Create Device Extensions
        let khr_swapchain = khr::Swapchain::new(&instance, &device);
        let khr_ray_tracing_pipeline;
        let khr_accel_struct;
        if support_raytracing {
            khr_ray_tracing_pipeline = Some(khr::RayTracingPipeline::new(&instance, &device));
            khr_accel_struct = Some(khr::AccelerationStructure::new(&instance, &device));
        } else {
            khr_ray_tracing_pipeline = None;
            khr_accel_struct = None;
        }

        // Create surface and swapchain
        let surface =
            swapchain::create_surface(&khr_win32_surface, config.app_handle, config.window_handle);
        let swapchain = swapchain::create_swapchain(
            &khr_surface,
            &khr_swapchain,
            &device,
            &physical,
            &surface,
            config.vsync,
        );

        Some(Self {
            entry,
            instance,
            physical,
            device,
            ext_debug_utils,
            khr_surface,
            khr_win32_surface,
            khr_swapchain,
            khr_ray_tracing_pipeline,
            khr_accel_struct,

            support_raytracing,

            gfx_queue_family_index,
            gfx_queue,
            surface,
            swapchain,
        })
    }
}

impl super::RenderDevice {
    pub fn create_fence(&self, signaled: bool) -> vk::Fence {
        unsafe {
            let create_info = vk::FenceCreateInfo::builder().flags(if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            });
            self.device.create_fence(&create_info, None)
        }
        .expect("Vulakn: failed to create fence")
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        unsafe {
            let create_info = vk::SemaphoreCreateInfo::builder();
            self.device.create_semaphore(&create_info, None)
        }
        .expect("Vulkan: failed to create semaphore")
    }

    pub fn create_event(&self) -> vk::Event {
        unsafe {
            let create_info = vk::EventCreateInfo::builder();
            self.device.create_event(&create_info, None)
        }
        .expect("Vulkan: failed to create event")
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        unsafe {
            self.device
                .create_command_pool(&create_info, None)
                .expect("Vulkan: failed to create command pool?!")
        }
    }

    pub fn create_command_buffer(&self, command_pool: vk::CommandPool) -> vk::CommandBuffer {
        let create_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();
        unsafe {
            self.device
                .allocate_command_buffers(&create_info)
                .expect("Vulkan: failed to allocated command buffer?!")[0]
        }
    }

    #[inline]
    pub fn begin_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }
    }

    #[inline]
    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();
        }
    }

    pub fn create_descriptor_pool(
        &self,
        flags: vk::DescriptorPoolCreateFlags,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> vk::DescriptorPool {
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(flags)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);
        unsafe { self.device.create_descriptor_pool(&create_info, None) }
            .expect("Vulkan: failed to create descriptor pool?!")
    }

    pub fn update_descriptor_sets(
        &self,
        descriptor_writes: &[vk::WriteDescriptorSet],
        descriptor_copies: &[vk::CopyDescriptorSet],
    ) {
        unsafe {
            self.device
                .update_descriptor_sets(descriptor_writes, descriptor_copies)
        }
    }
}
