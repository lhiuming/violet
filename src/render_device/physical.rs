use std::collections::HashSet;
use std::ffi::{CStr, CString};

use ash::vk;

pub struct PhysicalDevice {
    instance: ash::Instance, // keep a copy, for convinience
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    pub accel_struct_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl PhysicalDevice {
    pub fn new(instance: &ash::Instance) -> Self {
        // Pick PhysicalDevice
        let physical_device_handle = {
            let phy_devs = unsafe { instance.enumerate_physical_devices() }.unwrap();
            assert!(phy_devs.len() > 0);
            let picked = phy_devs
                .iter()
                .find(|phy_dev| {
                    let prop = unsafe { instance.get_physical_device_properties(**phy_dev) };
                    prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .or(phy_devs.last());

            // print some info
            if picked.is_some() {
                let phy_dev = *picked.unwrap();
                let prop = unsafe { instance.get_physical_device_properties(phy_dev) };
                let name = unsafe { CStr::from_ptr(prop.device_name.as_ptr()) };
                println!("Vulkan: using physical device {:?}", name);
            }

            *picked.expect("Vulkan: None physical device?!")
        };

        // Get PhysicalDevice extended properties
        let mut ray_tracing_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut accel_struct_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut ray_tracing_pipeline_properties)
            .push_next(&mut accel_struct_properties)
            .build();
        unsafe {
            instance.get_physical_device_properties2(physical_device_handle, &mut properties2);

            // clean up for safety
            properties2.p_next = std::ptr::null_mut();
            ray_tracing_pipeline_properties.p_next = std::ptr::null_mut();
            accel_struct_properties.p_next = std::ptr::null_mut();
        }

        // Get PhysicalDevice memory properties
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device_handle) };

        Self {
            instance: instance.clone(),
            handle: physical_device_handle,
            properties: properties2.properties,
            ray_tracing_pipeline_properties,
            accel_struct_properties,
            memory_properties,
        }
    }

    pub fn get_supported_device_extensions(&self) -> HashSet<CString> {
        // Get supported device extensions (for debug info)
        unsafe {
            self.instance
                .enumerate_device_extension_properties(self.handle)
                .unwrap()
                .iter()
                .map(|ext| CStr::from_ptr(ext.extension_name.as_ptr()).to_owned())
        }
        .collect()
    }

    pub fn support_timestamp(&self) -> bool {
        // Must have proper timestamp_period
        (self.properties.limits.timestamp_period > 0.0)
        // If support compute_and_graphics, we dont need to query support from each quere
        && (self.properties.limits.timestamp_compute_and_graphics == vk::TRUE)
    }

    pub fn pick_memory_type_index(
        &self,
        memory_type_bits: u32,
        property_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let memory_types = &self.memory_properties.memory_types;
        for i in 0..self.memory_properties.memory_type_count {
            if ((memory_type_bits & (1 << i)) != 0)
                && ((memory_types[i as usize].property_flags & property_flags) == property_flags)
            {
                return Some(i);
            }
        }

        let mut support_properties = Vec::new();
        for i in 0..self.memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0 {
                support_properties.push(memory_types[i as usize].property_flags);
            }
        }

        println!(
            "Vulkan: No compatible device memory type with required properties {:?}. Support types are: {:?}",
            property_flags, support_properties
        );
        return None;
    }

    #[inline]
    pub fn get_format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        unsafe {
            self.instance
                .get_physical_device_format_properties(self.handle, format)
        }
    }

    pub fn list_supported_image_formats(
        &self,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Vec<vk::Format> {
        let is_linear = tiling == vk::ImageTiling::LINEAR;
        let i_first = vk::Format::R4G4_UNORM_PACK8.as_raw();
        let i_last = vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw();
        let mut ret = Vec::new();
        for i in i_first..=i_last {
            let format = vk::Format::from_raw(i);
            let prop = self.get_format_properties(format);
            let supported = if is_linear {
                prop.linear_tiling_features.contains(features)
            } else {
                prop.optimal_tiling_features.contains(features)
            };
            if supported {
                ret.push(format)
            }
        }
        ret
    }
}

// Wrapper for PhysicalDevice features that are typically chained up.
pub struct PhysicalDeviceFeatures {
    // root
    features2: vk::PhysicalDeviceFeatures2,

    // components
    pub vulkan12: vk::PhysicalDeviceVulkan12Features,
    pub vulkan13: vk::PhysicalDeviceVulkan13Features,
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    pub ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR,
}

impl Default for PhysicalDeviceFeatures {
    fn default() -> Self {
        Self {
            features2: vk::PhysicalDeviceFeatures2::default(),
            vulkan12: vk::PhysicalDeviceVulkan12Features::default(),
            vulkan13: vk::PhysicalDeviceVulkan13Features::default(),
            ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default(),
            acceleration_structure: vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default(),
            ray_query: vk::PhysicalDeviceRayQueryFeaturesKHR::default(),
        }
    }
}

impl PhysicalDeviceFeatures {
    pub fn core(&self) -> &vk::PhysicalDeviceFeatures {
        &self.features2.features
    }

    pub fn core_mut(&mut self) -> &mut vk::PhysicalDeviceFeatures {
        &mut self.features2.features
    }

    pub fn chain(&mut self) -> &mut vk::PhysicalDeviceFeatures2 {
        // For safety
        self.unchain();
        // Chain the structs
        let features = self.features2.features;
        self.features2 = vk::PhysicalDeviceFeatures2::builder()
            .features(features)
            .push_next(&mut self.vulkan12)
            .push_next(&mut self.vulkan13)
            .push_next(&mut self.ray_tracing_pipeline)
            .push_next(&mut self.acceleration_structure)
            .push_next(&mut self.ray_query)
            .build();

        &mut self.features2
    }

    // Do this before move
    pub fn unchain(&mut self) {
        self.features2.p_next = std::ptr::null_mut();
        self.vulkan12.p_next = std::ptr::null_mut();
        self.vulkan13.p_next = std::ptr::null_mut();
        self.ray_tracing_pipeline.p_next = std::ptr::null_mut();
        self.acceleration_structure.p_next = std::ptr::null_mut();
        self.ray_query.p_next = std::ptr::null_mut();
    }
}

impl PhysicalDevice {
    // Get physical device supported features
    pub fn get_supported_features(&self) -> PhysicalDeviceFeatures {
        let mut features = PhysicalDeviceFeatures::default();
        unsafe {
            self.instance
                .get_physical_device_features2(self.handle, features.chain());
        };

        features.unchain();
        features
    }
}
