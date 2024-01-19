use std::ffi::c_char;

use ash::vk;

use super::physical::PhysicalDeviceFeatures;
use super::Buffer;

// Ray Tracing Device Extentions
pub const DEVICE_EXTENSIONS: [*const c_char; 4] = [
    // Raytracing Extensions (and dependencies under vulkan 1.2)
    vk::KhrRayTracingPipelineFn::name().as_ptr(),
    vk::KhrAccelerationStructureFn::name().as_ptr(),
    vk::KhrDeferredHostOperationsFn::name().as_ptr(),
    // Workaround a DXC bug (causing all raytracing shader requiring ray query extensions when it is not used at all)
    // ref: https://github.com/microsoft/DirectXShaderCompiler/commit/ce31e10902732c8cd8f6f3b5b78699110afddb2b#diff-44e37c9720575ff94b7842b9ceb70a87fe72486d2b5da2e3828512dc64a352e6R217-R222
    vk::KhrRayQueryFn::name().as_ptr(),
];

pub fn check_features(features: &PhysicalDeviceFeatures) -> bool {
    true
    // Ray Tracing
    && features.ray_tracing_pipeline.ray_tracing_pipeline == vk::TRUE
    && features.acceleration_structure.acceleration_structure == vk::TRUE
    // Buffer Device Address (required by ray tracing, to retrive device address for shader binding table)
    && features.vulkan12.buffer_device_address == vk::TRUE
}

// TODO: kind of duplicated with check_features
pub fn enable_features(features: &mut PhysicalDeviceFeatures) {
    features.ray_tracing_pipeline.ray_tracing_pipeline = vk::TRUE;
    features.acceleration_structure.acceleration_structure = vk::TRUE;

    features.vulkan12.buffer_device_address = vk::TRUE;

    // Workaround a DXC bug (causing all raytracing shader requiring ray query extensions when it is not used at all)
    features.ray_query.ray_query = vk::TRUE;
}

impl super::RenderDevice {
    pub fn get_shader_group_handles(
        &self,
        pipeline: vk::Pipeline,
        first_group: u32,
        group_count: u32,
    ) -> Vec<u8> {
        let handle_size = self
            .physical
            .ray_tracing_pipeline_properties
            .shader_group_handle_size;
        let data_size = group_count * handle_size;
        unsafe {
            self.khr_ray_tracing_pipeline
                .as_ref()
                .unwrap()
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    first_group,
                    group_count,
                    data_size as usize,
                )
                .unwrap()
        }
    }
}

#[derive(Clone, Copy)]
pub struct AccelerationStructure {
    pub buffer: Buffer,
    pub ty: vk::AccelerationStructureTypeKHR,
    pub handle: vk::AccelerationStructureKHR,
    pub device_address: vk::DeviceAddress, // used to fill vk::AccelerationStructureInstanceKHR::accelerationStructureReference
}

impl PartialEq for AccelerationStructure {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl super::RenderDevice {
    pub fn create_accel_struct(
        &self,
        buffer: Buffer,
        offset: u64,
        size: u64,
        ty: vk::AccelerationStructureTypeKHR,
    ) -> Option<AccelerationStructure> {
        let khr_accel_struct = self.khr_accel_struct.as_ref()?;

        assert!(buffer.desc.size >= (offset + size));
        // check alignment required by spec
        // (VUID-VkAccelerationStructureCreateInfoKHR-offset-03734)
        assert!(offset & 0xff == 0);
        let create_info: vk::AccelerationStructureCreateInfoKHRBuilder<'_> =
            vk::AccelerationStructureCreateInfoKHR::builder()
                .buffer(buffer.handle)
                .offset(offset)
                .size(size)
                .ty(ty);
        let handle = unsafe {
            khr_accel_struct
                .create_acceleration_structure(&create_info, None)
                .unwrap()
        };

        let device_address = unsafe {
            let info = vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                .acceleration_structure(handle);
            khr_accel_struct.get_acceleration_structure_device_address(&info)
        };

        Some(AccelerationStructure {
            buffer,
            ty,
            handle,
            device_address,
        })
    }
}
