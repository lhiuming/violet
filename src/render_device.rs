use ash::vk;

pub mod buffer;
pub mod core;
pub mod debug_utils;
pub mod physical;
pub mod raytracing;
pub mod swapchain;
pub mod texture;

// Re-exporting
pub use self::buffer::{Buffer, BufferDesc};
pub use self::core::{DeviceConfig, RenderDevice};
pub use self::physical::PhysicalDevice;
pub use self::raytracing::AccelerationStructure;
pub use self::swapchain::{Surface, Swapchain};
pub use self::texture::{Texture, TextureDesc, TextureView, TextureViewDesc};

/*
 * Shortcuts
 */
impl RenderDevice {
    #[inline]
    pub fn timestamp_period(&self) -> f32 {
        self.physical.properties.limits.timestamp_period
    }

    #[inline]
    pub fn min_uniform_buffer_offset_alignment(&self) -> u64 {
        self.physical
            .properties
            .limits
            .min_uniform_buffer_offset_alignment
    }

    #[inline]
    pub fn shader_group_handle_size(&self) -> u32 {
        self.physical
            .ray_tracing_pipeline_properties
            .shader_group_handle_size
    }

    #[inline]
    pub fn shader_group_handle_alignment(&self) -> u32 {
        self.physical
            .ray_tracing_pipeline_properties
            .shader_group_handle_alignment
    }

    #[inline]
    pub fn shader_group_base_alignment(&self) -> u32 {
        self.physical
            .ray_tracing_pipeline_properties
            .shader_group_base_alignment
    }

    #[inline]
    pub fn format_support_blending(&self, format: vk::Format) -> bool {
        let prop = self.physical.get_format_properties(format);
        // NOTE: currently all image are created as VK_IMAGE_TILING_OPTIMAL (including swapchain)
        prop.optimal_tiling_features
            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
    }
}
