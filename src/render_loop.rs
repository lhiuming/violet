use std::{mem::size_of, slice};

use ash::vk;
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::render_device::{Buffer, RenderDevice};

/*
 * Modules
 */
pub mod gbuffer_pass;

pub mod pbr_loop;
pub use pbr_loop::PhysicallyBasedRenderLoop;

/*
 * Basic Traits
 */

pub trait RenderLoop {
    fn new(rd: &RenderDevice) -> Self;

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut crate::shader::Shaders,
        scene: &crate::render_scene::RenderScene,
        view_info: &ViewInfo,
    );
}

/*
 * Common Types
 */

pub struct ViewInfo {
    pub view_position: Vec3,
    pub view_transform: Mat4,
    pub projection: Mat4,
    pub moved: bool,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FrameParams {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub view_pos: Vec4,
    pub view_ray_top_left: Vec4,
    pub view_ray_right_shift: Vec4,
    pub view_ray_down_shift: Vec4,

    pub sun_dir: Vec4,
    pub sun_inten: Vec4,
}

impl FrameParams {
    pub fn make(view_info: &ViewInfo, sun_dir: &Vec3, sun_inten: &Vec3) -> Self {
        // From row major float4x4 to column major Mat4
        let view_proj = view_info.projection * view_info.view_transform;
        let inv_proj = view_proj.inverse();
        let ndc_to_ray = |ndc: Vec4| {
            let pos_ws_h = inv_proj * ndc;
            let pos_ws = pos_ws_h.xyz() / pos_ws_h.w;
            pos_ws - view_info.view_position
        };
        let view_ray_top_left = ndc_to_ray(Vec4::new(-1.0, -1.0, 1.0, 1.0));
        let view_ray_right = ndc_to_ray(Vec4::new(1.0, 0.0, 1.0, 1.0));
        let view_ray_left = ndc_to_ray(Vec4::new(-1.0, 0.0, 1.0, 1.0));
        let view_ray_up = ndc_to_ray(Vec4::new(0.0, -1.0, 1.0, 1.0));
        let view_ray_down = ndc_to_ray(Vec4::new(0.0, 1.0, 1.0, 1.0));
        Self {
            view_proj: view_proj,
            inv_view_proj: view_proj.inverse(),
            view_pos: view_info.view_position.extend(0.0),
            view_ray_top_left: view_ray_top_left.extend(0.0),
            view_ray_right_shift: (view_ray_right - view_ray_left).extend(0.0),
            view_ray_down_shift: (view_ray_down - view_ray_up).extend(0.0),
            sun_dir: sun_dir.extend(0.0),
            sun_inten: sun_inten.extend(0.0),
        }
    }
}

pub const FRAME_DESCRIPTOR_SET_INDEX: u32 = 2;

pub const FRAMEPARAMS_BINDING_INDEX: u32 = 0;

// DescriptorSet that is allocated/updated per frame
pub struct RenderLoopDesciptorSets {
    pub descriptor_pool: vk::DescriptorPool,
    pub set_layout: vk::DescriptorSetLayout,
    pub frame_params_cb: Buffer,
    pub frame_params_stride: u32,
    pub sets: Vec<vk::DescriptorSet>,
}

impl RenderLoopDesciptorSets {
    pub fn new(rd: &RenderDevice, max_sets: u32) -> Self {
        let descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorPoolCreateFlags::empty(),
            max_sets,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: max_sets, // 1 dynamic cb per set
            }],
        );

        // Define set layout
        let set_layout = {
            let cbuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(FRAMEPARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let bindings = [*cbuffer];
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            unsafe {
                rd.device_entry
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("Failed to create scene descriptor set layout")
            }
        };

        // Frame parameter constant buffer
        let frame_params_align_mask = rd
            .physical_device
            .properties
            .limits
            .min_uniform_buffer_offset_alignment as u32
            - 1;
        let frame_params_stride =
            (size_of::<FrameParams>() as u32 + frame_params_align_mask) & !frame_params_align_mask;
        let frame_params_cb = rd
            .create_buffer(
                (frame_params_stride * max_sets) as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();

        // Create descriptor sets
        let sets = (0..max_sets)
            .map(|index| {
                // Create
                let layouts = [set_layout];
                let create_info = vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts);
                let descriptor_set = unsafe {
                    rd.device_entry
                        .allocate_descriptor_sets(&create_info)
                        .expect("Failed to create descriptor set for scene")[0]
                };

                // Fill
                let cbuffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(frame_params_cb.handle)
                    .offset((index * frame_params_stride) as vk::DeviceSize)
                    .range(vk::WHOLE_SIZE)
                    .build();
                let write_cbuffer = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(FRAMEPARAMS_BINDING_INDEX)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(slice::from_ref(&cbuffer_info))
                    .build();
                unsafe {
                    rd.device_entry
                        .update_descriptor_sets(&[write_cbuffer], &[]);
                }

                descriptor_set
            })
            .collect::<Vec<_>>();

        Self {
            descriptor_pool,
            set_layout,
            frame_params_cb,
            frame_params_stride,
            sets,
        }
    }

    pub fn update_frame_params(&self, index: u32, params: FrameParams) {
        let cb_offset = index * self.frame_params_stride;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                self.frame_params_cb.data.offset(cb_offset as isize) as *mut FrameParams,
                1,
            )
        };
        dst.copy_from_slice(std::slice::from_ref(&params));
    }
}

/*
 * Utilites.
 */

pub fn div_round_up<T>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + std::convert::From<u8>
        + Copy,
{
    (a + (b - T::from(1))) / b
}
