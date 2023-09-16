use std::{mem::size_of, slice};

use ash::vk;
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};

use crate::{
    imgui,
    render_device::{Buffer, BufferDesc, RenderDevice},
};

/*
 * Modules
 */
pub mod gbuffer_pass;
pub mod imgui_pass;

/*
 * Basic Traits
 */

pub trait RenderLoop: Sized {
    fn new(rd: &mut RenderDevice) -> Option<Self>;

    fn ui(&mut self, _ui: &mut imgui::Ui) {}

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut crate::shader::Shaders,
        scene: &crate::render_scene::RenderScene,
        view_info: &ViewInfo,
        imgui: Option<&imgui::ImGUIOuput>,
    );

    fn print_stat(&self) {}
}

/*
 * Common Types
 */

#[derive(Clone, Copy)]
pub struct ViewInfo {
    pub view_position: Vec3,
    pub view_transform: Mat4,
    pub projection: Mat4,
    pub moved: bool,
}

#[derive(Clone, Copy)]
pub struct JitterInfo {
    pub frame_index: u32,
    pub viewport_size: UVec2,
}

#[derive(Clone, Copy)]
pub struct PrevView {
    pub view_info: ViewInfo,
    pub jitter_info: Option<JitterInfo>,
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

    pub prev_view_proj: Mat4,

    pub jitter: Vec4,

    pub sun_dir: Vec4,
    pub sun_inten: Vec4,
}

impl FrameParams {
    // 2D jitter in range of [-1.0, 1.0]
    pub fn jitter(frame_index: u32) -> Vec2 {
        let cycle = 8;
        // NOTE: +1 to ignore the (0.0, 0.0) jitter
        let frame_index = frame_index % cycle + 1;
        let x = 2.0 * halton(frame_index, 2) - 1.0;
        let y = 2.0 * halton(frame_index, 3) - 1.0;
        Vec2::new(x, y)
    }

    pub fn make_view_proj(view_info: &ViewInfo, jitter_info: Option<&JitterInfo>) -> (Mat4, Vec2) {
        let mut view_proj = view_info.projection * view_info.view_transform;
        let mut jitter_ndc = Vec2::ZERO;
        if let Some(jitter_info) = jitter_info {
            let raster_size = jitter_info.viewport_size.as_vec2();
            jitter_ndc = Self::jitter(jitter_info.frame_index) / raster_size;
            view_proj = Mat4::from_translation(jitter_ndc.extend(0.0)) * view_proj;
        }
        (view_proj, jitter_ndc)
    }

    pub fn make(
        view_info: &ViewInfo,
        jitter_info: Option<&JitterInfo>,
        sun_dir: &Vec3,
        sun_inten: &Vec3,
        prev_view: Option<&PrevView>,
    ) -> Self {
        let (view_proj, jitter_ndc) = Self::make_view_proj(view_info, jitter_info);
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

        let (prev_view_proj, prev_jitter_ndc) = match prev_view {
            Some(prev) => Self::make_view_proj(&prev.view_info, prev.jitter_info.as_ref()),
            None => (view_proj, Vec2::ZERO),
        };

        let jitter_vec4 = Vec4::new(
            jitter_ndc.x,
            jitter_ndc.y,
            prev_jitter_ndc.x,
            prev_jitter_ndc.y,
        );

        Self {
            view_proj: view_proj,
            inv_view_proj: view_proj.inverse(),
            view_pos: view_info.view_position.extend(0.0),
            view_ray_top_left: view_ray_top_left.extend(0.0),
            view_ray_right_shift: (view_ray_right - view_ray_left).extend(0.0),
            view_ray_down_shift: (view_ray_down - view_ray_up).extend(0.0),
            prev_view_proj,
            jitter: jitter_vec4,
            sun_dir: sun_dir.extend(0.0),
            sun_inten: sun_inten.extend(0.0),
        }
    }
}

pub const FRAME_DESCRIPTOR_SET_INDEX: u32 = 2;

pub const FRAMEPARAMS_BINDING_INDEX: u32 = 0;
pub const SAMPLER_BINDING_INDEX_BEGIN: u32 = 1;

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

        // Static sampler
        let sampler_linear_clamp = unsafe {
            let create_info = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            rd.device.create_sampler(&create_info, None).unwrap()
        };
        let sampler_linear_wrap = unsafe {
            let create_info = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT);
            rd.device.create_sampler(&create_info, None).unwrap()
        };
        let sampler_nearest_clamp = unsafe {
            let create_info = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            rd.device.create_sampler(&create_info, None).unwrap()
        };

        // Define set layout
        let set_layout = {
            let cbuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(FRAMEPARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let sampler_lc = vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLER_BINDING_INDEX_BEGIN)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .immutable_samplers(std::slice::from_ref(&sampler_linear_clamp));
            let sampler_lw = vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLER_BINDING_INDEX_BEGIN + 1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .immutable_samplers(std::slice::from_ref(&sampler_linear_wrap));
            let sampler_nc = vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLER_BINDING_INDEX_BEGIN + 2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .immutable_samplers(std::slice::from_ref(&sampler_nearest_clamp));
            let bindings = [*cbuffer, *sampler_lc, *sampler_lw, *sampler_nc];
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            unsafe {
                rd.device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("Failed to create scene descriptor set layout")
            }
        };

        // Frame parameter constant buffer
        let frame_params_align_mask = rd.min_uniform_buffer_offset_alignment() as u32 - 1;
        let frame_params_stride =
            (size_of::<FrameParams>() as u32 + frame_params_align_mask) & !frame_params_align_mask;
        let frame_params_cb = rd
            .create_buffer(BufferDesc {
                size: (frame_params_stride * max_sets) as u64,
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
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
                    rd.device
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
                    rd.device.update_descriptor_sets(&[write_cbuffer], &[]);
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

pub const MAX_FRAMES_ON_THE_FLY: u32 = 2;

pub enum ResourceType {
    //Texture(Texture),
    Buffer(Buffer),
}

// Common resources to help render loop to support multi on-the-fly frames.
pub struct StreamLinedFrameResource {
    //command_pool: vk::CommandPool, // TODO destroy it

    // Index to access per-frame resources
    render_index: u32,

    // Per-frame resource that is accessed by render index.
    desciptor_sets: RenderLoopDesciptorSets,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_finished_fences: Vec<vk::Fence>,
    present_finished_fences: Vec<vk::Fence>,
    present_ready_semaphores: Vec<vk::Semaphore>,

    // Index to acess per-frame resources, but in double period
    double_render_index: u32,

    /// Resrouce that is released after MAX_FRAMES_ON_THE_FLY frames.
    /// Accessed by double render index (push: +MAX_FRAMES_ON_THE_FLY)
    delayed_release: Vec<Vec<ResourceType>>,
}

impl StreamLinedFrameResource {
    pub fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        let desciptor_sets = RenderLoopDesciptorSets::new(rd, MAX_FRAMES_ON_THE_FLY);

        let mut command_buffers = Vec::new();
        let mut command_buffer_finished_fences = Vec::new();
        let mut present_finished_fences = Vec::new();
        let mut present_ready_semaphores = Vec::new();
        for _ in 0..MAX_FRAMES_ON_THE_FLY {
            command_buffers.push(rd.create_command_buffer(command_pool));
            command_buffer_finished_fences.push(rd.create_fence(true));
            present_finished_fences.push(rd.create_fence(false));
            present_ready_semaphores.push(rd.create_semaphore());
        }

        let mut delayed_release = Vec::new();
        for _ in 0..MAX_FRAMES_ON_THE_FLY * 2 {
            delayed_release.push(Vec::new());
        }

        Self {
            //command_pool,
            render_index: 0,
            desciptor_sets,
            command_buffers,
            command_buffer_finished_fences,
            present_finished_fences,
            present_ready_semaphores,
            double_render_index: 0,
            delayed_release,
        }
    }

    // Index into per-render resource arrays (command_buffer, constant_buffer slot, etc.)
    pub fn advance_render_index(&mut self) -> u32 {
        self.render_index = (self.render_index + 1) % (MAX_FRAMES_ON_THE_FLY);
        self.double_render_index = (self.double_render_index + 1) % (MAX_FRAMES_ON_THE_FLY * 2);
        self.render_index
    }

    pub fn get_frame_desciptor_set(&self) -> vk::DescriptorSet {
        self.desciptor_sets.sets[self.render_index as usize]
    }

    pub fn get_set_layout(&self) -> vk::DescriptorSetLayout {
        self.desciptor_sets.set_layout
    }

    pub fn acquire_next_swapchain_image(&self, rd: &RenderDevice) -> u32 {
        self.acquire_next_swapchain_image_with_duration(rd).0
    }

    pub fn acquire_next_swapchain_image_with_duration(
        &self,
        rd: &RenderDevice,
    ) -> (u32, std::time::Duration) {
        // temporary fence for this image
        let swapchain_ready_fence = self.present_finished_fences[self.render_index as usize];

        let start = std::time::Instant::now();

        // TODO temporary semaphore for this image (for accss in GPU)?
        // ref: https://github.com/KhronosGroup/Vulkan-Docs/issues/1158#issuecomment-573874821
        let index = rd.acquire_next_swapchain_image(vk::Semaphore::null(), swapchain_ready_fence);

        let elapsed = start.elapsed();

        // Warn if acquire_next_swapchain_image takes too long
        let elapsed_us = elapsed.as_micros();
        if elapsed_us > 500 {
            println!("RenderDevice: acquire next image takes {} us!", elapsed_us);
        }

        (index, elapsed)
    }

    fn delay_release_buffer(&mut self, buffer: Buffer) {
        self.delay_release(ResourceType::Buffer(buffer));
    }

    fn delay_release(&mut self, res: ResourceType) {
        let future_index =
            (self.double_render_index + MAX_FRAMES_ON_THE_FLY) % (MAX_FRAMES_ON_THE_FLY * 2);
        self.delayed_release[future_index as usize].push(res);
    }

    fn release_resources(&mut self, rd: &RenderDevice) {
        let resources = &mut self.delayed_release[self.double_render_index as usize];
        for resource in resources.drain(0..) {
            match resource {
                ResourceType::Buffer(buffer) => {
                    rd.destroy_buffer(buffer);
                }
            }
        }
    }

    // Wait for command buffer finished and reset it for recording.
    pub fn wait_and_reset_command_buffer(&mut self, rd: &RenderDevice) -> vk::CommandBuffer {
        // Fence to make sure previous submit command buffer is finished
        let fence = self.command_buffer_finished_fences[self.render_index as usize];

        // Wait for the oldest command buffer finished
        let timeout = 5000 * 1000_000; // 5s
        let wait_begin = std::time::Instant::now();
        match unsafe {
            rd.device
                .wait_for_fences(slice::from_ref(&fence), true, timeout)
        } {
            Ok(_) => {
                let elapsed_ms = wait_begin.elapsed().as_micros() as f32 / 1000.0;
                let warning_ms = 33.3;
                if elapsed_ms > warning_ms {
                    println!(
                        "Vulkan: Wait oldest command buffer finished in {}ms",
                        elapsed_ms
                    );
                }
            }
            Err(err) => match err {
                vk::Result::TIMEOUT => {
                    println!(
                        "Vulkan: Failed to wait oldest command buffer finished in {}ms",
                        timeout / 1000_000
                    );
                }
                _ => {
                    println!(
                        "Vulkan: Wait oldest command buffer finished error: {:?}",
                        err
                    );
                }
            },
        }

        // Reset after use
        let command_buffer = self.command_buffers[self.render_index as usize];
        unsafe {
            rd.device.reset_fences(slice::from_ref(&fence)).unwrap();
            rd.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }

        // Good time to release resource if command buffer is finished
        self.release_resources(rd);

        command_buffer
    }

    pub fn update_frame_params(&self, params: FrameParams) {
        self.desciptor_sets
            .update_frame_params(self.render_index, params);
    }

    fn wait_and_reset_fence(rd: &RenderDevice, fence: vk::Fence) -> std::time::Duration {
        // Wait for the oldest image present finished
        let timeout = 5000 * 1000_000; // 5s
        let wait_begin = std::time::Instant::now();
        match unsafe {
            rd.device
                .wait_for_fences(slice::from_ref(&fence), true, timeout)
        } {
            Ok(_) => {
                let elapsed_ms = wait_begin.elapsed().as_micros() as f32 / 1000.0;
                let warning_ms = 33.3;
                if elapsed_ms > warning_ms {
                    println!("Vulkan: Wait oldest present finished in {}ms", elapsed_ms);
                }
            }
            Err(err) => match err {
                vk::Result::TIMEOUT => {
                    println!(
                        "Vulkan: Failed to wait oldest present finished in {}ms",
                        timeout / 1000_000
                    );
                }
                _ => {
                    println!("Vulkan: Wait oldest present finished error: {:?}", err);
                }
            },
        }

        // Reset after use
        unsafe {
            rd.device.reset_fences(slice::from_ref(&fence)).unwrap();
        }

        wait_begin.elapsed()
    }

    pub fn wait_and_submit_and_present(
        &self,
        rd: &RenderDevice,
        image_index: u32,
    ) -> (std::time::Duration, std::time::Duration) {
        // Wait for swapchain image ready (before submit the GPU works modifying the image)
        let wait_duration = {
            // the same temporary fence in acquire_next_swapchain_image
            let swapchain_ready_fence = self.present_finished_fences[self.render_index as usize];
            Self::wait_and_reset_fence(rd, swapchain_ready_fence)
        };

        // Submit the command buffer
        let present_ready = self.present_ready_semaphores[self.render_index as usize];
        {
            let command_buffer = self.command_buffers[self.render_index as usize];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(slice::from_ref(&command_buffer))
                .signal_semaphores(slice::from_ref(&present_ready))
                .build();
            let cb_finish_fence = self.command_buffer_finished_fences[self.render_index as usize];
            unsafe {
                rd.device
                    .queue_submit(rd.gfx_queue, slice::from_ref(&submit_info), cb_finish_fence)
                    .unwrap();
            }
        }

        // Present
        let present_duration = {
            let present_begin = std::time::Instant::now();

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(slice::from_ref(&present_ready))
                .swapchains(slice::from_ref(&rd.swapchain.handle))
                .image_indices(slice::from_ref(&image_index))
                .build();
            rd.queue_present(&present_info);

            present_begin.elapsed()
        };

        (wait_duration, present_duration)
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

pub fn div_round_up_uvec2(a: UVec2, b: UVec2) -> UVec2 {
    (a + (b - UVec2::new(1, 1))) / b
}

// Naive impelementation of Halton sequence; proper only for small indices.
// ref: https://en.wikipedia.org/wiki/Halton_sequence
pub fn halton(index: u32, base: u32) -> f32 {
    let mut i = index;
    let mut f = 1.0;
    let mut r = 0.0;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i = i / base;
    }
    r
}

pub mod rg_util {
    use crate::{
        render_device::{Texture, TextureDesc, TextureView},
        render_graph::{RGHandle, RenderGraphBuilder},
    };

    pub fn create_texture_and_view(
        rg: &mut RenderGraphBuilder,
        desc: TextureDesc,
    ) -> (RGHandle<Texture>, RGHandle<TextureView>) {
        let tex = rg.create_texutre(desc);
        let view = rg.create_texture_view(tex, None);
        (tex, view)
    }
}
