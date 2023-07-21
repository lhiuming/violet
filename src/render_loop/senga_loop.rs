use std::{f32::consts::PI, slice};

use ash::vk;
use glam::Vec3;

use crate::{
    render_device::RenderDevice,
    render_graph::{RenderGraphBuilder, RenderGraphCache},
    render_scene::RenderScene,
    shader::{Shaders, ShadersConfig},
};

use super::{FrameParams, RenderLoopDesciptorSets, ViewInfo};

const MAX_RENDERING_IN_FLIGHT: u32 = 2;

pub struct SengaRenderLoop {
    render_graph_cache: RenderGraphCache,
    shaders_config: ShadersConfig,
    command_pool: vk::CommandPool,

    // Per-render resource that is accessed by render index.
    desciptor_sets: RenderLoopDesciptorSets,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_finished_fences: Vec<vk::Fence>,

    next_render_index: u32,
}

impl SengaRenderLoop {
    pub fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        let desciptor_sets = RenderLoopDesciptorSets::new(rd, MAX_RENDERING_IN_FLIGHT);

        let mut command_buffers = Vec::new();
        let mut command_buffer_finished_fences = Vec::new();
        for i in 0..MAX_RENDERING_IN_FLIGHT {
            command_buffers.push(rd.create_command_buffer(command_pool));
            command_buffer_finished_fences.push(rd.create_fence(true));
        }

        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            shaders_config: Default::default(),
            command_pool,

            desciptor_sets,
            command_buffers,
            command_buffer_finished_fences,
            next_render_index: 0,
        }
    }

    // Index into per-render resource arrays (command_buffer, constant_buffer slot, etc.)
    pub fn acquire_render_index(&mut self, rd: &RenderDevice) -> u32 {
        let render_index = self.next_render_index;
        let fence = self.command_buffer_finished_fences[render_index as usize];

        // Wait for the oldest command buffer finished
        let mut timeout_in_ns = 500_000; // 0.5ms
        let timeout_max = 5000 * 1000_000; // 5s
        loop {
            match unsafe {
                rd.device_entry
                    .wait_for_fences(slice::from_ref(&fence), true, timeout_in_ns)
            } {
                Ok(_) => break,
                Err(_) => {
                    println!(
                        "Vulkan: Failed to wait oldest command buffer finished in {}ms, keep trying...",
                        timeout_in_ns / 1000000
                    );
                }
            }
            timeout_in_ns = std::cmp::min(timeout_in_ns * 2, timeout_max);
        }

        unsafe {
            rd.device_entry
                .reset_fences(slice::from_ref(&fence))
                .unwrap();
        }

        self.next_render_index = (self.next_render_index + 1) % (MAX_RENDERING_IN_FLIGHT);

        render_index
    }

    pub fn render(
        &mut self,
        rd: &RenderDevice,
        shaders: &Shaders,
        scene: &RenderScene,
        view_info: ViewInfo,
    ) {
        let render_index = self.acquire_render_index(rd);

        let mut rg = RenderGraphBuilder::new(&mut self.render_graph_cache);

        // Update FrameParams
        // TODO sun light config
        let exposure = 5.0;
        let sun_inten = Vec3::new(0.7, 0.7, 0.6) * PI * exposure;
        self.desciptor_sets.update_frame_params(
            render_index,
            FrameParams::make(&view_info, &scene.sun_dir, &sun_inten),
        );
    }
}
