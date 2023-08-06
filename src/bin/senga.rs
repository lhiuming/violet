use std::{f32::consts::PI, slice};

use ash::vk;
use glam::Vec3;

use violet::{
    command_buffer::*,
    render_device::RenderDevice,
    render_graph::*,
    render_loop::{
        gbuffer_pass::{add_gbuffer_pass, create_gbuffer_textures},
        *,
    },
    render_scene::{RenderScene, SCENE_DESCRIPTOR_SET_INDEX},
    shader::{ShaderDefinition, Shaders, ShadersConfig},
};

const MAX_RENDERING_IN_FLIGHT: u32 = 2;

pub struct SengaRenderLoop {
    render_graph_cache: RenderGraphCache,
    shaders_config: ShadersConfig,
    command_pool: vk::CommandPool,

    // Per-render resource that is accessed by render index.
    desciptor_sets: RenderLoopDesciptorSets,
    command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_finished_fences: Vec<vk::Fence>,
    present_finished_fences: Vec<vk::Fence>,
    present_ready_semaphores: Vec<vk::Semaphore>,

    next_render_index: u32,
}

// TODO move to StreamLinedResrouce.
impl SengaRenderLoop {
    // Index into per-render resource arrays (command_buffer, constant_buffer slot, etc.)
    fn acquire_render_index(&mut self) -> u32 {
        let render_index = self.next_render_index;
        self.next_render_index = (self.next_render_index + 1) % (MAX_RENDERING_IN_FLIGHT);
        render_index
    }

    // Wait for command buffer finished
    fn wait_and_reset_command_buffer(&self, rd: &RenderDevice, render_index: u32) {
        let fence = self.command_buffer_finished_fences[render_index as usize];

        // Wait for the oldest command buffer finished
        let timeout = 5000 * 1000_000; // 5s
        let wait_begin = std::time::Instant::now();
        match unsafe {
            rd.device_entry
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
        let command_buffer = self.command_buffers[render_index as usize];
        unsafe {
            rd.device_entry
                .reset_fences(slice::from_ref(&fence))
                .unwrap();
            rd.device_entry
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
    }

    fn wait_and_reset_fence(rd: &RenderDevice, fence: vk::Fence) {
        // Wait for the oldest image present finished
        let timeout = 5000 * 1000_000; // 5s
        let wait_begin = std::time::Instant::now();
        match unsafe {
            rd.device_entry
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
            rd.device_entry
                .reset_fences(slice::from_ref(&fence))
                .unwrap();
        }
    }
}

impl RenderLoop for SengaRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        let desciptor_sets = RenderLoopDesciptorSets::new(rd, MAX_RENDERING_IN_FLIGHT);

        let mut command_buffers = Vec::new();
        let mut command_buffer_finished_fences = Vec::new();
        let mut present_finished_fences = Vec::new();
        let mut present_ready_semaphores = Vec::new();
        for _ in 0..MAX_RENDERING_IN_FLIGHT {
            command_buffers.push(rd.create_command_buffer(command_pool));
            command_buffer_finished_fences.push(rd.create_fence(true));
            present_finished_fences.push(rd.create_fence(false));
            present_ready_semaphores.push(rd.create_semaphore());
        }

        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            shaders_config: Default::default(),
            command_pool,

            desciptor_sets,
            command_buffers,
            command_buffer_finished_fences,
            present_finished_fences,
            present_ready_semaphores,

            next_render_index: 0,
        }
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        let render_index = self.acquire_render_index();

        let mut rg = RenderGraphBuilder::new();

        let frame_descritpr_set = self.desciptor_sets.sets[render_index as usize];
        let common_sets = [
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descritpr_set),
        ];

        // Shader config
        let mut shader_config = ShadersConfig::default();
        shader_config
            .set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        shader_config
            .set_layout_override
            .insert(FRAME_DESCRIPTOR_SET_INDEX, self.desciptor_sets.set_layout);

        // Create GBuffer
        let gbuffer = create_gbuffer_textures(&mut rg, rd.swapchain.extent);

        // Pass: GBuffer
        add_gbuffer_pass(
            &mut rg,
            rd,
            shaders,
            &shader_config,
            &common_sets,
            scene,
            &gbuffer,
        );

        // Acquire swapchain image
        // TODO this wait can be delayed to the end of command buffer recording (during RG build)?
        let swapchain_ready_fence = self.present_finished_fences[render_index as usize];
        let swapchain_image_index =
            rd.acquire_next_swapchain_image(vk::Semaphore::null(), swapchain_ready_fence);

        let final_color =
            { rg.register_texture_view(rd.swapchain.image_view[swapchain_image_index as usize]) };

        // Pass: Image-Based Line-Drawing
        if let Some(pipeline) = shaders.create_compute_pipeline(
            ShaderDefinition::compute("image_based_line_drawing.hlsl", "main"),
            &shader_config,
        ) {
            rg.new_pass("ImageBasedLineDrawing", RenderPassType::Compute)
                .pipeline(pipeline)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .rw_texture("rwcolor", final_color)
                .render(move |cb, _, _| {
                    let x = div_round_up(gbuffer.size.width, 8);
                    let y = div_round_up(gbuffer.size.height, 8);
                    cb.dispatch(x, y, 1);
                });
        }

        // Pass: Output
        rg.new_pass("Present", RenderPassType::Present)
            .present_texture(final_color);

        // Prepare command buffer
        self.wait_and_reset_command_buffer(rd, render_index);

        // Update FrameParams
        // TODO sun light config
        let exposure = 5.0;
        let sun_inten = Vec3::new(0.7, 0.7, 0.6) * PI * exposure;
        self.desciptor_sets.update_frame_params(
            render_index,
            FrameParams::make(&view_info, &scene.sun_dir, &sun_inten),
        );

        // Execute the render graph, writing into command buffer
        let command_buffer = self.command_buffers[render_index as usize];
        {
            // Begin
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                rd.device_entry
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
            }

            let cb = CommandBuffer::new(rd, command_buffer);
            rg.execute(rd, &cb, shaders, &mut self.render_graph_cache);

            // End
            unsafe {
                rd.device_entry.end_command_buffer(command_buffer).unwrap();
            }
        }

        // Wait for swapchain image ready (before submit the GPU works modifying the image)
        // TODO maybe use a seperate submit for writing the swapchain image?
        Self::wait_and_reset_fence(rd, swapchain_ready_fence);

        // Submit the command buffer
        let present_ready = self.present_ready_semaphores[render_index as usize];
        {
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(slice::from_ref(&command_buffer))
                .signal_semaphores(slice::from_ref(&present_ready))
                .build();
            let cb_finish_fence = self.command_buffer_finished_fences[render_index as usize];
            unsafe {
                rd.device_entry
                    .queue_submit(rd.gfx_queue, slice::from_ref(&submit_info), cb_finish_fence)
                    .unwrap();
            }
        }

        // Present
        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(slice::from_ref(&present_ready))
                .swapchains(slice::from_ref(&rd.swapchain.handle))
                .image_indices(slice::from_ref(&swapchain_image_index))
                .build();
            unsafe {
                rd.swapchain_entry
                    .entry
                    .queue_present(rd.gfx_queue, &present_info)
                    .unwrap();
            }
        }
    }
}

fn main() {
    violet::app::run_with_renderloop::<SengaRenderLoop>();
}
