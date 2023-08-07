use std::f32::consts::PI;

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

pub struct SengaRenderLoop {
    render_graph_cache: RenderGraphCache,
    stream_lined: StreamLinedFrameResource,
}

impl RenderLoop for SengaRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            stream_lined: StreamLinedFrameResource::new(rd),
        }
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        self.stream_lined.advance_render_index();

        let mut rg = RenderGraphBuilder::new();

        let frame_descritpr_set = self.stream_lined.get_frame_desciptor_set();
        let common_sets = [
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descritpr_set),
        ];

        // Shader config
        let mut shader_config = ShadersConfig::default();
        shader_config
            .set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        shader_config.set_layout_override.insert(
            FRAME_DESCRIPTOR_SET_INDEX,
            self.stream_lined.get_set_layout(),
        );

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
        let swapchain_image_index = self.stream_lined.acquire_next_swapchain_image(rd);

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
        let command_buffer = self.stream_lined.wait_and_reset_command_buffer(rd);

        // Update FrameParams
        // TODO sun light config
        let exposure = 5.0;
        let sun_inten = Vec3::new(0.7, 0.7, 0.6) * PI * exposure;
        self.stream_lined.update_frame_params(FrameParams::make(
            &view_info,
            &scene.sun_dir,
            &sun_inten,
        ));

        // Execute the render graph, writing into command buffer
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

        // Wait for swapchain ready, submit, and present
        self.stream_lined
            .wait_and_submit_and_present(rd, swapchain_image_index);
    }
}

fn main() {
    violet::app::run_with_renderloop::<SengaRenderLoop>();
}