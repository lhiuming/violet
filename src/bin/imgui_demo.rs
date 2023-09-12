use ash::vk;
use violet::{
    command_buffer::*,
    imgui::ImGUIOuput,
    render_device::RenderDevice,
    render_graph::*,
    render_loop::{imgui_pass::ImGUIPass, *},
    render_scene::{RenderScene, UploadContext},
    shader::{Shaders, ShadersConfig},
};

pub struct ImGUIDemoRenderLoop {
    streamlined: StreamLinedFrameResource,
    render_graph_cache: RenderGraphCache,

    imgui_pass: ImGUIPass,

    upload_context: UploadContext,
}

impl RenderLoop for ImGUIDemoRenderLoop {
    fn new(rd: &RenderDevice) -> Option<Self> {
        Some(Self {
            streamlined: StreamLinedFrameResource::new(rd),
            render_graph_cache: RenderGraphCache::new(rd),
            imgui_pass: ImGUIPass::new(rd),
            upload_context: UploadContext::new(rd),
        })
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        _scene: &RenderScene,
        _view_info: &ViewInfo,
        imgui: Option<&ImGUIOuput>,
    ) {
        self.streamlined.advance_render_index();

        let mut shader_config = ShadersConfig::default();
        shader_config.set_layout_override.insert(
            FRAME_DESCRIPTOR_SET_INDEX,
            self.streamlined.get_set_layout(),
        );
        let mut rg = RenderGraphBuilder::new_with_shader_config(shader_config);
        rg.add_global_descriptor_sets(&[(
            FRAME_DESCRIPTOR_SET_INDEX,
            self.streamlined.get_frame_desciptor_set(),
        )]);

        let swapchain_image_index = self.streamlined.acquire_next_swapchain_image(rd);
        let target = rg.register_texture(rd.swapchain.textures[swapchain_image_index as usize]);

        // Draw ImGUI
        if let Some(imgui) = imgui {
            let clear_color = vk::ClearColorValue {
                float32: [0.5, 0.5, 0.7, 1.0],
            };
            self.imgui_pass.add(
                &mut rg,
                rd,
                &mut self.upload_context,
                target,
                imgui,
                Some(clear_color),
            )
        }

        // Output
        rg.present(target);

        // Execute render graph
        {
            let vk_command_buffer = self.streamlined.wait_and_reset_command_buffer(rd);

            let command_buffer = CommandBuffer::new(rd, vk_command_buffer);
            rd.begin_command_buffer(command_buffer.command_buffer);

            rg.execute(rd, &command_buffer, shaders, &mut self.render_graph_cache);

            rd.end_command_buffer(command_buffer.command_buffer);
        }

        // Present
        self.streamlined
            .wait_and_submit_and_present(rd, swapchain_image_index);
    }
}

fn main() {
    violet::app::run_with_renderloop::<ImGUIDemoRenderLoop>();
}
