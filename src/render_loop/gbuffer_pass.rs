use ash::vk;

use crate::{
    command_buffer::StencilOps,
    render_device::{RenderDevice, Texture, TextureDesc, TextureView},
    render_graph::*,
    render_scene::RenderScene,
    shader::PushConstantsBuilder,
};

type GBufferTexture = (RGHandle<Texture>, RGHandle<TextureView>);

pub struct GBuffer {
    pub depth: GBufferTexture,
    pub color: GBufferTexture,
    pub color_clear: vk::ClearColorValue,
    pub size: vk::Extent2D,
}

pub fn create_gbuffer_textures(rg: &mut RenderGraphBuilder, size: vk::Extent2D) -> GBuffer {
    // helper
    let mut create_gbuffer = |format: vk::Format| {
        let is_depth = crate::render_device::format_has_depth(format);
        let usage: vk::ImageUsageFlags = if is_depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } | vk::ImageUsageFlags::SAMPLED;
        let desc = TextureDesc::new_2d(size.width, size.height, format, usage);
        let texture = rg.create_texutre(desc);
        let view = rg.create_texture_view(texture, None);

        (texture, view)
    };

    // define
    let depth = create_gbuffer(vk::Format::D24_UNORM_S8_UINT);
    let color = create_gbuffer(vk::Format::R32G32B32A32_UINT);
    let color_clear = vk::ClearColorValue {
        uint32: [0, 0, 0, 0],
    };

    GBuffer {
        depth,
        color,
        color_clear,
        size,
    }
}

pub fn add_gbuffer_pass<'a>(
    rg: &mut RenderGraphBuilder<'a>,
    rd: &RenderDevice,
    scene: &'a RenderScene,
    gbuffer: &GBuffer,
) {
    let viewport_extent = rd.swapchain.extent;
    rg.new_graphics("GBuffer_Gen")
        .vertex_shader_with_ep("mesh_gbuffer.hlsl", "vs_main")
        .pixel_shader_with_ep("mesh_gbuffer.hlsl", "ps_main")
        .color_targets(&[ColorTarget {
            view: gbuffer.color.1,
            load_op: ColorLoadOp::Clear(gbuffer.color_clear),
        }])
        .depth_stencil(DepthStencilTarget {
            view: gbuffer.depth.1,
            load_op: DepthLoadOp::Clear(vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            }),
            store_op: vk::AttachmentStoreOp::STORE,
        })
        .render(move |cb, pipeline| {
            // set up raster state
            cb.set_viewport_0(vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: viewport_extent.width as f32,
                height: viewport_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            });
            cb.set_scissor_0(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: viewport_extent,
            });
            cb.set_depth_test_enable(true);
            cb.set_depth_write_enable(true);
            cb.set_stencil_test_enable(true);
            let face_mask = vk::StencilFaceFlags::FRONT_AND_BACK;
            let stencil_ops = StencilOps::write_on_pass(vk::CompareOp::ALWAYS);
            cb.set_stencil_op(face_mask, stencil_ops);
            cb.set_stencil_write_mask(face_mask, 0x01);
            cb.set_stencil_reference(face_mask, 0xFF);

            // Draw meshs
            cb.bind_index_buffer(scene.index_buffer.buffer.handle, 0, vk::IndexType::UINT16);
            cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            for (mesh_index, mesh_params) in scene.mesh_params.iter().enumerate() {
                // PushConstant for everything per-draw
                let model_xform = glam::Mat4::IDENTITY; // Not used for now
                let mesh_index = mesh_index as u32;
                let constants = PushConstantsBuilder::new()
                    .push(&model_xform.to_cols_array())
                    .push(&mesh_index)
                    .push(&mesh_params.material_index);
                cb.push_constants(
                    pipeline.layout,
                    pipeline.push_constant_ranges[0].stage_flags, // TODO
                    0,
                    &constants.build(),
                );

                cb.draw_indexed(mesh_params.index_count, 1, mesh_params.index_offset, 0, 0);
            }
        });
}
