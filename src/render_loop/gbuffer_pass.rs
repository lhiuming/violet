use ash::vk;
use glam::UVec2;

use crate::{
    command_buffer::StencilOps,
    render_device::{texture::TextureUsage, RenderDevice, Texture, TextureDesc},
    render_graph::*,
    render_scene::RenderScene,
    shader::PushConstantsBuilder,
};

pub static GBUFFER_COLOR_ARRAY_LEN: u32 = 4;

pub struct GBuffer {
    pub depth: RGHandle<Texture>,
    pub color: RGHandle<Texture>,
    pub color_clear: vk::ClearColorValue,
    pub size: UVec2,
}

pub fn create_gbuffer_textures(rg: &mut RenderGraphBuilder, size: UVec2) -> GBuffer {
    let depth = rg.create_texutre(TextureDesc::new_2d(
        size.x,
        size.y,
        vk::Format::D24_UNORM_S8_UINT,
        TextureUsage::new().depth_stencil().sampled().into(),
    ));
    let color = rg.create_texutre(TextureDesc::new_2d_array(
        size.x,
        size.y,
        GBUFFER_COLOR_ARRAY_LEN,
        vk::Format::R32_UINT,
        TextureUsage::new().color().sampled().into(),
    ));
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

pub fn add_gbuffer_pass<'a, 'render>(
    rg: &'a mut RenderGraphBuilder<'render>,
    rd: &RenderDevice,
    scene: &'render RenderScene,
    gbuffer: &GBuffer,
) {
    let viewport_extent = rd.swapchain.extent;
    let color_targets = (0..GBUFFER_COLOR_ARRAY_LEN)
        .map(|layer| ColorTarget {
            tex: gbuffer.color,
            layer,
            load_op: ColorLoadOp::Clear(gbuffer.color_clear),
            ..Default::default()
        })
        .collect::<Vec<_>>();

    rg.new_graphics("GBuffer_Gen")
        .vertex_shader_with_ep("mesh_gbuffer.hlsl", "vs_main")
        .pixel_shader_with_ep("mesh_gbuffer.hlsl", "ps_main")
        .color_targets(&color_targets)
        .depth_stencil(DepthStencilTarget {
            tex: gbuffer.depth,
            aspect: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
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
            cb.set_stencil_test_enable(false); // TODO move back to render pass desc
            let face_mask = vk::StencilFaceFlags::FRONT_AND_BACK;
            let stencil_ops = StencilOps::write_on_pass(vk::CompareOp::ALWAYS);
            cb.set_stencil_op(face_mask, stencil_ops);
            cb.set_stencil_write_mask(face_mask, 0x01);
            cb.set_stencil_reference(face_mask, 0xFF);

            // Draw meshs
            cb.bind_index_buffer(scene.index_buffer.buffer.handle, 0, vk::IndexType::UINT16);
            cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            for instance in &scene.instances {
                // Fetch geometry group instance
                let model_xform_rows = instance.transform.transpose().to_cols_array();
                let normal_xform_rows = instance.normal_transform.transpose().to_cols_array();
                let geometry_group =
                    &scene.geometry_group_params[instance.geometry_group_index as usize];
                // Draw the meshes in the group, with instanced transform
                let beg = geometry_group.geometry_index_offset as usize;
                let end = beg + geometry_group.geometry_count as usize;
                for mesh_index in beg..end {
                    let mesh = &scene.mesh_params[mesh_index];
                    // PushConstant for everything per-draw
                    let mesh_index = mesh_index as u32;
                    let constants = PushConstantsBuilder::new()
                        .push_slice(&model_xform_rows[0..12])
                        .push_slice(&normal_xform_rows[0..12])
                        .push(&mesh_index)
                        .push(&mesh.material_index);
                    cb.push_constants(
                        pipeline.layout,
                        pipeline.push_constant_ranges[0].stage_flags, // TODO
                        0,
                        &constants.build(),
                    );
                    // Dispatch
                    cb.draw_indexed(mesh.index_count, 1, mesh.index_offset, 0, 0);
                }
            }
        });
}
