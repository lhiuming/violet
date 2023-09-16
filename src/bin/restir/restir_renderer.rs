use ash::vk;
use glam::UVec2;

use violet::{
    command_buffer::StencilOps,
    imgui::Ui,
    render_device::{Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureViewDesc},
    render_graph::*,
    render_loop::{div_round_up, div_round_up_uvec2, gbuffer_pass::*},
    render_scene::RenderScene,
};

use crate::restir_render_loop::SceneRendererInput;

struct SampleGenPass {
    prev_reservoir_buffer: Option<RGTemporal<Buffer>>,
    prev_indirect_diffuse_texture: Option<RGTemporal<Texture>>,
}

impl SampleGenPass {
    fn new() -> Self {
        Self {
            prev_reservoir_buffer: None,
            prev_indirect_diffuse_texture: None,
        }
    }
}

struct TAAPass {
    prev_color: Option<RGTemporal<Texture>>,
    prev_depth: Option<RGTemporal<Texture>>,
}

impl TAAPass {
    fn new() -> Self {
        Self {
            prev_color: None,
            prev_depth: None,
        }
    }
}

pub struct RestirConfig {
    taa: bool,
}
impl Default for RestirConfig {
    fn default() -> Self {
        Self { taa: true }
    }
}

// Render the scene using ReSTIR lighting
pub struct RestirRenderer {
    config: RestirConfig,

    sample_gen: SampleGenPass,
    taa: TAAPass,
}

impl RestirRenderer {
    pub fn new() -> Self {
        Self {
            config: Default::default(),
            sample_gen: SampleGenPass::new(),
            taa: TAAPass::new(),
        }
    }

    pub fn taa(&self) -> bool {
        self.config.taa
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        let config = &mut self.config;
        ui.heading("restir renderer");
        ui.checkbox(&mut config.taa, "TAA");
    }

    pub fn add_passes<'render>(
        &mut self,
        rd: &mut RenderDevice,
        rg: &mut RenderGraphBuilder<'render>,
        scene: &'render RenderScene,
        input: SceneRendererInput<'_>,
    ) -> RGHandle<Texture> {
        let frame_index = input.frame_index;
        let main_size = input.main_size;
        let default_res = input.default_res;
        let skycube = input.sky_cube;
        let scene_tlas = input.scene_tlas;

        let main_size_flat = main_size.x * main_size.y;

        // Create GBuffer
        let gbuffer = create_gbuffer_textures(rg, main_size);

        // Pass: GBuffer
        add_gbuffer_pass(rg, rd, scene, &gbuffer);

        // depth buffer from last frame
        let has_prev_depth = self.taa.prev_depth.is_some();
        let prev_depth = match self.taa.prev_depth.take() {
            Some(tex) => {
                let tex = rg.convert_to_transient(tex);
                rg.create_texture_view(tex, None)
            }
            None => rg.register_texture_view(default_res.dummy_texture.1),
        };

        // indirect diffuse from last frame
        let has_prev_indirect_diffuse = self.sample_gen.prev_indirect_diffuse_texture.is_some();
        let prev_indirect_diffuse_texture =
            match self.sample_gen.prev_indirect_diffuse_texture.take() {
                Some(tex) => {
                    let tex = rg.convert_to_transient(tex);
                    rg.create_texture_view(tex, None)
                }
                None => rg.register_texture_view(default_res.dummy_texture.1),
            };

        // resercoir buffer from last frame
        let has_prev_reservoir = self.sample_gen.prev_reservoir_buffer.is_some();
        let prev_reservoir_buffer = match self.sample_gen.prev_reservoir_buffer {
            Some(buffer) => rg.convert_to_transient(buffer),
            None => rg.register_buffer(default_res.dummy_buffer),
        };

        let is_validation_frame = has_prev_reservoir && ((frame_index % 6) == 0);

        // Pass: Sample Generation (and temporal reusing)
        let has_new_sample;
        let new_sample_buffer;
        if !is_validation_frame {
            has_new_sample = 1u32;
            new_sample_buffer =
                rg.create_buffer(BufferDesc::compute(main_size_flat as u64 * 5 * 4 * 4));

            let has_prev_indirect = (has_prev_depth && has_prev_indirect_diffuse) as u32;

            rg.new_raytracing("Sample Gen")
                .raygen_shader("restir/sample_gen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth", prev_depth)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .buffer("rw_new_sample_buffer", new_sample_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&input.frame_index)
                .push_constant(&has_prev_indirect)
                .dimension(main_size.x, main_size.y, 1);
        }
        // Pass: Sample Validation
        else {
            has_new_sample = 0u32;
            new_sample_buffer = rg.register_buffer(default_res.dummy_buffer);

            rg.new_raytracing("Sample Validate")
                .raygen_shader("restir/sample_validate.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth", prev_depth)
                .rw_buffer("rw_prev_reservoir_buffer", prev_reservoir_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .dimension(main_size.x, main_size.y, 1);
        }

        let temporal_reservoir_buffer =
            rg.create_buffer(BufferDesc::compute(main_size_flat as u64 * 24 * 4));

        // Pass: Temporal Resampling
        rg.new_compute("Temporal Resample")
            .compute_shader("restir/temporal_resample.hlsl")
            .texture("prev_gbuffer_depth", prev_depth)
            .texture("gbuffer_depth", gbuffer.depth.1)
            .texture("gbuffer_color", gbuffer.color.1)
            .buffer("new_sample_buffer", new_sample_buffer)
            .buffer("prev_reservoir_buffer", prev_reservoir_buffer)
            .buffer("rw_temporal_reservoir_buffer", temporal_reservoir_buffer)
            .push_constant(&input.frame_index)
            .push_constant(&(has_prev_reservoir as u32))
            .push_constant(&has_new_sample)
            .group_count_uvec3(div_round_up_uvec2(main_size, UVec2::new(8, 4)).extend(1));

        // Pass: Spatial Resampling
        let indirect_diffuse;
        {
            indirect_diffuse = {
                let desc = TextureDesc::new_2d(
                    main_size.x,
                    main_size.y,
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT // sky rendering
                    | vk::ImageUsageFlags::STORAGE // compute lighting
                    | vk::ImageUsageFlags::SAMPLED, // taa
                );
                let texture = rg.create_texutre(desc);
                let view = rg.create_texture_view(texture, None);
                (texture, view)
            };

            let reservoir_buffer_desc = rg.get_buffer_desc(temporal_reservoir_buffer);
            let spatial_reservoir_buffer = rg.create_buffer(*reservoir_buffer_desc);

            rg.new_compute("Spatial Resampling")
                .compute_shader("restir/spatial_resampling.hlsl")
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .buffer("temporal_reservoir_buffer", temporal_reservoir_buffer)
                .rw_buffer("rw_spatial_reservoir_buffer", spatial_reservoir_buffer)
                .rw_texture("rw_lighting_texture", indirect_diffuse.1)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&frame_index)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );

            self.sample_gen
                .prev_reservoir_buffer
                .replace(rg.convert_to_temporal(spatial_reservoir_buffer));
            self.sample_gen
                .prev_indirect_diffuse_texture
                .replace(rg.convert_to_temporal(indirect_diffuse.0));
        }

        // Pass: Raytraced Shadow
        let raytraced_shadow_mask = {
            let tex = rg.create_texutre(TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ));
            (tex, rg.create_texture_view(tex, None))
        };
        {
            rg.new_raytracing("Raytraced Shadow")
                .raygen_shader("raytraced_shadow.hlsl")
                .miss_shader("raytrace/shadow.rmiss.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .rw_texture("rw_shadow", raytraced_shadow_mask.1)
                .dimension(main_size.x, main_size.y, 1);
        }

        let scene_color = {
            let desc = TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT // sky rendering
                    | vk::ImageUsageFlags::STORAGE // compute lighting
                    | vk::ImageUsageFlags::SAMPLED, // taa
            );
            let texture = rg.create_texutre(desc);
            let view = rg.create_texture_view(texture, None);
            (texture, view)
        };

        // Pass: Draw sky
        {
            let stencil = rg.create_texture_view(
                gbuffer.depth.0,
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::D24_UNORM_S8_UINT,
                    aspect: vk::ImageAspectFlags::STENCIL,
                    ..Default::default()
                }),
            );

            rg.new_graphics("Sky")
                .vertex_shader_with_ep("sky_vsps.hlsl", "vs_main")
                .pixel_shader_with_ep("sky_vsps.hlsl", "ps_main")
                .texture("skycube", skycube)
                .color_targets(&[ColorTarget {
                    view: scene_color.1,
                    load_op: ColorLoadOp::DontCare,
                }])
                .depth_stencil(DepthStencilTarget {
                    view: stencil,
                    load_op: DepthLoadOp::Load,
                    store_op: vk::AttachmentStoreOp::NONE,
                })
                .render(move |cb, pipeline| {
                    // Set up raster states
                    cb.set_depth_test_enable(false);
                    cb.set_depth_write_enable(false);
                    cb.set_stencil_test_enable(true);
                    let face_mask = vk::StencilFaceFlags::FRONT_AND_BACK;
                    let stencil_op = StencilOps::only_compare(vk::CompareOp::EQUAL);
                    cb.set_stencil_op(face_mask, stencil_op);
                    cb.set_stencil_compare_mask(face_mask, 0x01);
                    cb.set_stencil_reference(face_mask, 0x00);

                    // Draw
                    cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    cb.draw(3, 1, 0, 0);
                });
        }

        // Pass: Final Lighting (Combine)
        // TODO pixel shader to utilize the stencil buffer
        {
            rg.new_compute("Combined Lighting")
                .compute_shader("restir/final_lighting.hlsl")
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("shadow_mask_buffer", raytraced_shadow_mask.1)
                .texture("indirect_diffuse_buffer", indirect_diffuse.1)
                .rw_texture("rw_color_buffer", scene_color.1)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        let has_prev_color = self.taa.prev_color.is_some() as u32;
        let prev_color = match self.taa.prev_color.take() {
            Some(tex) => {
                let tex = rg.convert_to_transient(tex);
                rg.create_texture_view(tex, None)
            }
            None => rg.register_texture_view(default_res.dummy_texture.1),
        };

        // Pass: TAA
        let post_taa_color;
        if self.config.taa {
            post_taa_color = {
                let desc = TextureDesc::new_2d(
                    main_size.x,
                    main_size.y,
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::ImageUsageFlags::STORAGE // compute TAA
                    | vk::ImageUsageFlags::SAMPLED, // history and post
                );
                let texture = rg.create_texutre(desc);
                let view = rg.create_texture_view(texture, None);
                (texture, view)
            };

            rg.new_compute("Temporal AA")
                .compute_shader("temporal_aa.hlsl")
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("source_texture", scene_color.1)
                .texture("history_texture", prev_color)
                .rw_texture("rw_target", post_taa_color.1)
                .push_constant(&has_prev_color)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        } else {
            post_taa_color = scene_color;
        }

        // Cache scene buffer for next frame (TAA, temporal restir, etc.)
        self.taa
            .prev_color
            .replace(rg.convert_to_temporal(post_taa_color.0));
        self.taa
            .prev_depth
            .replace(rg.convert_to_temporal(gbuffer.depth.0));

        post_taa_color.0
    }
}
