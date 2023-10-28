use ash::vk;
use glam::UVec2;

use violet::{
    command_buffer::StencilOps,
    imgui::Ui,
    render_device::{
        texture::TextureUsage, Buffer, BufferDesc, RenderDevice, Texture, TextureDesc,
        TextureViewDesc,
    },
    render_graph::*,
    render_loop::{
        div_round_up, div_round_up_uvec2, gbuffer_pass::*, util_passes::clear_buffer, DivRoundUp,
    },
    render_scene::RenderScene,
};

use crate::restir_render_loop::SceneRendererInput;

static HG_CACHE_MAX_NUM_CELLS: u32 = 65536 * 16;

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

#[derive(Debug, PartialEq, Clone, Copy)]
enum DebugView {
    None,
    AO,
    AOFiltered,
    IndDiff,
    IndDiffFiltered,
    IndSpec,
    IndSpecFiltered,
    HashGridView,
}

pub struct RestirConfig {
    taa: bool,
    ao_radius: f32,
    ind_diff_validate: bool,
    ind_diff_taa: bool,
    ind_spec_validate: bool,
    ind_spec_taa: bool,
    debug_view: DebugView,
}
impl Default for RestirConfig {
    fn default() -> Self {
        Self {
            taa: true,
            ao_radius: 2.0,
            ind_diff_validate: true,
            ind_diff_taa: true,
            ind_spec_validate: true,
            ind_spec_taa: true,
            debug_view: DebugView::None,
        }
    }
}

struct IndirectSpecularTextures {
    reservoir: RGTemporal<Texture>,
    hit_pos: RGTemporal<Texture>,
    hit_radiance: RGTemporal<Texture>,
    lighting: RGTemporal<Texture>,
}

struct HashGridCacheResrouces<BufferHandle> {
    storage: BufferHandle,
    decay: BufferHandle,
}

// Render the scene using ReSTIR lighting
pub struct RestirRenderer {
    config: RestirConfig,
    taa: TAAPass,

    prev_diffuse_reservoir_buffer: Option<RGTemporal<Buffer>>,
    prev_indirect_diffuse_texture: Option<RGTemporal<Texture>>,

    prev_indirect_specular: Option<IndirectSpecularTextures>,
    prev_gbuffer_color: Option<RGTemporal<Texture>>,

    ao_history: Option<RGTemporal<Texture>>,

    hash_grid_cache_history: Option<HashGridCacheResrouces<RGTemporal<Buffer>>>,
    clear_hash_grid_cache: bool,
}

impl RestirRenderer {
    pub fn new() -> Self {
        Self {
            config: Default::default(),
            taa: TAAPass::new(),

            prev_diffuse_reservoir_buffer: None,
            prev_indirect_diffuse_texture: None,

            prev_indirect_specular: None,
            prev_gbuffer_color: None,

            ao_history: None,
            hash_grid_cache_history: None,
            clear_hash_grid_cache: false,
        }
    }

    pub fn taa(&self) -> bool {
        self.config.taa
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        let config = &mut self.config;
        ui.heading("RESTIR RENDERER");
        ui.checkbox(&mut config.taa, "taa");
        ui.checkbox(&mut config.ind_diff_validate, "ind.diff.s.validate");
        ui.checkbox(&mut config.ind_diff_taa, "ind.diff.taa");
        ui.checkbox(&mut config.ind_spec_validate, "ind.spec.s.validate");
        ui.checkbox(&mut config.ind_spec_taa, "ind.spec.taa");
        ui.add(egui::Slider::new(&mut config.ao_radius, 0.0..=5.0).text("ao radius"));

        egui::ComboBox::from_label("debug view")
            .selected_text(format!("{:?}", config.debug_view))
            .show_ui(ui, |ui| {
                let mut item = |view: DebugView| {
                    ui.selectable_value(&mut config.debug_view, view, format!("{:?}", view));
                    ui.set_min_width(128.0);
                };
                item(DebugView::None);
                item(DebugView::AO);
                item(DebugView::AOFiltered);
                item(DebugView::IndDiff);
                item(DebugView::IndDiffFiltered);
                item(DebugView::IndSpec);
                item(DebugView::IndSpecFiltered);
                item(DebugView::HashGridView);
            });

        if ui.button("Clear Hash Grid Cache").clicked() {
            self.clear_hash_grid_cache = true;
        }
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

        // Get Hash Grid Cache history
        let hg_cache_decay_format = vk::Format::R8_UINT;
        let hg_cache = match self.hash_grid_cache_history.take() {
            Some(temporal) => HashGridCacheResrouces {
                storage: rg.convert_to_transient(temporal.storage),
                decay: rg.convert_to_transient(temporal.decay),
            },
            None => {
                self.clear_hash_grid_cache = true; // reuse the flag for initialization
                HashGridCacheResrouces {
                    storage: rg
                        .create_buffer(BufferDesc::compute(HG_CACHE_MAX_NUM_CELLS as u64 * 4 * 4)),
                    decay: rg.create_buffer(BufferDesc::compute_with_usage(
                        HG_CACHE_MAX_NUM_CELLS as u64,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                    )),
                }
            }
        };
        if self.clear_hash_grid_cache {
            clear_buffer(rd, rg, hg_cache.storage, "Clear HashGridCache Storage");
            clear_buffer(rd, rg, hg_cache.decay, "Clear HashGridCache Decay");
            self.clear_hash_grid_cache = false;
        }

        // Pass: HashGridCache decay
        rg.new_compute("HashGridCache Decay")
            .compute_shader("restir/hash_grid_cache_decay.hlsl")
            .rw_buffer("rw_decay_buffer", hg_cache.decay)
            .rw_buffer("rw_storage_buffer", hg_cache.storage)
            // NOTE: see NUM_CELLS_PER_GROUP in shader
            .group_count(div_round_up(HG_CACHE_MAX_NUM_CELLS, 1024), 1, 1);

        // depth buffer from last frame
        let has_prev_depth = self.taa.prev_depth.is_some();
        let prev_depth = match self.taa.prev_depth.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_texture),
        };

        // indirect diffuse from last frame
        let has_prev_indirect_diffuse = self.prev_indirect_diffuse_texture.is_some();
        let prev_indirect_diffuse_texture = match self.prev_indirect_diffuse_texture.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_texture),
        };

        // indirect diffuse reservior buffer from last frame
        let has_prev_diffuse_reservoir = self.prev_diffuse_reservoir_buffer.is_some();
        let prev_diffuse_reservoir_buffer = match self.prev_diffuse_reservoir_buffer {
            Some(buffer) => rg.convert_to_transient(buffer),
            None => rg.register_buffer(default_res.dummy_buffer),
        };

        let is_indirect_diffuse_validation_frame = {
            let has_buffers = has_prev_depth && has_prev_diffuse_reservoir;
            has_buffers && ((frame_index % 6) == 0) && self.config.ind_diff_validate
        };

        // Pass: Indirect Diffuse ReSTIR Sample Generation
        let indirect_diffuse_has_new_sample;
        let indirect_diffuse_new_sample_buffer;
        if !is_indirect_diffuse_validation_frame {
            indirect_diffuse_has_new_sample = 1u32;
            indirect_diffuse_new_sample_buffer =
                rg.create_buffer(BufferDesc::compute(main_size_flat as u64 * 5 * 4 * 4));

            let has_prev_frame = (has_prev_depth && has_prev_indirect_diffuse) as u32;

            rg.new_raytracing("Ind. Diff. Sample Gen.")
                .raygen_shader("restir/ind_diff_sample_gen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture_view("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth_texture", prev_depth)
                .texture_view("gbuffer_depth", gbuffer.depth.1)
                .texture_view("gbuffer_color", gbuffer.color.1)
                .buffer("rw_new_sample_buffer", indirect_diffuse_new_sample_buffer)
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .rw_buffer_with_format(
                    "rw_hash_grid_decay_buffer",
                    hg_cache.decay,
                    hg_cache_decay_format,
                )
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&input.frame_index)
                .push_constant(&has_prev_frame)
                .dimension(main_size.x, main_size.y, 1);
        }
        // Pass: Indirect Diffuse Sample Validation
        else {
            indirect_diffuse_has_new_sample = 0u32;
            indirect_diffuse_new_sample_buffer = rg.register_buffer(default_res.dummy_buffer);

            rg.new_raytracing("Ind. Diff. Sample Validate")
                .raygen_shader("restir/ind_diff_sample_validate.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture_view("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth_texture", prev_depth)
                .rw_buffer("rw_prev_reservoir_buffer", prev_diffuse_reservoir_buffer)
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .rw_buffer_with_format(
                    "rw_hash_grid_decay_buffer",
                    hg_cache.decay,
                    hg_cache_decay_format,
                )
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .dimension(main_size.x, main_size.y, 1);
        }

        let indirect_diffuse_temporal_reservoir_buffer =
            rg.create_buffer(BufferDesc::compute(main_size_flat as u64 * 24 * 4));

        let has_prev_ao = self.ao_history.is_some() as u32;
        let prev_ao_texture = match self.ao_history.take() {
            Some(temporal) => rg.convert_to_transient(temporal),
            None => rg.register_texture(default_res.dummy_texture),
        };
        let curr_ao_texture = rg.create_texutre(TextureDesc::new_2d(
            main_size.x,
            main_size.y,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        ));

        // Pass: AO Gen and Temporal Filtering
        rg.new_compute("RayTraced AO Gen.")
            .compute_shader("restir/raytraced_ao_gen.hlsl")
            .buffer("new_sample_buffer", indirect_diffuse_new_sample_buffer)
            .texture_view("depth_buffer", gbuffer.depth.1)
            .texture("prev_ao_texture", prev_ao_texture)
            .rw_texture("rw_ao_texture", curr_ao_texture)
            .push_constant::<u32>(&indirect_diffuse_has_new_sample)
            .push_constant::<u32>(&has_prev_ao)
            .push_constant::<f32>(&self.config.ao_radius)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        let filtered_ao_texture = rg.create_texutre(TextureDesc::new_2d(
            main_size.x,
            main_size.y,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        ));

        // Pass: AO Spatial Filtering
        rg.new_compute("RayTraced AO Filter")
            .compute_shader("restir/raytraced_ao_filter.hlsl")
            .texture_view("depth_buffer", gbuffer.depth.1)
            .texture("ao_texture", curr_ao_texture)
            .rw_texture("rw_filtered_ao_texture", filtered_ao_texture)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        self.ao_history
            .replace(rg.convert_to_temporal(curr_ao_texture));

        // Pass: Indirect Diffuse Temporal Resampling
        rg.new_compute("Ind. Diff. Temporal Resample")
            .compute_shader("restir/ind_diff_temporal_resample.hlsl")
            .texture("prev_gbuffer_depth", prev_depth)
            .texture_view("gbuffer_depth", gbuffer.depth.1)
            .buffer("new_sample_buffer", indirect_diffuse_new_sample_buffer)
            .buffer("prev_reservoir_buffer", prev_diffuse_reservoir_buffer)
            .buffer(
                "rw_temporal_reservoir_buffer",
                indirect_diffuse_temporal_reservoir_buffer,
            )
            .push_constant(&input.frame_index)
            .push_constant(&(has_prev_diffuse_reservoir as u32))
            .push_constant(&indirect_diffuse_has_new_sample)
            .group_count_uvec3(div_round_up_uvec2(main_size, UVec2::new(8, 4)).extend(1));

        let indirect_diffuse = rg.create_texutre(TextureDesc::new_2d(
            main_size.x,
            main_size.y,
            vk::Format::B10G11R11_UFLOAT_PACK32,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        ));

        // Pass: Indirect Diffuse Spatial Resampling
        {
            let reservoir_buffer_desc =
                rg.get_buffer_desc(indirect_diffuse_temporal_reservoir_buffer);
            let spatial_reservoir_buffer = rg.create_buffer(*reservoir_buffer_desc);

            rg.new_compute("Ind. Diff. Spatial Resample")
                .compute_shader("restir/ind_diff_spatial_resample.hlsl")
                .texture_view("gbuffer_depth", gbuffer.depth.1)
                .texture_view("gbuffer_color", gbuffer.color.1)
                .texture("ao_texture", filtered_ao_texture)
                .buffer(
                    "temporal_reservoir_buffer",
                    indirect_diffuse_temporal_reservoir_buffer,
                )
                .rw_buffer("rw_spatial_reservoir_buffer", spatial_reservoir_buffer)
                .rw_texture("rw_lighting_texture", indirect_diffuse)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&frame_index)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );

            // NOTE: spatial reservoir is feedback to temporal reservoir (next frame)
            self.prev_diffuse_reservoir_buffer
                .replace(rg.convert_to_temporal(spatial_reservoir_buffer));
        }

        // Pass: Indirect Diffuse Temporal Filtering
        let filtered_indirect_diffuse;
        if has_prev_indirect_diffuse && self.config.ind_diff_taa {
            let desc = rg.get_texture_desc(indirect_diffuse);
            filtered_indirect_diffuse = rg.create_texutre(*desc);

            rg.new_compute("Ind. Diff. Temporal Filter")
                .compute_shader("restir/ind_diff_temporal_filter.hlsl")
                .texture("depth_buffer", gbuffer.depth.0)
                .texture("history_texture", prev_indirect_diffuse_texture)
                .texture("source_texture", indirect_diffuse)
                .rw_texture("rw_filtered_texture", filtered_indirect_diffuse)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        } else {
            filtered_indirect_diffuse = indirect_diffuse;
        }

        self.prev_indirect_diffuse_texture
            .replace(rg.convert_to_temporal(filtered_indirect_diffuse));

        let mut ind_spec_new_tex = |format: vk::Format| {
            let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
            rg.create_texutre(TextureDesc::new_2d(main_size.x, main_size.y, format, usage))
        };

        let ind_spec_hit_pos = ind_spec_new_tex(vk::Format::R32G32B32A32_SFLOAT);
        let ind_spec_hit_normal = ind_spec_new_tex(vk::Format::R32_UINT);
        let ind_spec_hit_radiance = ind_spec_new_tex(vk::Format::B10G11R11_UFLOAT_PACK32);

        let ind_spec_temporal_reservoir = ind_spec_new_tex(vk::Format::R32G32_UINT);
        let indirect_specular = ind_spec_new_tex(vk::Format::B10G11R11_UFLOAT_PACK32);

        // Pass: Indirect Specular Sample Genenration
        {
            let has_prev_frame = (has_prev_depth && has_prev_indirect_diffuse) as u32;

            rg.new_raytracing("Ind. Spec. Sample Gen.")
                .raygen_shader("restir/ind_spec_sample_gen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture_view("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth_texture", prev_depth)
                .texture("gbuffer_depth", gbuffer.depth.0)
                .texture("gbuffer_color", gbuffer.color.0)
                //.rw_texture("rw_origin_pos_texture", ind_spec_origin_pos)
                .rw_texture("rw_hit_pos_texture", ind_spec_hit_pos)
                .rw_texture("rw_hit_normal_texture", ind_spec_hit_normal)
                .rw_texture("rw_hit_radiance_texture", ind_spec_hit_radiance)
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .rw_buffer_with_format(
                    "rw_hash_grid_decay_buffer",
                    hg_cache.decay,
                    hg_cache_decay_format,
                )
                .push_constant::<u32>(&frame_index)
                .push_constant::<u32>(&has_prev_frame)
                .dimension(main_size.x, main_size.y, 1);
        }

        let has_prev_ind_spec = self.prev_indirect_specular.is_some();
        let prev_ind_spec_reservoir;
        let prev_ind_spec_hit_pos;
        let prev_ind_spec_hit_radiance;
        let prev_ind_spec;
        match self.prev_indirect_specular.take() {
            Some(ind_spec) => {
                prev_ind_spec_reservoir = rg.convert_to_transient(ind_spec.reservoir);
                prev_ind_spec_hit_pos = rg.convert_to_transient(ind_spec.hit_pos);
                prev_ind_spec_hit_radiance = rg.convert_to_transient(ind_spec.hit_radiance);
                prev_ind_spec = rg.convert_to_transient(ind_spec.lighting);
            }
            None => {
                prev_ind_spec_reservoir = rg.register_texture(default_res.dummy_uint_texture);
                prev_ind_spec_hit_pos = rg.register_texture(default_res.dummy_texture);
                prev_ind_spec_hit_radiance = rg.register_texture(default_res.dummy_texture);
                prev_ind_spec = rg.register_texture(default_res.dummy_texture);
            }
        }

        // Pass: Indirect Specular Sample Validation
        let has_spec_validate_textures =
            has_prev_depth && has_prev_indirect_diffuse && has_prev_ind_spec;
        if self.config.ind_spec_validate && has_spec_validate_textures {
            rg.new_raytracing("Ind. Spec. Sample Validate")
                .raygen_shader("restir/ind_spec_sample_validate.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture_view("skycube", skycube)
                .texture(
                    "prev_indirect_diffuse_texture",
                    prev_indirect_diffuse_texture,
                )
                .texture("prev_depth_texture", prev_depth)
                .rw_texture("rw_prev_reservoir_texture", prev_ind_spec_reservoir)
                .rw_texture("rw_prev_hit_pos_texture", prev_ind_spec_hit_pos)
                .rw_texture("rw_prev_hit_radiance_texture", prev_ind_spec_hit_radiance)
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .rw_buffer_with_format(
                    "rw_hash_grid_decay_buffer",
                    hg_cache.decay,
                    hg_cache_decay_format,
                )
                .dimension(main_size.x, main_size.y, 1);
        }

        let has_prev_gbuffer_color = self.prev_gbuffer_color.is_some();
        let prev_gbuffer_color = match self.prev_gbuffer_color.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_uint_texture),
        };

        // Pass: Indirect Specular Temporal Resampling
        {
            let has_prev_frame =
                (has_prev_depth && has_prev_ind_spec && has_prev_gbuffer_color) as u32;

            rg.new_compute("Ind. Spec. Temporal Resample")
                .compute_shader("restir/ind_spec_temporal_resample.hlsl")
                .texture("prev_gbuffer_depth", prev_depth)
                .texture("prev_gbuffer_color", prev_gbuffer_color)
                .texture("gbuffer_depth", gbuffer.depth.0)
                .texture("gbuffer_color", gbuffer.color.0)
                .texture("prev_reservoir_texture", prev_ind_spec_reservoir)
                .texture("prev_hit_pos_texture", prev_ind_spec_hit_pos)
                .texture("prev_hit_radiance_texture", prev_ind_spec_hit_radiance)
                .rw_texture("rw_reservoir_texture", ind_spec_temporal_reservoir)
                .rw_texture("rw_hit_pos_texture", ind_spec_hit_pos)
                .rw_texture("rw_hit_radiance_texture", ind_spec_hit_radiance)
                .push_constant::<u32>(&frame_index)
                .push_constant::<u32>(&has_prev_frame)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        // Pass: Indirect Specular Spatial Resampling
        rg.new_compute("Ind. Spec. Spaital Resample")
            .compute_shader("restir/ind_spec_spatial_resample.hlsl")
            .texture("gbuffer_depth", gbuffer.depth.0)
            .texture("gbuffer_color", gbuffer.color.0)
            .texture("reservoir_texture", ind_spec_temporal_reservoir)
            .texture("hit_pos_texture", ind_spec_hit_pos)
            .texture("hit_radiance_texture", ind_spec_hit_radiance)
            .rw_texture("rw_lighting_texture", indirect_specular)
            .push_constant::<u32>(&frame_index)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        // Pass: Indirect Specular Temporal Filtering
        let filtered_indirect_specular;
        if has_prev_ind_spec && self.config.ind_spec_taa {
            let desc = rg.get_texture_desc(indirect_specular);
            filtered_indirect_specular = rg.create_texutre(*desc);

            rg.new_compute("Ind. Spec. Temporal Filter")
                .compute_shader("restir/ind_spec_temporal_filter.hlsl")
                .texture("depth_buffer", gbuffer.depth.0)
                .texture("history_texture", prev_ind_spec)
                .texture("source_texture", indirect_specular)
                .rw_texture("rw_filtered_texture", filtered_indirect_specular)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        } else {
            filtered_indirect_specular = indirect_specular;
        }

        self.prev_indirect_specular
            .replace(IndirectSpecularTextures {
                reservoir: rg.convert_to_temporal(ind_spec_temporal_reservoir),
                hit_pos: rg.convert_to_temporal(ind_spec_hit_pos),
                hit_radiance: rg.convert_to_temporal(ind_spec_hit_radiance),
                lighting: rg.convert_to_temporal(filtered_indirect_specular),
            });
        self.prev_gbuffer_color
            .replace(rg.convert_to_temporal(gbuffer.color.0));

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
                .texture_view("gbuffer_depth", gbuffer.depth.1)
                .rw_texture_view("rw_shadow", raytraced_shadow_mask.1)
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
                .texture_view("skycube", skycube)
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
                .texture_view("gbuffer_color", gbuffer.color.1)
                .texture_view("shadow_mask_buffer", raytraced_shadow_mask.1)
                .texture("indirect_diffuse_texture", filtered_indirect_diffuse)
                .texture("indirect_specular_texture", filtered_indirect_specular)
                .rw_texture_view("rw_color_buffer", scene_color.1)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        let has_prev_color = self.taa.prev_color.is_some() as u32;
        let prev_color = match self.taa.prev_color.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_texture),
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
                .texture_view("gbuffer_depth", gbuffer.depth.1)
                .texture_view("source_texture", scene_color.1)
                .texture("history_texture", prev_color)
                .rw_texture_view("rw_target", post_taa_color.1)
                .push_constant(&has_prev_color)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        } else {
            post_taa_color = scene_color;
        }

        // Pass: HashGrid Debug Visualization
        let hash_grid_vis = if self.config.debug_view == DebugView::HashGridView {
            let show_color_color = 1u32;
            let tex = rg.create_texutre(TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                TextureUsage::compute().to_vk(),
            ));
            rg.new_compute("Hash Grid Vis")
                .compute_shader("restir/hash_grid_visualize.hlsl")
                .texture("gbuffer_depth", gbuffer.depth.0)
                .buffer("hash_grid_storage_buffer", hg_cache.storage)
                .rw_texture("rw_color", tex)
                .push_constant(&show_color_color)
                .group_count_uvec2(main_size.div_round_up(UVec2::new(8, 8)));
            tex
        } else {
            rg.register_texture(default_res.dummy_texture)
        };

        // Pass: debug view
        if self.config.debug_view != DebugView::None {
            let color_texture = match self.config.debug_view {
                DebugView::AO => curr_ao_texture,
                DebugView::AOFiltered => filtered_ao_texture,
                DebugView::IndDiff => indirect_diffuse,
                DebugView::IndDiffFiltered => filtered_indirect_diffuse,
                DebugView::IndSpec => indirect_specular,
                DebugView::IndSpecFiltered => filtered_indirect_specular,
                DebugView::HashGridView => hash_grid_vis,
                DebugView::None => rg.register_texture(default_res.dummy_texture),
            };
            rg.new_compute("Debug View")
                .compute_shader("restir/debug_view.hlsl")
                .texture("color_texture", color_texture)
                .rw_texture("rw_output_texture", post_taa_color.0)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        // Keep HashGridCache
        self.hash_grid_cache_history
            .replace(HashGridCacheResrouces {
                storage: rg.convert_to_temporal(hg_cache.storage),
                decay: rg.convert_to_temporal(hg_cache.decay),
            });

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
