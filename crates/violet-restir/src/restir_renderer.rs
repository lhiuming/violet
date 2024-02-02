use violet::{
    command_buffer::StencilOps,
    glam::UVec2,
    render_device::{
        texture::TextureUsage, AccelerationStructure, Buffer, BufferDesc, RenderDevice, Texture,
        TextureDesc, TextureView,
    },
    render_graph::*,
    render_scene::RenderScene,
    vk,
};
use violet_app::{
    imgui::{self, Ui, Widget},
    render_loop::{
        div_round_up, gbuffer_pass::*, jenkins_hash, util_passes::clear_buffer, DivRoundUp,
    },
};

use crate::denoising::{self, Denoiser};
use crate::restir_render_loop::SceneRendererInput;

// HASH_GRID_NUM_CELLS in shader
static HG_CACHE_NUM_CELLS: u64 = 65536 * 16;
// HASH_GRID_MAX_NUM_QUERIES in shader
static HG_CACHE_MAX_NUM_QUERIES: u64 = 128 * 1024;

// struct HashGridCell in shader
static HG_CACHE_CELL_SIZE: u64 = 4 * 5;
// HashGridQuery in shader
static HG_CACHE_QUERY_SIZE: u64 = 4 * 6;

#[derive(Debug, PartialEq, Clone, Copy)]
enum DebugView {
    None,
    Depth,
    Normal,
    Roughness,
    AO,
    AOFiltered,
    IndDiff,
    IndSpec,
    IndSpecRayLen,
    DenoiserHistLen,
    DenoiserVariance,
    DenoiserSpecFast,
    DenoiserDiffFast,
    HashGridCell,
    HashGridRadiance,
}

pub struct RestirConfig {
    taa: bool,
    jitter_cycle: u32,
    denoise: bool,
    ao_radius: f32,
    ind_diff_validate: bool,
    ind_spec_validate: bool,
    hash_grid_cache_decay: bool,
    debug_view: DebugView,
}
impl Default for RestirConfig {
    fn default() -> Self {
        Self {
            taa: true,
            jitter_cycle: 8,
            denoise: true,
            ao_radius: 2.0,
            ind_diff_validate: true,
            ind_spec_validate: true,
            hash_grid_cache_decay: true,
            debug_view: DebugView::None,
        }
    }
}

struct IndirectDiffuseTextures<TextureHandle> {
    reservoir: TextureHandle,
    hit_pos_normal: TextureHandle,
    hit_radiance: TextureHandle,
}

struct IndirectSpecularTextures {
    reservoir: RGTemporal<Texture>,
    hit_pos: RGTemporal<Texture>,
    hit_radiance: RGTemporal<Texture>,
}

struct HashGridCacheResrouces<BufferHandle> {
    storage: BufferHandle,
    decay: BufferHandle,
    query: BufferHandle,
    query_counter: BufferHandle,
}

/// Common resource for radiance tracing passes using the "raytrace.inc.hlsl".
struct RestirRaytraceResources {
    scene_tlas: RGHandle<AccelerationStructure>,
    skycube: RGHandle<TextureView>,
    prev_indirect_diffuse: RGHandle<Texture>,
    prev_depth: RGHandle<Texture>,
}

trait RadianceTracePassBuilder<'a>
where
    Self: PassBuilderTrait<'a>,
{
    /// Use by general raytracing passes (tlas, last frame screen-space radiance)
    fn restir_raytrace_resrouces(&mut self, res: &RestirRaytraceResources) -> &mut Self {
        self //
            .accel_struct("scene_tlas", res.scene_tlas)
            .texture_view("skycube", res.skycube)
            .texture("prev_indirect_diffuse_texture", res.prev_indirect_diffuse)
            .texture("prev_depth_texture", res.prev_depth)
    }

    /// Use by general raytracing passes (last frame work-space radiance)
    fn hg_cache_resources(
        &mut self,
        hg_cache: &HashGridCacheResrouces<RGHandle<Buffer>>,
    ) -> &mut Self {
        self //
            .buffer("hash_grid_storage_buffer", hg_cache.storage)
            .rw_buffer("rw_hash_grid_query_buffer", hg_cache.query)
            .rw_buffer("rw_hash_grid_query_counter_buffer", hg_cache.query_counter)
    }
}

impl<'a, 'render> RadianceTracePassBuilder<'render> for RaytracingPassBuilder<'a, 'render> {}

struct FrameRand {
    state: u32,
}

impl FrameRand {
    fn new(init: u32) -> Self {
        Self { state: init }
    }

    fn next(&mut self) -> u32 {
        self.state = jenkins_hash(self.state);
        self.state
    }
}

// Render the scene using ReSTIR lighting
pub struct RestirRenderer {
    config: RestirConfig,
    denoiser: Denoiser,

    prev_depth: Option<RGTemporal<Texture>>,
    prev_color: Option<RGTemporal<Texture>>,
    prev_gbuffer: Option<RGTemporal<Texture>>,

    prev_diffuse_reservoir_buffer: Option<IndirectDiffuseTextures<RGTemporal<Texture>>>,
    prev_indirect_diffuse_texture: Option<RGTemporal<Texture>>,

    prev_indirect_specular: Option<IndirectSpecularTextures>,

    ao_history: Option<RGTemporal<Texture>>,

    hash_grid_cache_history: Option<HashGridCacheResrouces<RGTemporal<Buffer>>>,
    clear_hash_grid_cache: bool,
}

impl RestirRenderer {
    pub fn new() -> Self {
        Self {
            config: Default::default(),
            denoiser: Denoiser::new(),

            prev_depth: None,
            prev_gbuffer: None,
            prev_color: None,

            prev_diffuse_reservoir_buffer: None,
            prev_indirect_diffuse_texture: None,

            prev_indirect_specular: None,

            ao_history: None,
            hash_grid_cache_history: None,
            clear_hash_grid_cache: false,
        }
    }

    pub fn taa(&self) -> bool {
        self.config.taa
    }

    pub fn jitter_cycle(&self) -> u32 {
        self.config.jitter_cycle
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        let config = &mut self.config;
        ui.heading("RESTIR RENDERER");
        ui.checkbox(&mut config.taa, "taa");
        ui.indent(imgui::Id::new("taa_child"), |ui| {
            ui.add_enabled_ui(config.taa, |ui| {
                ui.add(
                    imgui::Slider::new(&mut config.jitter_cycle, 8..=32)
                        .step_by(4.0)
                        .text("cycle"),
                );
            });
        });
        ui.checkbox(&mut config.denoise, "denoise");
        ui.indent(imgui::Id::new("denoise_child"), |ui| {
            ui.add_enabled_ui(config.denoise, |ui| {
                ui.checkbox(&mut self.denoiser.disocclusion_fix, "disocclusion fix");
                ui.checkbox(&mut self.denoiser.fast_history_clamp, "fast history clamp");
                imgui::Slider::new(&mut self.denoiser.atrous_iterations, 0..=5)
                    .text("atrous iter.")
                    .ui(ui);
            });
        });
        ui.checkbox(&mut config.ind_diff_validate, "ind.diff.s.validate");
        ui.checkbox(&mut config.ind_spec_validate, "ind.spec.s.validate");
        ui.checkbox(&mut config.hash_grid_cache_decay, "hg.cache.decay");
        ui.add(imgui::Slider::new(&mut config.ao_radius, 0.0..=5.0).text("ao radius"));

        // Debug options
        imgui::CollapsingHeader::new("debug options").show(ui, |ui| {
            if ui.button("Clear Hash Grid Cache").clicked() {
                self.clear_hash_grid_cache = true;
            }
        });

        // Debug view
        let response = imgui::CollapsingHeader::new("debug view").show(ui, |ui| {
            let mut item = |view: DebugView| {
                let selected = config.debug_view == view;
                let res = ui.selectable_label(selected, format!("{:?}", view));
                if res.clicked() {
                    config.debug_view = if selected { DebugView::None } else { view };
                }
            };
            item(DebugView::Depth);
            item(DebugView::Normal);
            item(DebugView::Roughness);
            item(DebugView::AO);
            item(DebugView::AOFiltered);
            item(DebugView::IndDiff);
            item(DebugView::IndSpec);
            item(DebugView::IndSpecRayLen);
            item(DebugView::DenoiserHistLen);
            item(DebugView::DenoiserVariance);
            item(DebugView::DenoiserDiffFast);
            item(DebugView::DenoiserSpecFast);
            item(DebugView::HashGridCell);
            item(DebugView::HashGridRadiance);
        });
        if !response.fully_open() {
            config.debug_view = DebugView::None;
        }
    }

    pub fn add_passes<'render>(
        &mut self,
        rd: &mut RenderDevice,
        rg: &mut RenderGraphBuilder<'render>,
        scene: &'render RenderScene,
        input: SceneRendererInput<'_>,
    ) -> RGHandle<Texture> {
        puffin::profile_function!();

        let frame_index = input.frame_index;
        let main_size = input.main_size;
        let default_res = input.default_res;
        let skycube = input.sky_cube;
        let scene_tlas = input.scene_tlas;

        // Use different hash in each pass to avoid correlation
        let mut frame_rand = FrameRand::new(frame_index);

        // Create GBuffer
        let gbuffer = create_gbuffer_textures(rg, main_size);

        // Pass: GBuffer
        add_gbuffer_pass(rg, rd, scene, &gbuffer);

        // Get Hash Grid Cache history
        let hg_cache_decay_format = vk::Format::R8_UINT;
        let mut hg_cache = match self.hash_grid_cache_history.take() {
            Some(temporal) => HashGridCacheResrouces {
                storage: rg.convert_to_transient(temporal.storage),
                decay: rg.convert_to_transient(temporal.decay),
                query: rg.convert_to_transient(temporal.query),
                query_counter: rg.convert_to_transient(temporal.query_counter),
            },
            None => {
                self.clear_hash_grid_cache = true; // reuse the flag for initialization
                HashGridCacheResrouces {
                    storage: rg.create_buffer(BufferDesc::compute(
                        HG_CACHE_NUM_CELLS * HG_CACHE_CELL_SIZE,
                    )),
                    decay: rg.create_buffer(BufferDesc::compute_with_usage(
                        HG_CACHE_NUM_CELLS,
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                    )),
                    query: rg.create_buffer(BufferDesc::compute(
                        HG_CACHE_MAX_NUM_QUERIES * HG_CACHE_QUERY_SIZE,
                    )),
                    query_counter: rg.create_buffer(BufferDesc::compute(4)),
                }
            }
        };
        if self.clear_hash_grid_cache {
            clear_buffer(rd, rg, hg_cache.storage, "Clear HashGridCache Storage");
            clear_buffer(rd, rg, hg_cache.decay, "Clear HashGridCache Decay");
            clear_buffer(
                rd,
                rg,
                hg_cache.query_counter,
                "Clear HashGridCache Query Counter",
            );
            self.clear_hash_grid_cache = false;
        }

        // Pass: HashGridCache Decay
        if self.config.hash_grid_cache_decay {
            rg.new_compute("HashGridCache Decay")
                .compute_shader("restir/hash_grid_decay.hlsl")
                .rw_buffer("rw_decay_buffer", hg_cache.decay)
                .rw_buffer("rw_storage_buffer", hg_cache.storage)
                // NOTE: see NUM_CELLS_PER_GROUP in shader
                .group_count(div_round_up(HG_CACHE_NUM_CELLS, 1024) as u32, 1, 1);
        }

        // depth buffer from last frame
        let has_prev_depth = self.prev_depth.is_some();
        let prev_depth = match self.prev_depth.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_texture),
        };

        // gbuffer from last frame (for specular temporal resampling)
        let has_prev_gbuffer = self.prev_gbuffer.is_some();
        let prev_gbuffer = match self.prev_gbuffer.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_uint_texture_array),
        };

        // indirect diffuse from last frame
        let has_prev_indirect_diffuse = self.prev_indirect_diffuse_texture.is_some();
        let prev_indirect_diffuse = match self.prev_indirect_diffuse_texture.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.register_texture(default_res.dummy_texture),
        };

        // Prepare common resources used by radiance tracing passes
        let restir_raytrace_res = RestirRaytraceResources {
            prev_depth,
            scene_tlas,
            skycube,
            prev_indirect_diffuse,
        };

        // HashGridCache Radiance Update Passes
        if has_prev_depth && has_prev_indirect_diffuse {
            let query_selection = rg.create_buffer(BufferDesc::compute(HG_CACHE_NUM_CELLS * 4));
            clear_buffer(rd, rg, query_selection, "Clear Query Selection Buffer");

            // Pass: HashGridCache Query Select
            rg.new_compute("HashGridCache Query Select")
                .compute_shader("restir/hash_grid_query_select.hlsl")
                .buffer("hash_grid_query_buffer", hg_cache.query)
                .buffer("hash_grid_query_counter_buffer", hg_cache.query_counter)
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .rw_buffer("rw_hash_grid_query_selection_buffer", query_selection)
                .push_constant(&frame_rand.next())
                .group_count(div_round_up(HG_CACHE_MAX_NUM_QUERIES, 32) as u32, 1, 1);

            // TODO size should be min(HG_CACHE_NUM_CELLS, HG_CACHE_MAX_NUM_QUERIES)
            let compact_query_index =
                rg.create_buffer(BufferDesc::compute(HG_CACHE_MAX_NUM_QUERIES * 4));
            let compact_query_cell_addr =
                rg.create_buffer(BufferDesc::compute(HG_CACHE_MAX_NUM_QUERIES * 4));
            let compact_query_counter = rg.create_buffer(BufferDesc::compute(4));
            clear_buffer(rd, rg, compact_query_counter, "Clear Compact Query Counter");

            // Pass: HashGridCache Query Compact
            rg.new_compute("HashGridCache Query Compact")
                .compute_shader("restir/hash_grid_query_compact.hlsl")
                .buffer("hash_grid_query_selection_buffer", query_selection)
                .rw_buffer("rw_compact_query_index_buffer", compact_query_index)
                .rw_buffer("rw_compact_query_cell_addr_buffer", compact_query_cell_addr)
                .rw_buffer("rw_compact_query_counter_buffer", compact_query_counter)
                // NOTE: see NUM_CELLS_PER_GROUP in compute shader
                .group_count(div_round_up(HG_CACHE_NUM_CELLS, 128) as u32, 1, 1);

            let new_query = rg.create_buffer(BufferDesc::compute(
                HG_CACHE_MAX_NUM_QUERIES * HG_CACHE_QUERY_SIZE,
            ));
            let new_query_counter = rg.create_buffer(BufferDesc::compute(4));
            clear_buffer(
                rd,
                rg,
                new_query_counter,
                "Clear New HashGridCache Query Counter",
            );

            // Pass: HashGridCache Raygen
            rg.new_raytracing("HashGridCache Raygen")
                .raygen_shader("restir/hash_grid_raygen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry_ind.rchit.hlsl")
                // raytrace.inc.hlsl
                .restir_raytrace_resrouces(&restir_raytrace_res)
                .rw_buffer("rw_hash_grid_query_buffer", new_query)
                .rw_buffer("rw_hash_grid_query_counter_buffer", new_query_counter)
                // hash_grid_radiance_update.hlsl
                .rw_buffer("rw_hash_grid_storage_buffer", hg_cache.storage)
                .buffer("hash_grid_query_buffer", hg_cache.query)
                .buffer("compact_query_counter_buffer", compact_query_counter)
                .buffer("compact_query_index_buffer", compact_query_index)
                .buffer("compact_query_cell_addr_buffer", compact_query_cell_addr)
                .rw_buffer_with_format(
                    "rw_hash_grid_decay_buffer",
                    hg_cache.decay,
                    hg_cache_decay_format,
                )
                .push_constant(&frame_rand.next())
                // TODO indirect trace
                .dimension(HG_CACHE_MAX_NUM_QUERIES as u32, 1, 1);

            // Replace with newd query buffer.
            // the old one is comsumed,
            // the new one has pending (indirect) queries in it.
            hg_cache.query = new_query;
            hg_cache.query_counter = new_query_counter;
        }

        // indirect diffuse reservior buffer from last frame
        let has_prev_diffuse_reservoir = self.prev_diffuse_reservoir_buffer.is_some();
        let ind_diff_prev = match self.prev_diffuse_reservoir_buffer.take() {
            Some(temp) => IndirectDiffuseTextures {
                reservoir: rg.convert_to_transient(temp.reservoir),
                hit_pos_normal: rg.convert_to_transient(temp.hit_pos_normal),
                hit_radiance: rg.convert_to_transient(temp.hit_radiance),
            },
            None => IndirectDiffuseTextures {
                reservoir: rg.register_texture(default_res.dummy_uint_texture),
                hit_pos_normal: rg.register_texture(default_res.dummy_uint_texture),
                hit_radiance: rg.register_texture(default_res.dummy_texture),
            },
        };

        let is_indirect_diffuse_validation_frame = {
            let has_buffers = has_prev_depth && has_prev_diffuse_reservoir;
            has_buffers && ((frame_index & 0x3) == 0) && self.config.ind_diff_validate
        };

        let ind_diff_has_new_sample = (!is_indirect_diffuse_validation_frame) as u32;

        // Pass: Indirect Diffuse ReSTIR Sample Generation
        let ind_diff_new_hit_pos_normal;
        let ind_diff_new_hit_radiance;
        if !is_indirect_diffuse_validation_frame {
            let hit_tex_desc = TextureDesc {
                width: main_size.x,
                height: main_size.y,
                usage: TextureUsage::new().storage().sampled().into(),
                ..Default::default()
            };
            ind_diff_new_hit_pos_normal = rg.create_texutre(TextureDesc {
                format: vk::Format::R32G32_UINT,
                ..hit_tex_desc
            });
            ind_diff_new_hit_radiance = rg.create_texutre(TextureDesc {
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                ..hit_tex_desc
            });

            let has_prev_frame = (has_prev_depth && has_prev_indirect_diffuse) as u32;

            rg.new_raytracing("Ind. Diff. Sample Gen.")
                .raygen_shader("restir/ind_diff_sample_gen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry_ind.rchit.hlsl")
                // raytrace.inc.hlsl
                .restir_raytrace_resrouces(&restir_raytrace_res)
                .hg_cache_resources(&hg_cache)
                // ind_diff_sample_gen.hlsl
                .texture("gbuffer_depth", gbuffer.depth)
                .texture("gbuffer_color", gbuffer.color)
                .rw_texture("rw_hit_pos_normal_texture", ind_diff_new_hit_pos_normal)
                .rw_texture("rw_hit_radiance_texture", ind_diff_new_hit_radiance)
                .push_constant(&frame_rand.next())
                .push_constant(&has_prev_frame)
                .dimension(main_size.x, main_size.y, 1);
        }
        // Pass: Indirect Diffuse Sample Validation
        else {
            ind_diff_new_hit_pos_normal = rg.register_texture(default_res.dummy_uint_texture);
            ind_diff_new_hit_radiance = rg.register_texture(default_res.dummy_texture);

            rg.new_raytracing("Ind. Diff. Sample Validate")
                .raygen_shader("restir/ind_diff_sample_validate.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry_ind.rchit.hlsl")
                // raytrace.inc.hlsl
                .restir_raytrace_resrouces(&restir_raytrace_res)
                .hg_cache_resources(&hg_cache)
                // ind_diff_sample_validate.hlsl
                .texture("depth_texture", prev_depth)
                .rw_texture("rw_reservoir_texture", ind_diff_prev.reservoir)
                .rw_texture("rw_hit_pos_normal_texture", ind_diff_prev.hit_pos_normal)
                .rw_texture("rw_hit_radiance_texture", ind_diff_prev.hit_radiance)
                //.push_constant(&(true as u32))
                .dimension(main_size.x, main_size.y, 1);
        }

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
            .texture("hit_pos_normal_texture", ind_diff_new_hit_pos_normal)
            .texture("depth_buffer", gbuffer.depth)
            .texture("prev_ao_texture", prev_ao_texture)
            .rw_texture("rw_ao_texture", curr_ao_texture)
            .push_constant::<u32>(&ind_diff_has_new_sample)
            .push_constant::<u32>(&has_prev_ao)
            .push_constant::<f32>(&self.config.ao_radius)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 8),
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
            .texture("depth_buffer", gbuffer.depth)
            .texture("ao_texture", curr_ao_texture)
            .rw_texture("rw_filtered_ao_texture", filtered_ao_texture)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        self.ao_history
            .replace(rg.convert_to_temporal(curr_ao_texture));

        let ind_diff_tex_desc = TextureDesc {
            width: main_size.x,
            height: main_size.y,
            usage: TextureUsage::new().storage().sampled().into(),
            ..Default::default()
        };

        // Indirect Diffuse reservoir texture after temporal resampling
        let ind_diff_temp = IndirectDiffuseTextures {
            reservoir: rg.create_texutre(TextureDesc {
                format: vk::Format::R32_UINT,
                ..ind_diff_tex_desc
            }),
            hit_pos_normal: rg.create_texutre(TextureDesc {
                format: vk::Format::R32G32_UINT,
                ..ind_diff_tex_desc
            }),
            hit_radiance: rg.create_texutre(TextureDesc {
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                ..ind_diff_tex_desc
            }),
        };

        // Pass: Indirect Diffuse Temporal Resampling
        rg.new_compute("Ind. Diff. Temporal Resample")
            .compute_shader("restir/ind_diff_temporal_resample.hlsl")
            .texture("prev_gbuffer_depth", prev_depth)
            .texture("gbuffer_depth", gbuffer.depth)
            // new samples
            .texture("new_hit_pos_normal_texture", ind_diff_new_hit_pos_normal)
            .texture("new_hit_radiance_texture", ind_diff_new_hit_radiance)
            // prev reservoir
            .texture("prev_reservoir_texture", ind_diff_prev.reservoir)
            .texture("prev_hit_pos_normal_texture", ind_diff_prev.hit_pos_normal)
            .texture("prev_hit_radiance_texture", ind_diff_prev.hit_radiance)
            // temporally resampled output
            .rw_texture("rw_reservoir_texture", ind_diff_temp.reservoir)
            .rw_texture("rw_hit_pos_normal_texture", ind_diff_temp.hit_pos_normal)
            .rw_texture("rw_hit_radiance_texture", ind_diff_temp.hit_radiance)
            .push_constant(&frame_rand.next())
            .push_constant(&(has_prev_diffuse_reservoir as u32))
            .push_constant(&ind_diff_has_new_sample)
            .group_count(main_size.x.div_round_up(8), main_size.y.div_round_up(8), 1);

        // Raw indirect diffuse lighting buffer, written by spatial resampling pass
        let indirect_diffuse = rg.create_texutre(TextureDesc {
            format: vk::Format::B10G11R11_UFLOAT_PACK32,
            ..ind_diff_tex_desc
        });

        // Indirect Diffuse reservoir textures after spatial resampling
        let ind_diff_spatial = IndirectDiffuseTextures {
            reservoir: rg.create_texutre(TextureDesc {
                format: vk::Format::R32_UINT,
                ..ind_diff_tex_desc
            }),
            hit_pos_normal: rg.create_texutre(TextureDesc {
                format: vk::Format::R32G32_UINT,
                ..ind_diff_tex_desc
            }),
            hit_radiance: rg.create_texutre(TextureDesc {
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                ..ind_diff_tex_desc
            }),
        };

        // Pass: Indirect Diffuse Spatial Resampling
        rg.new_compute("Ind. Diff. Spatial Resample")
            .compute_shader("restir/ind_diff_spatial_resample.hlsl")
            .texture("gbuffer_depth", gbuffer.depth)
            .texture("gbuffer_color", gbuffer.color)
            .texture("ao_texture", filtered_ao_texture)
            // temporally resampled reservoir
            .texture("temp_reservoir_texture", ind_diff_temp.reservoir)
            .texture("temp_hit_pos_normal_texture", ind_diff_temp.hit_pos_normal)
            .texture("temp_hit_radiance_texture", ind_diff_temp.hit_radiance)
            // spatially resampled output
            .rw_texture("rw_reservoir_texture", ind_diff_spatial.reservoir)
            .rw_texture("rw_hit_pos_normal_texture", ind_diff_spatial.hit_pos_normal)
            .rw_texture("rw_hit_radiance_texture", ind_diff_spatial.hit_radiance)
            .rw_texture("rw_lighting_texture", indirect_diffuse)
            //.rw_texture("rw_debug_texture", debug_texture.1)
            .push_constant(&frame_rand.next())
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 8),
                1,
            );

        // NOTE: spatial reservoir is feedback to temporal reservoir (next frame)
        self.prev_diffuse_reservoir_buffer = Some(IndirectDiffuseTextures {
            reservoir: rg.convert_to_temporal(ind_diff_spatial.reservoir),
            hit_pos_normal: rg.convert_to_temporal(ind_diff_spatial.hit_pos_normal),
            hit_radiance: rg.convert_to_temporal(ind_diff_spatial.hit_radiance),
        });

        let ind_spec_tex_desc = TextureDesc {
            width: main_size.x,
            height: main_size.y,
            format: vk::Format::R32G32B32A32_SFLOAT,
            usage: TextureUsage::new().storage().sampled().into(),
            ..Default::default()
        };
        let mut ind_spec_new_tex = |format: vk::Format| {
            rg.create_texutre(TextureDesc {
                format,
                ..ind_spec_tex_desc
            })
        };

        let ind_spec_hit_pos = ind_spec_new_tex(vk::Format::R32G32B32A32_SFLOAT);
        let ind_spec_hit_radiance = ind_spec_new_tex(vk::Format::B10G11R11_UFLOAT_PACK32);

        let ind_spec_temporal_reservoir = ind_spec_new_tex(vk::Format::R32G32_UINT);
        let indirect_specular = ind_spec_new_tex(vk::Format::B10G11R11_UFLOAT_PACK32);

        // Pass: Indirect Specular Sample Generation
        {
            let has_prev_frame = (has_prev_depth && has_prev_indirect_diffuse) as u32;

            rg.new_raytracing("Ind. Spec. Sample Gen.")
                .raygen_shader("restir/ind_spec_sample_gen.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry_ind.rchit.hlsl")
                // raytrace.inc.hlsl
                .restir_raytrace_resrouces(&restir_raytrace_res)
                .hg_cache_resources(&hg_cache)
                // ind_spec_sample_gen.hlsl
                .texture("gbuffer_depth", gbuffer.depth)
                .texture("gbuffer_color", gbuffer.color)
                //.rw_texture("rw_origin_pos_texture", ind_spec_origin_pos)
                //.rw_texture("rw_hit_normal_texture", ind_spec_hit_normal)
                .rw_texture("rw_hit_pos_texture", ind_spec_hit_pos)
                .rw_texture("rw_hit_radiance_texture", ind_spec_hit_radiance)
                .push_constant::<u32>(&frame_rand.next())
                .push_constant::<u32>(&has_prev_frame)
                .dimension(main_size.x, main_size.y, 1);
        }

        let has_prev_ind_spec = self.prev_indirect_specular.is_some();
        let prev_ind_spec_reservoir;
        let prev_ind_spec_hit_pos;
        let prev_ind_spec_hit_radiance;
        match self.prev_indirect_specular.take() {
            Some(ind_spec) => {
                prev_ind_spec_reservoir = rg.convert_to_transient(ind_spec.reservoir);
                prev_ind_spec_hit_pos = rg.convert_to_transient(ind_spec.hit_pos);
                prev_ind_spec_hit_radiance = rg.convert_to_transient(ind_spec.hit_radiance);
            }
            None => {
                prev_ind_spec_reservoir = rg.register_texture(default_res.dummy_uint_texture);
                prev_ind_spec_hit_pos = rg.register_texture(default_res.dummy_texture);
                prev_ind_spec_hit_radiance = rg.register_texture(default_res.dummy_texture);
            }
        }

        // Pass: Indirect Specular Sample Validation
        let has_spec_validate_textures =
            has_prev_depth && has_prev_indirect_diffuse && has_prev_ind_spec;
        if self.config.ind_spec_validate && has_spec_validate_textures {
            rg.new_raytracing("Ind. Spec. Sample Validate")
                .raygen_shader("restir/ind_spec_sample_validate.hlsl")
                .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
                .closest_hit_shader("raytrace/geometry_ind.rchit.hlsl")
                // raytrace.inc.hlsl
                .restir_raytrace_resrouces(&restir_raytrace_res)
                .hg_cache_resources(&hg_cache)
                // ind_spec_sample_validate.hlsl
                .rw_texture("rw_prev_reservoir_texture", prev_ind_spec_reservoir)
                .rw_texture("rw_prev_hit_pos_texture", prev_ind_spec_hit_pos)
                .rw_texture("rw_prev_hit_radiance_texture", prev_ind_spec_hit_radiance)
                .dimension(main_size.x, main_size.y, 1);
        }

        // Pass: Indirect Specular Temporal Resampling
        {
            let has_prev_frame = (has_prev_depth && has_prev_ind_spec && has_prev_gbuffer) as u32;

            rg.new_compute("Ind. Spec. Temporal Resample")
                .compute_shader("restir/ind_spec_temporal_resample.hlsl")
                .texture("prev_gbuffer_depth", prev_depth)
                .texture("prev_gbuffer_color", prev_gbuffer)
                .texture("gbuffer_depth", gbuffer.depth)
                .texture("gbuffer_color", gbuffer.color)
                .texture("prev_reservoir_texture", prev_ind_spec_reservoir)
                .texture("prev_hit_pos_texture", prev_ind_spec_hit_pos)
                .texture("prev_hit_radiance_texture", prev_ind_spec_hit_radiance)
                .rw_texture("rw_reservoir_texture", ind_spec_temporal_reservoir)
                .rw_texture("rw_hit_pos_texture", ind_spec_hit_pos)
                .rw_texture("rw_hit_radiance_texture", ind_spec_hit_radiance)
                .push_constant::<u32>(&frame_rand.next())
                .push_constant::<u32>(&has_prev_frame)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        let indirect_specular_ray_len = rg.create_texutre(TextureDesc {
            format: vk::Format::R16_SFLOAT,
            ..ind_spec_tex_desc
        });

        // Pass: Indirect Specular Spatial Resampling
        rg.new_compute("Ind. Spec. Spaital Resample")
            .compute_shader("restir/ind_spec_spatial_resample.hlsl")
            .texture("gbuffer_depth", gbuffer.depth)
            .texture("gbuffer_color", gbuffer.color)
            .texture("reservoir_texture", ind_spec_temporal_reservoir)
            .texture("hit_pos_texture", ind_spec_hit_pos)
            .texture("hit_radiance_texture", ind_spec_hit_radiance)
            .rw_texture("rw_lighting_texture", indirect_specular)
            .rw_texture("rw_ray_len_texture", indirect_specular_ray_len)
            .push_constant::<u32>(&frame_rand.next())
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        // Passes: ReLAX style denoiser
        let denoised_indirect_diffuse;
        let denoised_indirect_specular;
        let denoiser_dbv;
        if self.config.denoise && (has_prev_depth && has_prev_gbuffer) {
            let input = denoising::Input {
                depth: gbuffer.depth,
                gbuffer: gbuffer.color,
                prev_depth: prev_depth,
                prev_gbuffer: prev_gbuffer,
                diffuse: indirect_diffuse,
                specular: indirect_specular,
                specular_ray_len: indirect_specular_ray_len,
            };
            let denoised = self.denoiser.add_passes(rg, default_res, input);

            denoised_indirect_diffuse = denoised.diffuse;
            denoised_indirect_specular = denoised.specular;
            denoiser_dbv = denoised.debug_views;
        } else {
            self.denoiser.reset();
            let dummy = rg.register_texture(default_res.dummy_texture);
            denoiser_dbv = denoising::DebugViews {
                history_len: dummy,
                variance: dummy,
                spec_fast: dummy,
                diff_fast: dummy,
            };

            denoised_indirect_diffuse = indirect_diffuse;
            denoised_indirect_specular = indirect_specular;
        }

        self.prev_indirect_diffuse_texture
            .replace(rg.convert_to_temporal(denoised_indirect_diffuse));
        self.prev_indirect_specular
            .replace(IndirectSpecularTextures {
                reservoir: rg.convert_to_temporal(ind_spec_temporal_reservoir),
                hit_pos: rg.convert_to_temporal(ind_spec_hit_pos),
                hit_radiance: rg.convert_to_temporal(ind_spec_hit_radiance),
            });

        self.prev_gbuffer
            .replace(rg.convert_to_temporal(gbuffer.color));

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
                .texture("gbuffer_depth", gbuffer.depth)
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
            rg.create_texutre(desc)
        };

        // Pass: Draw sky
        {
            rg.new_graphics("Sky")
                .vertex_shader_with_ep("sky_vsps.hlsl", "vs_main")
                .pixel_shader_with_ep("sky_vsps.hlsl", "ps_main")
                .texture_view("skycube", skycube)
                .color_targets(&[ColorTarget {
                    tex: scene_color,
                    ..Default::default()
                }])
                .depth_stencil(DepthStencilTarget {
                    tex: gbuffer.depth,
                    aspect: vk::ImageAspectFlags::STENCIL,
                    load_op: DepthLoadOp::Load,
                    store_op: vk::AttachmentStoreOp::NONE, // keep but dont write deep
                    ..Default::default()
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
                .texture("gbuffer_color", gbuffer.color)
                .texture_view("shadow_mask_buffer", raytraced_shadow_mask.1)
                .texture("indirect_diffuse_texture", denoised_indirect_diffuse)
                .texture("indirect_specular_texture", denoised_indirect_specular)
                .rw_texture("rw_color_buffer", scene_color)
                .group_count(
                    div_round_up(main_size.x, 8),
                    div_round_up(main_size.y, 4),
                    1,
                );
        }

        let has_prev_color = self.prev_color.is_some() as u32;
        let prev_color = match self.prev_color.take() {
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
                rg.create_texutre(desc)
            };

            rg.new_compute("Temporal AA")
                .compute_shader("temporal_aa.hlsl")
                .texture("gbuffer_depth", gbuffer.depth)
                .texture("source_texture", scene_color)
                .texture("history_texture", prev_color)
                .rw_texture("rw_target", post_taa_color)
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
        let gen_hash_grid_vis = self.config.debug_view == DebugView::HashGridCell
            || self.config.debug_view == DebugView::HashGridRadiance;
        let hash_grid_vis = if gen_hash_grid_vis {
            let show_color_code = (self.config.debug_view == DebugView::HashGridCell) as u32;
            let tex = rg.create_texutre(TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                TextureUsage::new().storage().sampled().into(),
            ));
            rg.new_compute("Hash Grid Vis")
                .compute_shader("restir/hash_grid_visualize.hlsl")
                .texture("gbuffer_depth", gbuffer.depth)
                .buffer("hash_grid_storage_buffer", hg_cache.storage)
                .rw_texture("rw_color", tex)
                .push_constant(&show_color_code)
                .group_count_uvec2(main_size.div_round_up(UVec2::new(8, 8)));
            tex
        } else {
            rg.register_texture(default_res.dummy_texture)
        };

        // Pass: Debug View
        if self.config.debug_view != DebugView::None {
            let color_texture = match self.config.debug_view {
                DebugView::Depth => gbuffer.depth,
                DebugView::AO => curr_ao_texture,
                DebugView::AOFiltered => filtered_ao_texture,
                DebugView::IndDiff => denoised_indirect_diffuse,
                DebugView::IndSpec => denoised_indirect_specular,
                DebugView::IndSpecRayLen => indirect_specular_ray_len,
                DebugView::DenoiserHistLen => denoiser_dbv.history_len,
                DebugView::DenoiserVariance => denoiser_dbv.variance,
                DebugView::DenoiserDiffFast => denoiser_dbv.diff_fast,
                DebugView::DenoiserSpecFast => denoiser_dbv.spec_fast,
                DebugView::HashGridCell => hash_grid_vis,
                DebugView::HashGridRadiance => hash_grid_vis,
                _ => rg.register_texture(default_res.dummy_texture),
            };
            let uint_texture = match self.config.debug_view {
                _ => rg.register_texture(default_res.dummy_uint_texture),
            };
            let is_gbuffer: u32 = match self.config.debug_view {
                DebugView::Normal => 1,
                DebugView::Roughness => 2,
                _ => 0,
            };
            let is_uint = match self.config.debug_view {
                _ => false,
            } as u32;
            rg.new_compute("Debug View")
                .compute_shader("restir/debug_view.hlsl")
                .texture("gbuffer_texture", gbuffer.color)
                .texture("color_texture", color_texture)
                .texture("uint_texture", uint_texture)
                .rw_texture("rw_output_texture", post_taa_color)
                .push_constant(&is_gbuffer)
                .push_constant(&is_uint)
                .group_count_uvec2(main_size.div_round_up(UVec2::new(8, 8)));
        }

        // Keep HashGridCache
        self.hash_grid_cache_history
            .replace(HashGridCacheResrouces {
                storage: rg.convert_to_temporal(hg_cache.storage),
                decay: rg.convert_to_temporal(hg_cache.decay),
                query: rg.convert_to_temporal(hg_cache.query),
                query_counter: rg.convert_to_temporal(hg_cache.query_counter),
            });

        // Cache scene buffer for next frame (TAA, temporal restir, etc.)
        self.prev_color
            .replace(rg.convert_to_temporal(post_taa_color));
        self.prev_depth
            .replace(rg.convert_to_temporal(gbuffer.depth));

        post_taa_color
    }
}
