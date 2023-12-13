use ash::vk;
use violet::{
    render_device::{Texture, TextureDesc},
    render_graph::{PassBuilderTrait, RGHandle, RGTemporal, RenderGraphBuilder},
    render_loop::DivRoundUp,
};

use crate::restir_render_loop::DefaultResources;

pub struct Input {
    pub depth: RGHandle<Texture>,
    pub gbuffer: RGHandle<Texture>,
    pub prev_depth: RGHandle<Texture>,
    pub prev_gbuffer: RGHandle<Texture>,
    pub diffuse: RGHandle<Texture>, // diffuse component of (indirect) lighting
    pub specular: RGHandle<Texture>, // specular component of (indirect) lighting
}

pub struct DebugViews {
    pub history_len: RGHandle<Texture>,
    pub variance: RGHandle<Texture>,
}

pub struct Output {
    pub diffuse: RGHandle<Texture>,
    pub specular: RGHandle<Texture>,
    pub debug_views: DebugViews,
}

struct History<Handle> {
    diffuse: Handle,
    specular: Handle,
    moments: Handle,
    history_len: Handle,
    //fast: Handle, // fast history of diffuse and specular (packed)
}

pub struct Denoiser {
    // Config
    pub disocclusion_fix: bool, // ReLAX-style disocclusion fix pass
    pub atrous_iterations: u32, // number of a-trous iterations

    history: Option<History<RGTemporal<Texture>>>,
}

impl Denoiser {
    pub fn new() -> Self {
        Self {
            disocclusion_fix: true,
            atrous_iterations: 4,
            history: None,
        }
    }

    pub fn reset(&mut self) {
        self.history = None;
    }

    pub fn add_passes<'a>(
        &mut self,
        rg: &mut RenderGraphBuilder,
        default_res: &DefaultResources,
        input: Input,
    ) -> Output {
        // Sanity check
        let atrous_iterations = self.atrous_iterations.clamp(0, 5);

        let has_history = self.history.is_some() as u32;
        let history = self
            .history
            .take()
            .map(|temporal| History {
                diffuse: rg.convert_to_transient(temporal.diffuse),
                specular: rg.convert_to_transient(temporal.specular),
                moments: rg.convert_to_transient(temporal.moments),
                history_len: rg.convert_to_transient(temporal.history_len),
            })
            .unwrap_or_else(|| History {
                diffuse: rg.register_texture(default_res.dummy_texture),
                specular: rg.register_texture(default_res.dummy_texture),
                moments: rg.register_texture(default_res.dummy_texture),
                history_len: rg.register_texture(default_res.dummy_texture),
            });

        let (width, height) = {
            let desc = rg.get_texture_desc(input.diffuse);
            (desc.width, desc.height)
        };

        let color_desc = TextureDesc {
            width,
            height,
            format: vk::Format::B10G11R11_UFLOAT_PACK32,
            ..TextureDesc::compute_default()
        };

        // Pass: Temporal Filter (Reprojection)
        let diffuse = rg.create_texutre(color_desc);
        let specular = rg.create_texutre(color_desc);
        let moments = rg.create_texutre(TextureDesc {
            format: vk::Format::R16G16_SFLOAT,
            ..color_desc
        });
        let history_len = rg.create_texutre(TextureDesc {
            format: vk::Format::R8_UNORM,
            ..color_desc
        });
        rg.new_compute("Temporal Filter")
            .compute_shader("denoise/svgf_temporal_filter.hlsl")
            .texture("depth_texture", input.depth)
            .texture("gbuffer_texture", input.gbuffer)
            .texture("diff_source_texture", input.diffuse)
            .texture("spec_source_texture", input.specular)
            .texture("prev_depth_texture", input.prev_depth)
            .texture("prev_gbuffer_texture", input.prev_gbuffer)
            .texture("diff_history_texture", history.diffuse)
            .texture("spec_history_texture", history.specular)
            .texture("moments_history_texture", history.moments)
            .texture("history_len_texture", history.history_len)
            .rw_texture("rw_diff_texture", diffuse)
            .rw_texture("rw_spec_texture", specular)
            .rw_texture("rw_moments_texture", moments)
            .rw_texture("rw_history_len_texture", history_len)
            .push_constant(&[width as f32, height as f32])
            .push_constant(&has_history)
            .group_count(width.div_round_up(8), height.div_round_up(8), 1);

        // Pass: Disocclusion Fix
        let (diffuse, specular, moments) = if self.disocclusion_fix {
            let new_diffuse = rg.create_texutre(color_desc);
            let new_specular = rg.create_texutre(color_desc);
            let new_moments = rg.create_texutre(TextureDesc {
                format: vk::Format::R16G16_SFLOAT,
                ..color_desc
            });

            rg.new_compute("Disocclusion Fix")
                .compute_shader("denoise/relax_disocclusion_fix.hlsl")
                .texture("depth_texture", input.depth)
                .texture("gbuffer_texture", input.gbuffer)
                .texture("history_len_texture", history_len)
                .texture("diff_texture", diffuse)
                .texture("spec_texture", specular)
                .texture("moments_texture", moments)
                .rw_texture("rw_diff_texture", new_diffuse)
                .rw_texture("rw_spec_texture", new_specular)
                .rw_texture("rw_moments_texture", new_moments)
                .group_count(width.div_round_up(8), height.div_round_up(8), 1);

            (new_diffuse, new_specular, new_moments)
        } else {
            (diffuse, specular, moments)
        };

        // Temporal feedback before SVGF spatial passes
        // [2021 ReLAX]
        self.history = Some(History {
            diffuse: rg.convert_to_temporal(diffuse),
            specular: rg.convert_to_temporal(specular),
            moments: rg.convert_to_temporal(moments),
            history_len: rg.convert_to_temporal(history_len),
        });

        // Pass: Spatial Fallback (Spatial Variance Estimation)
        let variance;
        let (diffuse, specular) = {
            let new_diffuse = rg.create_texutre(color_desc);
            let new_specular = rg.create_texutre(color_desc);
            variance = rg.create_texutre(TextureDesc {
                format: vk::Format::R16G16_SFLOAT,
                ..color_desc
            });

            rg.new_compute("Spatial Fallback")
                .compute_shader("denoise/svgf_spatial_fallback_filter.hlsl")
                .texture("depth_texture", input.depth)
                .texture("gbuffer_texture", input.gbuffer)
                .texture("history_len_texture", history_len)
                .texture("diff_texture", diffuse)
                .texture("spec_texture", specular)
                .texture("moments_texture", moments)
                .rw_texture("rw_diff_texture", new_diffuse)
                .rw_texture("rw_spec_texture", new_specular)
                .rw_texture("rw_variance_texture", variance)
                .group_count(width.div_round_up(8), height.div_round_up(8), 1);

            (new_diffuse, new_specular)
        };

        // Pass: Spatial Filter (A-Trous), Iterated
        let diffuse_filtered;
        let specular_filtered;
        {
            let _feedback_iteration = 0;

            // Create ping-pong texture set
            let mut diff_src = diffuse;
            let mut spec_src = specular;
            let mut var_src = variance;
            let mut diff_dst = rg.create_texutre(color_desc);
            let mut spec_dst = rg.create_texutre(color_desc);
            let mut var_dst = rg.create_texutre(TextureDesc {
                format: vk::Format::R16G16_SFLOAT,
                ..color_desc
            });

            // Iterates (can be zero, e.g. in debug)
            for i in 0..atrous_iterations {
                // Each iteration introduces extra space of 2^(i-1)
                let step_size = 1 << i;

                // Dispatch
                rg.new_compute(&format!("Spatial Filter: {}", i))
                    .compute_shader("denoise/svgf_atrous_filter.hlsl")
                    .texture("depth_texture", input.depth)
                    .texture("gbuffer_texture", input.gbuffer)
                    .texture("diff_texture", diff_src)
                    .texture("spec_texture", spec_src)
                    .texture("variance_texture", var_src)
                    .rw_texture("rw_diff_texture", diff_dst)
                    .rw_texture("rw_spec_texture", spec_dst)
                    .rw_texture("rw_variance_texture", var_dst)
                    .push_constant(&step_size)
                    .group_count(width.div_round_up(8), height.div_round_up(8), 1);

                // Swap src and dst
                std::mem::swap(&mut diff_src, &mut diff_dst);
                std::mem::swap(&mut spec_src, &mut spec_dst);
                std::mem::swap(&mut var_src, &mut var_dst);
            }

            // NOTE: last dst is swapped to src
            diffuse_filtered = diff_src;
            specular_filtered = spec_src;
        }

        Output {
            diffuse: diffuse_filtered,
            specular: specular_filtered,
            debug_views: DebugViews {
                history_len,
                variance,
            },
        }
    }
}