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
    history: Option<History<RGTemporal<Texture>>>,
}

impl Denoiser {
    pub fn new() -> Self {
        Self { history: None }
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
                history_len: rg.register_texture(default_res.dummy_uint_texture),
            });

        let (width, height) = {
            let desc = rg.get_texture_desc(input.diffuse);
            (desc.width, desc.height)
        };

        // 1. Reprojection (temporal filter)
        let diffuse;
        let specular;
        let moments;
        let history_len;
        let variance;
        {
            let color_desc = TextureDesc {
                width,
                height,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                ..TextureDesc::compute_default()
            };
            diffuse = rg.create_texutre(color_desc);
            specular = rg.create_texutre(color_desc);
            moments = rg.create_texutre(TextureDesc {
                format: vk::Format::R16G16B16A16_SFLOAT,
                ..color_desc
            });
            history_len = rg.create_texutre(TextureDesc {
                format: vk::Format::R8_UINT,
                ..color_desc
            });
            variance = rg.create_texutre(TextureDesc {
                format: vk::Format::R16G16_SFLOAT,
                ..color_desc
            });

            rg.new_compute("Temporal Filter")
                .compute_shader("denoise/svgf_temporal_filter.hlsl")
                .texture("depth_buffer", input.depth)
                .texture("diff_source_texture", input.diffuse)
                .texture("spec_source_texture", input.specular)
                .texture("diff_history_texture", history.diffuse)
                .texture("spec_history_texture", history.specular)
                .texture("moments_history_texture", history.moments)
                .texture("history_len_texture", history.history_len)
                .rw_texture("rw_diff_texture", diffuse)
                .rw_texture("rw_spec_texture", specular)
                .rw_texture("rw_moments_texture", moments)
                .rw_texture("rw_history_len_texture", history_len)
                .rw_texture("rw_variance_texture", variance)
                .push_constant(&[width as f32, height as f32])
                .push_constant(&has_history)
                .group_count(width.div_round_up(8), height.div_round_up(8), 1);
        }

        self.history = Some(History {
            diffuse: rg.convert_to_temporal(diffuse),
            specular: rg.convert_to_temporal(specular),
            moments: rg.convert_to_temporal(moments),
            history_len: rg.convert_to_temporal(history_len),
        });

        Output {
            diffuse,
            specular,
            debug_views: DebugViews {
                history_len,
                variance,
            },
        }
    }
}
