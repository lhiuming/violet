use ash::vk;
use violet::{
    render_device::{texture::TextureUsage, Texture, TextureDesc},
    render_graph::{PassBuilderTrait, RGHandle, RGTemporal, RenderGraphBuilder},
    render_loop::DivRoundUp,
};

pub struct Input {
    pub depth: RGHandle<Texture>,
    pub gbuffer: RGHandle<Texture>,
    pub prev_depth: RGHandle<Texture>,
    pub prev_gbuffer: RGHandle<Texture>,
    pub diffuse: RGHandle<Texture>, // diffuse component of (indirect) lighting
    pub specular: RGHandle<Texture>, // specular component of (indirect) lighting
}

pub struct Output {
    pub diffuse: RGHandle<Texture>,
    pub specular: RGHandle<Texture>,
}

struct History<Handle> {
    diffuse: Handle,
    specular: Handle,
    //fast: Handle, // fast history of diffuse and specular (packed)
    //moment: Handle,
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

    pub fn add_passes<'a>(&mut self, rg: &mut RenderGraphBuilder, input: Input) -> Output {
        let history = self.history.take().map(|temporal| History {
            diffuse: rg.convert_to_transient(temporal.diffuse),
            specular: rg.convert_to_transient(temporal.specular),
        });

        let (width, height) = {
            let desc = rg.get_texture_desc(input.diffuse);
            (desc.width, desc.height)
        };

        // 1. Reprojection (temporal filter)
        let diff_filtered;
        let spec_filtered;
        if let Some(history) = history {
            let color_desc = TextureDesc {
                width,
                height,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                usage: TextureUsage::for_compute().into(),
                ..Default::default()
            };
            diff_filtered = rg.create_texutre(color_desc);
            spec_filtered = rg.create_texutre(color_desc);

            rg.new_compute("Temporal Filter")
                .compute_shader("denoise/temporal_filter.hlsl")
                .texture("depth_buffer", input.depth)
                .texture("diff_history_texture", history.diffuse)
                .texture("spec_history_texture", history.specular)
                .texture("diff_source_texture", input.diffuse)
                .texture("spec_source_texture", input.specular)
                .rw_texture("rw_diff_texture", diff_filtered)
                .rw_texture("rw_spec_texture", spec_filtered)
                .push_constant(&[width as f32, height as f32])
                .group_count(width.div_round_up(8), height.div_round_up(8), 1);
        } else {
            diff_filtered = input.diffuse;
            spec_filtered = input.specular;
        }

        self.history = Some(History {
            diffuse: rg.convert_to_temporal(diff_filtered),
            specular: rg.convert_to_temporal(spec_filtered),
        });

        Output {
            diffuse: diff_filtered,
            specular: spec_filtered,
        }
    }
}
