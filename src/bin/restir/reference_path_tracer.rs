use ash::vk;

use glam::{Mat4, Vec3};
use violet::{
    imgui::Ui,
    render_device::{RenderDevice, Texture, TextureDesc},
    render_graph::*,
    render_loop::ViewInfo,
    render_scene::RenderScene,
};

use crate::restir_render_loop::SceneRendererInput;

struct RenderParams {
    sun_dir: Vec3,
    view_position: Vec3,
    view_transform: Mat4,
}

impl RenderParams {
    pub fn new(sun_dir: &Vec3, view_info: &ViewInfo) -> Self {
        Self {
            sun_dir: *sun_dir,
            view_position: view_info.view_position,
            view_transform: view_info.view_transform,
        }
    }

    fn update(&mut self, sun_dir: &Vec3, view_info: &ViewInfo) -> bool {
        let changed = !self.sun_dir.eq(sun_dir)
            | !self.view_position.eq(&view_info.view_position)
            | !self.view_transform.eq(&view_info.view_transform);

        if changed {
            *self = Self::new(sun_dir, view_info);
        }

        changed
    }
}

pub struct ReferencePathTracer {
    accumulated_count: u32,
    accumulated_texture: Option<RGTemporal<Texture>>,
    accumulation_limit: u32,
    force_restart: bool,

    // track state
    last_params: Option<RenderParams>,
}

impl ReferencePathTracer {
    pub fn new() -> Self {
        Self {
            accumulated_count: 0,
            accumulated_texture: None,
            accumulation_limit: 8 * 1024,
            force_restart: false,
            last_params: None,
        }
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        ui.heading("PATHTRACER");
        ui.label(format!("- frame count: {}", self.accumulated_count));
        if ui.button("restart").clicked() {
            self.force_restart = true;
        }
    }

    pub fn add_passes(
        &mut self,
        _rd: &mut RenderDevice,
        rg: &mut RenderGraphBuilder,
        scene: &RenderScene,
        input: SceneRendererInput,
    ) -> RGHandle<Texture> {
        let main_size = input.main_size;

        let better_restart = if let Some(params) = &mut self.last_params {
            params.update(&scene.sun_dir, input.view_info)
        } else {
            self.last_params = Some(RenderParams::new(&scene.sun_dir, &input.view_info));
            false
        };

        if self.force_restart || better_restart {
            // Drop the temporal texture
            self.accumulated_texture.take().map(|tex| {
                rg.convert_to_transient(tex);
            });

            self.accumulated_count = 0;
            self.force_restart = false;
        }

        let lighting_texture = {
            let tex = rg.create_texutre(TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ));
            let view = rg.create_texture_view(tex, None);
            (tex, view)
        };

        let accumulated_texture = match self.accumulated_texture.take() {
            Some(tex) => rg.convert_to_transient(tex),
            None => rg.create_texutre(TextureDesc::new_2d(
                main_size.x,
                main_size.y,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE,
            )),
        };
        let accumulated_texture_view = rg.create_texture_view(accumulated_texture, None);

        let stop_accumulate = (self.accumulated_count >= self.accumulation_limit) as u32;

        rg.new_raytracing("Path Tracing")
            .raygen_shader("pathtraced_lighting.hlsl")
            .miss_shaders(&["raytrace/geometry.rmiss.hlsl", "raytrace/shadow.rmiss.hlsl"])
            .closest_hit_shader("raytrace/geometry.rchit.hlsl")
            .accel_struct("scene_tlas", input.scene_tlas)
            .texture("skycube", input.sky_cube)
            .rw_texture("rw_accumulated", accumulated_texture_view)
            .rw_texture("rw_lighting", lighting_texture.1)
            .push_constant::<u32>(&input.frame_index)
            .push_constant::<u32>(&self.accumulated_count)
            .push_constant::<u32>(&stop_accumulate)
            .dimension(main_size.x, main_size.y, 1);

        if stop_accumulate == 0 {
            self.accumulated_count += 1;
        }
        self.accumulated_texture
            .replace(rg.convert_to_temporal(accumulated_texture));

        lighting_texture.0
    }
}
