use ash::vk;
use glam::{UVec2, Vec3};

use violet::{
    command_buffer::CommandBuffer,
    gpu_profiling::NamedProfiling,
    imgui::{ImGUIOuput, Ui},
    render_device::{
        AccelerationStructure, Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView,
        TextureViewDesc,
    },
    render_graph::*,
    render_loop::{
        div_round_up, imgui_pass::ImGUIPass, FrameParams, JitterInfo, PrevView, RenderLoop,
        StreamLinedFrameResource, ViewInfo, FRAME_DESCRIPTOR_SET_INDEX,
    },
    render_scene::{RenderScene, UploadContext, SCENE_DESCRIPTOR_SET_INDEX},
    shader::{Shaders, ShadersConfig},
};

use crate::{reference_path_tracer::ReferencePathTracer, restir_renderer::RestirRenderer};

pub struct DefaultResources {
    pub dummy_buffer: Buffer,
    pub dummy_texture: Texture,
    pub dummy_uint_texture: Texture,
    //pub black_texture: (Texture, TextureView),
}

impl DefaultResources {
    pub fn new(rd: &RenderDevice) -> Self {
        let dummy_buffer = rd
            .create_buffer(BufferDesc {
                size: 1,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        let dummy_texture = rd
            .create_texture(TextureDesc::new_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED,
            ))
            .unwrap();

        let dummy_uint_texture = rd
            .create_texture(TextureDesc::new_2d(
                1,
                1,
                vk::Format::R8G8B8A8_UINT,
                vk::ImageUsageFlags::SAMPLED,
            ))
            .unwrap();

        Self {
            dummy_buffer,
            dummy_texture,
            dummy_uint_texture,
        }
    }
}

pub struct RenderConfig {
    reference: bool,
    exposure_stop: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            reference: false,
            exposure_stop: 0.0,
        }
    }
}

pub struct SceneRendererInput<'a> {
    pub frame_index: u32,
    pub main_size: UVec2,
    pub default_res: &'a DefaultResources,
    pub sky_cube: RGHandle<TextureView>,
    pub scene_tlas: RGHandle<AccelerationStructure>,
    pub view_info: &'a ViewInfo,
}

pub struct RestirRenderLoop {
    render_graph_cache: Option<RenderGraphCache>,
    stream_lined: StreamLinedFrameResource,
    default_res: DefaultResources,
    frame_index: u32,

    config: RenderConfig,

    restir_renderer: RestirRenderer,
    reference_path_tracer: ReferencePathTracer,
    prev_view: Option<PrevView>,
    imgui_pass: ImGUIPass,

    upload_context: UploadContext,

    last_start_time: Option<std::time::Instant>,
    total_frame_duration: std::time::Duration,
    total_frame_count: u32,
    total_acquire_duration: std::time::Duration,
    total_wait_duration: std::time::Duration,
    total_present_duration: std::time::Duration,
    last_ui_duration: std::time::Duration,
}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &mut RenderDevice) -> Option<Self> {
        if !rd.support_raytracing {
            return None;
        }

        Some(Self {
            render_graph_cache: Some(RenderGraphCache::new(rd)),
            stream_lined: StreamLinedFrameResource::new(rd),
            default_res: DefaultResources::new(rd),
            frame_index: 0,

            config: Default::default(),

            restir_renderer: RestirRenderer::new(),
            reference_path_tracer: ReferencePathTracer::new(),
            prev_view: None,
            imgui_pass: ImGUIPass::new(rd),

            upload_context: UploadContext::new(rd),

            last_start_time: None,
            total_frame_duration: std::time::Duration::ZERO,
            total_frame_count: 0,
            total_acquire_duration: std::time::Duration::ZERO,
            total_wait_duration: std::time::Duration::ZERO,
            total_present_duration: std::time::Duration::ZERO,
            last_ui_duration: std::time::Duration::ZERO,
        })
    }

    fn ui(&mut self, ui: &mut Ui) {
        let config = &mut self.config;
        ui.toggle_value(&mut config.reference, "reference");

        // Scene Renderer
        ui.add_enabled_ui(config.reference, |ui| {
            self.reference_path_tracer.ui(ui);
        });
        ui.add_enabled_ui(!config.reference, |ui| {
            self.restir_renderer.ui(ui);
        });

        // Post
        ui.heading("POSTPROCESSING");
        ui.horizontal(|ui| {
            let label = ui.label("exposure");
            let slider = egui::Slider::new(&mut config.exposure_stop, -3.0..=6.0).step_by(0.5);
            ui.add(slider).labelled_by(label.id);
        });
    }

    fn print_stat(&self) {
        println!("CPU Profiling:");
        let avg_ms = |name: &str, dur: std::time::Duration| {
            let ms = dur.as_secs_f64() * 1000.0 / self.total_frame_count as f64;
            println!("\t{:>16}: {:.4}ms", name, ms);
        };
        avg_ms("[Frame]", self.total_frame_duration);
        avg_ms("Acq. Swap.", self.total_acquire_duration);
        avg_ms("Wait Swap.", self.total_wait_duration);
        avg_ms("Present", self.total_present_duration);
        avg_ms("UI (last).", self.last_ui_duration);

        self.render_graph_cache
            .as_ref()
            .unwrap()
            .pass_profiling
            .print();
    }

    fn gpu_stat<'a>(&'a self) -> Option<&'a NamedProfiling> {
        if let Some(rgc) = self.render_graph_cache.as_ref() {
            Some(&rgc.pass_profiling)
        } else {
            None
        }
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
        imgui: Option<&ImGUIOuput>,
    ) {
        self.stream_lined.advance_render_index();

        // Update cpu profiling
        {
            let now = std::time::Instant::now();
            if let Some(last) = self.last_start_time {
                self.total_frame_duration += now - last;
                self.total_frame_count += 1;
            }
            self.last_start_time = Some(now);
        }

        // Shader config
        // TODO wrap into a ShaderPool/ShaderLibrary?
        let mut shader_config = ShadersConfig::default();
        shader_config
            .set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        shader_config.set_layout_override.insert(
            FRAME_DESCRIPTOR_SET_INDEX,
            self.stream_lined.get_set_layout(),
        );

        let main_extent = rd.swapchain.extent;
        let main_size = UVec2::new(main_extent.width, main_extent.height);

        let mut rg =
            RenderGraphBuilder::new(self.render_graph_cache.take().unwrap(), shader_config);

        // HACK: render graph should not use this; currently using it for SBT pooling
        rg.set_frame_index(self.frame_index);

        // Add persistent bindings
        let frame_descriptor_set = self.stream_lined.get_frame_desciptor_set();
        rg.add_global_descriptor_sets(&[
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set),
        ]);

        // Pass: Skycube update
        let skycube;
        {
            let width = 64;
            let desc = TextureDesc::new_2d_array(
                width,
                width,
                6,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            )
            .with_flags(vk::ImageCreateFlags::CUBE_COMPATIBLE);
            let skycube_texture = rg.create_texutre(desc);
            let uav = rg.create_texture_view(skycube_texture, None);

            rg.new_compute("Sky Cube")
                .compute_shader("sky_cube.hlsl")
                .rw_texture_view("rw_cube_texture", uav)
                .push_constant(&(width as f32))
                .group_count(width / 8, width / 4, 6);

            skycube = rg.create_texture_view(
                skycube_texture,
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::CUBE,
                    ..TextureViewDesc::auto(&desc)
                }),
            );
        }

        let scene_tlas = rg.register_accel_struct(scene.scene_top_level_accel_struct.unwrap());

        // Render the scene
        let scene_renderer_input = SceneRendererInput {
            frame_index: self.frame_index,
            main_size,
            sky_cube: skycube,
            scene_tlas: scene_tlas,
            default_res: &self.default_res,
            view_info,
        };
        let scene_color = if self.config.reference {
            self.reference_path_tracer
                .add_passes(rd, &mut rg, scene, scene_renderer_input)
        } else {
            self.restir_renderer
                .add_passes(rd, &mut rg, scene, scene_renderer_input)
        };

        // Wait swapchain image
        let (swapchain_image_index, acquire_swapchain_duratiaon) = self
            .stream_lined
            .acquire_next_swapchain_image_with_duration(rd);
        self.total_acquire_duration += acquire_swapchain_duratiaon;
        let present_target = (
            rg.register_texture(rd.swapchain.textures[swapchain_image_index as usize]),
            rg.register_texture_view(rd.swapchain.texture_views[swapchain_image_index as usize]),
        );

        let scene_color_view = rg.create_texture_view(scene_color, None);

        // Pass: Post Processing (write to swapchain)
        let exposure_scale: f32 = 2.0f32.powf(self.config.exposure_stop);
        rg.new_compute("Post Processing")
            .compute_shader("post_processing.hlsl")
            .texture_view("src_color_texture", scene_color_view)
            .rw_texture_view("rw_target_buffer", present_target.1)
            .push_constant(&exposure_scale)
            .group_count(
                div_round_up(main_size.x, 8),
                div_round_up(main_size.y, 4),
                1,
            );

        // Pass: UI
        if let Some(imgui) = imgui {
            let ui_time = std::time::Instant::now();
            self.imgui_pass.add(
                &mut rg,
                rd,
                &mut self.stream_lined,
                &mut self.upload_context,
                present_target.0,
                imgui,
                None,
            );
            self.last_ui_duration = ui_time.elapsed();
        }

        // Pass: Output
        rg.present(present_target.0);

        // Prepare command buffer
        let command_buffer = self.stream_lined.wait_and_reset_command_buffer(rd);

        // Update frame CB (before submit)
        let sun_inten = Vec3::new(1.0, 1.0, 0.85) * 20.0;
        let jitter_info = if self.restir_renderer.taa() {
            Some(JitterInfo {
                frame_index: self.frame_index,
                viewport_size: main_size,
            })
        } else {
            None
        };
        self.stream_lined.update_frame_params(FrameParams::make(
            &view_info,
            jitter_info.as_ref(),
            &scene.sun_dir,
            &sun_inten,
            self.prev_view.as_ref(),
        ));

        // only after last use of prev_view_info
        self.prev_view.replace(PrevView {
            view_info: *view_info,
            jitter_info,
        });

        // Execute render graph
        {
            // Begin
            rd.begin_command_buffer(command_buffer);

            let cb = CommandBuffer::new(rd, command_buffer);
            rg.execute(rd, &cb, shaders);

            // End
            rd.end_command_buffer(command_buffer);
        }
        self.render_graph_cache.replace(rg.done());

        // Submit, Present and stuff
        let (wait_duration, present_duration) = self
            .stream_lined
            .wait_and_submit_and_present(rd, swapchain_image_index);
        self.total_wait_duration += wait_duration;
        self.total_present_duration += present_duration;

        self.frame_index += 1;
    }
}
