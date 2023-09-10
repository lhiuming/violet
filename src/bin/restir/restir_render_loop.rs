use ash::vk;
use glam::{UVec2, Vec3};
use violet::{
    command_buffer::{CommandBuffer, StencilOps},
    imgui::ImGUIOuput,
    render_device::{
        Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
    },
    render_graph::*,
    render_loop::{
        div_round_up, imgui_pass::ImGUIPass, rg_util, FrameParams, JitterInfo, PrevView,
        RenderLoop, StreamLinedFrameResource, ViewInfo, FRAME_DESCRIPTOR_SET_INDEX,
    },
    render_loop::{div_round_up_uvec2, gbuffer_pass::*},
    render_scene::{RenderScene, UploadContext, SCENE_DESCRIPTOR_SET_INDEX},
    shader::{Shaders, ShadersConfig},
};

pub struct DefaultResources {
    pub dummy_buffer: Buffer,
    pub dummy_texture: (Texture, TextureView),
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

        let dummy_texture = {
            let tex = rd
                .create_texture(TextureDesc::new_2d(
                    1,
                    1,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::SAMPLED,
                ))
                .unwrap();
            let view_desc = TextureViewDesc {
                ..TextureViewDesc::auto(&tex.desc)
            };
            (tex, rd.create_texture_view(tex, view_desc).unwrap())
        };

        Self {
            dummy_buffer,
            dummy_texture,
        }
    }
}

struct SampleGenPass {
    prev_reservoir_buffer: Option<RGTemporal<Buffer>>,
}

impl SampleGenPass {
    fn new() -> Self {
        Self {
            prev_reservoir_buffer: None,
        }
    }
}

struct TAAPass {
    prev_view: Option<PrevView>,
    prev_color: Option<RGTemporal<Texture>>,
    prev_depth: Option<RGTemporal<Texture>>,
}

impl TAAPass {
    fn new() -> Self {
        Self {
            prev_view: None,
            prev_color: None,
            prev_depth: None,
        }
    }
}

pub struct RestirRenderLoop {
    render_graph_cache: RenderGraphCache,
    stream_lined: StreamLinedFrameResource,
    default_res: DefaultResources,

    frame_index: u32,

    sample_gen: SampleGenPass,
    taa: TAAPass,

    imgui_pass: ImGUIPass,

    upload_context: UploadContext,

    last_start_time: Option<std::time::Instant>,
    total_frame_duration: std::time::Duration,
    total_frame_count: u32,
    total_acquire_duration: std::time::Duration,
    total_wait_duration: std::time::Duration,
    total_present_duration: std::time::Duration,
}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &RenderDevice) -> Option<Self> {
        if !rd.support_raytracing {
            return None;
        }

        Some(Self {
            render_graph_cache: RenderGraphCache::new(rd),
            stream_lined: StreamLinedFrameResource::new(rd),
            default_res: DefaultResources::new(rd),
            frame_index: 0,
            sample_gen: SampleGenPass::new(),
            taa: TAAPass::new(),
            imgui_pass: ImGUIPass::new(rd),
            upload_context: UploadContext::new(rd),
            last_start_time: None,
            total_frame_duration: std::time::Duration::ZERO,
            total_frame_count: 0,
            total_acquire_duration: std::time::Duration::ZERO,
            total_wait_duration: std::time::Duration::ZERO,
            total_present_duration: std::time::Duration::ZERO,
        })
    }

    fn print_stat(&self) {
        println!("CPU Profiling:");
        let avg_ms = |name: &str, dur: std::time::Duration| {
            let ms = dur.as_secs_f64() * 1000.0 / self.total_frame_count as f64;
            println!("\t{:>24}: {:.2}ms", name, ms);
        };
        avg_ms("[Frame]", self.total_frame_duration);
        avg_ms("Acq. Swap.", self.total_acquire_duration);
        avg_ms("Wait Swap.", self.total_wait_duration);
        avg_ms("Present", self.total_present_duration);

        self.render_graph_cache.pass_profiling.print();
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

        let main_size = rd.swapchain.extent;
        let main_size_vec = UVec2::new(main_size.width, main_size.height);

        let mut rg = RenderGraphBuilder::new_with_shader_config(shader_config);

        // HACK: render graph should not use this; currently using it for SBT pooling
        rg.set_frame_index(self.frame_index);

        // Add persistent bindings
        let frame_descriptor_set = self.stream_lined.get_frame_desciptor_set();
        rg.add_global_descriptor_sets(&[
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set),
        ]);

        // a reused debug texture
        let debug_texture = rg_util::create_texture_and_view(
            &mut rg,
            TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ),
        );

        // Create GBuffer
        let gbuffer = create_gbuffer_textures(&mut rg, main_size);

        // Pass: GBuffer
        add_gbuffer_pass(&mut rg, rd, scene, &gbuffer);

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
                .rw_texture("rw_cube_texture", uav)
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

        let has_prev_frame: u32;
        let prev_color = match self.taa.prev_color {
            Some(tex) => {
                has_prev_frame = 1;
                let tex = rg.convert_to_transient(tex);
                rg.create_texture_view(tex, None)
            }
            None => {
                has_prev_frame = 0;
                rg.register_texture_view(self.default_res.dummy_texture.1)
            }
        };
        let prev_depth = match self.taa.prev_depth {
            Some(tex) => {
                let tex = rg.convert_to_transient(tex);
                rg.create_texture_view(tex, None)
            }
            None => rg.register_texture_view(self.default_res.dummy_texture.1),
        };

        let indirect_diffuse = {
            let desc = TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT // sky rendering
                    | vk::ImageUsageFlags::STORAGE // compute lighting
                    | vk::ImageUsageFlags::SAMPLED, // taa
            );
            let texture = rg.create_texutre(desc);
            let view = rg.create_texture_view(texture, None);
            (texture, view)
        };

        let main_len = main_size.width * main_size.height;
        let temporal_reservoir_buffer =
            rg.create_buffer(BufferDesc::compute(main_len as u64 * 24 * 4));

        // bind a dummy texture if we don't have prev frame reservoir
        let prev_reservoir_buffer = if let Some(buffer) = self.sample_gen.prev_reservoir_buffer {
            rg.convert_to_transient(buffer)
        } else {
            rg.register_buffer(self.default_res.dummy_buffer)
        };

        let is_validation_frame =
            self.sample_gen.prev_reservoir_buffer.is_some() && ((self.frame_index % 6) == 0);

        let has_new_sample;
        let new_sample_buffer;

        // Pass: Sample Generation (and temporal reusing)
        if !is_validation_frame {
            has_new_sample = 1u32;
            new_sample_buffer = rg.create_buffer(BufferDesc::compute(main_len as u64 * 5 * 4 * 4));

            rg.new_raytracing("Sample Gen")
                .raygen_shader("restir/sample_gen.hlsl")
                .miss_shader("raytrace/geometry.rmiss.hlsl")
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("skycube", skycube)
                .texture("prev_color", prev_color)
                .texture("prev_depth", prev_depth)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .buffer("rw_new_sample_buffer", new_sample_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&self.frame_index)
                .push_constant(&has_prev_frame)
                .dimension(main_size.width, main_size.height, 1);
        }
        // Pass: Sample Validation
        else {
            has_new_sample = 0u32;
            new_sample_buffer = rg.register_buffer(self.default_res.dummy_buffer);

            rg.new_raytracing("Sample Validate")
                .raygen_shader("restir/sample_validate.hlsl")
                .miss_shader("raytrace/geometry.rmiss.hlsl")
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("skycube", skycube)
                .texture("prev_color", prev_color)
                .texture("prev_depth", prev_depth)
                .rw_buffer("rw_prev_reservoir_buffer", prev_reservoir_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .dimension(main_size.width, main_size.height, 1);
        }

        // Pass: Temporal Resampling
        rg.new_compute("Temporal Resample")
            .compute_shader("restir/temporal_resample.hlsl")
            .texture("prev_gbuffer_depth", prev_depth)
            .texture("gbuffer_depth", gbuffer.depth.1)
            .texture("gbuffer_color", gbuffer.color.1)
            .buffer("new_sample_buffer", new_sample_buffer)
            .buffer("prev_reservoir_buffer", prev_reservoir_buffer)
            .buffer("rw_temporal_reservoir_buffer", temporal_reservoir_buffer)
            .push_constant(&self.frame_index)
            .push_constant(&has_prev_frame)
            .push_constant(&has_new_sample)
            .group_count_uvec3(div_round_up_uvec2(main_size_vec, UVec2::new(8, 4)).extend(1));

        // Pass: Spatial Resampling
        {
            let reservoir_buffer_desc = rg.get_buffer_desc(temporal_reservoir_buffer);
            let spatial_reservoir_buffer = rg.create_buffer(*reservoir_buffer_desc);

            rg.new_compute("Spatial Resampling")
                .compute_shader("restir/spatial_resampling.hlsl")
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .buffer("temporal_reservoir_buffer", temporal_reservoir_buffer)
                .rw_buffer("rw_spatial_reservoir_buffer", spatial_reservoir_buffer)
                .rw_texture("rw_lighting_texture", indirect_diffuse.1)
                .rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&self.frame_index)
                .group_count(
                    div_round_up(main_size.width, 8),
                    div_round_up(main_size.height, 4),
                    1,
                );

            let temporal =
                rg.convert_to_temporal(&mut self.render_graph_cache, spatial_reservoir_buffer);
            self.sample_gen.prev_reservoir_buffer.replace(temporal);
        }

        // Pass: Raytraced Shadow
        let raytraced_shadow_mask = {
            let tex = rg.create_texutre(TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ));
            (tex, rg.create_texture_view(tex, None))
        };
        {
            rg.new_raytracing("Raytraced Shadow")
                .raygen_shader_with_ep("raytraced_shadow.hlsl", "raygen")
                .miss_shader_with_ep("raytraced_shadow.hlsl", "miss")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .rw_texture("rw_shadow", raytraced_shadow_mask.1)
                .dimension(main_size.width, main_size.height, 1);
        }

        let scene_color = {
            let desc = TextureDesc::new_2d(
                main_size.width,
                main_size.height,
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
                    div_round_up(main_size.width, 8),
                    div_round_up(main_size.height, 4),
                    1,
                );
        }

        // Pass: TAA
        let post_taa_color = {
            let desc = TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                vk::ImageUsageFlags::STORAGE // compute TAA
                    | vk::ImageUsageFlags::SAMPLED, // history and post
            );
            let texture = rg.create_texutre(desc);
            let view = rg.create_texture_view(texture, None);
            (texture, view)
        };
        {
            rg.new_compute("Temporal AA")
                .compute_shader("temporal_aa.hlsl")
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("source_texture", scene_color.1)
                .texture("history_texture", prev_color)
                .rw_texture("rw_target", post_taa_color.1)
                .push_constant(&has_prev_frame)
                .group_count(
                    div_round_up(main_size.width, 8),
                    div_round_up(main_size.height, 4),
                    1,
                );

            self.taa
                .prev_color
                .replace(rg.convert_to_temporal(&mut self.render_graph_cache, post_taa_color.0));
            self.taa
                .prev_depth
                .replace(rg.convert_to_temporal(&mut self.render_graph_cache, gbuffer.depth.0));
        }

        let (swapchain_image_index, acquire_swapchain_duratiaon) = self
            .stream_lined
            .acquire_next_swapchain_image_with_duration(rd);
        self.total_acquire_duration += acquire_swapchain_duratiaon;
        let present_target =
            rg.register_texture_view(rd.swapchain.texture_views[swapchain_image_index as usize]);

        // Pass: Post Processing (write to swapchain)
        rg.new_compute("Post Processing")
            .compute_shader("post_processing.hlsl")
            .texture("src_color_texture", post_taa_color.1)
            .texture("debug_texture", debug_texture.1)
            .rw_texture("rw_target_buffer", present_target)
            .group_count(
                div_round_up(main_size.width, 8),
                div_round_up(main_size.height, 4),
                1,
            );

        // Pass: UI
        if let Some(imgui) = imgui {
            let target = rg.register_texture(rd.swapchain.textures[swapchain_image_index as usize]);
            self.imgui_pass
                .add(&mut rg, rd, &mut self.upload_context, target, imgui);
        }

        // Pass: Output
        rg.present(present_target);

        // Prepare command buffer
        let command_buffer = self.stream_lined.wait_and_reset_command_buffer(rd);

        // Update frame CB (before submit)
        let exposure = 20.0;
        let sun_inten = Vec3::new(1.0, 1.0, 0.85) * exposure;
        let jitter_info = Some(JitterInfo {
            frame_index: self.frame_index,
            viewport_size: main_size_vec,
        });
        self.stream_lined.update_frame_params(FrameParams::make(
            &view_info,
            jitter_info.as_ref(),
            &scene.sun_dir,
            &sun_inten,
            self.taa.prev_view.as_ref(),
        ));

        // only after last use of prev_view_info
        self.taa.prev_view.replace(PrevView {
            view_info: *view_info,
            jitter_info,
        });

        // Execute render graph
        {
            // Begin
            rd.begin_command_buffer(command_buffer);

            let cb = CommandBuffer::new(rd, command_buffer);
            rg.execute(rd, &cb, shaders, &mut self.render_graph_cache);

            // End
            rd.end_command_buffer(command_buffer);
        }

        // Submit, Present and stuff
        let (wait_duration, present_duration) = self
            .stream_lined
            .wait_and_submit_and_present(rd, swapchain_image_index);
        self.total_wait_duration += wait_duration;
        self.total_present_duration += present_duration;

        self.frame_index += 1;
    }
}
