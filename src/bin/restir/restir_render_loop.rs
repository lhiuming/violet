use ash::vk;
use glam::Vec3;
use violet::{
    command_buffer::{CommandBuffer, StencilOps},
    render_device::{
        Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
    },
    render_graph::*,
    render_loop::gbuffer_pass::*,
    render_loop::{
        div_round_up, rg_util, FrameParams, RenderLoop, StreamLinedFrameResource, ViewInfo,
        FRAME_DESCRIPTOR_SET_INDEX,
    },
    render_scene::{RenderScene, SCENE_DESCRIPTOR_SET_INDEX},
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
    prev_view_info: Option<ViewInfo>,
    prev_color: Option<RGTemporal<Texture>>,
    prev_depth: Option<RGTemporal<Texture>>,
}

impl TAAPass {
    fn new() -> Self {
        Self {
            prev_view_info: None,
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
}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            stream_lined: StreamLinedFrameResource::new(rd),
            default_res: DefaultResources::new(rd),
            frame_index: 0,
            sample_gen: SampleGenPass::new(),
            taa: TAAPass::new(),
        }
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        self.stream_lined.advance_render_index();

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

        //let mut rg = RenderGraphBuilder::new();
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
        /*
        let debug_texture = rg_util::create_texture_and_view(
            &mut rg,
            TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ),
        );
        */

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

        // Pass: Sample Generation (and temporal reusing)
        {
            // bind a dummy texture if we don't have prev frame reservoir
            let prev_reservoir_buffer = if let Some(buffer) = self.sample_gen.prev_reservoir_buffer
            {
                rg.convert_to_transient(buffer)
            } else {
                rg.register_buffer(self.default_res.dummy_buffer)
            };

            rg.new_raytracing("Sample Gen")
                .raygen_shader("restir/sample_gen.hlsl")
                .miss_shader("raytrace/geometry.rmiss.hlsl")
                .closest_hit_shader("raytrace/geometry.rchit.hlsl")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("skycube", skycube)
                .texture("prev_color", prev_color)
                .texture("prev_depth", prev_depth)
                .buffer("prev_reservoir_buffer", prev_reservoir_buffer)
                .rw_buffer("rw_temporal_reservoir_buffer", temporal_reservoir_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&self.frame_index)
                .push_constant(&has_prev_frame)
                .dimension(main_size.width, main_size.height, 1);
        }

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
                //.rw_texture("rw_debug_texture", debug_texture.1)
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

        let swapchain_image_index = self.stream_lined.acquire_next_swapchain_image(rd);
        let present_target =
            rg.register_texture_view(rd.swapchain.image_view[swapchain_image_index as usize]);

        // Pass: Post Processing (write to swapchain)
        rg.new_compute("Post Processing")
            .compute_shader("post_processing.hlsl")
            .texture("src_color_buffer", post_taa_color.1)
            .rw_texture("rw_target_buffer", present_target)
            .group_count(
                div_round_up(main_size.width, 8),
                div_round_up(main_size.height, 4),
                1,
            );

        // Pass: Output
        rg.present(present_target);

        // Prepare command buffer
        let command_buffer = self.stream_lined.wait_and_reset_command_buffer(rd);

        // Update frame CB (before submit)
        let exposure = 20.0;
        let sun_inten = Vec3::new(1.0, 1.0, 0.85) * exposure;
        self.stream_lined.update_frame_params(FrameParams::make(
            &view_info,
            self.taa.prev_view_info.as_ref(),
            &scene.sun_dir,
            &sun_inten,
        ));

        // only after last use of prev_view_info
        self.taa.prev_view_info.replace(*view_info);

        // Execute render graph
        {
            // Begin
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                rd.device_entry
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
            }

            let cb = CommandBuffer::new(rd, command_buffer);
            rg.execute(rd, &cb, shaders, &mut self.render_graph_cache);

            // End
            unsafe {
                rd.device_entry.end_command_buffer(command_buffer).unwrap();
            }
        }

        // Submit, Present and stuff
        self.stream_lined
            .wait_and_submit_and_present(rd, swapchain_image_index);

        self.frame_index += 1;
    }
}
