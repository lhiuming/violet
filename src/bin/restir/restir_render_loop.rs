use ash::vk;
use glam::Vec3;
use violet::{
    command_buffer::{CommandBuffer, StencilOps},
    render_device::{
        Buffer, BufferDesc, RenderDevice, ShaderBindingTableFiller, Texture, TextureDesc,
        TextureView, TextureViewDesc,
    },
    render_graph::*,
    render_loop::gbuffer_pass::*,
    render_loop::{
        div_round_up, rg_util, FrameParams, RenderLoop, StreamLinedFrameResource, ViewInfo,
        FRAME_DESCRIPTOR_SET_INDEX,
    },
    render_scene::{RenderScene, SCENE_DESCRIPTOR_SET_INDEX},
    shader::{Pipeline, PushConstantsBuilder, ShaderDefinition, Shaders, ShadersConfig},
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
    sbt: Buffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,

    prev_reservoir_buffer: Option<RGTemporal<Buffer>>,
}

impl SampleGenPass {
    fn new(rd: &RenderDevice) -> Self {
        let sbt = rd
            .create_buffer(BufferDesc {
                size: 256,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        let handle_size = rd.physical_device.shader_group_handle_size() as u64;
        let stride = std::cmp::max(
            handle_size,
            rd.physical_device.shader_group_base_alignment() as u64,
        );

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: handle_size,
            size: handle_size,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + stride,
            stride: handle_size,
            size: handle_size,
        };

        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + stride * 2,
            stride: handle_size,
            size: handle_size,
        };

        Self {
            sbt,
            raygen_region,
            miss_region,
            hit_region,
            prev_reservoir_buffer: None,
        }
    }

    fn update_shader_group_handles(&mut self, rd: &RenderDevice, pipeline: &Pipeline) {
        // TODO check shader change
        // TODO add to stream line to avoid write on GPU using
        let handle_data = rd.get_ray_tracing_shader_group_handles(pipeline.handle, 0, 3);
        let mut filler = ShaderBindingTableFiller::new(&rd.physical_device, self.sbt.data);
        filler.write_handles(&handle_data, 0, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 1, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 2, 1);
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

pub struct RaytracedShadow {
    pub sbt: Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
}

impl RaytracedShadow {
    pub fn new(rd: &RenderDevice) -> Self {
        let handle_size = rd.physical_device.shader_group_handle_size() as u64;
        let stride = std::cmp::max(
            handle_size,
            rd.physical_device.shader_group_base_alignment() as u64,
        );

        let sbt = rd
            .create_buffer(BufferDesc::shader_binding_table(handle_size + stride))
            .unwrap();

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: handle_size,
            size: handle_size,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + stride,
            stride: handle_size,
            size: handle_size,
        };

        Self {
            sbt,
            raygen_region,
            miss_region,
        }
    }

    pub fn update_shader_group_handle(&mut self, rd: &RenderDevice, pipeline: &Pipeline) {
        let handle_data = rd.get_ray_tracing_shader_group_handles(pipeline.handle, 0, 2);
        let mut filler = ShaderBindingTableFiller::new(&rd.physical_device, self.sbt.data);
        filler.write_handles(&handle_data, 0, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 1, 1);
    }
}

pub struct RestirRenderLoop {
    render_graph_cache: RenderGraphCache,
    stream_lined: StreamLinedFrameResource,
    default_res: DefaultResources,

    frame_index: u32,

    sample_gen: SampleGenPass,
    raytraced_shadow: RaytracedShadow,
    taa: TAAPass,
}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            stream_lined: StreamLinedFrameResource::new(rd),
            default_res: DefaultResources::new(rd),
            frame_index: 0,
            sample_gen: SampleGenPass::new(rd),
            raytraced_shadow: RaytracedShadow::new(rd),
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

        let mut rg = RenderGraphBuilder::new();

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
        add_gbuffer_pass(&mut rg, rd, shaders, &shader_config, &[], scene, &gbuffer);

        // Pass: Skycube update
        let skycube;
        {
            let pipeline = shaders
                .create_compute_pipeline(
                    ShaderDefinition::compute("sky_cube.hlsl", "main"),
                    &shader_config,
                )
                .unwrap();

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

            rg.new_pass("SkyCubeGen", RenderPassType::Compute)
                .pipeline(pipeline)
                .rw_texture("rw_cube_texture", uav)
                .push_constant(&(width as f32))
                .render(move |cb, _, _| {
                    cb.dispatch(width / 8, width / 4, 6);
                });

            skycube = rg.create_texture_view(
                skycube_texture,
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::CUBE,
                    ..TextureViewDesc::auto(&desc)
                }),
            );
        }

        let scene_tlas = rg.register_accel_struct(scene.scene_top_level_accel_struct.unwrap());

        let has_prev_frame;
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

        let temporal_on_temporal = false;

        // Pass: sample generation (and temporal reusing)
        let main_len = main_size.width * main_size.height;
        let temporal_reservoir_buffer =
            rg.create_buffer(BufferDesc::compute(main_len as u64 * 24 * 4));
        if let Some(pipeline) = shaders.create_raytracing_pipeline(
            ShaderDefinition::raygen("restir/sample_gen.hlsl", "raygen"),
            ShaderDefinition::miss("restir/sample_gen.hlsl", "miss"),
            Some(ShaderDefinition::closesthit(
                "restir/sample_gen.hlsl",
                "closesthit",
            )),
            &shader_config,
        ) {
            self.sample_gen
                .update_shader_group_handles(rd, shaders.get_pipeline(pipeline).unwrap());

            let frame_index = self.frame_index;
            let raygen_sbt = self.sample_gen.raygen_region;
            let miss_sbt = self.sample_gen.miss_region;
            let hit_sbt = self.sample_gen.hit_region;

            let use_prev_frame: u32;
            let prev_reservoir_buffer = if let Some(buffer) = self.sample_gen.prev_reservoir_buffer
            {
                use_prev_frame = 1;
                rg.convert_to_transient(buffer)
            } else {
                use_prev_frame = 0;
                rg.register_buffer(self.default_res.dummy_buffer)
            };

            rg.new_pass("Sample Gen", RenderPassType::RayTracing)
                .pipeline(pipeline)
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("skycube", skycube)
                .texture("prev_color", prev_color)
                .texture("prev_depth", prev_depth)
                .buffer("prev_reservoir_buffer", prev_reservoir_buffer)
                .rw_buffer("rw_temporal_reservoir_buffer", temporal_reservoir_buffer)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&frame_index)
                .push_constant(&use_prev_frame)
                .render(move |cb, _, _| {
                    cb.trace_rays(
                        &raygen_sbt,
                        &miss_sbt,
                        &hit_sbt,
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        main_size.width,
                        main_size.height,
                        1,
                    );
                });

            if temporal_on_temporal {
                let temporal =
                    rg.convert_to_temporal(&mut self.render_graph_cache, temporal_reservoir_buffer);
                self.sample_gen.prev_reservoir_buffer.replace(temporal);
            }
        }

        // Pass: Spatial Resampling
        {
            let reservoir_buffer_desc = rg.get_buffer_desc(temporal_reservoir_buffer);
            let spatial_reservoir_buffer = rg.create_buffer(*reservoir_buffer_desc);

            let pipeline = shaders
                .create_compute_pipeline(
                    ShaderDefinition::compute("restir/spatial_resampling.hlsl", "main"),
                    &shader_config,
                )
                .unwrap();

            rg.new_pass("Spatial Resampling", RenderPassType::Compute)
                .pipeline(pipeline)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .buffer("temporal_reservoir_buffer", temporal_reservoir_buffer)
                .rw_buffer("rw_spatial_reservoir_buffer", spatial_reservoir_buffer)
                .rw_texture("rw_lighting_texture", indirect_diffuse.1)
                //.rw_texture("rw_debug_texture", debug_texture.1)
                .push_constant(&self.frame_index)
                .render(|cb, _, _| {
                    cb.dispatch(
                        div_round_up(main_size.width, 8),
                        div_round_up(main_size.height, 4),
                        1,
                    );
                });

            if !temporal_on_temporal {
                let temporal =
                    rg.convert_to_temporal(&mut self.render_graph_cache, spatial_reservoir_buffer);
                self.sample_gen.prev_reservoir_buffer.replace(temporal);
            }
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
            let pipeline = shaders
                .create_raytracing_pipeline(
                    ShaderDefinition::raygen("raytraced_shadow.hlsl", "raygen"),
                    ShaderDefinition::miss("raytraced_shadow.hlsl", "miss"),
                    None,
                    &shader_config,
                )
                .unwrap();

            self.raytraced_shadow
                .update_shader_group_handle(rd, shaders.get_pipeline(pipeline).unwrap());
            let raygen_sbt = self.raytraced_shadow.raygen_region;
            let miss_sbt = self.raytraced_shadow.miss_region;

            rg.new_pass("Raytraced Shadow", RenderPassType::RayTracing)
                .pipeline(pipeline)
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .rw_texture("rw_shadow", raytraced_shadow_mask.1)
                .render(move |cb, _, _| {
                    let none_sbt = vk::StridedDeviceAddressRegionKHR::default();
                    cb.trace_rays(
                        &raygen_sbt,
                        &miss_sbt,
                        &none_sbt,
                        &none_sbt,
                        main_size.width,
                        main_size.height,
                        1,
                    );
                });
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
        if let Some(pipeline) = shaders.create_gfx_pipeline(
            ShaderDefinition::vert("sky_vsps.hlsl", "vs_main"),
            ShaderDefinition::frag("sky_vsps.hlsl", "ps_main"),
            &shader_config,
        ) {
            let stencil = rg.create_texture_view(
                gbuffer.depth.0,
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::D24_UNORM_S8_UINT,
                    aspect: vk::ImageAspectFlags::STENCIL,
                    ..Default::default()
                }),
            );

            rg.new_pass("Sky", RenderPassType::Graphics)
                .pipeline(pipeline)
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
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();

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
            let pipeline = shaders
                .create_compute_pipeline(
                    ShaderDefinition::compute("restir/final_lighting.hlsl", "main"),
                    &shader_config,
                )
                .unwrap();

            rg.new_pass("Combined Lighting", RenderPassType::Compute)
                .pipeline(pipeline)
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("shadow_mask_buffer", raytraced_shadow_mask.1)
                .texture("indirect_diffuse_buffer", indirect_diffuse.1)
                .rw_texture("rw_color_buffer", scene_color.1)
                .render(|cb, _, _| {
                    cb.dispatch(
                        div_round_up(main_size.width, 8),
                        div_round_up(main_size.height, 4),
                        1,
                    );
                });
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
        if let Some(pipeline) = shaders.create_compute_pipeline(
            ShaderDefinition::compute("temporal_aa.hlsl", "main"),
            &shader_config,
        ) {
            rg.new_pass("Temporal AA", RenderPassType::Compute)
                .pipeline(pipeline)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("source_texture", scene_color.1)
                .texture("history_texture", prev_color)
                .rw_texture("rw_target", post_taa_color.1)
                .push_constant(&has_prev_frame)
                .render(move |cb, shaders, _| {
                    let group_count_x = div_round_up(main_size.width, 8);
                    let group_count_y = div_round_up(main_size.height, 4);
                    cb.dispatch(group_count_x, group_count_y, 1);
                });

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
        {
            let pipeline = shaders
                .create_compute_pipeline(
                    ShaderDefinition::compute("post_processing.hlsl", "main"),
                    &shader_config,
                )
                .unwrap();

            rg.new_pass("Post Processing", RenderPassType::Compute)
                .pipeline(pipeline)
                .texture("src_color_buffer", post_taa_color.1)
                .rw_texture("rw_target_buffer", present_target)
                .render(|cb, _, _| {
                    cb.dispatch(
                        div_round_up(main_size.width, 8),
                        div_round_up(main_size.height, 4),
                        1,
                    );
                });
        }

        // Pass: Output
        rg.new_pass("Present", RenderPassType::Present)
            .present_texture(present_target);

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
