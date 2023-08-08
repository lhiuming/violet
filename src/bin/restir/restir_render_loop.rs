use std::f32::consts::PI;

use ash::vk;
use glam::Vec3;
use violet::{
    command_buffer::{CommandBuffer, StencilOps},
    render_device::{
        Buffer, RenderDevice, ShaderBindingTableFiller, Texture, TextureDesc, TextureViewDesc,
    },
    render_graph::*,
    render_loop::gbuffer_pass::*,
    render_loop::{
        div_round_up, FrameParams, RenderLoop, StreamLinedFrameResource, ViewInfo,
        FRAME_DESCRIPTOR_SET_INDEX,
    },
    render_scene::{RenderScene, SCENE_DESCRIPTOR_SET_INDEX},
    shader::{Pipeline, PushConstantsBuilder, ShaderDefinition, Shaders, ShadersConfig},
};

struct SampleGenPass {
    sbt: Buffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl SampleGenPass {
    fn new(rd: &RenderDevice) -> Self {
        let sbt = rd
            .create_buffer(
                256,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
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
    prev_color: RGTemporal<Texture>,
}

impl TAAPass {
    fn new() -> Self {
        Self {
            prev_color: RGTemporal::null(),
        }
    }
}

pub struct RestirRenderLoop {
    render_graph_cache: RenderGraphCache,
    stream_lined: StreamLinedFrameResource,

    frame_index: u32,

    sample_gen: SampleGenPass,
    taa: TAAPass,
}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        Self {
            render_graph_cache: RenderGraphCache::new(rd),
            stream_lined: StreamLinedFrameResource::new(rd),
            frame_index: 0,
            sample_gen: SampleGenPass::new(rd),
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

        // Persistent binding stuff
        let frame_descriptor_set = self.stream_lined.get_frame_desciptor_set();
        let common_sets = [
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set),
        ];

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

        // Create GBuffer
        let gbuffer = create_gbuffer_textures(&mut rg, main_size);

        // Pass: GBuffer
        add_gbuffer_pass(
            &mut rg,
            rd,
            shaders,
            &shader_config,
            &common_sets,
            scene,
            &gbuffer,
        );

        // Pass: Skycube update
        let mut skycube = RGHandle::null();
        if let Some(pipeline) = shaders.create_compute_pipeline(
            ShaderDefinition::compute("sky_cube.hlsl", "main"),
            &shader_config,
        ) {
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
                .render(move |cb, shaders, _| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();

                    let pc = PushConstantsBuilder::new()
                        .pushv(width as f32)
                        .push(&scene.sun_dir);
                    cb.push_constants(
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &pc.build(),
                    );

                    cb.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline.handle);
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
                .descritpro_sets(&common_sets)
                .texture("skycube", skycube)
                .color_targets(&[ColorTarget {
                    view: scene_color.1,
                    load_op: ColorLoadOp::Clear(vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    }),
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

        let scene_tlas = rg.register_accel_struct(scene.scene_top_level_accel_struct.unwrap());

        // Pass: sample generation
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

            rg.new_pass("Sample_Gen", RenderPassType::RayTracing)
                .pipeline(pipeline)
                .descritpro_sets(&common_sets)
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("skycube", skycube)
                .rw_texture("rw_debug_color", scene_color.1)
                .render(move |cb, shaders, _| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();
                    cb.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.handle);

                    let pc = PushConstantsBuilder::new().pushv(frame_index);
                    cb.push_constants(
                        pipeline.layout,
                        pipeline.push_constant_ranges[0].stage_flags,
                        0,
                        pc.build(),
                    );

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
        }

        let post_taa_color = {
            let desc = TextureDesc::new_2d(
                main_size.width,
                main_size.height,
                rd.swapchain.image[0].desc.format,
                vk::ImageUsageFlags::STORAGE // compute TAA
                    | vk::ImageUsageFlags::SAMPLED // history
                    | vk::ImageUsageFlags::TRANSFER_SRC, // copy to swapchain
            );
            let texture = rg.create_texutre(desc);
            let view = rg.create_texture_view(texture, None);
            (texture, view)
        };

        // Pass: TAA
        if self.taa.prev_color.is_null() {
            self.taa.prev_color =
                rg.convert_to_temporal(&mut self.render_graph_cache, scene_color.0);
        } else if let Some(pipeline) = shaders.create_compute_pipeline(
            ShaderDefinition::compute("temporal_aa.hlsl", "main"),
            &shader_config,
        ) {
            let history_tex = rg.convert_to_transient(self.taa.prev_color);
            let history = rg.create_texture_view(history_tex, None);

            rg.new_pass("Temporal AA", RenderPassType::Compute)
                .pipeline(pipeline)
                //.descritpro_sets(&common_sets)
                .texture("source", scene_color.1)
                .texture("history", history)
                .rw_texture("rw_target", post_taa_color.1)
                .render(move |cb, _, _| {
                    let group_count_x = div_round_up(main_size.width, 8);
                    let group_count_y = div_round_up(main_size.height, 4);
                    cb.dispatch(group_count_x, group_count_y, 1);
                });

            self.taa.prev_color =
                rg.convert_to_temporal(&mut self.render_graph_cache, post_taa_color.0);
        }

        let swapchain_image_index = self.stream_lined.acquire_next_swapchain_image(rd);
        let present_target =
            rg.register_texture_view(rd.swapchain.image_view[swapchain_image_index as usize]);

        // Pass: Copy to swapchain
        rg.new_pass("Copy for Present", RenderPassType::Copy)
            .copy_src(post_taa_color.1)
            .copy_dst(present_target);

        // Pass: Output
        rg.new_pass("Present", RenderPassType::Present)
            .present_texture(present_target);

        // Prepare command buffer
        let command_buffer = self.stream_lined.wait_and_reset_command_buffer(rd);

        // Update frame CB (before submit)
        let exposure = 5.0f32;
        let sun_inten = Vec3::new(0.7, 0.7, 0.6) * PI * exposure;
        self.stream_lined.update_frame_params(FrameParams::make(
            &view_info,
            &scene.sun_dir,
            &sun_inten,
        ));

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
