use ash::vk;
use glam::Vec3;

use crate::{
    command_buffer::*,
    imgui,
    render_device::{
        Buffer, BufferDesc, RenderDevice, ShaderBindingTableFiller, Texture, TextureDesc,
        TextureView, TextureViewDesc,
    },
    render_graph,
    render_scene::*,
    shader::{Handle, Pipeline, RayTracingDesc, ShaderDefinition, Shaders, ShadersConfig},
};

use super::{RenderLoopDesciptorSets, ViewInfo, FRAME_DESCRIPTOR_SET_INDEX};

use super::{gbuffer_pass::*, RenderLoop};

pub struct RayTracedShadowResources {
    pub shader_binding_table: Buffer,
    pub prev_pipeline_handle: Handle<Pipeline>,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
}

impl RayTracedShadowResources {
    pub fn new(rd: &RenderDevice) -> RayTracedShadowResources {
        let sbt_size = 256; // should be big engough
        let sbt = rd
            .create_buffer(BufferDesc {
                size: sbt_size,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        let handle_size = rd
            .physical_device
            .ray_tracing_pipeline_properties
            .shader_group_handle_size as vk::DeviceSize;
        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: handle_size,
            size: handle_size,
        };

        let group_alignment = rd
            .physical_device
            .ray_tracing_pipeline_properties
            .shader_group_base_alignment as vk::DeviceSize;
        let miss_offset = std::cmp::max(handle_size, group_alignment);
        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + miss_offset,
            stride: handle_size,
            size: handle_size,
        };

        RayTracedShadowResources {
            shader_binding_table: sbt,
            prev_pipeline_handle: Handle::null(),
            raygen_region,
            miss_region,
        }
    }

    pub fn update_shader_group_handles(
        &mut self,
        rd: &RenderDevice,
        shaders: &Shaders,
        pipeline_handle: Handle<Pipeline>,
    ) {
        if self.prev_pipeline_handle != pipeline_handle {
            if let Some(pipeline) = shaders.get_pipeline(pipeline_handle) {
                self.fill_shader_group_handles(rd, pipeline);
                self.prev_pipeline_handle = pipeline_handle;
            }
        }
    }

    fn fill_shader_group_handles(&self, rd: &RenderDevice, pipeline: &Pipeline) {
        let handle_data = rd.get_ray_tracing_shader_group_handles(pipeline.handle, 0, 2);

        let mut filler =
            ShaderBindingTableFiller::new(&rd.physical_device, self.shader_binding_table.data);
        filler.write_handles(&handle_data, 0, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 1, 1);
    }
}

pub struct PathTracedLightingResources {
    pub accumulated_texture: Texture,
    pub accumulated_texture_view: TextureView,
    pub accumulated_count: u32,

    pub prev_sun_dir: Vec3,

    pub shader_binding_table: Buffer,
    pub prev_pipeline_handle: Handle<Pipeline>,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl PathTracedLightingResources {
    pub fn new(image_size: vk::Extent2D, rd: &RenderDevice) -> Self {
        let handle_size = rd
            .physical_device
            .ray_tracing_pipeline_properties
            .shader_group_handle_size as vk::DeviceSize;
        let group_alignment = rd
            .physical_device
            .ray_tracing_pipeline_properties
            .shader_group_base_alignment as vk::DeviceSize;
        let group_stride =
            ((handle_size + group_alignment - 1) / group_alignment) * group_alignment;

        let sbt_size = group_stride * 3;
        let sbt = rd
            .create_buffer(BufferDesc {
                size: sbt_size,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: handle_size,
            size: handle_size,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + group_stride,
            stride: handle_size,
            size: handle_size,
        };

        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + group_stride * 2,
            stride: handle_size,
            size: handle_size,
        };

        let accumulated_texture = rd
            .create_texture(TextureDesc::new_2d(
                image_size.width,
                image_size.height,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            ))
            .unwrap();
        let accumulated_texture_view = rd
            .create_texture_view(
                accumulated_texture,
                TextureViewDesc::auto(&accumulated_texture.desc),
            )
            .unwrap();

        Self {
            shader_binding_table: sbt,
            prev_pipeline_handle: Handle::null(),
            raygen_region,
            miss_region,
            hit_region,
            prev_sun_dir: Vec3::ZERO,
            accumulated_texture,
            accumulated_texture_view,
            accumulated_count: 0,
        }
    }

    pub fn update_shader_group_handles(
        &mut self,
        rd: &RenderDevice,
        shaders: &Shaders,
        pipeline_handle: Handle<Pipeline>,
    ) {
        // Likely path
        if self.prev_pipeline_handle == pipeline_handle {
            return;
        }

        let pipeline = match shaders.get_pipeline(pipeline_handle) {
            Some(p) => p,
            None => return,
        };

        let handle_data = rd.get_ray_tracing_shader_group_handles(pipeline.handle, 0, 3);

        let mut filler =
            ShaderBindingTableFiller::new(&rd.physical_device, self.shader_binding_table.data);
        filler.write_handles(&handle_data, 0, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 1, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 2, 1);

        self.prev_pipeline_handle = pipeline_handle;
    }
}

pub struct PhysicallyBasedRenderLoop {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub descriptor_sets: RenderLoopDesciptorSets,

    pub present_finished_fence: vk::Fence,
    pub present_finished_semephore: vk::Semaphore,
    pub present_ready_semaphore: vk::Semaphore,
    pub command_buffer_finished_fence: vk::Fence,

    pub frame_index: u32,

    pub render_graph_cache: render_graph::RenderGraphCache,

    pub raytraced_shadow: RayTracedShadowResources,
    pub pathtraced_lighting: PathTracedLightingResources,
}

impl RenderLoop for PhysicallyBasedRenderLoop {
    fn new(rd: &RenderDevice) -> PhysicallyBasedRenderLoop {
        let command_pool = rd.create_command_pool();
        let command_buffer = rd.create_command_buffer(command_pool);
        let descriptor_sets = RenderLoopDesciptorSets::new(rd, 1);
        let image_size = rd.swapchain.extent;
        PhysicallyBasedRenderLoop {
            command_pool,
            command_buffer,
            descriptor_sets,
            present_finished_fence: rd.create_fence(false),
            present_finished_semephore: rd.create_semaphore(),
            present_ready_semaphore: rd.create_semaphore(),
            command_buffer_finished_fence: rd.create_fence(true),
            render_graph_cache: render_graph::RenderGraphCache::new(rd),
            frame_index: 0,
            raytraced_shadow: RayTracedShadowResources::new(rd),
            pathtraced_lighting: PathTracedLightingResources::new(image_size, rd),
        }
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
        _imgui: Option<&imgui::ImGUIOuput>,
    ) {
        let command_buffer = self.command_buffer;
        let frame_descriptor_set = self.descriptor_sets.sets[0];

        use render_graph::*;
        let mut rg = RenderGraphBuilder::new();
        rg.add_global_descriptor_sets(&[
            (SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set),
            (FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set),
        ]);

        // Stupid shader compiling hack
        let mut hack = ShadersConfig {
            set_layout_override: std::collections::HashMap::new(),
        };
        hack.set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        hack.set_layout_override
            .insert(FRAME_DESCRIPTOR_SET_INDEX, self.descriptor_sets.set_layout);

        // Acquire target image
        let (image_index, b_image_suboptimal) = unsafe {
            rd.swapchain_entry.acquire_next_image(
                rd.swapchain.handle,
                u64::MAX,
                vk::Semaphore::null(),
                self.present_finished_fence,
            )
        }
        .expect("Vulkan: failed to acquire swapchain image");
        if b_image_suboptimal {
            println!("Vulkan: suboptimal image is get (?)");
        }

        // Wait for the fence such that the command buffer must be done
        unsafe {
            /*
            let fences = [
                self.present_finished_fence,
                self.command_buffer_finished_fence,
            ];
            let timeout_in_ns = 500000; // 500ms
            device
                .wait_for_fences(&fences, true, timeout_in_ns)
                .expect("Vulkan: wait for fence timtout for 500000");
             */

            let wait_fence = |fence, msg| {
                let fences = [fence];
                let mut timeout_in_ns = 100 * 1000000; // 100ms
                let timeout_max = 5000 * 1000000; // 5s
                loop {
                    match rd
                        .device_entry
                        .wait_for_fences(&fences, true, timeout_in_ns)
                    {
                        Ok(_) => return,
                        Err(_) => {
                            println!(
                                "Vulkan: Failed to wait {} in {}ms, keep trying...",
                                msg,
                                timeout_in_ns / 1000000
                            );
                        }
                    }
                    timeout_in_ns = std::cmp::min(timeout_in_ns * 2, timeout_max);
                }
            };

            wait_fence(self.present_finished_fence, "present_finished");
            wait_fence(
                self.command_buffer_finished_fence,
                "command_buffer_finished",
            );

            // Reset the fence
            rd.device_entry
                .reset_fences(&[
                    self.present_finished_fence,
                    self.command_buffer_finished_fence,
                ])
                .unwrap();
        }

        // Reuse the command buffer
        unsafe {
            rd.device_entry
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Vulkan: Reset command buffer failed???");
        };

        // Update GPU ViewParams const buffer
        /*
        {
            // TODO
            let exposure = 20.0;
            let sun_inten = Vec3::new(1.0, 1.0, 0.85) * exposure;

            let params = FrameParams::make(view_info, None, &scene.sun_dir, &sun_inten);
            self.descriptor_sets.update_frame_params(0, params);
        }
        */

        // Being command recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                rd.device_entry
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
            }
        }

        // Update sky IBL cube
        let skycube_size = 64u32;
        let mut skycube_texture = None;
        let skycube_gen = shaders
            .create_compute_pipeline(ShaderDefinition::compute("sky_cube.hlsl", "main"), &hack);
        if let Some(_) = skycube_gen {
            skycube_texture = Some(rg.create_texutre(TextureDesc {
                width: skycube_size,
                height: skycube_size,
                layer_count: 6,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::CUBE_COMPATIBLE, // required for viewed as cube
                ..Default::default()
            }));
            /*
            let array_uav = rg.create_texture_view(
                skycube_texture.unwrap(),
                Some(TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                    format: vk::Format::B10G11R11_UFLOAT_PACK32,
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..Default::default()
                }),
            );

            rg.new_pass("Sky IBL gen", RenderPassType::Compute)
                .pipeline(pipeline)
                .descritpro_set(FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set)
                .rw_texture("rw_cube_texture", array_uav)
                .push_constant(&(skycube_size as f32))
                .render(move |cb, _, _| {
                    cb.dispatch(skycube_size / 8, skycube_size / 4, 6);
                });
             */
        }

        let skycube = rg.create_texture_view(
            skycube_texture.unwrap(),
            Some(TextureViewDesc {
                view_type: vk::ImageViewType::CUBE,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            }),
        );

        // Define GBuffer
        let gbuffer = create_gbuffer_textures(&mut rg, rd.swapchain.extent);

        /*
        // Draw mesh
        add_gbuffer_pass(&mut rg, &rd, hack, &common_sets, scene, &gbuffer);
        */

        let final_color =
            rg.register_texture_view(rd.swapchain.texture_views[image_index as usize]);

        /*
        // Draw (procedure) Sky
        let sky_pipeline = shaders.create_gfx_pipeline(
            ShaderDefinition::vert("sky_vsps.hlsl", "vs_main"),
            ShaderDefinition::frag("sky_vsps.hlsl", "ps_main"),
            &hack,
        );
        {
            let pipeline = sky_pipeline.unwrap();

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
                .descritpro_set(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set)
                .descritpro_set(FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set)
                .texture("skycube", skycube)
                .color_targets(&[ColorTarget {
                    view: final_color,
                    load_op: ColorLoadOp::Load,
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
        */

        let scene_tlas = rg.register_accel_struct(scene.scene_top_level_accel_struct.unwrap());

        /*
        let raytraced_shadow_mask = {
            let tex_desc = TextureDesc::new_2d(
                gbuffer.size.width,
                gbuffer.size.height,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            );
            let rt_shadow_tex = rg.create_texutre(tex_desc);
            rg.create_texture_view(rt_shadow_tex, None)
        };

        // Ray Traced Shadow
        if let Some(raytraced_shadow) = shaders.create_raytracing_pipeline(
            ShaderDefinition::raygen("raytraced_shadow.hlsl", "raygen"),
            ShaderDefinition::miss("raytraced_shadow.hlsl", "miss"),
            None,
            &hack,
        ) {
            self.raytraced_shadow
                .update_shader_group_handles(rd, shaders, raytraced_shadow);
            let raygen_sbt = self.raytraced_shadow.raygen_region;
            let miss_sbt = self.raytraced_shadow.miss_region;

            rg.new_pass("RayTracedShadow", RenderPassType::RayTracing)
                .pipeline(raytraced_shadow)
                .descritpro_set(FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set)
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .rw_texture("rw_shadow", raytraced_shadow_mask)
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(raytraced_shadow).unwrap();
                    cb.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.handle);
                    cb.trace_rays(
                        &raygen_sbt,
                        &miss_sbt,
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        gbuffer.size.width,
                        gbuffer.size.height,
                        1,
                    )
                });
        }
        */

        /*
        // GBuffer Lighting
        let gbuffer_lighting = shaders.create_compute_pipeline(
            ShaderDefinition::compute("gbuffer_lighting.hlsl", "main"),
            &hack,
        );
        if let Some(pipeline) = gbuffer_lighting {
            rg.new_pass("GBuffer_Lighting", RenderPassType::Compute)
                .pipeline(pipeline)
                .descritpro_set(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set)
                .descritpro_set(FRAME_DESCRIPTOR_SET_INDEX, frame_descriptor_set)
                .texture("gbuffer_depth", gbuffer.depth.1)
                .texture("gbuffer_color", gbuffer.color.1)
                .texture("skycube", skycube)
                .texture("shadow_mask", raytraced_shadow_mask)
                .rw_texture("out_lighting", final_color)
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();
                    cb.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline.handle);
                    let dispatch_x = (gbuffer.size.width + 7) / 8;
                    let diapatch_y = (gbuffer.size.height + 7) / 8;
                    cb.dispatch(dispatch_x, diapatch_y, 1);
                });
        }
        */

        // Reset path tracing if camera or sun moved
        if view_info.moved {
            self.pathtraced_lighting.accumulated_count = 0;
        }
        if !self
            .pathtraced_lighting
            .prev_sun_dir
            .abs_diff_eq(scene.sun_dir, 0.000001)
        {
            self.pathtraced_lighting.accumulated_count = 0;
            self.pathtraced_lighting.prev_sun_dir = scene.sun_dir;
        }

        // Pathtraced Lighting
        if let Some(pipeline) = shaders.create_raytracing_pipeline(
            ShaderDefinition::raygen("pathtraced_lighting.hlsl", "raygen"),
            ShaderDefinition::miss("pathtraced_lighting.hlsl", "miss"),
            Some(ShaderDefinition::closesthit(
                "pathtraced_lighting.hlsl",
                "closesthit",
            )),
            &RayTracingDesc::default(),
            &hack,
        ) {
            self.pathtraced_lighting
                .update_shader_group_handles(&rd, &shaders, pipeline);
            let raygen_region = self.pathtraced_lighting.raygen_region;
            let miss_region = self.pathtraced_lighting.miss_region;
            let hit_region = self.pathtraced_lighting.hit_region;

            self.pathtraced_lighting.accumulated_count += 1;

            let frame_index = self.frame_index;
            let accumulated_count = self.pathtraced_lighting.accumulated_count;
            let rw_accumulated =
                rg.register_texture_view(self.pathtraced_lighting.accumulated_texture_view);

            // Trace
            rg.new_raytracing("PathTracedLighting")
                .raygen_shader_with_ep("pathtraced_lighting.hlsl", "raygen")
                .miss_shader_with_ep("pathtraced_lighting.hlsl", "miss")
                .closest_hit_shader_with_ep("pathtraced_lighting.hlsl", "closesthit")
                .accel_struct("scene_tlas", scene_tlas)
                .texture("skycube", skycube)
                .rw_texture("rw_accumulated", rw_accumulated)
                .rw_texture("rw_lighting", final_color)
                .push_constant(&frame_index)
                .push_constant(&accumulated_count)
                .render(move |cb, _| {
                    cb.trace_rays(
                        &raygen_region,
                        &miss_region,
                        &hit_region,
                        &Default::default(),
                        gbuffer.size.width,
                        gbuffer.size.height,
                        1,
                    );
                });
        }

        // Insert output pass
        let swapchain_view_handle =
            rg.register_texture_view(rd.swapchain.texture_views[image_index as usize]);
        rg.present(swapchain_view_handle);

        // Run render graph and fianlize
        {
            let cb = CommandBuffer::new(&rd, command_buffer);
            rg.execute(rd, &cb, shaders, &mut self.render_graph_cache);
        }

        // End command recoding
        unsafe {
            rd.device_entry.end_command_buffer(command_buffer).unwrap();
        }

        // Submit
        {
            let command_buffers = [command_buffer];
            let signal_semaphores = [self.present_ready_semaphore];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                let rt = rd.device_entry.queue_submit(
                    rd.gfx_queue,
                    &[*submit_info],
                    self.command_buffer_finished_fence,
                );
                if let Err(e) = rt {
                    match e {
                        vk::Result::ERROR_DEVICE_LOST => {
                            // Try nv tool
                            let len = rd
                                .nv_diagnostic_checkpoints_entry
                                .get_queue_checkpoint_data_len(rd.gfx_queue);
                            let mut cp = Vec::new();
                            cp.resize(len, vk::CheckpointDataNV::default());
                            rd.nv_diagnostic_checkpoints_entry
                                .get_queue_checkpoint_data(rd.gfx_queue, &mut cp);
                            println!("cp: {:?}", cp);
                        }
                        _ => {}
                    }
                }
                rt.unwrap()
            }
        }

        // Present
        {
            let mut present_info = vk::PresentInfoKHR::default();
            present_info.wait_semaphore_count = 1;
            present_info.p_wait_semaphores = &self.present_ready_semaphore;
            present_info.swapchain_count = 1;
            present_info.p_swapchains = &rd.swapchain.handle;
            present_info.p_image_indices = &image_index;
            unsafe {
                rd.swapchain_entry
                    .queue_present(rd.gfx_queue, &present_info)
                    .unwrap_or_else(|e| panic!("Failed to present: {:?}", e));
            }
        }

        self.frame_index += 1;
    }
}
