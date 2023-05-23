use std::mem::{self, size_of};

use ash::vk;

use crate::float4x4;
use crate::gltf_asset::GLTF;
use crate::render_device::{create_buffer, Buffer, RenderDevice};
use crate::shader::Pipeline;

// Allocatable buffer. Alway aligned to 4 bytes.
pub struct AllocBuffer 
{
    pub buffer: Buffer,
    next_pos: u32,
}

impl AllocBuffer {
    pub fn new(buffer: Buffer) -> AllocBuffer {
        AllocBuffer { buffer: buffer, next_pos: 0 }
    }

    pub fn alloc<'a, T>(&mut self, count: u32) -> (&'a mut [T], u32) {
        assert!((size_of::<T>() as u32 * count)<= (self.buffer.size as u32 - self.next_pos));
        let pos = self.next_pos;
        let size = size_of::<T>() as u32 * count; 
        self.next_pos += (size + 3) & !3; // always aligned to 4 bytes
        let slice = unsafe {
            let ptr = self.buffer.data.offset(pos as isize);
            std::slice::from_raw_parts_mut(ptr as *mut T, count as usize)
        };
        (slice, pos)
    }
}

// Contain everything to be rendered
pub struct RenderScene {
    // Global buffer to store all loaded meshes
    pub vertex_buffer: AllocBuffer,
    pub index_buffer: AllocBuffer,

    // Default shaders/pipelines
    pub mesh_gfx_pipeline: Option<Pipeline>,
    pub mesh_cs_pipeline: Option<Pipeline>,

    // The loaded GLFT. We have only one :)
    pub gltf: Option<GLTF>,
}

#[repr(C)]
pub struct ViewParams {
    pub view_proj: float4x4,
}

pub struct RednerLoop {
    pub view_params_cb: Buffer,
    pub depth_buffer: vk::Image,
    pub depth_buffer_view: vk::ImageView,
}

impl RednerLoop {
    pub fn new(rd: &RenderDevice) -> RednerLoop {
        let surface_size = rd.surface.query_size(&rd.surface_entry, &rd.physical_device);
        let swapchain_size = rd.swapchain.extent;
        assert_eq!(surface_size, swapchain_size);

        // View parameter constant buffer
        let view_params_cb = create_buffer(
            &rd,
            mem::size_of::<ViewParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::Format::UNDEFINED,
        )
        .unwrap();

        // Create depth buffer
        let create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::D16_UNORM)
        .extent(vk::Extent3D {
            width: surface_size.width,
            height: surface_size.height,
            depth: 1,
        })
        .array_layers(1)
        .mip_levels(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        ;
        let depth_buffer = unsafe { rd.device.create_image(&create_info, None) }.unwrap();

        // Bind depth buffer memory
        let depth_buffer_memory = {
            let mem_requirements = unsafe { rd.device.get_image_memory_requirements(depth_buffer) };
            let momory_type_index = rd.pick_memory_type_index(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap(); // TODO
            let create_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(momory_type_index);
            unsafe { rd.device.allocate_memory(&create_info, None) }.unwrap()
        };
        unsafe { rd.device.bind_image_memory(depth_buffer, depth_buffer_memory, 0) }.unwrap();

        let create_info = vk::ImageViewCreateInfo::builder()
        .image(depth_buffer)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk::Format::D16_UNORM)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        ;
        let depth_buffer_view = unsafe { rd.device.create_image_view(&create_info, None) }.unwrap();

        RednerLoop { view_params_cb, depth_buffer, depth_buffer_view }
    }

    pub fn render(&self, rd: &RenderDevice, scene: &RenderScene, view_proj: float4x4) {
        let device = &rd.device;
        let physical_device = &rd.physical_device;
        let surface_entry = &rd.surface_entry;
        let swapchain_entry = &rd.swapchain_entry;
        let dynamic_rendering_entry = &rd.dynamic_rendering_entry;
        let gfx_queue = &rd.gfx_queue;
        let cmd_buf = rd.cmd_buf;
        let command_buffer = rd.cmd_buf; // rust naming convention
        let surface = &rd.surface;
        let swapchain = &rd.swapchain;
        let present_semaphore = &rd.present_semaphore;

        let surface_size = surface.query_size(&surface_entry, physical_device);

        // wait idle (for now)
        unsafe {
            device
                .reset_command_buffer(cmd_buf, vk::CommandBufferResetFlags::empty())
                .expect("Vulkan: Reset command buffer failed???");
        };

        // Update GPU ViewParams const buffer
        {
            let view_params = ViewParams {
                view_proj: view_proj,
            };

            unsafe {
                std::ptr::copy_nonoverlapping(
                    std::ptr::addr_of!(view_params),
                    self.view_params_cb.data as *mut ViewParams,
                    mem::size_of::<ViewParams>(),
                );
            }
        }

        // Acquire target image
        let (image_index, b_image_suboptimal) = unsafe {
            swapchain_entry.entry.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                *present_semaphore,
                vk::Fence::default(),
            )
        }
        .expect("Vulkan: failed to acquire swapchain image");
        if b_image_suboptimal {
            println!("Vulkan: suboptimal image is get (?)");
        }

        // SIMPLE CONFIGURATION
        let b_clear_using_render_pass = true;
        let clear_color = vk::ClearColorValue {
            float32: [
                0x5A as f32 / 255.0,
                0x44 as f32 / 255.0,
                0x94 as f32 / 255.0,
                0xFF as f32 / 255.0,
            ],
        };

        // Being command recording
        {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                device.begin_command_buffer(cmd_buf, &begin_info).unwrap();
            }
        }

        let mut swapchain_image_layout = vk::ImageLayout::UNDEFINED;

        // Transition for render
        if b_clear_using_render_pass {
            let sub_res_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(swapchain_image_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(*sub_res_range)
                .image(swapchain.image[image_index as usize]);
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier],
                );
            }

            swapchain_image_layout = image_barrier.new_layout;
        }

        // Begin render pass
        if rd.b_support_dynamic_rendering {
            let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(swapchain.image_view[image_index as usize])
                .image_layout(swapchain_image_layout)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color });
            let color_attachments = [*color_attachment];
            let depth_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(self.depth_buffer_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            let rendering_info = vk::RenderingInfoKHR::builder()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_size,
                })
                .layer_count(1)
                .view_mask(0)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);
            unsafe {
                dynamic_rendering_entry.cmd_begin_rendering(cmd_buf, &rendering_info);
            }
        }

        // Draw mesh
        if let Some(pipeline) = &scene.mesh_gfx_pipeline {
            // Set viewport and scissor
            {
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: swapchain.extent.width as f32,
                    height: swapchain.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let viewports = [viewport];
                unsafe {
                    device.cmd_set_viewport(cmd_buf, 0, &viewports);
                }

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain.extent,
                };
                let scissors = [scissor];
                unsafe {
                    device.cmd_set_scissor(cmd_buf, 0, &scissors);
                }
            }

            // Bind shader resources
            if let Some(vb_srv) = scene.vertex_buffer.buffer.srv {
                // TODO map desriptor binding name
                let buffer_views = [vb_srv];
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(pipeline.descriptor_sets[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                    .texel_buffer_view(&buffer_views);
                let cb_info = vk::DescriptorBufferInfo::builder()
                    .buffer(self.view_params_cb.buffer)
                    .range(vk::WHOLE_SIZE);
                let cb_infos = [*cb_info];
                let write_cb = vk::WriteDescriptorSet::builder()
                    .dst_set(pipeline.descriptor_sets[1])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&cb_infos);
                let writes = [*write, *write_cb];
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &pipeline.descriptor_sets,
                        &[],
                    );
                }
            }

            // Bind index buffer
            unsafe {
                device.cmd_bind_index_buffer(
                    cmd_buf,
                    scene.index_buffer.buffer.buffer,
                    0,
                    vk::IndexType::UINT16,
                );
            }

            // Set pipeline and Draw
            unsafe {
                device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            }
            unsafe {
                //device.cmd_draw(cmd_buf, 3, 1, 0, 0);
                //device.cmd_draw_indexed(cmd_buf, scene.index_count, 1, 0, 0, 0);
                if let Some(gltf) = &scene.gltf {
                    for scene in &gltf.scenes {
                        for node in &scene.nodes {
                            if let Some(mesh_index) = node.mesh_index {
                                // Send transform with PushConstnat
                                // NOTE: 12 floats for model transform, 1 uint for pos_offset, 1 uint for uv_offset
                                let mut constants: [u8; 4 * 12] = mem::zeroed();
                                {
                                    // model_transform
                                    let dst_ptr = constants.as_mut_ptr() as *mut f32;
                                    let dst = std::slice::from_raw_parts_mut(dst_ptr.offset(0), 4);
                                    node.transform.row(0).write_to_slice(dst);
                                    let dst = std::slice::from_raw_parts_mut(dst_ptr.offset(4), 4);
                                    node.transform.row(1).write_to_slice(dst);
                                    let dst = std::slice::from_raw_parts_mut(dst_ptr.offset(8), 4);
                                    node.transform.row(2).write_to_slice(dst);
                                }
                                device.cmd_push_constants(
                                    cmd_buf, 
                                    pipeline.layout,
                                    vk::ShaderStageFlags::VERTEX,
                                    0, 
                                    &constants);

                                // Draw mesh primitives
                                let mesh = &gltf.meshes[mesh_index as usize];
                                for primitive in &mesh.primitives {
                                    // Geomtry data offset is per primitive
                                    let mut constants: [u8; 4 * 2] = mem::zeroed();
                                    {
                                        let dst = std::slice::from_raw_parts_mut(constants.as_mut_ptr() as *mut u32, 2);
                                        dst[0] = primitive.positions_offset;
                                        dst[1] = primitive.texcoords_offsets[0];
                                    }
                                    device.cmd_push_constants(
                                        command_buffer, 
                                        pipeline.layout,
                                        vk::ShaderStageFlags::VERTEX,
                                        4 * 12, 
                                        &constants);

                                    device.cmd_draw_indexed(
                                        cmd_buf,
                                        primitive.index_count,
                                        1,
                                        primitive.index_offset,
                                        0,
                                        0,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // End render pass
        unsafe {
            dynamic_rendering_entry.cmd_end_rendering(cmd_buf);
        }

        // Draw something with compute
        if let Some(pipeline) = &scene.mesh_cs_pipeline {
            // Transition for compute
            {
                let sub_res_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1);
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(swapchain_image_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(*sub_res_range)
                    .image(swapchain.image[image_index as usize]);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd_buf,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[*image_barrier],
                    );
                }
                swapchain_image_layout = image_barrier.new_layout;
            }

            // Set and dispatch compute
            {
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.handle,
                    )
                }

                // Bind the swapchain image (the only descriptor)
                unsafe {
                    let image_info = vk::DescriptorImageInfo::builder()
                        .image_view(swapchain.image_view[image_index as usize])
                        .image_layout(swapchain_image_layout);
                    let image_infos = [image_info.build()];
                    let write = vk::WriteDescriptorSet::builder()
                        .dst_set(pipeline.descriptor_sets[0])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_infos);
                    let writes = [write.build()];
                    device.update_descriptor_sets(&writes, &[]);
                }

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout,
                        0,
                        &pipeline.descriptor_sets,
                        &[],
                    )
                }

                let dispatch_x = (swapchain.extent.width + 7) / 8;
                let dispatch_y = (swapchain.extent.height + 3) / 4;
                unsafe {
                    device.cmd_dispatch(cmd_buf, dispatch_x, dispatch_y, 1);
                }
            }
        }

        // Transition for present
        if swapchain_image_layout != vk::ImageLayout::PRESENT_SRC_KHR {
            let sub_res_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(swapchain_image_layout)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(*sub_res_range)
                .image(swapchain.image[image_index as usize]);
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier],
                );
            }
            //swapchain_image_layout = image_barrier.new_layout;
        }

        // End command recoding
        unsafe {
            device.end_command_buffer(cmd_buf).unwrap();
        }

        // Submit
        {
            let command_buffers = [cmd_buf];
            let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
            unsafe {
                device
                    .queue_submit(*gfx_queue, &[*submit_info], vk::Fence::null())
                    .unwrap();
            }
        }

        // Present
        {
            let mut present_info = vk::PresentInfoKHR::default();
            present_info.wait_semaphore_count = 1;
            present_info.p_wait_semaphores = present_semaphore;
            present_info.swapchain_count = 1;
            present_info.p_swapchains = &swapchain.handle;
            present_info.p_image_indices = &image_index;
            unsafe {
                swapchain_entry
                    .entry
                    .queue_present(*gfx_queue, &present_info)
                    .unwrap();
            }
        }

        // Wait brute-force (ATM)
        unsafe {
            device.device_wait_idle().unwrap();
        }
    }
}
