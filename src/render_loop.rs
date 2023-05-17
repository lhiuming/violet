use std::mem;

use ash::vk;

use crate::float4x4;
use crate::render_device::{create_buffer, Buffer, RenderDevice};
use crate::shader::Pipeline;

pub struct RenderScene {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub mesh_gfx_pipeline: Option<Pipeline>,
    pub mesh_cs_pipeline: Option<Pipeline>,
}

#[repr(C)]
pub struct ViewParams {
    pub view_proj: float4x4,
}

pub struct RednerLoop {
    pub view_params_cb: Buffer,
}

impl RednerLoop {
    pub fn new(rd: &RenderDevice) -> RednerLoop {
        // View parameter constant buffer
        let view_params_cb = create_buffer(
            &rd,
            mem::size_of::<ViewParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::Format::UNDEFINED,
        )
        .unwrap();

        RednerLoop { view_params_cb }
    }

    pub fn render(&self, rd: &RenderDevice, scene: &RenderScene, view_proj: float4x4) {
        let device = &rd.device;
        let physical_device = &rd.physical_device;
        let surface_entry = &rd.surface_entry;
        let swapchain_entry = &rd.swapchain_entry;
        let dynamic_rendering_entry = &rd.dynamic_rendering_entry;
        let gfx_queue = &rd.gfx_queue;
        let cmd_buf = rd.cmd_buf;
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
                .resolve_mode(vk::ResolveModeFlags::NONE)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color });
            let color_attachments = [*color_attachment];
            let rendering_info = vk::RenderingInfoKHR::builder()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: surface_size,
                })
                .layer_count(1)
                .view_mask(0)
                .color_attachments(&color_attachments);
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
            if let Some(vb_srv) = scene.vertex_buffer.srv {
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
                    scene.index_buffer.buffer,
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
                device.cmd_draw_indexed(cmd_buf, scene.index_count, 1, 0, 0, 0);
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
