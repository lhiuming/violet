use std::collections::HashMap;

use ash::vk;
use glam::Vec2;

use crate::{
    imgui::ImGUIOuput,
    render_device::{Buffer, BufferDesc, RenderDevice, Texture, TextureDesc},
    render_graph::{ColorLoadOp, ColorTarget, PassBuilderTrait, RGHandle, RenderGraphBuilder},
    render_scene::UploadContext,
    shader::PushConstantsBuilder,
};

pub struct ImGUIPass {
    index_buffer: Buffer,
    vertex_buffer: Buffer,
    // TODO should do altas (easier binding!)
    textures: HashMap<u64, Texture>,
}

impl ImGUIPass {
    pub fn new(rd: &RenderDevice) -> Self {
        let index_buffer = rd
            .create_buffer(BufferDesc {
                size: 1024, // TODO growing?
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();
        let vertex_buffer = rd
            .create_buffer(BufferDesc {
                size: 2048, // TODO growing?
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | // sampled as raw buffer
                vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();
        Self {
            index_buffer,
            vertex_buffer,
            textures: HashMap::new(),
        }
    }

    pub fn add(
        &self,
        rg: &mut RenderGraphBuilder,
        rd: &RenderDevice,
        upload_context: &mut UploadContext,
        target: RGHandle<Texture>,
        imgui: &ImGUIOuput,
    ) {
        // TODO calculate this properly
        let index_data_size = 1024;
        let vertex_data_size = 2048;

        // Upload buffer / texture-delta to GPU

        let (ind_staging_buffer, ind_staging_event) =
            upload_context.borrow_staging_buffer(rd, index_data_size);
        let (vert_staging_buffer, vert_staging_event) =
            upload_context.borrow_staging_buffer(rd, vertex_data_size);

        let index_data = unsafe {
            std::slice::from_raw_parts_mut(
                ind_staging_buffer.data as *mut u32,
                (index_data_size / 4) as usize,
            )
        };
        let mut index_pos = 0;
        let vertex_data = unsafe {
            let ptr = vert_staging_buffer.data as *mut egui::epaint::Vertex;
            let len = vertex_data_size as usize / std::mem::size_of::<egui::epaint::Vertex>();
            std::slice::from_raw_parts_mut(ptr, len)
        };
        let mut vertex_pos = 0;

        // upload buffer data
        let mut mesh_offsets = Vec::new();
        let mut mesh_index_count = Vec::new();
        for cp in &imgui.clipped_primitives {
            match &cp.primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    mesh.texture_id;

                    mesh_offsets.push((index_pos as u32, vertex_pos as u32));
                    mesh_index_count.push(mesh.indices.len() as u32);

                    let ind_pos_end = index_pos + mesh.indices.len();
                    index_data[index_pos..ind_pos_end].copy_from_slice(&mesh.indices);
                    index_pos = ind_pos_end;

                    let vert_pos_end = vertex_pos + mesh.vertices.len();
                    vertex_data[vertex_pos..vert_pos_end].copy_from_slice(&mesh.vertices);
                    vertex_pos = vert_pos_end;
                }
                egui::epaint::Primitive::Callback(_) => {
                    mesh_offsets.push((u32::MAX, u32::MAX));
                    mesh_index_count.push(0);
                    unimplemented!()
                }
            }
        }

        let image_staging_buffer_size = imgui
            .textures_delta
            .set
            .iter()
            .map(|set| -> usize {
                let delta = set.1;
                let size = delta.image.size();
                let bytesize = delta.image.bytes_per_pixel() * size[0] * size[1];
                (bytesize + 3) & !3 // align to 4 bytes
            })
            .sum::<usize>();
        let (image_staging_buffer, image_staging_event) =
            upload_context.borrow_staging_buffer(rd, image_staging_buffer_size as u64);

        // upload image data
        let mut copy_buffer_to_images = Vec::new();
        let mut image_staging_byte_pos = 0;
        for (tex_id, delta) in &imgui.textures_delta.set {
            let tex_id = match tex_id {
                egui::TextureId::Managed(id) => *id,
                egui::TextureId::User(_) => todo!(),
            };

            // find or create the texture
            let image;
            match self.textures.entry(tex_id) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    image = entry.get().image;
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let format = match delta.image {
                        egui::ImageData::Color(_) => vk::Format::R8G8B8A8_SRGB,
                        egui::ImageData::Font(_) => vk::Format::R32_SFLOAT,
                    };
                    let texture = rd
                        .create_texture(TextureDesc::new_2d(
                            delta.image.width() as u32,
                            delta.image.height() as u32,
                            format,
                            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        ))
                        .unwrap();
                    image = texture.image;
                    entry.insert(texture);
                }
            }

            let data_bytesize;
            match delta.image {
                egui::ImageData::Color(color) => {
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            image_staging_buffer.data.offset(image_staging_byte_pos)
                                as *mut egui::Color32,
                            color.width() * color.height(),
                        )
                    };
                    dst.copy_from_slice(&color.pixels);
                    data_bytesize =
                        color.width() * color.height() * std::mem::size_of::<egui::Color32>();
                }
                egui::ImageData::Font(font) => {
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            image_staging_buffer.data.offset(image_staging_byte_pos) as *mut f32,
                            font.width() * font.height(),
                        )
                    };
                    dst.copy_from_slice(&font.pixels);
                    data_bytesize = font.width() * font.height() * std::mem::size_of::<f32>();
                }
            }
            let image_offset = match delta.pos {
                Some(pos) => vk::Offset3D {
                    x: pos[0] as i32,
                    y: pos[1] as i32,
                    z: 0,
                },
                None => vk::Offset3D::default(),
            };
            let image_extent = vk::Extent3D {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                depth: 1,
            };

            copy_buffer_to_images.push((
                image,
                vk::BufferImageCopy {
                    buffer_offset: image_staging_byte_pos as u64,
                    buffer_row_length: 0,   //tight
                    buffer_image_height: 0, //tight
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset,
                    image_extent,
                },
            ));

            image_staging_byte_pos += (data_bytesize as isize + 3) & !3;
            assert!(image_staging_buffer.desc.size >= image_staging_byte_pos as u64);
        }

        if imgui.textures_delta.free.len() > 0 {
            println!(
                "Warning: ImGUI: need to free {} textures.",
                imgui.textures_delta.free.len()
            );
        }

        let rg_vertex_buffer = rg.register_buffer(self.vertex_buffer);
        let target_view = rg.create_texture_view(target, None);

        // Copy from staging buffer to rendering buffers/textures
        // TODO dont need to wait submit done; just sync before the ui draw
        upload_context.immediate_submit(rd, |command_buffer| {
            // image transition (for transfer)
            {
                let mut image_memory_barriers = Vec::new();
                for (dst_image, region) in &copy_buffer_to_images {
                    image_memory_barriers.push(
                        vk::ImageMemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::SHADER_READ)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .image(*dst_image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .build(),
                    );
                }
                unsafe {
                    rd.device_entry.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::ALL_GRAPHICS,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &image_memory_barriers,
                    );
                }
            }

            // copy from staging to target
            unsafe {
                rd.device_entry.cmd_copy_buffer(
                    command_buffer,
                    ind_staging_buffer.handle,
                    self.index_buffer.handle,
                    std::slice::from_ref(&vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: index_data_size as u64,
                    }),
                );

                rd.device_entry.cmd_copy_buffer(
                    command_buffer,
                    vert_staging_buffer.handle,
                    self.vertex_buffer.handle,
                    std::slice::from_ref(&vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: vertex_data_size as u64,
                    }),
                );

                for (dst_image, region) in &copy_buffer_to_images {
                    rd.device_entry.cmd_copy_buffer_to_image(
                        command_buffer,
                        image_staging_buffer.handle,
                        *dst_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(region),
                    );
                }
            }

            // sync buffers
            unsafe {
                let index_bmb = vk::BufferMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(0)
                    .dst_queue_family_index(0) // same
                    .buffer(self.index_buffer.handle)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                let vertex_bmb = vk::BufferMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(0)
                    .dst_queue_family_index(0) // same
                    .buffer(self.vertex_buffer.handle)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                let buffer_memory_barriers = [index_bmb.build(), vertex_bmb.build()];
                rd.device_entry.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_GRAPHICS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &buffer_memory_barriers,
                    &[],
                );
            }

            // sync images
            {
                // image transition (for transfer)
                let mut image_memory_barriers = Vec::new();
                for (dst_image, region) in &copy_buffer_to_images {
                    image_memory_barriers.push(
                        vk::ImageMemoryBarrier::builder()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image(*dst_image)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            })
                            .build(),
                    );
                }
                unsafe {
                    rd.device_entry.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::ALL_GRAPHICS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &image_memory_barriers,
                    );
                }
            }

            // done using staging buffer
            // TODO is this used corretly? any sync required?
            // TODO share the staging buffer?
            unsafe {
                rd.device_entry.cmd_set_event(
                    command_buffer,
                    ind_staging_event,
                    vk::PipelineStageFlags::TRANSFER,
                );
                rd.device_entry.cmd_set_event(
                    command_buffer,
                    vert_staging_event,
                    vk::PipelineStageFlags::TRANSFER,
                );
                rd.device_entry.cmd_set_event(
                    command_buffer,
                    image_staging_event,
                    vk::PipelineStageFlags::TRANSFER,
                );
            }
        });

        // Render
        let viewport_extent = rd.swapchain.extent;
        let texel_size = Vec2::new(
            1.0 / viewport_extent.width as f32,
            1.0 / viewport_extent.height as f32,
        );
        let index_buffer_handle = self.index_buffer.handle;
        rg.new_graphics("ImGUI")
            .vertex_shader_with_ep("imgui_vsps.hlsl", "vs_main")
            .pixel_shader_with_ep("imgui_vsps.hlsl", "ps_main")
            .color_targets(&[ColorTarget {
                view: target_view,
                load_op: ColorLoadOp::Load,
            }])
            .buffer("vertex_buffer", rg_vertex_buffer) // TODO kind of hack; but this make binding easier
            .render(move |cb, pipeline| {
                // setup raster state
                cb.set_viewport_0(vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: viewport_extent.width as f32,
                    height: viewport_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                });
                cb.set_scissor_0(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: viewport_extent,
                });

                // draw each mesh
                cb.bind_index_buffer(index_buffer_handle, 0, vk::IndexType::UINT32);
                for i in 0..mesh_offsets.len() {
                    let (index_offset, vertex_offset) = mesh_offsets[i];
                    let index_count = mesh_index_count[i];
                    if pipeline.push_constant_ranges.len() > 0 {
                        let pc = PushConstantsBuilder::new()
                            .push(&vertex_offset)
                            .push(&texel_size);
                        cb.push_constants(
                            pipeline.layout,
                            pipeline.push_constant_ranges[0].stage_flags,
                            0,
                            pc.build(),
                        );
                    }
                    cb.draw_indexed(
                        index_count,
                        1,
                        index_offset,
                        0, /* not add to index for shader */
                        0,
                    );
                }
            });
    }
}
