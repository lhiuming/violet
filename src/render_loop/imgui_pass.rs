use std::collections::HashMap;

use ash::vk;
use glam::Vec2;

use crate::{
    command_buffer::StencilOps,
    imgui::ImGUIOuput,
    render_device::{
        Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
    },
    render_graph::{
        ColorLoadOp, ColorTarget, PassBuilderTrait, RGHandle, RenderGraphBuilder,
        DESCRIPTOR_SET_INDEX_UNUSED,
    },
    render_scene::UploadContext,
    shader::PushConstantsBuilder,
};

// see also: imgui_vsps.hlsl
static IMGUI_DESCRIPTOR_SET_INDEX: u32 = 0;

pub struct ImGUIPass {
    index_buffer: Buffer,
    vertex_buffer: Buffer,
    ui_textures: Vec<Texture>,
    ui_texture_views: Vec<TextureView>,
    tex_id_to_slot: HashMap<u64, u32>,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
}

impl ImGUIPass {
    pub fn new(rd: &RenderDevice) -> Self {
        let index_buffer = rd
            .create_buffer(BufferDesc {
                size: 1024 * 8, // TODO growing?
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();
        let vertex_buffer = rd
            .create_buffer(BufferDesc {
                size: 1024 * 16, // TODO growing?
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | // sampled as raw buffer
                vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        let descriptor_set_pool = rd.create_descriptor_pool(
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            1,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1, // vertex buffer
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: 1024, // TODO bindless
                },
            ],
        );
        let set_layout = {
            let stage_flags = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            let bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(stage_flags)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1024)
                    .stage_flags(stage_flags)
                    .build(),
            ];
            let binding_falgs = [
                vk::DescriptorBindingFlags::default(),
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            ];
            let mut flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_falgs);
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .push_next(&mut flags_create_info);
            unsafe {
                rd.device
                    .create_descriptor_set_layout(&create_info, None)
                    .unwrap()
            }
        };
        let descriptor_set = unsafe {
            let allocate_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_set_pool)
                .set_layouts(std::slice::from_ref(&set_layout));
            rd.device.allocate_descriptor_sets(&allocate_info).unwrap()[0]
        };
        {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(vertex_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let descriptor_writes = [vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info))
                .build()];
            unsafe {
                rd.device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        Self {
            index_buffer,
            vertex_buffer,
            ui_textures: Vec::new(),
            ui_texture_views: Vec::new(),
            tex_id_to_slot: HashMap::new(),
            descriptor_set_layout: set_layout,
            descriptor_set,
        }
    }

    pub fn add(
        &mut self,
        rg: &mut RenderGraphBuilder,
        rd: &RenderDevice,
        upload_context: &mut UploadContext,
        target: RGHandle<Texture>,
        imgui: &ImGUIOuput,
        clear: Option<vk::ClearColorValue>,
    ) {
        // TODO calculate this properly
        let (index_data_size, vertex_data_size) = {
            let mut index_data_size = 0;
            let mut vertex_data_size = 0;
            for cp in &imgui.clipped_primitives {
                match &cp.primitive {
                    egui::epaint::Primitive::Mesh(mesh) => {
                        index_data_size += mesh.indices.len() * std::mem::size_of::<u32>() as usize;
                        vertex_data_size += mesh.vertices.len()
                            * std::mem::size_of::<egui::epaint::Vertex>() as usize;
                    }
                    egui::epaint::Primitive::Callback(_) => {
                        unimplemented!()
                    }
                }
            }
            (index_data_size as u64, vertex_data_size as u64)
        };

        // TOOD grow the buffers?
        assert!(index_data_size <= self.index_buffer.desc.size);
        assert!(vertex_data_size <= self.vertex_buffer.desc.size);

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

        let image_staging_buffer_size = imgui
            .textures_delta
            .set
            .iter()
            .map(|set| -> usize {
                let delta = &set.1;
                let size = delta.image.size();
                let bytesize = delta.image.bytes_per_pixel() * size[0] * size[1];
                (bytesize + 3) & !3 // align to 4 bytes
            })
            .sum::<usize>();
        let (image_staging_buffer, image_staging_event) =
            upload_context.borrow_staging_buffer(rd, image_staging_buffer_size as u64);

        // upload image data
        let mut write_bindless = Vec::<(u32, vk::DescriptorImageInfo)>::new();
        let mut copy_buffer_to_images = Vec::new();
        let mut image_staging_byte_pos = 0;
        for (tex_id, delta) in &imgui.textures_delta.set {
            let tex_id = match tex_id {
                egui::TextureId::Managed(id) => *id,
                egui::TextureId::User(_) => todo!(),
            };

            // find or create the texture
            let old_layout;
            let tex_slot = match self.tex_id_to_slot.entry(tex_id) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    old_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    *entry.get()
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
                    let texture_view = rd
                        .create_texture_view(texture, TextureViewDesc::auto(&texture.desc))
                        .unwrap();
                    // TODO find free slot
                    let slot = self.ui_textures.len() as u32;
                    {
                        self.ui_textures.push(texture);
                        self.ui_texture_views.push(texture_view);
                    }
                    entry.insert(slot);
                    // TODO batch the writes to ranges of updates
                    write_bindless.push((
                        slot as u32,
                        vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: texture_view.image_view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                    ));
                    old_layout = vk::ImageLayout::UNDEFINED;
                    slot
                }
            };

            let image = self.ui_textures[tex_slot as usize].image;

            let data_bytesize;
            match &delta.image {
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
                old_layout,
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

        // upload buffer data
        // NOTE: done after uploading image data, to query proper tex_id_to_slot
        let mut mesh_offsets = Vec::new();
        let mut mesh_index_count = Vec::new();
        let mut texture_metas = Vec::new();
        for cp in &imgui.clipped_primitives {
            match &cp.primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    let tex_index = match mesh.texture_id {
                        egui::TextureId::Managed(tex_id) => {
                            *self.tex_id_to_slot.get(&tex_id).unwrap()
                        }
                        egui::TextureId::User(_) => todo!(),
                    };
                    let mut tex_meta = tex_index & 0xFFFF;
                    let is_font =
                        self.ui_textures[tex_index as usize].desc.format == vk::Format::R32_SFLOAT;
                    if is_font {
                        tex_meta |= 0x10000;
                    }

                    texture_metas.push(tex_meta);

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
                    texture_metas.push(u32::MAX);
                    unimplemented!()
                }
            }
        }

        // Add new texture view to bindless texture array
        if write_bindless.len() > 0 {
            let writes: Vec<vk::WriteDescriptorSet> = write_bindless
                .iter()
                .map(|(slot, info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(self.descriptor_set)
                        .dst_binding(1)
                        .dst_array_element(*slot)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(std::slice::from_ref(info))
                        .build()
                })
                .collect();
            unsafe {
                rd.device.update_descriptor_sets(&writes, &[]);
            }
        }

        // Copy from staging buffer to rendering buffers/textures
        // TODO dont need to wait submit done; just sync before the ui draw
        if (index_data_size > 0) || (vertex_data_size > 0) || (copy_buffer_to_images.len() > 0) {
            upload_context.immediate_submit(rd, |command_buffer| {
                // image transition (for transfer)
                if copy_buffer_to_images.len() > 0 {
                    let mut image_memory_barriers = Vec::new();
                    for (dst_image, old_layout, _region) in &copy_buffer_to_images {
                        image_memory_barriers.push(
                            vk::ImageMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::SHADER_READ)
                                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .old_layout(*old_layout)
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
                        rd.device.cmd_pipeline_barrier(
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
                    if index_data_size > 0 {
                        rd.device.cmd_copy_buffer(
                            command_buffer,
                            ind_staging_buffer.handle,
                            self.index_buffer.handle,
                            std::slice::from_ref(&vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: index_data_size as u64,
                            }),
                        );
                    }

                    if vertex_data_size > 0 {
                        rd.device.cmd_copy_buffer(
                            command_buffer,
                            vert_staging_buffer.handle,
                            self.vertex_buffer.handle,
                            std::slice::from_ref(&vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: vertex_data_size as u64,
                            }),
                        );
                    }

                    for (dst_image, _old_layout, region) in &copy_buffer_to_images {
                        rd.device.cmd_copy_buffer_to_image(
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
                    rd.device.cmd_pipeline_barrier(
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
                if copy_buffer_to_images.len() > 0 {
                    // image transition (for transfer)
                    let mut image_memory_barriers = Vec::new();
                    for (dst_image, _old_layout, _region) in &copy_buffer_to_images {
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
                        rd.device.cmd_pipeline_barrier(
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
                    rd.device.cmd_set_event(
                        command_buffer,
                        ind_staging_event,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                    rd.device.cmd_set_event(
                        command_buffer,
                        vert_staging_event,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                    rd.device.cmd_set_event(
                        command_buffer,
                        image_staging_event,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }
            });
        }

        // Render
        if mesh_offsets.len() > 0 {
            let viewport_extent = rd.swapchain.extent;
            let texel_size = Vec2::new(
                1.0 / viewport_extent.width as f32,
                1.0 / viewport_extent.height as f32,
            );
            let index_buffer_handle = self.index_buffer.handle;
            let imgui_descriptor_set = self.descriptor_set;
            let target_view = rg.create_texture_view(target, None);
            let load_op = match clear {
                Some(color) => ColorLoadOp::Clear(color),
                None => ColorLoadOp::Load,
            };

            /*
            // check
            let target_format = rg.get_texture_desc(target).format;
            if !rd.format_support_blending(target_format) {
                panic!("ImGUI: target format {:?} does not support blending; GUI may not composed corecctly.", target_format);
            }
            */

            rg.new_graphics("ImGUI")
                .vertex_shader_with_ep("imgui_vsps.hlsl", "vs_main")
                .pixel_shader_with_ep("imgui_vsps.hlsl", "ps_main")
                .set_layout_override(IMGUI_DESCRIPTOR_SET_INDEX, self.descriptor_set_layout)
                .descriptor_set_index(DESCRIPTOR_SET_INDEX_UNUSED)
                .color_targets(&[ColorTarget {
                    view: target_view,
                    load_op,
                }])
                .blend_enabled(true)
                .render(move |cb, pipeline| {
                    // TODO make into gfx config
                    cb.set_depth_test_enable(false);
                    cb.set_depth_write_enable(false);
                    cb.set_stencil_test_enable(false);
                    cb.set_stencil_op(
                        vk::StencilFaceFlags::FRONT,
                        StencilOps::only_compare(vk::CompareOp::ALWAYS),
                    );

                    cb.bind_descriptor_set(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        IMGUI_DESCRIPTOR_SET_INDEX,
                        imgui_descriptor_set,
                        None,
                    );

                    // draw each mesh
                    cb.bind_index_buffer(index_buffer_handle, 0, vk::IndexType::UINT32);
                    for i in 0..mesh_offsets.len() {
                        let (first_index, vertex_offset) = mesh_offsets[i];
                        let index_count = mesh_index_count[i];
                        let texture_meta = texture_metas[i];
                        if pipeline.push_constant_ranges.len() > 0 {
                            let pc = PushConstantsBuilder::new()
                                .push(&vertex_offset)
                                .push(&texture_meta)
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
                            first_index,
                            0, /* not add to index for shader */
                            0,
                        );
                    }

                    // TODO should block further GPU mut to the buffers and textures..
                });
        }
    }
}
