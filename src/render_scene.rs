use std::mem::size_of;
use std::slice;

use ash::vk;
use glam::Vec3;

use crate::{
    model::Model,
    render_device::{
        AccelerationStructure, Buffer, BufferDesc, RenderDevice, Texture, TextureDesc, TextureView,
        TextureViewDesc,
    },
};

// Allocatable buffer. Alway aligned to 4 bytes.
pub struct AllocBuffer {
    pub buffer: Buffer,
    next_pos: u32,
}

impl AllocBuffer {
    pub fn new(buffer: Buffer) -> AllocBuffer {
        AllocBuffer {
            buffer,
            next_pos: 0,
        }
    }

    pub fn alloc<'a, T>(&mut self, count: u32) -> (&'a mut [T], u32) {
        assert!((size_of::<T>() as u32 * count) <= (self.buffer.desc.size as u32 - self.next_pos));
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

// Struct for asset uploading to GPU
pub struct UploadContext {
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub finished_fences: Vec<vk::Fence>,
    pub staging_buffers: Vec<Buffer>,
    pub staging_finished: Vec<vk::Event>,
}

impl UploadContext {
    pub fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        Self {
            command_pool,
            command_buffers: Vec::new(),
            finished_fences: Vec::new(),
            staging_buffers: Vec::new(),
            staging_finished: Vec::new(),
        }
    }

    fn pick_command_buffer(&mut self, rd: &RenderDevice) -> (vk::CommandBuffer, vk::Fence) {
        // Find an existing command buffer that's ready to be used
        for i in 0..self.command_buffers.len() {
            let fence = self.finished_fences[i];
            let finished = unsafe { rd.device.get_fence_status(fence) }.unwrap();
            if finished {
                let fences = [fence];
                unsafe { rd.device.reset_fences(&fences) }.unwrap();
                return (self.command_buffers[i], fence);
            }
        }

        // If every buffer is pending, allocate a new one
        let command_buffer = rd.create_command_buffer(self.command_pool);
        let fence = rd.create_fence(false);

        self.command_buffers.push(command_buffer);
        self.finished_fences.push(fence);

        return (command_buffer, fence);
    }

    pub fn immediate_submit<F>(&mut self, rd: &RenderDevice, f: F)
    where
        F: FnOnce(vk::CommandBuffer),
    {
        // TODO we will need semaphore to sync with rendering which will depends on operation here
        let device = &rd.device;
        let (command_buffer, finished_fence) = self.pick_command_buffer(rd);

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();

        f(command_buffer);

        unsafe { device.end_command_buffer(command_buffer) }.unwrap();

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        unsafe {
            device
                .queue_submit(rd.gfx_queue, &[*submit_info], finished_fence)
                .unwrap();

            // debug wait
            let fences = [finished_fence];
            let timeout = 1000 * 1000000; // 1s
            device.wait_for_fences(&fences, true, timeout).unwrap();
        }
    }

    // Signaled the event to return the fence back to the pool
    pub fn borrow_staging_buffer(
        &mut self,
        rd: &RenderDevice,
        buffer_size: u64,
    ) -> (Buffer, vk::Event) {
        // quantize the size
        let buffer_size = buffer_size.next_power_of_two();
        // Find reusable one
        // TODO something better than linear search?
        for (i, buffer) in self.staging_buffers.iter().enumerate() {
            if buffer.desc.size == buffer_size {
                let event = self.staging_finished[i];
                let finished = unsafe { rd.device.get_event_status(event).unwrap() };
                if finished {
                    // todo reset in bunch?
                    unsafe { rd.device.reset_event(event).unwrap() };
                    return (*buffer, event);
                }
            }
        }

        // Crate new
        let buffer = rd
            .create_buffer(BufferDesc {
                size: buffer_size,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();
        let event = rd.create_event();
        self.staging_buffers.push(buffer);
        self.staging_finished.push(event);
        (buffer, event)
    }
}

#[repr(C)]
pub struct MaterialParams {
    pub base_color_index: u32,
    pub metallic_roughness_index: u32,
    pub normal_index: u32,
    pub pad: u32,
}

pub struct MeshParams {
    pub index_offset: u32,
    pub index_count: u32,

    pub positions_offset: u32,
    pub texcoords_offset: u32,
    pub normals_offset: u32,
    pub tangents_offset: u32,

    pub material_index: u32,

    pub pad: u32,
}

// Matching constants in `shader/scene_bindings.hlsl`
pub const SCENE_DESCRIPTOR_SET_INDEX: u32 = 1;
pub const VERTEX_BUFFER_BINDING_INDEX: u32 = 0;
pub const MATERIAL_PARAMS_BINDING_INDEX: u32 = 1;
pub const BINDLESS_TEXTURE_BINDING_INDEX: u32 = 2;
//pub const VIEWPARAMS_BINDING_INDEX: u32 = 3;
pub const MESH_PARAMS_BINDING_INDEX: u32 = 4;
pub const INDEX_BUFFER_BINDING_INDEX: u32 = 5;

// Contain everything to be rendered
pub struct RenderScene {
    pub upload_context: UploadContext,

    // Scene Descriptor Set
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,

    // Global buffer to store all loaded meshes
    pub vertex_buffer: AllocBuffer,
    pub index_buffer: AllocBuffer,

    // Global texture arrays, mapping the bindless textures
    pub material_textures: Vec<Texture>,
    pub material_texture_views: Vec<TextureView>,

    // Stuff to be rendered
    pub material_parmas: Vec<MaterialParams>,
    pub mesh_params: Vec<MeshParams>,

    // Global material parameter buffer for all loaded mesh; map to `material_params`
    pub material_param_buffer: Buffer,

    // Global mesh paramter buffer for all loaded mesh; map to `mesh_params`
    pub mesh_param_buffer: Buffer,

    // Ray Tracing Acceleration structures for whole scene
    pub mesh_bottom_level_accel_structs: Vec<AccelerationStructure>,
    pub scene_top_level_accel_struct: Option<AccelerationStructure>,

    // Lighting settings
    pub sun_dir: Vec3,
}

impl RenderScene {
    pub fn new(rd: &RenderDevice) -> RenderScene {
        // Buffer for whole scene
        let ib_size = 4 * 1024 * 1024;
        let vb_size = 16 * 1024 * 1024;
        let accel_strut_usafe = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let index_buffer = AllocBuffer::new(
            rd.create_buffer(BufferDesc {
                size: ib_size,
                usage: vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | accel_strut_usafe,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staing buffer
            })
            .unwrap(),
        );
        let index_buffer_view = rd
            .create_buffer_view(index_buffer.buffer.handle, vk::Format::R16_UINT)
            .unwrap();
        let vertex_buffer = AllocBuffer::new(
            rd.create_buffer(BufferDesc {
                size: vb_size,
                usage: vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER | accel_strut_usafe,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staging buffer
            })
            .unwrap(),
        );
        let vertex_buffer_view = rd
            .create_buffer_view(vertex_buffer.buffer.handle, vk::Format::R32_UINT)
            .unwrap();

        // Material Parameters buffer
        let material_param_size = std::mem::size_of::<MaterialParams>() as vk::DeviceSize;
        let material_param_buffer = rd
            .create_buffer(BufferDesc {
                size: material_param_size * 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staging buffer
            })
            .unwrap();

        // Mesh Paramters buffer
        let mesh_param_size = std::mem::size_of::<MeshParams>() as vk::DeviceSize;
        let mesh_param_buffer = rd
            .create_buffer(BufferDesc {
                size: mesh_param_size * 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        // Descriptor pool for whole scene bindless texture
        // TODO specific size and stuff
        let descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
            64,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1024 + 64,
            }],
        );

        // Create the scene/persistent binding set
        // see also: `shader/scene_bindings.hlsl`
        let descriptor_set_layout = {
            let vbuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(VERTEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let ibuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(INDEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL); // TODO only hit shader
            let mat_buffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(MATERIAL_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let bindless_textures = vk::DescriptorSetLayoutBinding::builder()
                .binding(BINDLESS_TEXTURE_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1024) // TODO size?
                .stage_flags(vk::ShaderStageFlags::ALL);
            let bindless_textures_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
                //| vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
            let mesh_buffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(MESH_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let bindings = [
                *vbuffer,
                *ibuffer,
                *mat_buffer,
                *bindless_textures,
                *mesh_buffer,
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::default(),
                vk::DescriptorBindingFlags::default(),
                vk::DescriptorBindingFlags::default(),
                bindless_textures_flags,
                vk::DescriptorBindingFlags::default(),
            ];
            assert_eq!(bindings.len(), binding_flags.len());
            let mut flags_create_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flags);
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(&bindings)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL) // for bindless
                .push_next(&mut flags_create_info);
            unsafe {
                rd.device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("Failed to create scene descriptor set layout")
            }
        };
        let descriptor_set = {
            let layouts = [descriptor_set_layout];
            let create_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&layouts);
            unsafe {
                rd.device
                    .allocate_descriptor_sets(&create_info)
                    .expect("Failed to create descriptor set for scene")[0]
            }
        };

        // Initialize descriptor set
        {
            let write_vbuffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(VERTEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .texel_buffer_view(slice::from_ref(&vertex_buffer_view))
                .build();
            let write_ibuffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(INDEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .texel_buffer_view(slice::from_ref(&index_buffer_view))
                .build();
            let mat_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(material_param_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let write_mat_buffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(MATERIAL_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(&mat_buffer_info))
                .build();
            let mesh_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(mesh_param_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let write_mesh_buffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(MESH_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(&mesh_buffer_info))
                .build();
            unsafe {
                rd.device.update_descriptor_sets(
                    &[
                        write_vbuffer,
                        write_ibuffer,
                        write_mat_buffer,
                        write_mesh_buffer,
                    ],
                    &[],
                )
            }
        }

        RenderScene {
            upload_context: UploadContext::new(rd),
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            vertex_buffer,
            index_buffer,
            material_textures: Vec::new(),
            material_texture_views: Vec::new(),
            material_parmas: Vec::new(),
            mesh_params: Vec::new(),
            material_param_buffer,
            mesh_param_buffer,
            mesh_bottom_level_accel_structs: Vec::new(),
            scene_top_level_accel_struct: None,
            sun_dir: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn add(&mut self, rd: &RenderDevice, model: &Model) {
        let upload_context = &mut self.upload_context;
        //let material_texture = &mut self.material_texture;
        let index_buffer = &mut self.index_buffer;
        let vertex_buffer = &mut self.vertex_buffer;

        // log
        let total_vert_count = model
            .meshes
            .iter()
            .map(|m| m.positions.len())
            .sum::<usize>();
        println!("Total Vertex: {}", total_vert_count);

        let texture_index_offset = self.material_textures.len() as u32;
        for image in &model.images {
            let texel_count = image.width * image.height;

            // Create texture object
            let texture = rd
                .create_texture(
                    TextureDesc::new_2d(
                        image.width,
                        image.height,
                        vk::Format::R8G8B8A8_UINT,
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                    )
                    .with_flags(vk::ImageCreateFlags::MUTABLE_FORMAT), // performance cost?
                )
                .unwrap();

            // Create staging buffer
            let staging_buffer = rd
                .create_buffer(BufferDesc {
                    size: texel_count as u64 * 4,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                        | vk::MemoryPropertyFlags::HOST_COHERENT,
                })
                .unwrap();

            // Read to staging buffer
            let staging_slice = unsafe {
                std::slice::from_raw_parts_mut(staging_buffer.data, texel_count as usize * 4)
            };
            staging_slice.copy_from_slice(&image.data);

            // Transfer to texture
            upload_context.immediate_submit(rd, |command_buffer| {
                // Transfer to proper image layout
                unsafe {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .image(texture.image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    rd.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[*barrier],
                    );
                }

                // Copy
                let dst_image_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                let region = vk::BufferImageCopy::builder()
                    .buffer_row_length(0) // zero means tightly packed
                    .buffer_image_height(0) // zero means tightly packed
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: image.width,
                        height: image.height,
                        depth: 1,
                    });
                let regions = [*region];
                unsafe {
                    rd.device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer.handle,
                        texture.image,
                        dst_image_layout,
                        &regions,
                    )
                }

                // Transfer to shader ready layout
                unsafe {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .image(texture.image)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        });
                    rd.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[*barrier],
                    );
                }
            });

            // Queue to upload to scene data

            self.material_textures.push(texture);
        }

        let texture_view_index_offset = self.material_texture_views.len() as u32;
        let material_index_offset = self.material_parmas.len() as u32;
        for material in &model.materials {
            // Create texture view for each material and add to scene
            // TODO collect and reduce duplicate in view?
            // TODO default textures
            let mut resolve = |map: &Option<crate::model::MaterialMap>, is_srgb: bool| match map {
                Some(map) => {
                    let texture = {
                        let texture_index = texture_index_offset + map.image_index;
                        self.material_textures[texture_index as usize]
                    };
                    let format = if is_srgb {
                        vk::Format::R8G8B8A8_SRGB
                    } else {
                        vk::Format::R8G8B8A8_UNORM
                    };
                    let texture_view = rd
                        .create_texture_view(
                            texture,
                            TextureViewDesc::with_format(&texture.desc, format),
                        )
                        .unwrap();
                    let texture_view_index = self.material_texture_views.len() as u32;
                    self.material_texture_views.push(texture_view);
                    texture_view_index
                }
                None => 0,
            };

            self.material_parmas.push(MaterialParams {
                base_color_index: resolve(&material.base_color_map, true),
                metallic_roughness_index: resolve(&material.metallic_roughness_map, true),
                normal_index: resolve(&material.normal_map, false),
                pad: 0,
            });
        }

        // Upload new material params
        unsafe {
            let param_size = std::mem::size_of::<MaterialParams>();
            let param_count = self.material_parmas.len() - material_index_offset as usize;
            let data_offset = material_index_offset as isize * param_size as isize;
            let data_size = param_size * param_count;
            let src = std::slice::from_raw_parts(
                (self.material_parmas.as_ptr() as *const u8).offset(data_offset),
                data_size,
            );
            let dst = std::slice::from_raw_parts_mut(
                self.material_param_buffer.data.offset(data_offset),
                data_size,
            );
            dst.copy_from_slice(src);
        }

        // Add new texture view to bindless texture array
        {
            let image_infos: Vec<vk::DescriptorImageInfo> = self.material_texture_views
                [texture_index_offset as usize..]
                .iter()
                .map(|view| {
                    vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(view.image_view)
                        .sampler(vk::Sampler::null())
                        .build()
                })
                .collect();
            let write = vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set)
                .dst_binding(BINDLESS_TEXTURE_BINDING_INDEX)
                .dst_array_element(texture_view_index_offset)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_infos);
            unsafe {
                rd.device.update_descriptor_sets(&[*write], &[]);
            }
        }

        let mut blas_geometries = Vec::<vk::AccelerationStructureGeometryKHR>::new();
        let mut blas_range_infos = Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();
        let mut max_primitive_counts = Vec::<u32>::new(); // or blas_primitive_counts, since the scratch buffer is allocated for this blas exclusively
        let mesh_index_offset = self.mesh_params.len();
        for (material_index, mesh) in model.meshes.iter().enumerate() {
            // Upload indices
            let index_offset;
            let index_offset_bytes;
            let index_count;
            {
                index_count = mesh.indicies.len() as u32;
                let (dst, offset) = index_buffer.alloc(index_count);
                index_offset = offset / 2;
                dst.copy_from_slice(&mesh.indicies);
                index_offset_bytes = offset;
            }
            // Upload position
            let positions_offset;
            let positions_offset_bytes;
            {
                let (dst, offset) = vertex_buffer.alloc::<[f32; 3]>(mesh.positions.len() as u32);
                positions_offset = offset / 4;
                dst.copy_from_slice(&mesh.positions);
                positions_offset_bytes = offset;
            }
            // Upload normal
            let normals_offset;
            if let Some(normals) = &mesh.normals {
                let (dst, offset) = vertex_buffer.alloc::<[f32; 3]>(normals.len() as u32);
                normals_offset = offset / 4;
                dst.copy_from_slice(&normals);
            } else {
                normals_offset = u32::MAX;
            }
            // Upload texcoords
            let texcoords_offset;
            if let Some(texcoords) = &mesh.texcoords {
                let (dst, offset) = vertex_buffer.alloc::<[f32; 2]>(texcoords.len() as u32);
                texcoords_offset = offset / 4;
                dst.copy_from_slice(&texcoords);
            } else {
                texcoords_offset = u32::MAX;
            }
            // Upload tangents
            let tangents_offset;
            if let Some(tangents) = &mesh.tangents {
                let (dst, offset) = vertex_buffer.alloc::<[f32; 4]>(tangents.len() as u32);
                tangents_offset = offset / 4;
                dst.copy_from_slice(&tangents);
            } else {
                tangents_offset = u32::MAX;
            }

            self.mesh_params.push(MeshParams {
                index_offset,
                index_count,
                positions_offset,
                normals_offset,
                texcoords_offset,
                tangents_offset,
                material_index: material_index_offset + material_index as u32,
                pad: 0,
            });

            let triangle_count = index_count / 3;

            // TODO assert index type
            let vertex_data = vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_buffer.buffer.device_address.unwrap()
                    + positions_offset_bytes as u64,
            };
            let index_data = vk::DeviceOrHostAddressConstKHR {
                device_address: index_buffer.buffer.device_address.unwrap()
                    + index_offset_bytes as u64,
            };
            let geo_data = vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vertex_data)
                    .vertex_stride(4 * 3) // f32 * 3
                    .max_vertex(mesh.positions.len() as u32)
                    .index_type(vk::IndexType::UINT16)
                    .index_data(index_data)
                    .transform_data(vk::DeviceOrHostAddressConstKHR::default()) // no xform
                    .build(),
            };
            let blas_geo = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(geo_data)
                .flags(vk::GeometryFlagsKHR::OPAQUE) // todo check material
                .build();
            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(triangle_count)
                .primitive_offset(0) // a.k.a index_offset_bytes for indexed triangle geomtry; not using because already offset via address in GeometryDataKHR.index_data
                .first_vertex(0) // a.k.a vertex_offset for indexed triangle geometry; not using because already offset via address in GeometryDataKHR.vertex_data
                .transform_offset(0) // no xform
                .build();
            blas_geometries.push(blas_geo);
            blas_range_infos.push(range_info);
            max_primitive_counts.push(triangle_count);
        }

        // Upload new mesh params
        unsafe {
            let param_size = std::mem::size_of::<MeshParams>();
            let param_count = self.mesh_params.len() - mesh_index_offset;
            let data_offset = mesh_index_offset as isize * param_size as isize;
            let data_size = param_size * param_count;
            let src = std::slice::from_raw_parts(
                (self.mesh_params.as_ptr() as *const u8).offset(data_offset),
                data_size,
            );
            let dst = std::slice::from_raw_parts_mut(
                self.mesh_param_buffer.data.offset(data_offset),
                data_size,
            );
            dst.copy_from_slice(src);
        }

        // Build BLAS for all added meshes
        if let Some(khr_accel_struct) = rd.khr_accel_struct.as_ref() {
            // Get build size
            // NOTE: actual buffer addresses in blas_geometrys are ignored
            let build_size_info = unsafe {
                assert_eq!(blas_geometries.len(), max_primitive_counts.len());
                let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .geometries(&blas_geometries)
                    .build();
                khr_accel_struct.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &max_primitive_counts,
                )
            };

            // Create buffer
            let buffer = rd
                .create_buffer(BufferDesc {
                    size: build_size_info.acceleration_structure_size,
                    usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                })
                .unwrap();

            // Create AS
            let accel_struct = rd
                .create_accel_struct(
                    buffer,
                    0,
                    build_size_info.acceleration_structure_size,
                    vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                )
                .unwrap();

            // Create Scratch buffer
            let scratch_buffer = rd
                .create_buffer(BufferDesc {
                    size: build_size_info.build_scratch_size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                })
                .unwrap();

            upload_context.immediate_submit(rd, move |cb| {
                let scratch_data = vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address.unwrap(),
                };
                assert_eq!(blas_geometries.len(), blas_range_infos.len());
                let geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(accel_struct.handle)
                    .geometries(&blas_geometries)
                    .scratch_data(scratch_data)
                    .build();
                let blas_range_infos = blas_range_infos.as_slice();
                unsafe {
                    khr_accel_struct.cmd_build_acceleration_structures(
                        cb,
                        slice::from_ref(&geo_info),
                        slice::from_ref(&blas_range_infos),
                    )
                };
            });

            self.mesh_bottom_level_accel_structs.push(accel_struct);
        }
    }

    pub fn rebuild_top_level_accel_struct(&mut self, rd: &RenderDevice) -> Option<()> {
        let khr_accel_struct = rd.khr_accel_struct.as_ref()?;

        // Create instance buffer
        let num_blas = self.mesh_bottom_level_accel_structs.len();
        let instance_buffer = rd
            .create_buffer(BufferDesc {
                size: (size_of::<vk::AccelerationStructureInstanceKHR>() * num_blas) as u64,
                usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        // Traverse all BLAS to fill the instance buffer
        for (index, blas) in self.mesh_bottom_level_accel_structs.iter().enumerate() {
            // Fill instance buffer (3x4 row-major affine transform)
            let xform = vk::TransformMatrixKHR {
                matrix: [
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                ],
            };
            let instance = vk::AccelerationStructureInstanceKHR {
                transform: xform,
                instance_custom_index_and_mask: vk::Packed24_8::new(
                    0, // custom index is not used
                    0xFF,
                ),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0, // TODO not used hit shader
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FRONT_COUNTERCLOCKWISE.as_raw() as u8, // Keep consistent with rasterization
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.device_address,
                },
            };
            let dst = unsafe {
                std::slice::from_raw_parts_mut::<vk::AccelerationStructureInstanceKHR>(
                    instance_buffer.data as _,
                    instance_buffer.desc.size as usize,
                )
            };
            dst[index] = instance;
        }

        let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: instance_buffer.device_address.unwrap(),
        };

        // Make geometry info (instance buffer)
        let instances_geo = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES) // NOTE: only allow INSTANCES if TLAS
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(instance_data_device_address)
                    .array_of_pointers(false)
                    .build(),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE) // todo check material
            .build();

        // Calculate build size
        // NOTE: actual buffer addresses in geometrys are ignored
        let build_size_info = unsafe {
            let max_primitive_count = 1;
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(slice::from_ref(&instances_geo))
                .build();
            khr_accel_struct.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &slice::from_ref(&max_primitive_count),
            )
        };

        // Create buffer
        let buffer = rd
            .create_buffer(BufferDesc {
                size: build_size_info.acceleration_structure_size,
                usage: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        // Create AS
        let accel_struct = rd
            .create_accel_struct(
                buffer,
                0,
                build_size_info.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            )
            .unwrap();

        // Create Scratch buffer
        let scratch_buffer = rd
            .create_buffer(BufferDesc {
                size: build_size_info.build_scratch_size,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        // Build
        self.upload_context.immediate_submit(rd, move |cb| {
            let geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(accel_struct.handle)
                .geometries(slice::from_ref(&instances_geo)) // NOTE: must be one if TLAS
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address.unwrap(),
                })
                .build();
            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(1)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0)
                .build();
            let range_infos = slice::from_ref(&range_info);
            unsafe {
                khr_accel_struct.cmd_build_acceleration_structures(
                    cb,
                    slice::from_ref(&geo_info),
                    slice::from_ref(&range_infos),
                )
            };
        });

        self.scene_top_level_accel_struct.replace(accel_struct);

        Some(())
    }
}
