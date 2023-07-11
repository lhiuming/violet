use std::mem::{self, size_of};
use std::slice;

use ash::vk;
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};

use crate::command_buffer::{CommandBuffer, StencilOps};
use crate::model::Model;
use crate::render_device::{
    AccelerationStructure, Buffer, RenderDevice, ShaderBindingTableFiller, Texture, TextureDesc,
    TextureView, TextureViewDesc,
};
use crate::render_graph::{self};
use crate::shader::{HackStuff, Handle, Pipeline, PushConstantsBuilder, ShaderDefinition, Shaders};

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
        assert!((size_of::<T>() as u32 * count) <= (self.buffer.size as u32 - self.next_pos));
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
}

impl UploadContext {
    pub fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        Self {
            command_pool,
            command_buffers: Vec::new(),
            finished_fences: Vec::new(),
        }
    }

    fn pick_command_buffer(&mut self, rd: &RenderDevice) -> (vk::CommandBuffer, vk::Fence) {
        // Find an existing command buffer that's ready to be used
        for i in 0..self.command_buffers.len() {
            let fence = self.finished_fences[i];
            let finished = unsafe { rd.device_entry.get_fence_status(fence) }.unwrap();
            if finished {
                let fences = [fence];
                unsafe { rd.device_entry.reset_fences(&fences) }.unwrap();
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
        let device = &rd.device_entry;
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
}

// Matching constants in `shader/scene_bindings.hlsl`
const SCENE_DESCRIPTOR_SET_INDEX: u32 = 1;
const VERTEX_BUFFER_BINDING_INDEX: u32 = 0;
const MATERIAL_PARAMS_BINDING_INDEX: u32 = 1;
const BINDLESS_TEXTURE_BINDING_INDEX: u32 = 2;
const VIEWPARAMS_BINDING_INDEX: u32 = 3;
const SAMPLER_BINDING_INDEX: u32 = 4;

// Contain everything to be rendered
pub struct RenderScene {
    pub upload_context: UploadContext,

    pub view_params_cb: Buffer,

    // Shared samplers
    pub shared_sampler: vk::Sampler,

    // Scene Descriptor Set
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,

    // Global buffer to store all loaded meshes
    pub vertex_buffer: AllocBuffer,
    pub index_buffer: AllocBuffer,

    // Global material parameter buffer for all loaded mesh; map to `material_params`
    pub material_param_buffer: Buffer,

    // Global texture arrays, mapping the bindless textures
    pub material_textures: Vec<Texture>,
    pub material_texture_views: Vec<TextureView>,

    // Stuff to be rendered
    pub material_parmas: Vec<MaterialParams>,
    pub mesh_params: Vec<MeshParams>,

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
            rd.create_buffer(
                ib_size,
                vk::BufferUsageFlags::INDEX_BUFFER | accel_strut_usafe,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staing buffer
            )
            .unwrap(),
        );
        let vertex_buffer = AllocBuffer::new(
            rd.create_buffer(
                vb_size,
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER | accel_strut_usafe,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staging buffer
            )
            .unwrap(),
        );
        let vertex_buffer_view = rd
            .create_buffer_view(vertex_buffer.buffer.handle, vk::Format::R32_UINT)
            .unwrap();

        // Material Parameters buffer
        let material_param_size = 3 * 4;
        let material_param_buffer = rd
            .create_buffer(
                material_param_size * 1024,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, // TODO staging buffer
            )
            .unwrap();

        // View parameter constant buffer
        let view_params_cb = rd
            .create_buffer(
                mem::size_of::<ViewParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();

        // Shared samplers
        let shared_sampler = unsafe {
            let create_info = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR);
            rd.device_entry.create_sampler(&create_info, None).unwrap()
        };

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
            let cbuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(VIEWPARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let samplers = [shared_sampler];
            let sampler_ll = vk::DescriptorSetLayoutBinding::builder()
                .binding(SAMPLER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .immutable_samplers(&samplers);
            let bindings = [
                *vbuffer,
                *mat_buffer,
                *bindless_textures,
                *cbuffer,
                *sampler_ll,
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::default(),
                vk::DescriptorBindingFlags::default(),
                bindless_textures_flags,
                vk::DescriptorBindingFlags::default(),
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
                rd.device_entry
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
                rd.device_entry
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
            let cbuffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(view_params_cb.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let write_cbuffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(VIEWPARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&cbuffer_info))
                .build();
            unsafe {
                rd.device_entry
                    .update_descriptor_sets(&[write_vbuffer, write_mat_buffer, write_cbuffer], &[])
            }
        }

        RenderScene {
            upload_context: UploadContext::new(rd),
            view_params_cb,
            shared_sampler,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            vertex_buffer,
            index_buffer,
            material_param_buffer,
            material_textures: Vec::new(),
            material_texture_views: Vec::new(),
            material_parmas: Vec::new(),
            mesh_params: Vec::new(),
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
                .create_buffer(
                    texel_count as u64 * 4,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
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
                    rd.device_entry.cmd_pipeline_barrier(
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
                    rd.device_entry.cmd_copy_buffer_to_image(
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
                    rd.device_entry.cmd_pipeline_barrier(
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
                rd.device_entry.update_descriptor_sets(&[*write], &[]);
            }
        }

        let mut blas_geometrys = Vec::<vk::AccelerationStructureGeometryKHR>::new();
        let mut blas_range_info = Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();
        let mut max_primitive_counts = Vec::<u32>::new();
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
            });

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
                .primitive_count(index_count / 3)
                .primitive_offset(0) // a.k.a index_offset_bytes for indexed triangle geomtry; not using because already offset via address in GeometryDataKHR.index_data
                .first_vertex(0) // a.k.a vertex_offset for indexed triangle geometry; not using because already offset via address in GeometryDataKHR.vertex_data
                .transform_offset(0) // no xform
                .build();
            blas_geometrys.push(blas_geo);
            blas_range_info.push(range_info);
            max_primitive_counts.push(index_count / 3);
        }

        // Build BLAS for all added meshes
        {
            // Get build size
            // NOTE: actual buffer addresses in blas_geometrys are ignored
            let build_size_info = unsafe {
                let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .geometries(&blas_geometrys)
                    .build();
                rd.acceleration_structure_entry
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &build_info,
                        &max_primitive_counts,
                    )
            };

            // Create buffer
            let buffer = rd
                .create_buffer(
                    build_size_info.acceleration_structure_size,
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
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
                .create_buffer(
                    build_size_info.build_scratch_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .unwrap();

            upload_context.immediate_submit(rd, move |cb| {
                let scratch_data = vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address.unwrap(),
                };
                let geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(accel_struct.handle)
                    .geometries(&blas_geometrys)
                    .scratch_data(scratch_data)
                    .build();
                unsafe {
                    rd.acceleration_structure_entry
                        .cmd_build_acceleration_structures(
                            cb,
                            slice::from_ref(&geo_info),
                            &[&blas_range_info],
                        )
                };
            });

            self.mesh_bottom_level_accel_structs.push(accel_struct);
        }
    }

    pub fn rebuild_top_level_accel_struct(&mut self, rd: &RenderDevice) {
        // Create instance buffer
        let num_blas = self.mesh_bottom_level_accel_structs.len();
        let instance_buffer = rd
            .create_buffer(
                (size_of::<vk::AccelerationStructureInstanceKHR>() * num_blas) as u64,
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();

        // Traverse all BLAS
        let mut geometries = Vec::with_capacity(num_blas);
        let mut range_infos = Vec::with_capacity(num_blas);
        let mut max_primitive_counts = Vec::with_capacity(num_blas);
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
                    instance_buffer.size as usize,
                )
            };
            dst[index] = instance;

            // Add geometry entry
            let instance_data_device_address = vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address.unwrap()
                    + (size_of::<vk::AccelerationStructureInstanceKHR>() * index) as u64,
            };
            let geo_data = vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                    .data(instance_data_device_address)
                    .array_of_pointers(false)
                    .build(),
            };
            let blas_geo = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(geo_data)
                .flags(vk::GeometryFlagsKHR::OPAQUE) // todo check material
                .build();
            geometries.push(blas_geo);

            // Add (trivial) range info
            let range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(1)
                .primitive_offset(0)
                .first_vertex(0)
                .transform_offset(0)
                .build();
            range_infos.push(range_info);
            max_primitive_counts.push(1);
        }

        // Get build size
        // NOTE: actual buffer addresses in geometrys are ignored
        let build_size_info = unsafe {
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .geometries(&geometries)
                .build();
            rd.acceleration_structure_entry
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &max_primitive_counts,
                )
        };

        // Create buffer
        let buffer = rd
            .create_buffer(
                build_size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
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
            .create_buffer(
                build_size_info.build_scratch_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();

        self.upload_context.immediate_submit(rd, move |cb| {
            let geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(accel_struct.handle)
                .geometries(&geometries)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address.unwrap(),
                })
                .build();
            unsafe {
                rd.acceleration_structure_entry
                    .cmd_build_acceleration_structures(
                        cb,
                        slice::from_ref(&geo_info),
                        &[&range_infos],
                    )
            };
        });

        self.scene_top_level_accel_struct.replace(accel_struct);
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ViewParams {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub view_pos: Vec3,
    pub pad0: f32,
    pub view_dir_top_left: Vec3,
    pub pad1: f32,
    pub view_dir_right_shift: Vec3,
    pub pad2: f32,
    pub view_dir_down_shift: Vec3,
    pub pad3: f32,
    pub sun_dir: Vec3,
}

pub struct ViewInfo {
    pub view_position: Vec3,
    pub view_transform: Mat4,
    pub projection: Mat4,
}

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
            .create_buffer(
                sbt_size,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
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

pub struct RednerLoop {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub present_finished_fence: vk::Fence,
    pub present_finished_semephore: vk::Semaphore,
    pub present_ready_semaphore: vk::Semaphore,
    pub command_buffer_finished_fence: vk::Fence,

    pub render_graph_cache: render_graph::RenderGraphCache,

    pub raytraced_shadow: RayTracedShadowResources,
}

impl RednerLoop {
    pub fn new(rd: &RenderDevice) -> RednerLoop {
        let command_pool = rd.create_command_pool();
        let command_buffer = rd.create_command_buffer(command_pool);

        RednerLoop {
            command_pool,
            command_buffer,
            present_finished_fence: rd.create_fence(false),
            present_finished_semephore: rd.create_semaphore(),
            present_ready_semaphore: rd.create_semaphore(),
            command_buffer_finished_fence: rd.create_fence(true),
            render_graph_cache: render_graph::RenderGraphCache::new(rd),
            raytraced_shadow: RayTracedShadowResources::new(rd),
        }
    }

    pub fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        let command_buffer = self.command_buffer;

        use render_graph::*;
        let mut rg = RenderGraphBuilder::new(&mut self.render_graph_cache);

        // Stupid shader compiling hack
        let mut hack = HackStuff {
            bindless_size: 1024,
            set_layout_override: std::collections::HashMap::new(),
            ray_recursiion_depth: 0,
        };

        // Acquire target image
        let (image_index, b_image_suboptimal) = unsafe {
            rd.swapchain_entry.entry.acquire_next_image(
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
                let mut timeout_in_ns = 5 * 1000000; // 5ms
                let timeout_max = 500 * 1000000; // 500ms
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
        {
            // From row major float4x4 to column major Mat4
            let view_proj = view_info.projection * view_info.view_transform;
            let inv_proj = view_proj.inverse();
            let ndc_to_ray = |ndc: Vec4| {
                let pos_ws_h = inv_proj * ndc;
                let pos_ws = pos_ws_h.xyz() / pos_ws_h.w;
                pos_ws - view_info.view_position
            };
            let view_dir_top_left = ndc_to_ray(Vec4::new(-1.0, -1.0, 1.0, 1.0));
            let view_dir_right = ndc_to_ray(Vec4::new(1.0, 0.0, 1.0, 1.0));
            let view_dir_left = ndc_to_ray(Vec4::new(-1.0, 0.0, 1.0, 1.0));
            let view_dir_up = ndc_to_ray(Vec4::new(0.0, -1.0, 1.0, 1.0));
            let view_dir_down = ndc_to_ray(Vec4::new(0.0, 1.0, 1.0, 1.0));
            let view_params = ViewParams {
                view_proj: view_proj,
                inv_view_proj: view_proj.inverse(),
                view_pos: view_info.view_position,
                view_dir_top_left,
                view_dir_right_shift: view_dir_right - view_dir_left,
                view_dir_down_shift: view_dir_down - view_dir_up,
                sun_dir: scene.sun_dir,
                pad0: 0.0,
                pad1: 0.0,
                pad2: 0.0,
                pad3: 0.0,
            };

            let dst = unsafe {
                std::slice::from_raw_parts_mut(scene.view_params_cb.data as *mut ViewParams, 1)
            };
            dst.copy_from_slice(std::slice::from_ref(&view_params));
        }

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
        let mut skycube_texture = RGHandle::null();
        let skycube_gen = shaders
            .create_compute_pipeline(ShaderDefinition::compute("sky_cube.hlsl", "main"), &hack);
        if let Some(pipeline) = skycube_gen {
            skycube_texture = rg.create_texutre(TextureDesc {
                width: skycube_size,
                height: skycube_size,
                layer_count: 6,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::CUBE_COMPATIBLE, // required for viewed as cube
                ..Default::default()
            });
            let array_uav = rg.create_texture_view(
                skycube_texture,
                TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                    format: vk::Format::B10G11R11_UFLOAT_PACK32,
                    aspect: vk::ImageAspectFlags::COLOR,
                    ..Default::default()
                },
            );

            rg.new_pass("Sky IBL gen", RenderPassType::Compute)
                .pipeline(pipeline)
                .rw_texture("rw_cube_texture", array_uav)
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();

                    let pc = PushConstantsBuilder::new()
                        .pushv(skycube_size as f32)
                        .push(&scene.sun_dir);
                    cb.push_constants(
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &pc.build(),
                    );

                    cb.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline.handle);
                    cb.dispatch(skycube_size / 8, skycube_size / 4, 6);
                });
        }

        let skycube = rg.create_texture_view(
            skycube_texture,
            TextureViewDesc {
                view_type: vk::ImageViewType::CUBE,
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                aspect: vk::ImageAspectFlags::COLOR,
                ..Default::default()
            },
        );

        // Define GBuffer
        let gbuffer_size = rd.swapchain.extent;
        let gbuffer_depth;
        let gbuffer_color;
        let gbuffer_color_clear;
        {
            // helper
            let mut create_gbuffer = |format: vk::Format| {
                let is_depth = crate::render_device::format_has_depth(format);
                let usage: vk::ImageUsageFlags = if is_depth {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                } | vk::ImageUsageFlags::SAMPLED;
                let texture_desc =
                    TextureDesc::new_2d(gbuffer_size.width, gbuffer_size.height, format, usage);
                let texture_view_desc = TextureViewDesc::auto(&texture_desc);
                let texture = rg.create_texutre(texture_desc);
                let view = rg.create_texture_view(texture, texture_view_desc);

                (texture, view)
            };

            // define
            gbuffer_depth = create_gbuffer(vk::Format::D24_UNORM_S8_UINT);
            gbuffer_color = create_gbuffer(vk::Format::R32G32B32A32_UINT);
            gbuffer_color_clear = vk::ClearColorValue {
                uint32: [0, 0, 0, 0],
            };
        }

        // Draw mesh
        hack.set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        let mesh_gfx_pipeline = shaders.create_gfx_pipeline(
            ShaderDefinition::vert("mesh_gbuffer.hlsl", "vs_main"),
            ShaderDefinition::frag("mesh_gbuffer.hlsl", "ps_main"),
            &hack,
        );
        if let Some(pipeline) = mesh_gfx_pipeline {
            let viewport_extent = rd.swapchain.extent;
            rg.new_pass("GBuffer_Gen", RenderPassType::Graphics)
                .pipeline(pipeline)
                .descritpro_set(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set)
                .descriptor_set_index(2)
                .color_targets(&[ColorTarget {
                    view: gbuffer_color.1,
                    load_op: ColorLoadOp::Clear(gbuffer_color_clear),
                }])
                .depth_stencil(DepthStencilTarget {
                    view: gbuffer_depth.1,
                    load_op: DepthLoadOp::Clear(vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    }),
                    store_op: vk::AttachmentStoreOp::STORE,
                })
                .texture("skycube", skycube)
                .render(move |cb, shaders, _pass| {
                    // set up raster state
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
                    cb.set_depth_test_enable(true);
                    cb.set_depth_write_enable(true);
                    cb.set_stencil_test_enable(true);
                    let face_mask = vk::StencilFaceFlags::FRONT_AND_BACK;
                    let stencil_ops = StencilOps::write_on_pass(vk::CompareOp::ALWAYS);
                    cb.set_stencil_op(face_mask, stencil_ops);
                    cb.set_stencil_write_mask(face_mask, 0x01);
                    cb.set_stencil_reference(face_mask, 0xFF);

                    let pipeline = shaders.get_pipeline(pipeline).unwrap();

                    // Draw meshs
                    cb.bind_index_buffer(
                        scene.index_buffer.buffer.handle,
                        0,
                        vk::IndexType::UINT16,
                    );
                    cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    for mesh_params in &scene.mesh_params {
                        // PushConstant for everything per-draw
                        let model_xform = glam::Mat4::IDENTITY; // Not used for now
                        let constants = PushConstantsBuilder::new()
                            // model_transform
                            .push(&model_xform.to_cols_array())
                            // vertex attributes
                            .push(&mesh_params.positions_offset)
                            .push(&mesh_params.texcoords_offset)
                            .push(&mesh_params.normals_offset)
                            .push(&mesh_params.tangents_offset)
                            // material params
                            .push(&mesh_params.material_index);
                        cb.push_constants(
                            pipeline.layout,
                            pipeline.push_constant_ranges[0].stage_flags, // TODO
                            0,
                            &constants.build(),
                        );

                        cb.draw_indexed(mesh_params.index_count, 1, mesh_params.index_offset, 0, 0);
                    }
                });
        }

        let final_color = rg.register_texture_view(rd.swapchain.image_view[image_index as usize]);

        // Draw (procedure) Sky
        let sky_pipeline = shaders.create_gfx_pipeline(
            ShaderDefinition::vert("sky_vsps.hlsl", "vs_main"),
            ShaderDefinition::frag("sky_vsps.hlsl", "ps_main"),
            &hack,
        );
        {
            let pipeline = sky_pipeline.unwrap();

            let stencil = rg.create_texture_view(
                gbuffer_depth.0,
                TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::D24_UNORM_S8_UINT,
                    aspect: vk::ImageAspectFlags::STENCIL,
                    ..Default::default()
                },
            );

            rg.new_pass("Sky", RenderPassType::Graphics)
                .pipeline(pipeline)
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

                    // Bind resources
                    cb.bind_descriptor_set(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        SCENE_DESCRIPTOR_SET_INDEX,
                        scene.descriptor_set,
                        None,
                    );

                    // Draw
                    cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    cb.draw(3, 1, 0, 0);
                });
        }

        let scene_tlas = rg.register_accel_struct(scene.scene_top_level_accel_struct.unwrap());

        let raytraced_shadow_mask = {
            let tex_desc = TextureDesc::new_2d(
                gbuffer_size.width,
                gbuffer_size.height,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            );
            let rt_shadow_tex = rg.create_texutre(tex_desc);
            rg.create_texture_view(rt_shadow_tex, TextureViewDesc::auto(&tex_desc))
        };

        // Ray Traced Shadow
        if let Some(raytraced_shadow) = shaders.create_raytracing_pipeline(
            ShaderDefinition::raygen("raytraced_shadow.hlsl", "raygen"),
            ShaderDefinition::miss("raytraced_shadow.hlsl", "miss"),
            &hack,
        ) {
            self.raytraced_shadow
                .update_shader_group_handles(rd, shaders, raytraced_shadow);
            let raygen_sbt = self.raytraced_shadow.raygen_region;
            let miss_sbt = self.raytraced_shadow.miss_region;

            rg.new_pass("RayTracedShadow", RenderPassType::RayTracing)
                .pipeline(raytraced_shadow)
                .descritpro_set(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set)
                .accel_struct("scene_tlas", scene_tlas)
                .texture("gbuffer_depth", gbuffer_depth.1)
                .rw_texture("rw_shadow", raytraced_shadow_mask)
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(raytraced_shadow).unwrap();
                    cb.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.handle);
                    cb.trace_rays(
                        &raygen_sbt,
                        &miss_sbt,
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        gbuffer_size.width,
                        gbuffer_size.height,
                        1,
                    )
                });
        }

        // GBuffer Lighting
        let gbuffer_lighting = shaders.create_compute_pipeline(
            ShaderDefinition::compute("gbuffer_lighting.hlsl", "main"),
            &hack,
        );
        if let Some(pipeline) = gbuffer_lighting {
            rg.new_pass("GBuffer_Lighting", RenderPassType::Compute)
                .pipeline(pipeline)
                .descritpro_set(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set)
                .texture("gbuffer_depth", gbuffer_depth.1)
                .texture("gbuffer_color", gbuffer_color.1)
                .texture("skycube", skycube)
                .texture("shadow_mask", raytraced_shadow_mask)
                .rw_texture("out_lighting", final_color)
                .render(move |cb, shaders, _pass| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();
                    cb.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline.handle);
                    let dispatch_x = (gbuffer_size.width + 7) / 8;
                    let diapatch_y = (gbuffer_size.height + 7) / 8;
                    cb.dispatch(dispatch_x, diapatch_y, 1);
                });
        }

        // TODO define an extra Output node instead of hacking RenderPass node
        let swapchain_view_handle =
            rg.register_texture_view(rd.swapchain.image_view[image_index as usize]);
        rg.new_pass("Present", RenderPassType::Present)
            .present_texture(swapchain_view_handle)
            .render(|_, _, _| {});

        // Run render graph and fianlize
        {
            let cb = CommandBuffer {
                device: rd.device_entry.clone(),
                raytracing_pipeline: rd.raytracing_pipeline_entry.clone(),
                nv_diagnostic_checkpoints: rd.nv_diagnostic_checkpoints_entry.clone(),
                command_buffer,
            };
            rg.execute(rd, &cb, &shaders);
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
                    .entry
                    .queue_present(rd.gfx_queue, &present_info)
                    .unwrap_or_else(|e| panic!("Failed to present: {:?}", e));
            }
        }
    }
}
