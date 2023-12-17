use std::mem::size_of;
use std::slice;

use ash::vk;
use glam::{Mat4, Vec3};

use crate::{
    model::{ImageFormat, Model},
    render_device::{
        texture::TextureUsage, AccelerationStructure, Buffer, BufferDesc, RenderDevice, Texture,
        TextureDesc, TextureView, TextureViewDesc,
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

    pub fn unused_space(&self) -> usize {
        self.buffer.desc.size as usize - self.next_pos as usize
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

struct BufferCopiesBuilder {
    staging_buffer: Buffer,
    offset: u64,
    alignemnt_mask: u64,
    buffer_copies: Vec<vk::BufferCopy>,
}

impl BufferCopiesBuilder {
    fn new(buffer: Buffer, alignment: u64) -> BufferCopiesBuilder {
        assert!(alignment.is_power_of_two());
        assert!(alignment >= 1);
        BufferCopiesBuilder {
            staging_buffer: buffer,
            offset: 0,
            buffer_copies: Vec::new(),
            alignemnt_mask: alignment - 1,
        }
    }

    fn alloc<T>(&mut self, count: usize, dst_offset: u64) -> &mut [T] {
        let size = count as u64 * size_of::<T>() as u64;
        assert!(self.offset + size <= self.staging_buffer.desc.size);
        // new copy task
        self.buffer_copies.push(vk::BufferCopy {
            src_offset: self.offset,
            dst_offset,
            size,
        });
        // make host alice
        let dst = unsafe {
            let dst_ptr = self.staging_buffer.data.offset(self.offset as isize);
            std::slice::from_raw_parts_mut(dst_ptr as *mut T, count)
        };
        // bump
        self.offset += (size + self.alignemnt_mask) & !self.alignemnt_mask;
        dst
    }
}

// Struct for asset uploading to GPU
pub struct UploadContext {
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub finished_fences: Vec<vk::Fence>,
    pub staging_buffers: Vec<Buffer>,
    pub staging_finished: Vec<vk::Event>,
    pub general_events: Vec<(vk::Event, vk::Event)>,
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
            general_events: Vec::new(),
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

    pub fn immediate_submit_no_wait<F>(&mut self, rd: &RenderDevice, f: F)
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

    // Borrow an general event object, with a auxilary event to signal when it's ready to be reused
    pub fn borrow_event(&mut self, rd: &RenderDevice) -> (vk::Event, vk::Event) {
        // Find reusable one
        for (event, recycle) in self.general_events.iter() {
            let can_recycle = unsafe { rd.device.get_event_status(*recycle).unwrap() };
            if can_recycle {
                unsafe {
                    rd.device.reset_event(*event).unwrap();
                    rd.device.reset_event(*recycle).unwrap();
                }
                return (*event, *recycle);
            }
        }

        // Create New
        let event = rd.create_event();
        let recycle_state = rd.create_event();

        (event, recycle_state)
    }
}

// TODO inlining in MeshParams?
#[repr(C)]
pub struct MaterialParams {
    pub color_metalrough_index_packed: u32,
    pub normal_index: u32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

impl MaterialParams {
    pub fn new(
        base_color_index: u32,
        metallic_roughness_index: u32,
        normal_index: u32,
        metallic_factor: f32,
        roughness_factor: f32,
    ) -> Self {
        let color_metalrough_index_packed =
            base_color_index & 0xFFFF | (metallic_roughness_index & 0xFFFF) << 16;
        MaterialParams {
            color_metalrough_index_packed,
            normal_index,
            metallic_factor,
            roughness_factor,
        }
    }
}

// Representing a triangle mesh with a materal.
#[repr(C)]
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

// A group of geomtry/mesh that is clustered together.
// Think about "an object".
// This is the (default?) unit to build a BLAS.
#[repr(C)]
pub struct GeometryGroupParams {
    pub geometry_index_offset: u32, // first (global) geomtry index in the group
    pub geometry_count: u32,        // number of geomtries in the group
    pub _pad0: u32,
    pub _pad1: u32,
}

pub struct Instance {
    pub geometry_group_index: u32, // This is also the BLAS index
    pub transform: Mat4,
    pub normal_transform: Mat4,
}

// Matching constants in `shader/scene_bindings.hlsl`
pub const SCENE_DESCRIPTOR_SET_INDEX: u32 = 1;
pub const INDEX_BUFFER_BINDING_INDEX: u32 = 0;
pub const VERTEX_BUFFER_BINDING_INDEX: u32 = 1;
pub const MATERIAL_PARAMS_BINDING_INDEX: u32 = 2;
pub const BINDLESS_TEXTURE_BINDING_INDEX: u32 = 3;
pub const MESH_PARAMS_BINDING_INDEX: u32 = 4;
pub const GEOMETRY_GROUP_PARAMS_BINDING_INDEX: u32 = 5;

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
    pub geometry_group_params: Vec<GeometryGroupParams>,
    pub instances: Vec<Instance>,

    // Global material parameter buffer for all loaded mesh; map to `material_params`
    pub material_param_buffer: Buffer,

    // Global mesh paramter buffer for all loaded mesh; map to `mesh_params`
    pub mesh_param_buffer: Buffer,

    // Global geomtry group paramter buffer for all loaded mesh; map to `geometry_group_params`
    pub geometry_group_param_buffer: Buffer,

    // Ray Tracing Acceleration structures for whole scene
    pub mesh_bottom_level_accel_structs: Vec<AccelerationStructure>,
    pub scene_top_level_accel_struct: Option<AccelerationStructure>,

    // Lighting settings
    pub sun_dir: Vec3,
}

impl RenderScene {
    pub fn new(rd: &RenderDevice) -> RenderScene {
        // Buffer for whole scene
        let ib_size = 8 * 1024 * 1024;
        let vb_size = 128 * 1024 * 1024;
        let accel_struct_usage = if rd.support_raytracing {
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let index_buffer = AllocBuffer::new(
            rd.create_buffer(BufferDesc {
                size: ib_size,
                usage: vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | accel_struct_usage,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap(),
        );
        let index_buffer_view = rd
            .create_buffer_view(index_buffer.buffer, vk::Format::R16_UINT)
            .unwrap();
        let vertex_buffer = AllocBuffer::new(
            rd.create_buffer(BufferDesc {
                size: vb_size,
                usage: vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | accel_struct_usage,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap(),
        );
        let vertex_buffer_view = rd
            .create_buffer_view(vertex_buffer.buffer, vk::Format::R32_UINT)
            .unwrap();

        // Material Parameters buffer
        let material_param_size = std::mem::size_of::<MaterialParams>() as vk::DeviceSize;
        let material_param_buffer = rd
            .create_buffer(BufferDesc {
                size: material_param_size * 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        // Mesh Paramters buffer
        let mesh_param_size = std::mem::size_of::<MeshParams>() as vk::DeviceSize;
        let mesh_param_buffer = rd
            .create_buffer(BufferDesc {
                size: mesh_param_size * 2048,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            })
            .unwrap();

        // GeometryGroup Paramters buffer
        let geometry_group_param_size =
            std::mem::size_of::<GeometryGroupParams>() as vk::DeviceSize;
        let geometry_group_param_buffer = rd
            .create_buffer(BufferDesc {
                size: geometry_group_param_size * 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
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
            let geometry_group_buffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(GEOMETRY_GROUP_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL);
            let bindings = [
                *vbuffer,
                *ibuffer,
                *mat_buffer,
                *bindless_textures,
                *mesh_buffer,
                *geometry_group_buffer,
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::default(),
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
                .texel_buffer_view(slice::from_ref(&vertex_buffer_view.handle))
                .build();
            let write_ibuffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(INDEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .texel_buffer_view(slice::from_ref(&index_buffer_view.handle))
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
            let geometry_group_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(geometry_group_param_buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let write_geometry_group_buffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(GEOMETRY_GROUP_PARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(slice::from_ref(&geometry_group_buffer_info))
                .build();
            unsafe {
                rd.device.update_descriptor_sets(
                    &[
                        write_vbuffer,
                        write_ibuffer,
                        write_mat_buffer,
                        write_mesh_buffer,
                        write_geometry_group_buffer,
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
            geometry_group_params: Vec::new(),
            instances: Vec::new(),
            material_param_buffer,
            mesh_param_buffer,
            geometry_group_param_buffer,
            mesh_bottom_level_accel_structs: Vec::new(),
            scene_top_level_accel_struct: None,
            sun_dir: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    pub fn add(&mut self, rd: &RenderDevice, model: &Model) {
        let upload_context = &mut self.upload_context;

        let index_buffer = &mut self.index_buffer;
        let vertex_buffer = &mut self.vertex_buffer;

        let global_texture_index_offset = self.material_textures.len() as u32;
        let global_texture_view_index_offset = self.material_texture_views.len() as u32;
        let global_material_index_offset = self.material_parmas.len() as u32;
        let global_mesh_index_offset = self.mesh_params.len();
        let global_geometry_group_index_offset = self.geometry_group_params.len();

        // log
        let total_vert_count = model
            .geometry_groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|mesh| mesh.1.positions.len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        println!("Total Vertex: {}", total_vert_count);

        // Upload textures
        for image in &model.images {
            let format = match image.format {
                ImageFormat::R8G8B8A8Unorm => vk::Format::R8G8B8A8_UNORM,
                ImageFormat::R8G8B8A8Srgb => vk::Format::R8G8B8A8_SRGB,
                ImageFormat::BC5Unorm => vk::Format::BC5_UNORM_BLOCK,
                ImageFormat::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,
                ImageFormat::BC7Srgb => vk::Format::BC7_SRGB_BLOCK,
            };
            let mip_level_count = image.mips_data.len() as u32;

            // Create texture object
            let texture = rd
                .create_texture(TextureDesc {
                    width: image.width,
                    height: image.height,
                    mip_level_count,
                    format,
                    usage: TextureUsage::new().sampled().transfer_dst().into(),
                    ..Default::default()
                })
                .unwrap();

            let buffer_size = image.mips_data.iter().map(Vec::len).sum();

            // Create staging buffer
            let staging_buffer = upload_context.borrow_staging_buffer(rd, buffer_size as u64);

            // Write to staging buffer
            let mut buffer_offsets = Vec::with_capacity(image.mips_data.len());
            {
                let staging_slice =
                    unsafe { std::slice::from_raw_parts_mut(staging_buffer.0.data, buffer_size) };
                let mut dst = 0;
                for mip in image.mips_data.iter() {
                    let dst_end = dst + mip.len();
                    staging_slice[dst..dst_end].copy_from_slice(mip);
                    buffer_offsets.push(dst);
                    dst = dst_end;
                }
            }

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
                            level_count: mip_level_count,
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
                let mut width = image.width;
                let mut height = image.height;
                let mut regions = Vec::with_capacity(mip_level_count as usize);
                for mip_level in 0..mip_level_count {
                    let region = vk::BufferImageCopy::builder()
                        .buffer_offset(buffer_offsets[mip_level as usize] as vk::DeviceSize)
                        .buffer_row_length(0) // zero means tightly packed
                        .buffer_image_height(0) // zero means tightly packed
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        });
                    regions.push(region.build());

                    // ref: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#resources-image-mip-level-sizing
                    width = (width / 2).max(1);
                    height = (height / 2).max(1);
                }
                let dst_image_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                unsafe {
                    rd.device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer.0.handle,
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
                            level_count: mip_level_count,
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

                // Return the staging buffer
                unsafe {
                    rd.device.cmd_set_event(
                        command_buffer,
                        staging_buffer.1,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }
            });

            // Queue to upload to scene data

            self.material_textures.push(texture);
        }

        // Collect materials
        for material in &model.materials {
            // Create texture view for each material and add to scene
            // TODO collect and reduce duplicate in view?
            // TODO default textures
            let mut resolve = |map: &Option<crate::model::MaterialMap>| match map {
                Some(map) => {
                    let texture = {
                        let texture_index = global_texture_index_offset + map.image_index;
                        self.material_textures[texture_index as usize]
                    };
                    let view_desc = TextureViewDesc {
                        view_type: vk::ImageViewType::TYPE_2D,
                        format: texture.desc.format,
                        aspect: vk::ImageAspectFlags::COLOR,
                        ..Default::default()
                    };
                    let texture_view = rd.create_texture_view(texture, view_desc).unwrap();
                    let texture_view_index = self.material_texture_views.len() as u32;
                    self.material_texture_views.push(texture_view);
                    texture_view_index
                }
                None => 0,
            };

            self.material_parmas.push(MaterialParams::new(
                resolve(&material.base_color_map),
                resolve(&material.metallic_roughness_map),
                resolve(&material.normal_map),
                material.metallic_factor,
                material.roughness_factor,
            ));
        }

        // Upload new material params
        {
            let param_size = std::mem::size_of::<MaterialParams>();
            let param_count = self.material_parmas.len() - global_material_index_offset as usize;
            let data_offset = global_material_index_offset as isize * param_size as isize;
            let data_size = param_size * param_count;

            let staging_buffer = upload_context.borrow_staging_buffer(rd, data_size as u64);

            unsafe {
                let src = std::slice::from_raw_parts(
                    (self.material_parmas.as_ptr() as *const u8).offset(data_offset),
                    data_size,
                );
                let dst = std::slice::from_raw_parts_mut(
                    //self.material_param_buffer.data.offset(data_offset),
                    staging_buffer.0.data,
                    data_size,
                );
                dst.copy_from_slice(src);
            }

            // Transfer to global buffer
            upload_context.immediate_submit(rd, |command_buffer| {
                // Copy buffer
                let buffer_copy = vk::BufferCopy::builder()
                    .dst_offset(data_offset as u64)
                    .size(data_size as u64)
                    .size(data_size as u64)
                    .build();
                unsafe {
                    rd.device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer.0.handle,
                        self.material_param_buffer.handle,
                        std::slice::from_ref(&buffer_copy),
                    )
                }

                // Return the staging buffer
                unsafe {
                    rd.device.cmd_set_event(
                        command_buffer,
                        staging_buffer.1,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }
            })
        }

        // Add new texture view to bindless texture array
        {
            let image_infos: Vec<vk::DescriptorImageInfo> = self.material_texture_views
                [global_texture_index_offset as usize..]
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
                .dst_array_element(global_texture_view_index_offset)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_infos);
            unsafe {
                rd.device.update_descriptor_sets(&[*write], &[]);
            }
        }

        let align_4 = |pos: usize| (pos + 3) & !3;

        // Collect mesh data sizes
        let mut total_index_bytesize = 0usize;
        let mut total_vertex_bytesize = 0usize;
        for group in &model.geometry_groups {
            for (_mat_index, mesh) in group {
                let index_count = mesh.indicies.len();
                total_index_bytesize += align_4(index_count * size_of::<u16>());
                let vertex_count = mesh.positions.len();
                let vertex_size = size_of::<[f32; 3]>() // position
                + (mesh.normals.is_some() as usize) * size_of::<[f32; 3]>() // normal
                + (mesh.texcoords.is_some() as usize) * size_of::<[f32; 2]>() // texcoords
                + (mesh.tangents.is_some() as usize) * size_of::<[f32; 4]>() // tangent
                ;
                total_vertex_bytesize += vertex_count * vertex_size;
            }
        }

        // TODO auto growing the buffer?
        assert!(total_vertex_bytesize <= vertex_buffer.unused_space());
        assert!(total_index_bytesize <= index_buffer.unused_space());

        // Get staging buffer for all trangle meshes
        let index_staging_buffer =
            upload_context.borrow_staging_buffer(rd, total_index_bytesize as u64);
        let vertex_staging_buffer =
            upload_context.borrow_staging_buffer(rd, total_vertex_bytesize as u64);
        let mut index_copy_buider = BufferCopiesBuilder::new(index_staging_buffer.0, 4);
        let mut vertex_copy_buider = BufferCopiesBuilder::new(vertex_staging_buffer.0, 4);

        // Upload mesh data, and prepare BLAS builds
        // We orgnize such that:
        //  - one GeometryGroup <-> one BLAS, and
        //  - one TriangleMesh in a MeshGroup <-> one 'geometry' in the BLAS.
        struct BlasParams {
            beg: usize,
            end: usize,
        }
        let mut all_blas_geometries = Vec::<vk::AccelerationStructureGeometryKHR>::new();
        let mut all_blas_build_range_infos =
            Vec::<vk::AccelerationStructureBuildRangeInfoKHR>::new();
        let mut all_blas_geometry_primitive_counts = Vec::<u32>::new(); // sub set of `blas_build_ranges`
        let mut blas_params = Vec::<BlasParams>::new();
        for group in &model.geometry_groups {
            // Add GeometryGroup
            self.geometry_group_params.push(GeometryGroupParams {
                geometry_index_offset: self.mesh_params.len() as u32,
                geometry_count: group.len() as u32,
                _pad0: 0,
                _pad1: 0,
            });

            // Add each geometry (TriangleMesh)
            let blas_geometry_offset = all_blas_geometries.len();
            for (material_index, mesh) in group {
                // Upload indices
                let index_count = mesh.indicies.len();
                let index_offset_by_u16;
                let index_offset_bytes;
                {
                    // allocate
                    let (_, offset) = index_buffer.alloc::<u16>(index_count as u32);
                    index_offset_bytes = offset;
                    index_offset_by_u16 = offset / 2;
                    // copy
                    let dst = index_copy_buider.alloc::<u16>(index_count, offset as u64);
                    dst.copy_from_slice(&mesh.indicies);
                }
                // Upload position
                let positions_offset_by_f32;
                let positions_offset_bytes;
                {
                    // alloc
                    let count = mesh.positions.len();
                    let (_, offset) = vertex_buffer.alloc::<[f32; 3]>(count as u32);
                    positions_offset_by_f32 = offset / 4;
                    positions_offset_bytes = offset;
                    // copy
                    let dst = vertex_copy_buider.alloc::<[f32; 3]>(count, offset as u64);
                    dst.copy_from_slice(&mesh.positions);
                }
                // Upload normal
                let normals_offset_by_f32;
                if let Some(normals) = &mesh.normals {
                    // alloc
                    let count = normals.len();
                    let (_, offset) = vertex_buffer.alloc::<[f32; 3]>(count as u32);
                    normals_offset_by_f32 = offset / 4;
                    // copy
                    let dst = vertex_copy_buider.alloc::<[f32; 3]>(count, offset as u64);
                    dst.copy_from_slice(&normals);
                } else {
                    normals_offset_by_f32 = u32::MAX;
                }
                // Upload texcoords
                let texcoords_offset_by_f32;
                if let Some(texcoords) = &mesh.texcoords {
                    // alloc
                    let count = texcoords.len();
                    let (_, offset) = vertex_buffer.alloc::<[f32; 2]>(count as u32);
                    texcoords_offset_by_f32 = offset / 4;
                    // copy
                    let dst = vertex_copy_buider.alloc::<[f32; 2]>(count, offset as u64);
                    dst.copy_from_slice(&texcoords);
                } else {
                    texcoords_offset_by_f32 = u32::MAX;
                }
                // Upload tangents
                let tangents_offset_by_f32;
                if let Some(tangents) = &mesh.tangents {
                    // alloc
                    let count = tangents.len();
                    let (_, offset) = vertex_buffer.alloc::<[f32; 4]>(count as u32);
                    tangents_offset_by_f32 = offset / 4;
                    // copy
                    let dst = vertex_copy_buider.alloc::<[f32; 4]>(count, offset as u64);
                    dst.copy_from_slice(&tangents);
                } else {
                    tangents_offset_by_f32 = u32::MAX;
                }

                self.mesh_params.push(MeshParams {
                    index_offset: index_offset_by_u16,
                    index_count: index_count as u32,
                    positions_offset: positions_offset_by_f32,
                    normals_offset: normals_offset_by_f32,
                    texcoords_offset: texcoords_offset_by_f32,
                    tangents_offset: tangents_offset_by_f32,
                    material_index: global_material_index_offset + *material_index as u32,
                    pad: 0,
                });

                if !rd.support_raytracing {
                    continue;
                }

                // Collect BLAS info //

                let triangle_count = (index_count / 3) as u32;

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
                        // NOTE transform is per instance, not applied in the BLAS
                        .transform_data(vk::DeviceOrHostAddressConstKHR::default()) // no xform
                        .build(),
                };
                let blas_geo = vk::AccelerationStructureGeometryKHR::builder()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(geo_data)
                    .flags(vk::GeometryFlagsKHR::OPAQUE) // todo check material
                    .build();
                let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(triangle_count)
                    .primitive_offset(0) // a.k.a index_offset_bytes for indexed triangle geomtry; not using because already offset via address in GeometryDataKHR.index_data
                    .first_vertex(0) // a.k.a vertex_offset for indexed triangle geometry; not using because already offset via address in GeometryDataKHR.vertex_data
                    .transform_offset(0) // no xform
                    .build();
                all_blas_geometries.push(blas_geo);
                all_blas_build_range_infos.push(build_range_info);
                all_blas_geometry_primitive_counts.push(triangle_count);
            }
            assert!(all_blas_geometries.len() == all_blas_build_range_infos.len());
            assert!(all_blas_geometries.len() == all_blas_geometry_primitive_counts.len());

            let beg = blas_geometry_offset;
            let end = all_blas_geometries.len();
            blas_params.push(BlasParams { beg, end });
        }

        // Transfer mesh data to global buffer
        upload_context.immediate_submit(rd, |command_buffer| {
            // index
            unsafe {
                rd.device.cmd_copy_buffer(
                    command_buffer,
                    index_staging_buffer.0.handle,
                    index_buffer.buffer.handle,
                    &index_copy_buider.buffer_copies,
                );
            }
            // vertex
            unsafe {
                rd.device.cmd_copy_buffer(
                    command_buffer,
                    vertex_staging_buffer.0.handle,
                    vertex_buffer.buffer.handle,
                    &vertex_copy_buider.buffer_copies,
                );
            }
            // return staging buffers
            unsafe {
                rd.device.cmd_set_event(
                    command_buffer,
                    index_staging_buffer.1,
                    vk::PipelineStageFlags::TRANSFER,
                );
                rd.device.cmd_set_event(
                    command_buffer,
                    vertex_staging_buffer.1,
                    vk::PipelineStageFlags::TRANSFER,
                );
            }
        });

        // Upload new mesh params
        {
            let param_size = std::mem::size_of::<MeshParams>();
            let param_count = self.mesh_params.len() - global_mesh_index_offset;
            let data_offset = global_mesh_index_offset as usize * param_size;
            let data_size = param_size * param_count;

            // TODO auto grow buffer size?
            assert!(data_offset + data_size <= self.mesh_param_buffer.desc.size as usize);

            let staging_buffer = upload_context.borrow_staging_buffer(rd, data_size as u64);

            unsafe {
                let src = std::slice::from_raw_parts(
                    (self.mesh_params.as_ptr() as *const u8).offset(data_offset as isize),
                    data_size,
                );
                let dst = std::slice::from_raw_parts_mut(staging_buffer.0.data, data_size);
                dst.copy_from_slice(src);
            }

            upload_context.immediate_submit(rd, |command_buffer| {
                let buffer_copy = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: data_offset as u64,
                    size: data_size as u64,
                };
                unsafe {
                    rd.device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer.0.handle,
                        self.mesh_param_buffer.handle,
                        slice::from_ref(&buffer_copy),
                    );
                }

                unsafe {
                    rd.device.cmd_set_event(
                        command_buffer,
                        staging_buffer.1,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }
            })
        }

        // Upload new GeometryGroup params
        {
            let param_size = std::mem::size_of::<GeometryGroupParams>();
            let param_count = self.geometry_group_params.len() - global_geometry_group_index_offset;
            let data_offset = global_geometry_group_index_offset as usize * param_size;
            let data_size = param_size * param_count;

            let staging_buffer = upload_context.borrow_staging_buffer(rd, data_size as u64);

            unsafe {
                let src = std::slice::from_raw_parts(
                    (self.geometry_group_params.as_ptr() as *const u8).offset(data_offset as isize),
                    data_size,
                );
                let dst = std::slice::from_raw_parts_mut(staging_buffer.0.data, data_size);
                dst.copy_from_slice(src);
            }

            upload_context.immediate_submit(rd, |command_buffer| {
                let buffer_copy = vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: data_offset as u64,
                    size: data_size as u64,
                };
                unsafe {
                    rd.device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer.0.handle,
                        self.geometry_group_param_buffer.handle,
                        slice::from_ref(&buffer_copy),
                    );
                }

                unsafe {
                    rd.device.cmd_set_event(
                        command_buffer,
                        staging_buffer.1,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }
            })
        }

        // Collect instances
        for instance in &model.instances {
            let geometry_group_index =
                instance.geometry_group_index + global_geometry_group_index_offset as u32;
            self.instances.push(Instance {
                geometry_group_index,
                transform: instance.transform,
                normal_transform: instance.transform.inverse().transpose(),
            });
        }

        if !rd.support_raytracing {
            return;
        }

        // GeomtryGroup <-> BLAS
        assert_eq!(
            self.mesh_bottom_level_accel_structs.len(),
            global_geometry_group_index_offset
        );

        // Build BLASes
        if let Some(khr_accel_struct) = rd.khr_accel_struct.as_ref() {
            // Gather all build info
            let mut scratch_offset_curr = 0;
            let mut scratch_offsets = Vec::<u64>::new();
            let mut blas_sizes = Vec::<u64>::new();
            assert_eq!(
                all_blas_geometries.len(),
                all_blas_geometry_primitive_counts.len()
            );
            for params in &blas_params {
                // Get build size
                // NOTE: actual buffer addresses in blas_geometrys are ignored
                let geometries = &all_blas_geometries[params.beg..params.end];
                // NOTE: this is max, because we are not reusing the scratch buffer for other build
                let max_primitive_counts =
                    &all_blas_geometry_primitive_counts[params.beg..params.end];
                let build_size_info = unsafe {
                    let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                        .geometries(geometries)
                        .build();
                    khr_accel_struct.get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &build_info,
                        max_primitive_counts,
                    )
                };

                let scratch_offset = {
                    let alignment = rd
                        .physical
                        .accel_struct_properties
                        .min_acceleration_structure_scratch_offset_alignment;
                    let mask = (alignment - 1) as u64;
                    (scratch_offset_curr + mask) & !mask
                };
                scratch_offsets.push(scratch_offset);
                scratch_offset_curr = scratch_offset + build_size_info.build_scratch_size;

                blas_sizes.push(build_size_info.acceleration_structure_size);
            }
            let total_scratch_size = scratch_offset_curr;

            // Create scratch buffer (batching all BLAS builds)
            let scratch_buffer = rd
                .create_buffer(BufferDesc {
                    size: total_scratch_size,
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    memory_property: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                })
                .unwrap();
            let scratch_device_address = scratch_buffer.device_address.unwrap();

            // Collect all BLAS builds
            let mut blas_build_infos = Vec::<vk::AccelerationStructureBuildGeometryInfoKHR>::new();
            let mut blas_build_range_info_slices =
                Vec::<&[vk::AccelerationStructureBuildRangeInfoKHR]>::new();
            for i in 0..blas_params.len() {
                let params = &blas_params[i];
                let blas_size = blas_sizes[i];

                // Create buffer
                let buffer = rd
                    .create_buffer(BufferDesc {
                        size: blas_size,
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
                        blas_size,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    )
                    .unwrap();

                // Record it
                self.mesh_bottom_level_accel_structs.push(accel_struct);

                // Setup BLAS build info
                let geometries = &all_blas_geometries[params.beg..params.end];
                let scrath_offset = scratch_offsets[i];
                let geo_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(accel_struct.handle)
                    .geometries(geometries)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch_device_address + scrath_offset,
                    })
                    .build();
                blas_build_infos.push(geo_info);

                let build_range_info_slice = &all_blas_build_range_infos[params.beg..params.end];
                blas_build_range_info_slices.push(build_range_info_slice);
            }

            // Build
            upload_context.immediate_submit(rd, move |cb| {
                unsafe {
                    khr_accel_struct.cmd_build_acceleration_structures(
                        cb,
                        &blas_build_infos,
                        &blas_build_range_info_slices,
                    )
                };
            });

            // Clean up
            rd.destroy_buffer(scratch_buffer);
        }
    }

    pub fn rebuild_top_level_accel_struct(&mut self, rd: &RenderDevice) -> Option<()> {
        if !rd.support_raytracing {
            return None;
        }

        // GeometryGroup <-> BLAS
        assert_eq!(
            self.geometry_group_params.len(),
            self.mesh_bottom_level_accel_structs.len()
        );

        let khr_accel_struct = rd.khr_accel_struct.as_ref()?;

        // Create instance buffer
        let num_blas_instances = self.instances.len();
        let instance_buffer = rd
            .create_buffer(BufferDesc {
                size: (size_of::<vk::AccelerationStructureInstanceKHR>() * num_blas_instances)
                    as u64,
                usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        // Traverse all GeometryGroup instances to fill the instance buffer
        for (index, instance) in self.instances.iter().enumerate() {
            // BLAS <-> GeometryGroup
            let blas_index = instance.geometry_group_index;
            let blas = self.mesh_bottom_level_accel_structs[blas_index as usize];

            // Fill instance buffer (3x4 row-major affine transform)
            let mut matrix = [0.0f32; 12];
            {
                let row_data = instance.transform.transpose().to_cols_array();
                matrix[0..12].copy_from_slice(&row_data[0..12]);
            }
            let instance = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix },
                instance_custom_index_and_mask: vk::Packed24_8::new(
                    blas_index, // used to access BLAS/GeomtryGroup data
                    0xFF,
                ),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0, // TODO not used in hit shader
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
            let max_primitive_count = num_blas_instances as u32;
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
                .primitive_count(num_blas_instances as u32)
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

        rd.destroy_buffer(instance_buffer);
        rd.destroy_buffer(scratch_buffer);

        self.scene_top_level_accel_struct.replace(accel_struct);

        Some(())
    }
}
