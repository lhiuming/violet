use std::mem::{self, size_of};
use std::slice;

use ash::vk::{self};
use glam::{Mat4, UVec2, Vec3};

use crate::command_buffer::StencilOps;
use crate::model::Model;
use crate::render_device::{
    Buffer, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
};
use crate::render_graph::{RGHandle, RenderGraph, RenderPass};
use crate::shader::{
    HackStuff, Pipeline, PushConstantsBuilder, ShaderDefinition, ShaderStage, Shaders,
};

// Allocatable buffer. Alway aligned to 4 bytes.
pub struct AllocBuffer {
    pub buffer: Buffer,
    next_pos: u32,
}

impl AllocBuffer {
    pub fn new(buffer: Buffer) -> AllocBuffer {
        AllocBuffer {
            buffer: buffer,
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

// Use a u64 to represent a 8x8 sub-tree,
// each node is represented as a bit, each parent is the "and" of its children.
// TODO actually implement a tree
struct BooleanQuadtree {
    max_depth: u32,
    data: u64,
}

// Interleave the binary representation by zeros
// ref: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
fn part1by1(mut x: u32) -> u32 {
    x = x & 0x0000ffff;
    x = (x | x << 8) & 0x00ff00ff;
    x = (x | x << 4) & 0x0f0f0f0f;
    x = (x | x << 2) & 0x33333333; // 0x3=0b0011
    x = (x | x << 1) & 0x55555555; // 0x5=0b0101
    x
}

fn compact1by1(mut x: u32) -> u32 {
    x = x & 0x55555555; // 0x5=0x0101
    x = (x | (x >> 1)) & 0x33333333; // 0x3=0b0011
    x = (x | (x >> 2)) & 0x0f0f0f0f;
    x = (x | (x >> 4)) & 0x00ff00ff;
    x = (x | (x >> 8)) & 0x0000ffff;
    x
}

impl BooleanQuadtree {
    fn new(max_depth: u32) -> BooleanQuadtree {
        assert!(max_depth <= 4);
        BooleanQuadtree { max_depth, data: 0 }
    }

    fn find_empty_and_set(&mut self, depth: u32) -> Option<(u32, u32)> {
        assert!(depth < 4);
        let data = self.data;
        let height = 3 - depth;
        let stride = 1 << (height * 2); // 0->1, 1->4, 2->16, 3->64
        let mask: u64 = 0xFFFFFFFFFFFFFFFF >> (64 - stride); // 1->1, 4->3, 16->15, 64->63

        // Linear saerch on the nodes, find an empty one
        let count = 1u32 << depth;
        for i in 0..count {
            let pos = i * stride;
            let data = (data >> pos) & mask;
            if data == 0 {
                // Found an empty node, set it to 1
                self.data = data | (mask << pos);

                // Calculate 2D position from morton code position i
                let x = compact1by1(i);
                let y = compact1by1(i >> 1);
                return Some((x, y));
            }
        }

        None
    }
}

// Allocatable texture. Always aligned to fix-size rectangle tile.
// Allocate a morton code order. (but why?)
pub struct AllocTexture2D {
    pub texture: Texture,
    pub view: TextureView,
    tree_depth: u32,
    occupation_per_layer: Vec<BooleanQuadtree>,
}

impl AllocTexture2D {
    pub fn new(texture: Texture, view: TextureView) -> AllocTexture2D {
        let desc = &texture.desc;

        let tile_size = 256;
        assert!(desc.width >= tile_size);
        assert!(desc.height >= tile_size);
        assert!(desc.width.is_power_of_two());
        assert!(desc.height.is_power_of_two());

        // Size of the tile map
        let width: u32 = desc.width / tile_size;
        let height: u32 = desc.height / tile_size;

        // Fit it to a square such that morton code works
        let width = std::cmp::min(width, height);
        let height = std::cmp::min(width, height);
        if (height * tile_size != desc.height) || (width * tile_size != desc.width) {
            println!(
                "Warning: texture size is not optimal. {}x{} -> {}x{}. Some memory is watsted",
                desc.width,
                desc.height,
                width * tile_size,
                height * tile_size
            );
        }

        let tree_depth = width.trailing_zeros() + 1;
        let occupation_per_layer = (0..desc.array_len)
            .map(|_| BooleanQuadtree::new(tree_depth))
            .collect();

        AllocTexture2D {
            texture,
            view,
            tree_depth,
            occupation_per_layer,
        }
    }

    pub fn size(&self) -> UVec2 {
        UVec2 {
            x: self.texture.desc.width,
            y: self.texture.desc.height,
        }
    }

    pub fn alloc<'a>(&mut self, width: u32, height: u32) -> Option<(u32, u32, u32)> {
        let tile_size = 256;

        let size = std::cmp::max(width, height);
        let size_in_tile = (size + tile_size - 1) / tile_size;
        let height_in_tree = size_in_tile.trailing_zeros();
        assert!(height_in_tree < self.tree_depth);
        let depth = self.tree_depth - 1 - height_in_tree;

        for layer in 0..self.occupation_per_layer.len() {
            let tree = &mut self.occupation_per_layer[layer];
            if let Some((x, y)) = tree.find_empty_and_set(depth) {
                let node_size = tile_size * size_in_tile;
                return Some((x * node_size, y * node_size, layer as u32));
            }
        }
        None
    }
}

fn clamp<T>(v: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if min > v {
        return min;
    }
    if max < v {
        return max;
    }
    v
}

fn float_to_unorm(v: f32) -> u32 {
    unsafe { clamp((v * 255.0).round().to_int_unchecked(), 0, 255) }
}

fn pack_unorm(x: f32, y: f32, z: f32, w: f32) -> u32 {
    float_to_unorm(x)
        | (float_to_unorm(y) << 8)
        | (float_to_unorm(z) << 16)
        | (float_to_unorm(w) << 24)
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
        }
    }
}

pub struct TextureParams {
    //pub index: u32,
}

pub struct MaterialParams {
    pub base_color_index: u32,
    pub metallic_roughness_index: u32,
    pub normal_index: u32,
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

const SCENE_DESCRIPTOR_SET_INDEX: u32 = 1;
const VERTEX_BUFFER_BINDING_INDEX: u32 = 0;
const BINDLESS_TEXTURE_BINDING_INDEX: u32 = 1;
const VIEWPARAMS_BINDING_INDEX: u32 = 2;
const SAMPLER_BINDING_INDEX: u32 = 3;

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

    // Global texture to store all loaded textures
    //pub material_texture: AllocTexture2D,

    // Global texture arrays, mapping the bindless textures
    pub material_textures: Vec<Texture>,
    pub material_texture_views: Vec<TextureView>,

    // Stuff to be rendered
    //pub texture_params: Vec<TextureParams>,
    pub material_parmas: Vec<MaterialParams>,
    pub mesh_params: Vec<MeshParams>,
}

impl RenderScene {
    pub fn new(rd: &RenderDevice) -> RenderScene {
        // Buffer for whole scene
        let ib_size = 4 * 1024 * 1024;
        let vb_size = 4 * 1024 * 1024;
        let index_buffer = AllocBuffer::new(
            rd.create_buffer(
                ib_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::Format::UNDEFINED,
            )
            .unwrap(),
        );
        let vertex_buffer = AllocBuffer::new(
            rd.create_buffer(
                vb_size,
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                vk::Format::R32_UINT,
            )
            .unwrap(),
        );

        // View parameter constant buffer
        let view_params_cb = rd
            .create_buffer(
                mem::size_of::<ViewParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::Format::UNDEFINED,
            )
            .unwrap();

        /*
        // Texture for whole scene
        let tex_width = 2048;
        let tex_height = 2048;
        let tex_array_len = 5;
        let material_texture = {
            let texture = rd
                .create_texture(TextureDesc::new_2d_array(
                    tex_width,
                    tex_height,
                    tex_array_len,
                    vk::Format::R8G8B8A8_SRGB,
                    vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ))
                .unwrap();
            let texture_view = rd
                .create_texture_view(&texture, TextureViewDesc::default(&texture))
                .unwrap();
            AllocTexture2D::new(texture, texture_view)
        };
        */

        // Shared samplers
        let shared_sampler = unsafe {
            let create_info = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR);
            rd.device.create_sampler(&create_info, None).unwrap()
        };

        // Descriptor pool for whole scene bindless texture
        // TODO specific size and stuff
        let descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
        );

        // Create the scene/persistent binding set
        let descriptor_set_layout = {
            let vbuffer = vk::DescriptorSetLayoutBinding::builder()
                .binding(VERTEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
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
            let bindings = [*vbuffer, *bindless_textures, *cbuffer, *sampler_ll];
            let binding_flags = [
                vk::DescriptorBindingFlags::default(),
                bindless_textures_flags,
                vk::DescriptorBindingFlags::default(),
                vk::DescriptorBindingFlags::default(),
            ];
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
            let write_buffer_view = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(VERTEX_BUFFER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
                .texel_buffer_view(slice::from_ref(&vertex_buffer.buffer.srv.unwrap()))
                .build();
            let cbuffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(view_params_cb.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build();
            let write_cbuffer = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(VIEWPARAMS_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(slice::from_ref(&cbuffer_info))
                .build();
            let sampler_info = vk::DescriptorImageInfo::builder()
                .sampler(shared_sampler)
                .build();
            /* Using immutable samplers; no write.
            let write_sampler = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(SAMPLER_BINDING_INDEX)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(slice::from_ref(&sampler_info))
                .build();
             */
            unsafe {
                rd.device
                    .update_descriptor_sets(&[write_buffer_view, write_cbuffer], &[])
            }
        }

        RenderScene {
            upload_context: UploadContext::new(rd),
            view_params_cb,
            shared_sampler,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_set,
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            //material_texture: material_texture,
            material_textures: Vec::new(),
            material_texture_views: Vec::new(),
            material_parmas: Vec::new(),
            mesh_params: Vec::new(),
        }
    }

    pub fn add(&mut self, rd: &RenderDevice, model: &Model) {
        let upload_context = &mut self.upload_context;
        //let material_texture = &mut self.material_texture;
        let index_buffer = &mut self.index_buffer;
        let vertex_buffer = &mut self.vertex_buffer;

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
                    vk::Format::UNDEFINED,
                )
                .unwrap();

            // Read to staging buffer
            let staging_slice = unsafe {
                std::slice::from_raw_parts_mut(
                    staging_buffer.data as *mut u8,
                    texel_count as usize * 4,
                )
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
                        staging_buffer.buffer,
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
                        &self.material_textures[texture_index as usize]
                    };
                    let format = if is_srgb {
                        vk::Format::R8G8B8A8_SRGB
                    } else {
                        vk::Format::R8G8B8A8_UNORM
                    };
                    let texture_view = rd
                        .create_texture_view(texture, TextureViewDesc::with_format(texture, format))
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
            });
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

        for (material_index, mesh) in model.meshes.iter().enumerate() {
            // Upload indices
            let index_offset;
            let index_count;
            {
                index_count = mesh.indicies.len() as u32;
                let (dst, offset) = index_buffer.alloc(index_count);
                index_offset = offset / 4;
                dst.copy_from_slice(&mesh.indicies);
            }
            // Upload position
            let positions_offset;
            {
                let (dst, offset) = vertex_buffer.alloc::<[f32; 3]>(mesh.positions.len() as u32);
                positions_offset = offset / 4;
                dst.copy_from_slice(&mesh.positions);
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
        }
    }
}

#[repr(C)]
pub struct ViewParams {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub view_pos: Vec3,
    pub padding: f32,
}

pub struct ViewInfo {
    pub view_position: Vec3,
    pub view_transform: Mat4,
    pub projection: Mat4,
}

pub struct RednerLoop {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub present_finished_fence: vk::Fence,
    pub present_finished_semephore: vk::Semaphore,
    pub present_ready_semaphore: vk::Semaphore,
    pub command_buffer_finished_fence: vk::Fence,

    pub depth_stencil_buffer: Texture,
    pub depth_stencil_buffer_view: TextureView,
}

impl RednerLoop {
    pub fn new(rd: &RenderDevice) -> RednerLoop {
        let command_pool = rd.create_command_pool();
        let command_buffer = rd.create_command_buffer(command_pool);

        let surface_size = rd
            .surface
            .query_size(&rd.surface_entry, &rd.physical_device);
        let swapchain_size = rd.swapchain.extent;
        assert_eq!(surface_size, swapchain_size);

        // Create depth buffer
        let depth_stencil_buffer = rd
            .create_texture(TextureDesc::new_2d(
                surface_size.width,
                surface_size.height,
                vk::Format::D24_UNORM_S8_UINT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap();
        let depth_stencil_buffer_view = rd
            .create_texture_view(
                &depth_stencil_buffer,
                TextureViewDesc::default(&depth_stencil_buffer),
            )
            .unwrap();

        RednerLoop {
            command_pool,
            command_buffer,
            present_finished_fence: rd.create_fence(false),
            present_finished_semephore: rd.create_semaphore(),
            present_ready_semaphore: rd.create_semaphore(),
            command_buffer_finished_fence: rd.create_fence(true),
            depth_stencil_buffer,
            depth_stencil_buffer_view,
        }
    }

    pub fn render(
        &self,
        rd: &RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        let device = &rd.device;
        let physical_device = &rd.physical_device;
        let surface_entry = &rd.surface_entry;
        let swapchain_entry = &rd.swapchain_entry;
        let gfx_queue = &rd.gfx_queue;
        let command_buffer = self.command_buffer;
        let surface = &rd.surface;
        let swapchain = &rd.swapchain;

        let surface_size = surface.query_size(&surface_entry, physical_device);

        let mut rg = RenderGraph::new();
        let use_rg = false;

        // Acquire target image
        let (image_index, b_image_suboptimal) = unsafe {
            swapchain_entry.entry.acquire_next_image(
                swapchain.handle,
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
                let timeout_in_ns = 500000; // 500ms
                loop {
                    match device.wait_for_fences(&fences, true, timeout_in_ns) {
                        Ok(_) => return,
                        Err(_) => {
                            println!("Vulkan: Faild to wait {}, keep trying...", msg)
                        }
                    }
                }
            };

            wait_fence(self.present_finished_fence, "present_finished");
            wait_fence(
                self.command_buffer_finished_fence,
                "command_buffer_finished",
            );

            // Reset the fence
            device
                .reset_fences(&[
                    self.present_finished_fence,
                    self.command_buffer_finished_fence,
                ])
                .unwrap();
        }

        // Reuse the command buffer
        // TODO check or wait for fence
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Vulkan: Reset command buffer failed???");
        };

        // Update GPU ViewParams const buffer
        {
            // From row major float4x4 to column major Mat4
            let view_proj = view_info.projection * view_info.view_transform;
            let view_params = ViewParams {
                view_proj: view_proj,
                inv_view_proj: view_proj.inverse(),
                view_pos: view_info.view_position,
                padding: 0f32,
            };

            unsafe {
                std::ptr::copy_nonoverlapping(
                    std::ptr::addr_of!(view_params),
                    scene.view_params_cb.data as *mut ViewParams,
                    1,
                );
            }
        }

        // SIMPLE CONFIGURATION
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
                device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
            }
        }

        let mut swapchain_image_layout = vk::ImageLayout::UNDEFINED;

        // Transition for render
        if !use_rg {
            let sub_res_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1);
            let texture = rd.res.get_texture(swapchain.image[image_index as usize]);
            let image_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(swapchain_image_layout)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(*sub_res_range)
                .image(texture.image);
            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
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
        if !use_rg {
            let texture_view = rd
                .res
                .get_texture_view(swapchain.image_view[image_index as usize]);
            let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(texture_view.image_view)
                .image_layout(swapchain_image_layout)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color });
            let color_attachments = [*color_attachment];
            let depth_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(self.depth_stencil_buffer_view.image_view)
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
                device.cmd_begin_rendering(command_buffer, &rendering_info);

                device.cmd_set_depth_test_enable(command_buffer, true);
                device.cmd_set_depth_write_enable(command_buffer, true);
                device.cmd_set_stencil_test_enable(command_buffer, true);
                device.cmd_set_stencil_op(
                    command_buffer,
                    vk::StencilFaceFlags::FRONT_AND_BACK,
                    vk::StencilOp::KEEP,
                    vk::StencilOp::REPLACE,
                    vk::StencilOp::KEEP,
                    vk::CompareOp::ALWAYS,
                );
                device.cmd_set_stencil_write_mask(
                    command_buffer,
                    vk::StencilFaceFlags::FRONT_AND_BACK,
                    0x01,
                );
                device.cmd_set_stencil_reference(
                    command_buffer,
                    vk::StencilFaceFlags::FRONT_AND_BACK,
                    0x01,
                );
            }
        }

        // Draw mesh
        let mut hack = HackStuff {
            bindless_size: 1024,
            set_layout_override: std::collections::HashMap::new(),
        };
        hack.set_layout_override
            .insert(SCENE_DESCRIPTOR_SET_INDEX, scene.descriptor_set_layout);
        let mesh_gfx_pipeline = shaders.create_gfx_pipeline(
            &ShaderDefinition::new("MeshVSPS.hlsl", "vs_main", ShaderStage::Vert),
            &ShaderDefinition::new("MeshVSPS.hlsl", "ps_main", ShaderStage::Frag),
            &hack,
        );

        if !use_rg {
            if let Some(create_pipeline) = mesh_gfx_pipeline {
                let pipeline = shaders.get_pipeline(create_pipeline).unwrap();
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
                        device.cmd_set_viewport(command_buffer, 0, &viewports);
                    }

                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain.extent,
                    };
                    let scissors = [scissor];
                    unsafe {
                        device.cmd_set_scissor(command_buffer, 0, &scissors);
                    }
                }

                // Bind scene resources
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        SCENE_DESCRIPTOR_SET_INDEX,
                        &[scene.descriptor_set],
                        &[],
                    );
                }

                // Bind shader resources
                /*
                DescriptorSetUpdater::new(pipeline)
                    .buffer("vertex_buffer", &scene.vertex_buffer.buffer.srv.unwrap())
                    .constant_buffer("view_params", &self.view_params_cb.buffer)
                    .image("material_texture", &scene.material_texture.view.image_view)
                    .sampler("material_texture_sampler", &self.shared_sampler)
                    .update(&device);
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &pipeline.descriptor_sets,
                        &[],
                    );
                }
                */

                // Bind index buffer
                unsafe {
                    device.cmd_bind_index_buffer(
                        command_buffer,
                        scene.index_buffer.buffer.buffer,
                        0,
                        vk::IndexType::UINT16,
                    );
                }

                // Set pipeline and Draw
                unsafe {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle,
                    );
                }
                self.draw_geometry(device, command_buffer, &pipeline, scene);
            }

            // End render pass
            unsafe { device.cmd_end_rendering(command_buffer) }
        }

        let mut main_depth_stencil = RGHandle::<Texture>::null();
        if use_rg {
            let pipeline = mesh_gfx_pipeline.unwrap();

            let color = rg.register_texture_view(swapchain.image_view[image_index as usize]);
            main_depth_stencil = rg.create_texutre(TextureDesc::new_2d(
                swapchain.extent.width,
                swapchain.extent.height,
                vk::Format::D24_UNORM_S8_UINT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ));
            let ds = rg.create_texture_view(
                main_depth_stencil,
                TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::D24_UNORM_S8_UINT,
                    aspect: vk::ImageAspectFlags::DEPTH,
                },
            );

            let mut pass = RenderPass::new("Base");
            &mut pass
                //rg.new_pass("Base Pass")
                .color_targets(&[color])
                .depth_stencil(ds)
                .logic(move |cb, shaders, pass| {
                    let pipeline = shaders.get_pipeline(pipeline).unwrap();

                    // set up raster state
                    cb.set_viewport_0(vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: swapchain.extent.width as f32,
                        height: swapchain.extent.height as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    });
                    cb.set_scissor_0(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain.extent,
                    });
                    cb.set_depth_test_enable(true);
                    cb.set_depth_write_enable(true);
                    cb.set_stencil_test_enable(true);
                    let face_mask = vk::StencilFaceFlags::FRONT_AND_BACK;
                    let stencil_ops = StencilOps::write_on_pass(vk::CompareOp::ALWAYS);
                    cb.set_stencil_op(face_mask, stencil_ops);
                    cb.set_stencil_write_mask(face_mask, 0x01);
                    cb.set_stencil_reference(face_mask, 0xFF);

                    // Bind scene resources
                    cb.bind_descriptor_set(
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        SCENE_DESCRIPTOR_SET_INDEX,
                        scene.descriptor_set,
                        None,
                    );

                    // Draw meshs
                    cb.bind_index_buffer(
                        scene.index_buffer.buffer.buffer,
                        0,
                        vk::IndexType::UINT16,
                    );
                    cb.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
                    for mesh_params in &scene.mesh_params {
                        let material_params =
                            &scene.material_parmas[mesh_params.material_index as usize];
                        // PushConstant for everything per-draw
                        let mut constants = PushConstantsBuilder::new(pipeline);
                        {
                            // model_transform
                            let model_xform = glam::Mat4::IDENTITY; // Not used for now
                            constants.push(&model_xform.to_cols_array());
                            // vertex attributes
                            constants.push(&mesh_params.positions_offset);
                            constants.push(&mesh_params.texcoords_offset);
                            constants.push(&mesh_params.normals_offset);
                            constants.push(&mesh_params.tangents_offset);
                            // material params
                            constants.push(&material_params.base_color_index);
                            constants.push(&material_params.normal_index);
                            constants.push(&material_params.metallic_roughness_index);
                        }
                        cb.push_constants(
                            pipeline.layout,
                            pipeline.push_constant_ranges[0].stage_flags, // TODO
                            0,
                            &constants.build(),
                        );

                        cb.draw_indexed(mesh_params.index_count, 1, mesh_params.index_offset, 0, 0);
                    }
                });
            rg.add_pass(pass);
        }

        // Draw something with compute
        let mesh_cs_pipeline = shaders.create_compute_pipeline(&ShaderDefinition::new(
            "MeshCS.hlsl",
            "main",
            ShaderStage::Compute,
        ));
        if let Some(created_pipeline) = mesh_cs_pipeline {
            /*
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
                        command_buffer,
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
                        command_buffer,
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
                        command_buffer,
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
                    device.cmd_dispatch(command_buffer, dispatch_x, dispatch_y, 1);
                }
            }
            */
        }

        // Draw (procedure) Sky
        let sky_pipeline = shaders.create_gfx_pipeline(
            &ShaderDefinition::new("sky_vsps.hlsl", "vs_main", ShaderStage::Vert),
            &ShaderDefinition::new("sky_vsps.hlsl", "ps_main", ShaderStage::Frag),
            &hack,
        );
        if !use_rg {
            if let Some(created_pipeline) = sky_pipeline {
                let pipeline = shaders.get_pipeline(created_pipeline).unwrap();
                // Begin pass
                let image_view = rd
                    .res
                    .get_texture_view(swapchain.image_view[image_index as usize]);
                let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
                    .image_view(image_view.image_view)
                    .image_layout(swapchain_image_layout)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE);
                let color_attachments = [*color_attachment];
                let stencil_attachment = vk::RenderingAttachmentInfoKHR::builder()
                    .image_view(self.depth_stencil_buffer_view.image_view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::NONE);
                let rendering_info = vk::RenderingInfoKHR::builder()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: surface_size,
                    })
                    .layer_count(1)
                    .view_mask(0)
                    .color_attachments(&color_attachments)
                    .stencil_attachment(&stencil_attachment);
                unsafe {
                    device.cmd_begin_rendering(command_buffer, &rendering_info);

                    device.cmd_set_depth_test_enable(command_buffer, false);
                    device.cmd_set_depth_write_enable(command_buffer, false);
                    device.cmd_set_stencil_test_enable(command_buffer, true);
                    device.cmd_set_stencil_op(
                        command_buffer,
                        vk::StencilFaceFlags::FRONT_AND_BACK,
                        vk::StencilOp::KEEP,
                        vk::StencilOp::KEEP,
                        vk::StencilOp::KEEP,
                        vk::CompareOp::EQUAL,
                    );
                    device.cmd_set_stencil_compare_mask(
                        command_buffer,
                        vk::StencilFaceFlags::FRONT_AND_BACK,
                        0x01,
                    );
                    device.cmd_set_stencil_reference(
                        command_buffer,
                        vk::StencilFaceFlags::FRONT_AND_BACK,
                        0x00,
                    );
                }

                // Bind scene resources
                // TODO Need to bind everying render pass ?
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        SCENE_DESCRIPTOR_SET_INDEX,
                        &[scene.descriptor_set],
                        &[],
                    );
                }

                // Draw
                unsafe {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle,
                    );
                    device.cmd_draw(command_buffer, 3, 1, 0, 0);
                }

                // End pass
                unsafe { device.cmd_end_rendering(command_buffer) }
            };
        }

        // Transition for present
        if !use_rg {
            if swapchain_image_layout != vk::ImageLayout::PRESENT_SRC_KHR {
                let sub_res_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1);
                let texture = rd.res.get_texture(swapchain.image[image_index as usize]);
                let image_barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(swapchain_image_layout)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .subresource_range(*sub_res_range)
                    .image(texture.image);
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
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
        }

        if use_rg {
            let pipeline = sky_pipeline.unwrap();

            let color_target = rg.register_texture_view(swapchain.image_view[image_index as usize]);
            let stencil = rg.create_texture_view(
                main_depth_stencil,
                TextureViewDesc {
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: vk::Format::D24_UNORM_S8_UINT,
                    aspect: vk::ImageAspectFlags::STENCIL,
                },
            );

            let mut pass = RenderPass::new("Sky");
            &mut pass
                //rg.new_pass("Sky")
                .color_targets(&[color_target])
                .depth_stencil(stencil)
                .logic(move |cb, shaders, pass| {
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
            rg.add_pass(pass);
        }

        if use_rg {
            rg.execute();
        }

        // End command recoding
        unsafe {
            device.end_command_buffer(command_buffer).unwrap();
        }

        // Submit
        {
            let command_buffers = [command_buffer];
            let signal_semaphores = [self.present_ready_semaphore];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            unsafe {
                device
                    .queue_submit(
                        *gfx_queue,
                        &[*submit_info],
                        self.command_buffer_finished_fence,
                    )
                    .unwrap();
            }
        }

        // Present
        {
            let mut present_info = vk::PresentInfoKHR::default();
            present_info.wait_semaphore_count = 1;
            present_info.p_wait_semaphores = &self.present_ready_semaphore;
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
        //unsafe { device.device_wait_idle().unwrap(); }
    }

    fn draw_geometry(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        pipeline: &Pipeline,
        render_scene: &RenderScene,
    ) {
        for mesh_params in &render_scene.mesh_params {
            let material_params =
                &render_scene.material_parmas[mesh_params.material_index as usize];
            // PushConstant for everything per-draw
            let mut constants = PushConstantsBuilder::new(pipeline);
            {
                // model_transform
                let model_xform = glam::Mat4::IDENTITY; // Not used for now
                constants.push(&model_xform.to_cols_array());
                // vertex attributes
                constants.push(&mesh_params.positions_offset);
                constants.push(&mesh_params.texcoords_offset);
                constants.push(&mesh_params.normals_offset);
                constants.push(&mesh_params.tangents_offset);
                // material params
                constants.push(&material_params.base_color_index);
                constants.push(&material_params.normal_index);
                constants.push(&material_params.metallic_roughness_index);
            }
            unsafe {
                device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    pipeline.push_constant_ranges[0].stage_flags, // TODO finer flags
                    0,
                    &constants.build(),
                );

                device.cmd_draw_indexed(
                    command_buffer,
                    mesh_params.index_count,
                    1,
                    mesh_params.index_offset,
                    0,
                    0,
                );
            }
        }
    }
}
