use std::mem::{self, size_of};

use ash::vk::{self};
use glam::{Mat4, UVec2, Vec2};

use crate::model::Model;
use crate::render_device::{
    Buffer, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
};
use crate::shader::{Pipeline, ShaderDefinition, ShaderStage, Shaders};

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
    pub size: UVec2,
    pub offset: UVec2,
    pub layer: u32,
    pub uv_scale: Vec2,
    pub uv_offset: Vec2,
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

struct PushConstantsBuilder {
    data: Vec<u8>,
}

impl PushConstantsBuilder {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push<T>(&mut self, value: &T) -> &mut Self {
        let size = std::mem::size_of::<T>();
        let offset = self.data.len();
        self.data.resize(offset + size, 0);
        self.data[offset..offset + size].copy_from_slice(unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, size)
        });
        self
    }

    pub fn build(&self) -> &[u8] {
        &self.data
    }
}

// Contain everything to be rendered
pub struct RenderScene {
    pub upload_context: UploadContext,

    // Global buffer to store all loaded meshes
    pub vertex_buffer: AllocBuffer,
    pub index_buffer: AllocBuffer,

    // Global texture to store all loaded textures
    pub material_texture: AllocTexture2D,

    // Stuff to be rendered
    pub texture_params: Vec<TextureParams>,
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

        RenderScene {
            upload_context: UploadContext::new(rd),
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            material_texture: material_texture,
            texture_params: Vec::new(),
            material_parmas: Vec::new(),
            mesh_params: Vec::new(),
        }
    }

    pub fn add(&mut self, rd: &RenderDevice, model: &Model) {
        let upload_context = &mut self.upload_context;
        let material_texture = &mut self.material_texture;
        let index_buffer = &mut self.index_buffer;
        let vertex_buffer = &mut self.vertex_buffer;

        let texture_index_offset = self.texture_params.len() as u32;
        for image in &model.images {
            let texel_count = image.width * image.height;

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

            // Transfer to material texture
            let (dst_x, dst_y, dst_layer) =
                material_texture.alloc(image.width, image.height).unwrap();
            println!(
                "/t Writing to material texture: x {}, y {}, layer {}",
                dst_x, dst_y, dst_layer
            );
            upload_context.immediate_submit(rd, |command_buffer| {
                // Transfer to proper image layout
                unsafe {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .image(material_texture.texture.image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: dst_layer,
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
                        base_array_layer: dst_layer,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D {
                        x: dst_x as i32,
                        y: dst_y as i32,
                        z: 0,
                    })
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
                        material_texture.texture.image,
                        dst_image_layout,
                        &regions,
                    )
                }

                // Transfer to shader ready layout
                unsafe {
                    let barrier = vk::ImageMemoryBarrier::builder()
                        .image(material_texture.texture.image)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: dst_layer,
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

            let size = UVec2::new(image.width, image.height);
            let offset = UVec2 { x: dst_x, y: dst_y };
            self.texture_params.push(TextureParams {
                size,
                offset,
                layer: dst_layer,
                uv_scale: size.as_vec2() / material_texture.size().as_vec2(),
                uv_offset: offset.as_vec2() / material_texture.size().as_vec2(),
            });
        }

        let material_index_offset = self.material_parmas.len() as u32;
        for material in &model.materials {
            // TODO default textures
            let resolve = |map: &Option<crate::model::MaterialMap>| match map {
                Some(map) => texture_index_offset + map.image_index,
                None => 0,
            };

            self.material_parmas.push(MaterialParams {
                base_color_index: resolve(&material.base_color_map),
                metallic_roughness_index: resolve(&material.metallic_roughness_map),
                normal_index: resolve(&material.normal_map),
            });
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
}

pub struct RednerLoop {
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,

    pub present_finished_fence: vk::Fence,
    pub present_finished_semephore: vk::Semaphore,
    pub present_ready_semaphore: vk::Semaphore,
    pub command_buffer_finished_fence: vk::Fence,

    pub shared_sampler: vk::Sampler,

    pub view_params_cb: Buffer,
    pub depth_buffer: Texture,
    pub depth_buffer_view: TextureView,
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

        // View parameter constant buffer
        let view_params_cb = rd
            .create_buffer(
                mem::size_of::<ViewParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::Format::UNDEFINED,
            )
            .unwrap();

        // Create depth buffer
        let depth_buffer = rd
            .create_texture(TextureDesc::new_2d(
                surface_size.width,
                surface_size.height,
                vk::Format::D16_UNORM,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ))
            .unwrap();
        let depth_buffer_view = rd
            .create_texture_view(&depth_buffer, TextureViewDesc::default(&depth_buffer))
            .unwrap();

        // Shared samplers
        let shared_sampler = unsafe {
            let create_info = vk::SamplerCreateInfo::builder();
            rd.device.create_sampler(&create_info, None).unwrap()
        };

        RednerLoop {
            command_pool,
            command_buffer,
            present_finished_fence: rd.create_fence(false),
            present_finished_semephore: rd.create_semaphore(),
            present_ready_semaphore: rd.create_semaphore(),
            command_buffer_finished_fence: rd.create_fence(true),
            shared_sampler,
            view_params_cb,
            depth_buffer,
            depth_buffer_view,
        }
    }

    pub fn render(
        &self,
        rd: &RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_proj: Mat4,
    ) {
        let device = &rd.device;
        let physical_device = &rd.physical_device;
        let surface_entry = &rd.surface_entry;
        let swapchain_entry = &rd.swapchain_entry;
        let dynamic_rendering_entry = &rd.dynamic_rendering_entry;
        let gfx_queue = &rd.gfx_queue;
        let command_buffer = self.command_buffer;
        let surface = &rd.surface;
        let swapchain = &rd.swapchain;

        let surface_size = surface.query_size(&surface_entry, physical_device);

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
            let view_params = ViewParams { view_proj };

            unsafe {
                std::ptr::copy_nonoverlapping(
                    std::ptr::addr_of!(view_params),
                    self.view_params_cb.data as *mut ViewParams,
                    mem::size_of::<ViewParams>(),
                );
            }
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
                device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .unwrap();
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
        if rd.b_support_dynamic_rendering {
            let color_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(swapchain.image_view[image_index as usize])
                .image_layout(swapchain_image_layout)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: clear_color });
            let color_attachments = [*color_attachment];
            let depth_attachment = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(self.depth_buffer_view.image_view)
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
                dynamic_rendering_entry.cmd_begin_rendering(command_buffer, &rendering_info);
            }
        }

        // Draw mesh
        let mesh_gfx_pipeline = shaders.get_gfx_pipeline(
            &ShaderDefinition::new("MeshVSPS.hlsl", "vs_main", ShaderStage::Vert),
            &ShaderDefinition::new("MeshVSPS.hlsl", "ps_main", ShaderStage::Frag),
        );
        if let Some(pipeline) = mesh_gfx_pipeline {
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

                let image_info = vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(scene.material_texture.view.image_view);
                let image_infos = [*image_info];

                let write_image = vk::WriteDescriptorSet::builder()
                    .dst_set(pipeline.descriptor_sets[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_infos);

                let sampler_info = vk::DescriptorImageInfo::builder().sampler(self.shared_sampler);
                let sampler_infos = [*sampler_info];
                let write_sampler = vk::WriteDescriptorSet::builder()
                    .dst_set(pipeline.descriptor_sets[0])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_infos);

                let writes = [*write, *write_cb, *write_image, *write_sampler];
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }

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
            }

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
        unsafe {
            dynamic_rendering_entry.cmd_end_rendering(command_buffer);
        }

        // Draw something with compute
        let mesh_cs_pipeline = shaders.get_compute_pipeline(&ShaderDefinition::new(
            "MeshCS.hlsl",
            "main",
            ShaderStage::Compute,
        ));
        if let Some(pipeline) = mesh_cs_pipeline {
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
            let mut constants = PushConstantsBuilder::new();
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
                let encode_tex = |tex_index| {
                    let tex_params = &render_scene.texture_params[tex_index as usize];
                    let scale_offset = pack_unorm(
                        tex_params.uv_scale.x,
                        tex_params.uv_scale.y,
                        tex_params.uv_offset.x,
                        tex_params.uv_offset.y,
                    );
                    UVec2 {
                        x: scale_offset,
                        y: tex_params.layer,
                    }
                };
                constants.push(&encode_tex(material_params.base_color_index));
                constants.push(&encode_tex(material_params.normal_index));
                constants.push(&encode_tex(material_params.metallic_roughness_index));
            }
            unsafe {
                device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, // TODO use reflection to automate this?
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
