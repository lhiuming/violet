use std::{array, mem::size_of};

use crate::{render_loop::{AllocBuffer, AllocTexture2D}, render_device::RenderDevice};
extern crate gltf as gltf_rs;

extern crate glam;
use ash::vk::{self, BufferUsageFlags};
use glam::{Mat4, Quat, Vec3};

pub struct Material {}

pub struct Primitive {
    pub index_offset: u32,
    pub index_count: u32,

    pub vertex_count: u32,
    pub positions_offset: u32,
    pub texcoords_offsets: [u32; 8],
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

pub struct Node {
    pub transform: Mat4,
    pub mesh_index: Option<u32>,
    pub material: Option<Material>,
}

pub struct Scene {
    pub nodes: Vec<Node>,
}

pub struct GLTF {
    pub scenes: Vec<Scene>,
    pub meshes: Vec<Mesh>,
}

pub struct UploadContext
{
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
}

impl UploadContext
{
    pub fn new(rd: &RenderDevice) -> Self {
        let command_pool = rd.create_command_pool();
        let command_buffer = rd.create_command_buffer(command_pool);
        Self {
            command_pool,
            command_buffer,
        }
    }

    pub fn immediate_submit<F>(&self, rd: &RenderDevice, f: F) where F: FnOnce(vk::CommandBuffer) {
        // TOOD we will need fence for multiple immediate submit or reusing command buffer
        // TODO we will need semaphore to sync with rendering which will depends on operation here

        let device = &rd.device;
        let command_buffer = self.command_buffer;

        // TODO check finished?

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info) }.unwrap();

        f(command_buffer);

        unsafe { device.end_command_buffer(command_buffer) }.unwrap();

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        unsafe {
            device.queue_submit(rd.gfx_queue, &[*submit_info], vk::Fence::null()).unwrap();
        }
    }

}

// Load a GLTF file as a bunch of meshes, materials, etc.
pub fn load(path: &String, rd: &RenderDevice, upload_context: &UploadContext, index_buffer: &mut AllocBuffer, vertex_buffer: &mut AllocBuffer, material_texture: &mut AllocTexture2D) -> Option<GLTF> {
    // Read the document (structure) and blob data (buffers, imanges)
    // NOTE: gltf::Gltf::open only load the document
    let path = std::path::Path::new(&path);
    let (document, buffers, images) = gltf::import(path).ok()?;

    let mut meshes = Vec::<Mesh>::new();

    // pre-load meshes
    for mesh in document.meshes() {
        let mut primitives = Vec::<Primitive>::new();

        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            // Load indices
            let index_offset;
            let index_count;
            if let Some(indices) = reader.read_indices() {
                //println!("{:?}", indices);
                match indices {
                    gltf::mesh::util::ReadIndices::U8(_) => todo!("Read u8 indices"),
                    gltf::mesh::util::ReadIndices::U16(iter) => {
                        index_count = iter.len() as u32;
                        let (ib_u16, pos) = index_buffer.alloc::<u16>(index_count);
                        for (i, ind) in iter.enumerate() {
                            ib_u16[i] = ind;
                        }
                        index_offset = pos / size_of::<u16>() as u32;
                    }
                    gltf::mesh::util::ReadIndices::U32(_) => todo!(),
                }
            } else {
                index_offset = u32::MAX;
                index_count = 0;
            }

            // Load positions 
            let positions_offset;
            let vertex_count;
            if let Some(iter) = reader.read_positions() {
                vertex_count = iter.len() as u32;
                let (vb_f32, pos) = vertex_buffer.alloc::<f32>(vertex_count * 3);
                let mut write_offset = 0;
                for vert_pos in iter {
                    vb_f32[write_offset + 0] = vert_pos[0];
                    vb_f32[write_offset + 1] = vert_pos[1];
                    vb_f32[write_offset + 2] = vert_pos[2];
                    write_offset += 3;
                }
                positions_offset = pos / size_of::<f32>() as u32;
            } else {
                vertex_count = 0;
                positions_offset = u32::MAX;
            }

            // Load texcoords
            let texcoords_offsets: [u32;8] = array::from_fn(|set_index| {
                if let Some(read_texcoord) = reader.read_tex_coords(set_index as u32) {
                    match read_texcoord {
                        gltf_rs::mesh::util::ReadTexCoords::U8(_) => todo!("Read u8 texcoord"),
                        gltf_rs::mesh::util::ReadTexCoords::U16(_) => todo!("Read u16 texcoord"),
                        gltf_rs::mesh::util::ReadTexCoords::F32(texcoord_data) => {
                            let f32_count = texcoord_data.len() as u32 * 2;
                            let (vb_f32, pos) = vertex_buffer.alloc::<f32>(f32_count);
                            let mut write_offset = 0;
                            for texcoord in texcoord_data {
                                vb_f32[write_offset + 0] = texcoord[0];
                                vb_f32[write_offset + 1] = texcoord[1];
                                write_offset += 2;
                            }
                            assert_eq!(f32_count / 2, vertex_count);
                            pos / size_of::<f32>() as u32
                        },
                    }
                } else {
                    0
                }
            });

            if reader.read_colors(0).is_some() {
                println!("GLTF Loader: Colors are ignored")
            }
            if reader.read_joints(0).is_some() {
                println!("GLTF Loader: Joints are ignored")
            }
            if reader.read_morph_targets().count() != 0 {
                println!("GLTF Loader: morph targets are ignored")
            }
            if reader.read_normals().is_some() {
                println!("GLTF Loader: normals are ignored")
            }
            if reader.read_tangents().is_some() {
                println!("GLTF Loader: tangents are ignored")
            }
            if reader.read_weights(0).is_some() {
                println!("GLTF Loader: weights are ignored")
            }

            primitives.push(Primitive {
                index_offset,
                index_count,
                vertex_count,
                positions_offset,
                texcoords_offsets,
            });
        } // end for primitives

        meshes.push(Mesh { primitives })
    } // end for meshes

    // pre-load textures
    for image in images {
        println!(
            "Loading Image {} {}, {:?}: {:?}",
            image.width,
            image.height,
            image.format,
            image.pixels.split_at(8).0
        );

        if (image.format == gltf_rs::image::Format::R8G8B8)
        && (material_texture.texture.srv_format == vk::Format::R8G8B8A8_SRGB) {

            let texel_count = image.width * image.height;

            // Create staging buffer
            let staging_buffer = rd.create_buffer(texel_count as u64 * 4, BufferUsageFlags::TRANSFER_SRC, vk::Format::UNDEFINED).unwrap();

            // Read to staging buffer
            let staging_slice = unsafe { std::slice::from_raw_parts_mut(staging_buffer.data as *mut u8, texel_count as usize * 4) };
            for y in 0..image.height {
                for x in 0..image.width {
                    let offset = (y * image.width + x) as usize;
                    let (r, g, b) = (
                        image.pixels[offset * 3 + 0],
                        image.pixels[offset * 3 + 1],
                        image.pixels[offset * 3 + 2],
                    );
                    staging_slice[offset * 4 + 0] = r;
                    staging_slice[offset * 4 + 1] = g;
                    staging_slice[offset * 4 + 2] = b;
                    staging_slice[offset * 4 + 3] = 255;
                }
            }

            // Transfer to material texture
            let offset = material_texture.alloc(image.width, image.height);
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
                        &[*barrier]);
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
                .image_offset(vk::Offset3D { x: offset.0 as i32, y: offset.1 as i32, z: 0 })
                .image_extent(vk::Extent3D { width: image.width, height: image.height, depth: 1, });
                let regions = [*region];
                unsafe { rd.device.cmd_copy_buffer_to_image(command_buffer, staging_buffer.buffer, material_texture.texture.image, dst_image_layout, &regions) }

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
                        &[*barrier]);
                }
            });

        } else {
            println!("Warning: GLTF Loader: Unsupported texture format: {:?} to {:?}", image.format, material_texture.texture.srv_format);
        }
    }

    // Load nodes in the scenes
    let mut scenes = Vec::<Scene>::new();
    for scene in document.scenes().by_ref() {
        let mut nodes = Vec::<Node>::new();
        for root in scene.nodes().by_ref() {
            // Create a statck to fake recursion
            let mut stack = Vec::<(gltf_rs::Node, Mat4)>::new();
            stack.push((root, Mat4::IDENTITY));

            // Load nodes recursively
            while let Some((node, parent_transform)) = stack.pop() {
                // Get flat transform
                let local_transform;
                match node.transform() {
                    gltf::scene::Transform::Matrix { matrix } => {
                        println!("Transform Matrix{:?}", matrix);
                        local_transform = Mat4::from_cols_array_2d(&matrix);
                    }
                    gltf::scene::Transform::Decomposed {
                        translation,
                        rotation,
                        scale,
                    } => {
                        println!(
                            "Transform TRS: {:?}, {:?}, {:?}",
                            translation, rotation, scale
                        );
                        local_transform = Mat4::from_scale_rotation_translation(
                            Vec3::from_array(scale),
                            Quat::from_array(rotation),
                            Vec3::from_array(translation),
                        );
                    }
                }
                let transform = parent_transform * local_transform;
                // Add the node to output
                nodes.push(Node {
                    transform,
                    mesh_index: node.mesh().map(|mesh| mesh.index() as u32),
                    material: None,
                });
                // Recursively load children
                for child in node.children().by_ref() {
                    stack.push((child, transform));
                }
            }
        }
        scenes.push(Scene { nodes });
    }

    Some(GLTF { scenes, meshes })
}

