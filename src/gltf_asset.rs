use std::{array, mem::size_of};

use crate::{render_loop::AllocBuffer};
extern crate gltf as gltf_rs;

extern crate glam;
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

// Load a GLTF file as a bunch of meshes, materials, etc.
pub fn load(path: &String, index_buffer: &mut AllocBuffer, vertex_buffer: &mut AllocBuffer) -> Option<GLTF> {
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
            "Image {} {}, {:?}: {:?}",
            image.width,
            image.height,
            image.format,
            image.pixels.split_at(8).0
        );
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

