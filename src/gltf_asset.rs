use crate::render_device::Buffer;
extern crate gltf as gltf_rs;

extern crate glam;
use glam::{Mat4, Quat, Vec3};

pub struct Material {}

pub struct Primitive {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
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
pub fn load(path: &String, index_buffer: &Buffer, vertex_buffer: &Buffer) -> Option<GLTF> {
    // TODO should be passed in if mutiple glTF
    let mut index_buffer_offet = 0u32;
    let mut vertex_buffer_offset = 0u32;

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
            let mut index_count = 0;
            let mut index_offset = index_buffer_offet;
            if let Some(indices) = reader.read_indices() {
                // Fill GPU buffer
                let mut write_offset = index_offset;
                //println!("{:?}", indices);
                match indices {
                    gltf::mesh::util::ReadIndices::U8(_) => todo!(),
                    gltf::mesh::util::ReadIndices::U16(iter) => {
                        let ib_u16 = unsafe {
                            std::slice::from_raw_parts_mut(
                                index_buffer.data as *mut u16,
                                index_buffer.size as usize / 2, // 2 bytes per index
                            )
                        };
                        for ind in iter {
                            ib_u16[write_offset as usize] = ind;
                            write_offset += 1;
                        }
                    }
                    gltf::mesh::util::ReadIndices::U32(_) => todo!(),
                }

                // Allcoate
                index_count = write_offset - index_buffer_offet;
                index_buffer_offet = write_offset;
            }

            // Load vertices
            let mut vertex_count = 0;
            let mut vertex_offset = vertex_buffer_offset;
            if let Some(iter) = reader.read_positions() {
                // Fill GPU buffer
                let mut write_offset = vertex_offset as usize;
                let vb_f32 = unsafe {
                    std::slice::from_raw_parts_mut(
                        vertex_buffer.data as *mut f32,
                        vertex_buffer.size as usize / 4, // 4 bytes per f32
                    )
                };
                for vert_pos in iter {
                    //println!("{:?}", vert_pos);
                    vb_f32[write_offset + 0] = vert_pos[0];
                    vb_f32[write_offset + 1] = vert_pos[1];
                    vb_f32[write_offset + 2] = vert_pos[2];
                    write_offset += 3;
                }

                // Allocate
                vertex_count = write_offset as u32 - vertex_offset;
                vertex_buffer_offset = write_offset as u32;
            }

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
            if reader.read_tex_coords(0).is_some() {
                println!("GLTF Loader: tex coords are ignored")
            }
            if reader.read_weights(0).is_some() {
                println!("GLTF Loader: weights are ignored")
            }

            primitives.push(Primitive {
                vertex_offset,
                index_offset,
                index_count,
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
