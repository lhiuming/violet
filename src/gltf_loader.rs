use crate::render_device::Buffer;
use gltf;

pub struct Material {}

pub struct Primitive {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
}

pub struct GLTF {
    pub meshes: Vec<Mesh>,
}

// Load a GLTF file as a bunch of meshes, materials, etc.
pub fn load_gltf(path: &String, index_buffer: &Buffer, vertex_buffer: &Buffer) -> Option<GLTF> {
    // TODO should be passed in if mutiple glTF
    let mut index_buffer_offet = 0u32;
    let mut vertex_buffer_offset = 0u32;

    // Read the document (structure) and blob data (buffers, imanges)
    // NOTE: gltf::Gltf::open only load the document
    let path = std::path::Path::new(&path);
    let (document, buffers, images) = gltf::import(path).ok()?;

    let mut meshes = Vec::<Mesh>::new();

    // load meshes
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

    Some(GLTF { meshes })
}
