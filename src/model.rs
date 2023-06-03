use glam::{Mat4, Vec3, Vec4};
use rkyv::{ser::Serializer, Archive, Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Archive, Deserialize, Serialize)]
pub struct Model {
    pub meshes: Vec<TriangleMesh>, // aggregated by material index
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
}

// TODO geometry compression/32bit-packing
#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub indicies: Vec<u16>,
    pub normals: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
}

impl TriangleMesh {
    fn empty() -> TriangleMesh {
        TriangleMesh {
            positions: Vec::new(),
            indicies: Vec::new(),
            normals: Vec::new(),
            texcoords: Vec::new(),
            tangents: Vec::new(),
        }
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct Material {
    pub base_color_map: Option<MaterialMap>,
    pub metallic_roughness_map: Option<MaterialMap>,
    pub normal_map: Option<MaterialMap>,
    pub metaliic_factor: f32,
    pub roughness_factor: f32,
}

#[derive(Archive, Deserialize, Serialize)]
pub struct MaterialMap {
    pub image_index: u32,
}

// TODO image compression
#[derive(Archive, Deserialize, Serialize)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    NotExist(String),
    General(String),
}

fn iterate_on_nodes<F>(node: &gltf::Node, xform: Mat4, f: &mut F)
where
    F: FnMut(&gltf::Node, Mat4) -> Mat4,
{
    let xform = f(node, xform);
    for child in node.children() {
        iterate_on_nodes(&child, xform, f);
    }
}

struct MikktspaceGeometry<'a> {
    pub num_triangle: usize,
    pub indices: &'a Vec<u16>,
    pub positions: &'a Vec<[f32; 3]>,
    pub normals: &'a Vec<[f32; 3]>,
    pub texcoords: &'a Vec<[f32; 2]>,
    pub debug_tangent_counts: Vec<u32>,
    pub out_tangents: &'a mut Vec<[f32; 4]>,
}

impl mikktspace::Geometry for MikktspaceGeometry<'_> {
    fn num_faces(&self) -> usize {
        self.num_triangle
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.texcoords[self.indices[face * 3 + vert] as usize]
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = self.indices[face * 3 + vert] as usize;

        // Debug log
        self.debug_tangent_counts[index] += 1;
        if self.debug_tangent_counts[index] > 1 {
            let a = Vec4::from_array(self.out_tangents[index]);
            let b = Vec4::from_array(tangent);
            if !Vec4::abs_diff_eq(a, b, 0.0000001) {
                println!(
                    "Writing to same tangent index with different value ({} times): {} -> {}",
                    self.debug_tangent_counts[index], a, b
                )
            }
        }

        self.out_tangents[index] = tangent;
    }
}

// Load a model from cache, or
// if model is not in cache, load the original gltf file and store it in cache.
pub fn load(path: &Path) -> Result<Model> {
    // Try to load from caceh
    match try_load_from_cache(path) {
        Ok(model) => Ok(model),
        Err(err) => {
            match err {
                Error::NotExist(cache_path) => {
                    // Import if cache not exist
                    println!(
                        "Model cache ({}) not exist ... import the gltf.",
                        cache_path
                    );
                    import_gltf(path)
                }
                Error::General(msg) => Err(Error::General(msg)),
            }
        }
    }
}

pub fn to_cache_path(path: &Path) -> Result<PathBuf> {
    let asset_name = if path.is_absolute() {
        let cur_dir = match std::env::current_dir() {
            Ok(dir) => dir,
            Err(err) => {
                return Err(Error::General(err.to_string()));
            }
        };
        let asset_dir = cur_dir.join("asset");
        path.strip_prefix(&asset_dir)
            .expect("Faild to strip asset dir")
    } else {
        let asset_dir = Path::new("./assets");
        path.strip_prefix(&asset_dir).unwrap_or(path)
    };

    let mut cache_path = PathBuf::new();
    cache_path.push(".");
    cache_path.push("cache");
    cache_path.push(asset_name);
    cache_path.set_extension("cache");
    Ok(cache_path)
}

pub fn try_load_from_cache(path: &Path) -> Result<Model> {
    let cache_path = to_cache_path(path).expect(&format!(
        "Failed to get cache path for asset: {}",
        path.to_string_lossy()
    ));
    if cache_path.exists() {
        return Model::new_from_file(&cache_path);
    } else {
        return Err(Error::NotExist(cache_path.to_string_lossy().to_string()));
    }
}

impl Model {
    fn save_to_file(&self, path: &Path) {
        let mut serializer = rkyv::ser::serializers::AllocSerializer::<4096>::default();
        let size = serializer
            .serialize_value(self)
            .expect("Failed to serialize model");

        std::fs::create_dir_all(path.parent().unwrap()).expect(&format!(
            "Failed to create dirs for: {}",
            path.to_string_lossy()
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(path)
            .expect(&format!(
                "Failed to create file at: {}",
                path.to_string_lossy()
            ));

        file.write_all(serializer.into_serializer().into_inner().as_ref())
            .expect("Failed to write to file");

        println!("Saved model to cache: {}. size {}", path.display(), size);
    }

    fn new_from_file(path: &Path) -> Result<Model> {
        let mut file = File::open(path).expect("Failed to open file");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("Failed to read file");
        let model = unsafe { rkyv::from_bytes_unchecked::<Model>(&buf) }
            .expect("Failed to deserialize model from cache");

        println!("Loaded model from cache: {}", path.display());
        Ok(model)
    }
}

pub fn import_gltf(path: &Path) -> Result<Model> {
    let model = import_gltf_uncached(path);
    if let Ok(model) = &model {
        let cache_path = to_cache_path(path).expect(&format!(
            "Failed to get cached path for: {}",
            path.to_string_lossy()
        ));
        model.save_to_file(&cache_path);
    }
    model
}

pub fn import_gltf_uncached(path: &Path) -> Result<Model> {
    // Read the document (structure) and blob data (buffers, imanges)
    let (document, buffers, images) = match gltf::import(path) {
        Ok(ret) => ret,
        Err(_) => {
            return Err(Error::General(format!(
                "Failed to load glTF file: {}",
                path.display()
            )));
        }
    };
    println!("Loaded glTF file: {}", path.display());
    if document.extensions_required().count() > 0 {
        println!(
            "Required extensions: {}",
            document
                .extensions_required()
                .collect::<Vec<&str>>()
                .join(", ")
        );
    }
    if document.extensions_used().count() > 0 {
        println!(
            "Used extensions: {}",
            document.extensions_used().collect::<Vec<&str>>().join(", ")
        );
    }

    // Load (and flatten) meshes
    let mut model_meshes = document
        .materials()
        .map(|_| TriangleMesh::empty())
        .collect::<Vec<_>>();
    let mut mesh_instance_count = vec![0u32; document.meshes().count()];
    if let Some(scene) = document.default_scene() {
        let mut process_node = |node: &gltf::Node, xform: Mat4| -> Mat4 {
            // Get flatten transform
            let local_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
            let xform = xform * local_xform;

            if node.mesh().is_none() {
                return xform;
            }
            let mesh = node.mesh().unwrap();

            // count instance (for stat)
            mesh_instance_count[mesh.index()] += 1;

            // Load each primitive
            for primitive in mesh.primitives() {
                // Check
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    println!("Warning: Mesh primitive is not triangulated. Ignored.");
                    continue;
                }
                if primitive.material().index().is_none() {
                    println!("Warning: Mesh primitive has no material. Ignored.");
                    continue;
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // Load indicies (required)
                let mut model_indicies = Vec::<u16>::new();
                if let Some(indices) = reader.read_indices() {
                    match indices {
                        gltf::mesh::util::ReadIndices::U8(_) => todo!("Read u8 indices"),
                        gltf::mesh::util::ReadIndices::U16(iter) => {
                            model_indicies.append(&mut iter.collect());
                        }
                        gltf::mesh::util::ReadIndices::U32(_) => todo!(),
                    }
                    assert!(model_indicies.len() % 3 == 0);
                } else {
                    println!("Warning: Mesh primitive has no indices.");
                    continue;
                }

                // Load positions (required)
                let mut model_positions = Vec::<[f32; 3]>::new();
                if let Some(positions) = reader.read_positions() {
                    model_positions.append(&mut positions.collect());
                } else {
                    println!("Warning: Mesh primitive has no positions.");
                    continue;
                }

                let vertex_count = model_positions.len();

                // Load texcoords
                let mut model_texcoords = Vec::<[f32; 2]>::new();
                if let Some(read_texcoords) = reader.read_tex_coords(0) {
                    match read_texcoords {
                        gltf::mesh::util::ReadTexCoords::U8(_) => todo!(),
                        gltf::mesh::util::ReadTexCoords::U16(_) => todo!(),
                        gltf::mesh::util::ReadTexCoords::F32(data) => {
                            model_texcoords.append(&mut data.collect());
                        }
                    }
                    assert_eq!(model_texcoords.len(), vertex_count);
                }

                // Load normals
                let mut model_normals = Vec::<[f32; 3]>::new();
                if let Some(normals) = reader.read_normals() {
                    model_normals.append(&mut normals.collect());
                    assert_eq!(model_normals.len(), vertex_count);
                }

                // Load tangent
                let mut model_tangent = Vec::<[f32; 4]>::new();
                if let Some(tangents) = reader.read_tangents() {
                    model_tangent.append(&mut tangents.collect());
                    assert_eq!(model_tangent.len(), vertex_count);
                }

                // Generated tangents if proper
                if model_tangent.len() == 0 {
                    if model_texcoords.len() > 0 && model_normals.len() > 0 {
                        model_tangent.resize(vertex_count, [0.0; 4]);
                        let mut geometry = MikktspaceGeometry {
                            num_triangle: model_indicies.len() / 3,
                            indices: &model_indicies,
                            positions: &model_positions,
                            normals: &model_normals,
                            texcoords: &model_texcoords,
                            debug_tangent_counts: vec![0u32; model_positions.len()],
                            out_tangents: &mut model_tangent,
                        };
                        let ret = mikktspace::generate_tangents(&mut geometry);
                        if !ret {
                            println!("Warning: Failed to generate tangents.");
                            model_tangent.clear();
                        }
                    }
                }

                // Log
                if reader.read_colors(0).is_some() {
                    println!("Warning: Mesh primitive has vertex colors. Ignored.")
                }

                // Merge the mesh primitive to model mesh
                let model_mesh = &mut model_meshes[primitive.material().index().unwrap()];
                let indice_offset = model_mesh.positions.len() as u16;
                model_mesh.positions.append(
                    &mut model_positions
                        .into_iter()
                        .map(|x| xform.transform_point3(Vec3::from(x)).to_array())
                        .collect(),
                );
                model_mesh.indicies.append(
                    &mut model_indicies
                        .into_iter()
                        .map(|x| x + indice_offset)
                        .collect(),
                );
                model_mesh.texcoords.append(&mut model_texcoords);
                model_mesh.normals.append(
                    &mut model_normals
                        .into_iter()
                        .map(|v| xform.transform_vector3(Vec3::from(v)).to_array())
                        .collect(),
                );
                model_mesh.tangents.append(
                    &mut model_tangent
                        .into_iter()
                        .map(|v| {
                            xform
                                .transform_vector3(Vec4::from(v).truncate())
                                .extend(v[3])
                                .to_array()
                        })
                        .collect(),
                );
            }

            return xform;
        };

        // Rotate the model to match the coordinate system
        // glTF: Y up, right-handed
        // Here: Z up, right-handed
        // So we gonna rotate the model by 90 degrees around X axis
        let root_xform = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2);

        for root in scene.nodes() {
            iterate_on_nodes(&root, root_xform, &mut process_node);
        }
    } else {
        return Err(Error::General("No default scene found.".to_string()));
    }

    // Clean up meshes
    for (index, model_mesh) in model_meshes.iter_mut().enumerate() {
        let vertex_count = model_mesh.positions.len();
        if model_mesh.texcoords.len() != vertex_count {
            model_mesh.texcoords.clear();
            model_mesh.tangents.clear();
            println!("Warning: Mesh {} has incomplete texcoords.", index);
        }
        if model_mesh.normals.len() != vertex_count {
            model_mesh.normals.clear();
            model_mesh.tangents.clear();
            println!("Warning: Mesh {} has incomplete normals.", index);
        }
        if model_mesh.tangents.len() != vertex_count {
            model_mesh.tangents.clear();
            println!("Warning: Mesh {} has incomplete tangents.", index);
        }
    }

    // Log instance count
    for (index, count) in mesh_instance_count.iter().enumerate() {
        if *count > 1 {
            println!("Mesh {} has {} instances.", index, count);
        }
    }

    // Load images
    let mut model_images = Vec::<Image>::new();
    for image in images {
        // Log
        println!(
            "Loading Image {} {}, {:?}: {:?}",
            image.width,
            image.height,
            image.format,
            image.pixels.split_at(8).0
        );

        // Load
        let texel_count = image.width * image.height;
        let mut data;
        match image.format {
            gltf::image::Format::R8G8B8 => {
                data = Vec::with_capacity(texel_count as usize * 4);
                for pixel in image.pixels.chunks(3) {
                    data.push(pixel[0]);
                    data.push(pixel[1]);
                    data.push(pixel[2]);
                    data.push(255);
                }
            }
            gltf::image::Format::R8G8B8A8 => {
                data = image.pixels.to_vec();
            }
            _ => todo!(),
        }
        assert_eq!(data.len(), texel_count as usize * 4);

        model_images.push(Image {
            width: image.width,
            height: image.height,
            data,
        });
    }

    // Looa materials
    let mut model_materials = Vec::<Material>::new();
    for material in document.materials() {
        let pbr_texs = material.pbr_metallic_roughness();

        let base_color_map = if let Some(base_color) = pbr_texs.base_color_texture() {
            if base_color.tex_coord() != 0 {
                println!("Warning: GLTF Loader: Only texture coordinate 0 is supported");
            }
            println!(
                "Material base color texture: {}",
                base_color.texture().source().index(),
            );

            Some(MaterialMap {
                image_index: base_color.texture().source().index() as u32,
            })
        } else {
            None
        };

        let metallic_roughness_map =
            if let Some(metal_rough) = pbr_texs.metallic_roughness_texture() {
                if metal_rough.tex_coord() != 0 {
                    println!("Warning: GLTF Loader: Only texture coordinate 0 is supported");
                }
                println!(
                    "Material metal rough texture: {}",
                    metal_rough.texture().index()
                );

                Some(MaterialMap {
                    image_index: metal_rough.texture().source().index() as u32,
                })
            } else {
                None
            };

        let normal_map = if let Some(normal) = material.normal_texture() {
            if normal.tex_coord() != 0 {
                println!("Warning: GLTF Loader: Only texture coordinate 0 is supported");
            }
            println!("Material normal texture: {}", normal.texture().index());

            Some(MaterialMap {
                image_index: normal.texture().index() as u32,
            })
        } else {
            None
        };

        // Log ignored
        if material.emissive_texture().is_some() {
            println!("Warning: GLTF Loader: Emissive textures are not supported yet");
        }
        if material.occlusion_texture().is_some() {
            println!("Warning: GLTF Loader: Occlusion textures are not supported yet");
        }

        model_materials.push(Material {
            base_color_map,
            metallic_roughness_map,
            normal_map,
            metaliic_factor: pbr_texs.metallic_factor(),
            roughness_factor: pbr_texs.roughness_factor(),
        });
    }

    return Ok(Model {
        meshes: model_meshes,
        images: model_images,
        materials: model_materials,
    });
}
