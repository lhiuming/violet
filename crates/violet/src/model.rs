use std::{
    collections::{hash_map, HashMap},
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
};

use glam::{Mat4, Vec3, Vec4};
use intel_tex_2::{bc4, bc5, bc7};
use log::{info, trace, warn};
use rkyv::{
    ser::Serializer,
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Archive, Archived, Deserialize, Fallible, Resolver, Serialize,
};

#[derive(Debug, Clone, Copy)]
pub struct LoadConfig {
    pub force_reimport: bool,
    pub tex_compression: bool,
    pub scale: f32,
}

impl Default for LoadConfig {
    fn default() -> Self {
        LoadConfig {
            force_reimport: false,
            tex_compression: true,
            scale: 1.0,
        }
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct Model {
    pub instances: Vec<GeometryGroupInstance>,
    pub geometry_groups: Vec<Vec<(u32, TriangleMesh)>>, // mesh_group made of meshes (called "primitive" in glTF), each mesh is linked with a material
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
}

// rkyv support for Mat4
pub struct ArchiveMat4;

impl ArchiveWith<Mat4> for ArchiveMat4 {
    type Archived = Archived<[f32; 16]>;
    type Resolver = Resolver<[f32; 16]>;

    unsafe fn resolve_with(
        field: &Mat4,
        pos: usize,
        _: Resolver<[(); 16]>,
        out: *mut Self::Archived,
    ) {
        let array = field.to_cols_array();
        array.resolve(pos, [(); 16], out);
    }
}

impl<S: Fallible + ?Sized> SerializeWith<Mat4, S> for ArchiveMat4
where
    [f32; 16]: Serialize<S>,
{
    fn serialize_with(
        field: &Mat4,
        serializer: &mut S,
    ) -> std::result::Result<Self::Resolver, S::Error> {
        let array = field.to_cols_array();
        array.serialize(serializer)
    }
}

impl<D: Fallible + ?Sized> DeserializeWith<Archived<[f32; 16]>, Mat4, D> for ArchiveMat4
where
    Archived<i32>: Deserialize<i32, D>,
{
    fn deserialize_with(
        field: &Archived<[f32; 16]>,
        deserializer: &mut D,
    ) -> std::result::Result<Mat4, D::Error> {
        let array = field.deserialize(deserializer)?;
        Ok(Mat4::from_cols_array(&array))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct GeometryGroupInstance {
    pub geometry_group_index: u32,
    // TODO just store glam::Affine3A
    #[with(ArchiveMat4)]
    pub transform: Mat4,
}

// TODO geometry compression/32bit-packing
#[derive(Archive, Deserialize, Serialize, Debug)]
pub struct TriangleMesh {
    pub positions: Vec<[f32; 3]>,
    pub indicies: Vec<u16>,
    pub normals: Option<Vec<[f32; 3]>>,
    pub texcoords: Option<Vec<[f32; 2]>>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub bounds: ([f32; 3], [f32; 3]),
}

#[derive(Archive, Deserialize, Serialize)]
pub struct Material {
    pub base_color_map: Option<MaterialMap>,
    pub metallic_roughness_map: Option<MaterialMap>,
    pub normal_map: Option<MaterialMap>,
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,

    pub transmission_map: Option<MaterialMap>,
    pub transmission_factor: f32,
}

impl Material {
    pub fn maps_mut<'a>(&'a mut self) -> [&'a mut Option<MaterialMap>; 4] {
        [
            &mut self.base_color_map,
            &mut self.metallic_roughness_map,
            &mut self.normal_map,
            &mut self.transmission_map,
        ]
    }
}

#[derive(Archive, Deserialize, Serialize)]
pub struct MaterialMap {
    pub image_index: u32,
}

#[derive(Archive, Deserialize, Serialize, Clone)]
pub enum ImageFormat {
    R8G8B8A8Unorm,
    R8G8B8A8Srgb,
    BC4Unorm,
    BC5Unorm,
    BC7Unorm,
    BC7Srgb,
}

#[derive(Archive, Deserialize, Serialize, Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub mips_data: Vec<Vec<u8>>,
}

impl Image {
    pub fn empty() -> Self {
        Image {
            width: 0,
            height: 0,
            format: ImageFormat::R8G8B8A8Unorm,
            mips_data: Default::default(),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    NotExist(String),
    GLTF(gltf::Error),
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
                warn!(
                    "Writing to same tangent index with different value ({} times): {} -> {}",
                    self.debug_tangent_counts[index], a, b
                )
            }
        }

        self.out_tangents[index] = tangent;
    }
}

pub type Result<T> = std::result::Result<T, Error>;

// Load a model from cache, or
// if model is not in cache, load the original gltf file and store it in cache.
pub fn load(path: &Path, config: LoadConfig) -> Result<Model> {
    if config.force_reimport {
        return import_gltf(path, config);
    }

    // Try to load from caceh
    match try_load_from_cache(path) {
        Ok(model) => Ok(model),
        Err(err) => {
            match err {
                Error::NotExist(cache_path) => {
                    // Import if cache not exist
                    info!(
                        "Model cache ({}) not exist ... import the gltf.",
                        cache_path
                    );
                    import_gltf(path, config)
                }
                _ => Err(err),
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

        info!("Saved model to cache: {}. size {}", path.display(), size);
    }

    fn new_from_file(path: &Path) -> Result<Model> {
        let mut file = File::open(path).expect("Failed to open file");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("Failed to read file");
        let model = unsafe { rkyv::from_bytes_unchecked::<Model>(&buf) }
            .expect("Failed to deserialize model from cache");

        info!("Loaded model from cache: {}", path.display());
        Ok(model)
    }
}

pub fn import_gltf(path: &Path, config: LoadConfig) -> Result<Model> {
    let model = import_gltf_uncached(path, config);
    if let Ok(model) = &model {
        let cache_path = to_cache_path(path).expect(&format!(
            "Failed to get cached path for: {}",
            path.to_string_lossy()
        ));
        model.save_to_file(&cache_path);
    }
    model
}

bitflags::bitflags! {
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
struct ImportedImageUsage: u8 {
    const BaseColor    = 0b00000001;
    const Normal       = 0b00000010;
    const MetalRough   = 0b00000100;
    const Transmission = 0b00001000;
}
}

// Align image to 4x4 texel blocks, ready for block compression
struct BlockAlignedImage {
    image: image::RgbaImage,
}

impl BlockAlignedImage {
    pub fn from(image: image::RgbaImage) -> BlockAlignedImage {
        assert!(image.width() > 0 && image.height() > 0);
        if image.width() % 4 != 0 || image.height() % 4 != 0 {
            // Create aligned image by Padding (repeating)
            let dst_width = (image.width() + 3) & !3;
            let dst_height = (image.height() + 3) & !3;
            let mut dst = Vec::<u8>::with_capacity(dst_width as usize * dst_height as usize * 4);
            for y in 0..image.height() {
                let dst_offset = dst.len();
                // copy a row
                let src = image.as_raw();
                let src_stride = image.width() as usize * 4;
                let src_beg = src_stride * y as usize;
                let src_end = src_beg + src_stride;
                dst.extend_from_slice(&src[src_beg..src_end]);
                // wrapping from left, to fill remaining pixels
                let mut curr_width = image.width();
                while curr_width < dst_width {
                    let remaining_pixels = dst_width - curr_width;
                    let pixels = remaining_pixels.min(curr_width); // can't copy more than what we have
                    dst.extend_from_within(dst_offset..(dst_offset + pixels as usize * 4));
                    curr_width += pixels;
                }
            }
            // wrapping from top, to fill remaining rows
            let mut curr_height = image.height();
            while curr_height < dst_height {
                let remaining_rows = dst_height - curr_height;
                let rows = remaining_rows.min(curr_height); // can't copy more than what we have
                dst.extend_from_within(..(rows * dst_width) as usize * 4);
                curr_height += rows;
            }
            assert!(dst.len() == dst_width as usize * dst_height as usize * 4);
            BlockAlignedImage {
                image: image::RgbaImage::from_raw(dst_width, dst_height, dst).unwrap(),
            }
        } else {
            // Use as-it-is
            BlockAlignedImage { image }
        }
    }

    pub fn as_surface<'a>(&'a self) -> intel_tex_2::RgbaSurface<'a> {
        intel_tex_2::RgbaSurface {
            data: &self.image.as_raw(),
            width: self.image.width(),
            height: self.image.height(),
            stride: self.image.width() * 4,
        }
    }
}

pub fn import_gltf_uncached(path: &Path, config: LoadConfig) -> Result<Model> {
    // Read the document (structure) and blob data (buffers, imanges)
    let (document, buffers, images) = match gltf::import(path) {
        Ok(ret) => ret,
        Err(gltf_err) => {
            return Err(Error::GLTF(gltf_err));
        }
    };
    info!("Loaded glTF file: {}", path.display());
    if document.extensions_required().count() > 0 {
        info!(
            "Required extensions: {}",
            document
                .extensions_required()
                .collect::<Vec<&str>>()
                .join(", ")
        );
    }
    if document.extensions_used().count() > 0 {
        info!(
            "Used extensions: {}",
            document.extensions_used().collect::<Vec<&str>>().join(", ")
        );
    }

    // Log
    info!("Meshes: {}", document.meshes().len());
    info!("Images: {}", document.images().len());

    // Load geometries (witn material index) in the file
    // Each glTF mesh is a "geometry group" (or just "mesh group"?) here, wich consists of multiple glTF primitives
    let geometry_groups = document
        .meshes()
        .map(|geometry_group| {
            let mut materialed_geometries = Vec::<(u32, TriangleMesh)>::new();
            // Load each primitive
            for primitive in geometry_group.primitives() {
                // Check
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    warn!("Mesh primitive is not triangulated. Ignored.");
                    continue;
                }
                if primitive.material().index().is_none() {
                    warn!("Mesh primitive has no material. Ignored.");
                    continue;
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                // Load indicies (required)
                let mut model_indicies = Vec::<u16>::new();
                if let Some(indices) = reader.read_indices() {
                    match indices {
                        gltf::mesh::util::ReadIndices::U8(iter) => {
                            model_indicies.append(&mut iter.map(|i| i as u16).collect());
                        }
                        gltf::mesh::util::ReadIndices::U16(iter) => {
                            model_indicies.append(&mut iter.collect());
                        }
                        gltf::mesh::util::ReadIndices::U32(iter) => {
                            model_indicies.append(&mut iter.map(|i| i as u16).collect());
                        }
                    }
                    assert!(model_indicies.len() % 3 == 0);
                } else {
                    warn!("Mesh primitive has no indices.");
                    continue;
                }

                // Load positions (required)
                let mut model_positions = Vec::<[f32; 3]>::new();
                if let Some(positions) = reader.read_positions() {
                    model_positions.append(&mut positions.collect());
                } else {
                    warn!("Mesh primitive has no positions.");
                    continue;
                }

                let vertex_count = model_positions.len();

                // Load texcoords
                let model_texcoords = reader.read_tex_coords(0).map(|texcoords| match texcoords {
                    gltf::mesh::util::ReadTexCoords::U8(_) => todo!(),
                    gltf::mesh::util::ReadTexCoords::U16(_) => todo!(),
                    gltf::mesh::util::ReadTexCoords::F32(data) => data.collect(),
                });

                // Load normals
                let model_normals = reader
                    .read_normals()
                    .map(|normals| normals.collect::<Vec<[f32; 3]>>());

                // Load tangent
                let model_tangents = reader
                    .read_tangents()
                    .map(|tangents| tangents.collect::<Vec<[f32; 4]>>())
                    .or_else(|| {
                        // Generated tangents if suitable (need texcoords and normals)
                        if let Some((texcoords, normals)) =
                            model_texcoords.as_ref().zip(model_normals.as_ref())
                        {
                            let mut tangents = vec![[0.0; 4]; vertex_count];
                            let mut geometry = MikktspaceGeometry {
                                num_triangle: model_indicies.len() / 3,
                                indices: &model_indicies,
                                positions: &model_positions,
                                normals: normals,
                                texcoords: texcoords,
                                debug_tangent_counts: vec![0u32; vertex_count],
                                out_tangents: &mut tangents,
                            };
                            if mikktspace::generate_tangents(&mut geometry) {
                                return Some(tangents);
                            } else {
                                warn!("mikktspace Failed to generate tangents.");
                            }
                        }
                        None
                    });

                // Log unsupported
                if reader.read_colors(0).is_some() {
                    warn!("Mesh primitive has vertex colors. Ignored.")
                }

                // get local bouding box
                let bounds = primitive.bounding_box();

                let material_index = primitive.material().index().unwrap() as u32;
                materialed_geometries.push((
                    material_index,
                    TriangleMesh {
                        positions: model_positions,
                        indicies: model_indicies,
                        texcoords: model_texcoords,
                        normals: model_normals,
                        tangents: model_tangents,
                        bounds: (bounds.min, bounds.max),
                    },
                ));
            }

            materialed_geometries
        })
        .collect::<Vec<_>>();

    // Load instances (nodes); also flatten the hierarchy
    let mut instance_count_per_group = vec![0u32; document.meshes().count()];
    let mut instances = Vec::<GeometryGroupInstance>::new();
    if let Some(scene) = document.default_scene() {
        // iterative function along the node hierarchy
        let mut process_node = |node: &gltf::Node, xform: Mat4| -> Mat4 {
            // Get flatten transform
            let local_xform = Mat4::from_cols_array_2d(&node.transform().matrix());
            let xform = xform * local_xform;
            let _xform_normal = xform.inverse().transpose();

            if node.mesh().is_none() {
                return xform;
            }
            let mesh = node.mesh().unwrap();
            let group_index = mesh.index();

            // count instance (for stat)
            instance_count_per_group[group_index] += 1;

            instances.push(GeometryGroupInstance {
                geometry_group_index: mesh.index() as u32,
                transform: xform,
            });

            return xform;
        };

        // Rotate the model to match the coordinate system unsed in violet
        // glTF: Y up, right-handed
        // Here: Z up, right-handed
        // So we gonna rotate the model by 90 degrees around X axis
        let root_xform = Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2)
            * Mat4::from_scale(Vec3::splat(config.scale));

        for root in scene.nodes() {
            iterate_on_nodes(&root, root_xform, &mut process_node);
        }
    } else {
        return Err(Error::General("No default scene found.".to_string()));
    }

    // Log instance count
    for (index, count) in instance_count_per_group.iter().enumerate() {
        if *count > 1 {
            trace!("glTF Mesh {} has {} instances.", index, count);
        }
        if *count == 0 {
            trace!("glTF Mesh {} has no instances.", index);
        }
    }

    let fmt_sampler = |sampler: &gltf::texture::Sampler<'_>| {
        format!(
            "index:{:?}-{:?}, <min:{:?}, max:{:?}, wrap_s:{:?}, wrap_t:{:?}>",
            sampler.index(),
            sampler.name(),
            sampler.min_filter(),
            sampler.mag_filter(),
            sampler.wrap_s(),
            sampler.wrap_t(),
        )
    };

    type ImageIndex = u32;
    let mut image_usages = HashMap::<ImageIndex, ImportedImageUsage>::new();
    let mut mark_image_usage = |index, usage| match image_usages.entry(index) {
        hash_map::Entry::Occupied(entry) => {
            entry.into_mut().extend(usage);
        }
        hash_map::Entry::Vacant(entry) => {
            entry.insert(usage);
        }
    };

    // Looa materials
    let mut model_materials = Vec::<Material>::new();
    for material in document.materials() {
        let pbr_texs = material.pbr_metallic_roughness();

        let base_color_map = if let Some(base_color) = pbr_texs.base_color_texture() {
            if base_color.tex_coord() != 0 {
                warn!("GLTF Loader: Only texture coordinate 0 is supported");
            }

            let image_index = base_color.texture().source().index() as ImageIndex;
            trace!(
                "Material base color texture: {}, sampler {}",
                image_index,
                fmt_sampler(&base_color.texture().sampler())
            );

            mark_image_usage(image_index, ImportedImageUsage::BaseColor);

            Some(MaterialMap { image_index })
        } else {
            trace!("Material base color texture is empty.");
            None
        };

        let metallic_roughness_map =
            if let Some(metal_rough) = pbr_texs.metallic_roughness_texture() {
                if metal_rough.tex_coord() != 0 {
                    warn!("GLTF Loader: Only texture coordinate 0 is supported");
                }

                let image_index = metal_rough.texture().source().index() as ImageIndex;
                trace!(
                    "Material metal rough texture: {}, sampler {:?}",
                    image_index,
                    fmt_sampler(&metal_rough.texture().sampler()),
                );

                mark_image_usage(image_index, ImportedImageUsage::MetalRough);

                Some(MaterialMap { image_index })
            } else {
                trace!("Material metal rough texture is empty.");
                None
            };

        let normal_map = if let Some(normal) = material.normal_texture() {
            if normal.tex_coord() != 0 {
                warn!("GLTF Loader: Only texture coordinate 0 is supported");
            }

            let image_index = normal.texture().source().index() as ImageIndex;
            trace!(
                "Material normal texture: {}, sampler {:?}",
                image_index,
                fmt_sampler(&normal.texture().sampler()),
            );

            if normal.scale() != 1.0f32 {
                warn!(
                    "\t normal texture required scale {}, is ignored.",
                    normal.scale()
                );
            }

            mark_image_usage(image_index, ImportedImageUsage::Normal);

            Some(MaterialMap { image_index })
        } else {
            trace!("Material normal texture is empty.");
            None
        };

        let (transmission_map, transmission_factor) =
            if let Some(transmission) = material.transmission() {
                let transmission_map = if let Some(trans) = transmission.transmission_texture() {
                    if trans.tex_coord() != 0 {
                        warn!("GLTF Loader: Only texture cooridate 0 is supported");
                    }

                    let image_index = trans.texture().source().index() as ImageIndex;
                    trace!(
                        "Material transmission texture: {}, sampler {}",
                        image_index,
                        fmt_sampler(&trans.texture().sampler())
                    );

                    mark_image_usage(image_index, ImportedImageUsage::Transmission);

                    Some(MaterialMap { image_index })
                } else {
                    None
                };
                (transmission_map, transmission.transmission_factor())
            } else {
                (None, 0.0f32)
            };

        // Log ignored
        if material.emissive_texture().is_some() {
            warn!("GLTF Loader: Emissive textures are not supported yet");
        }
        if material.occlusion_texture().is_some() {
            warn!("GLTF Loader: Occlusion textures are not supported yet");
        }

        model_materials.push(Material {
            base_color_map,
            metallic_roughness_map,
            normal_map,
            base_color_factor: pbr_texs.base_color_factor(),
            metallic_factor: pbr_texs.metallic_factor(),
            roughness_factor: pbr_texs.roughness_factor(),
            transmission_factor,
            transmission_map,
        });
    }

    // Filter-out unused image
    let mut used_image_indices = Vec::with_capacity(image_usages.len());
    let mut image_indices_remap = Vec::with_capacity(images.len());
    let mut next_image_index = 0;
    for i in 0..images.len() as u32 {
        let new_index;
        if image_usages.contains_key(&i) {
            new_index = next_image_index;
            next_image_index += 1;
            used_image_indices.push(i);
        } else {
            new_index = u32::MAX;
        }
        image_indices_remap.push(new_index);
    }

    // Re-map the material images
    let image_index_remap = |opt: &mut Option<MaterialMap>| {
        if let Some(opt) = opt.as_mut() {
            let old_index = opt.image_index;
            let new_index = image_indices_remap[old_index as usize];
            assert!(new_index != u32::MAX);
            opt.image_index = new_index;
        }
    };
    for mat in &mut model_materials {
        for map in mat.maps_mut() {
            image_index_remap(map);
        }
    }

    // Load, Create mipmap, and Compress (used) images
    let image_work_func = |image_index: usize,
                           image: &gltf::image::Data,
                           imported_usage: ImportedImageUsage,
                           config: &LoadConfig|
     -> Image {
        let image_index = image_index as ImageIndex;

        // Log
        trace!(
            "Loading Image {} {}, {:?}, usage {:?}: {:?}",
            image.width,
            image.height,
            image.format,
            imported_usage,
            image.pixels.split_at(8).0
        );

        if imported_usage.bits().count_ones() > 1 {
            warn!(
                "Image {} is used for multiple purposes ({:?}) !!!",
                image_index, imported_usage
            );
        }

        // For metalllic_roughness, we swizzle the image to reduce channels (for compression)
        // Original: G: roughness, B: metallic
        // After: R: metallic, G: roughness, BA: dont care
        let swizzle_bgxx = imported_usage == ImportedImageUsage::MetalRough;

        // Load
        let texel_count = image.width * image.height;
        let mut raw_data;
        match image.format {
            gltf::image::Format::R8G8B8 => {
                raw_data = Vec::with_capacity(texel_count as usize * 4);
                for pixel in image.pixels.chunks(3) {
                    if swizzle_bgxx {
                        raw_data.push(pixel[2]); // b
                        raw_data.push(pixel[1]); // g
                        raw_data.push(pixel[0]); // r
                        raw_data.push(255); // one
                    } else {
                        raw_data.push(pixel[0]);
                        raw_data.push(pixel[1]);
                        raw_data.push(pixel[2]);
                        raw_data.push(255);
                    }
                }
            }
            gltf::image::Format::R8G8B8A8 => {
                if swizzle_bgxx {
                    raw_data = Vec::with_capacity(texel_count as usize * 4);
                    for pixel in image.pixels.chunks(4) {
                        raw_data.push(pixel[2]); // b
                        raw_data.push(pixel[1]); // g
                        raw_data.push(pixel[0]); // r
                        raw_data.push(pixel[3]); // a
                    }
                } else {
                    raw_data = image.pixels.to_vec();
                }
            }
            _ => todo!(),
        }
        assert_eq!(raw_data.len(), texel_count as usize * 4);

        // Cretae mipmaps
        let last_mip_levels = image
            .width
            .max(image.height)
            .next_power_of_two()
            .trailing_zeros() as usize;
        let mut raw_mips = Vec::<image::RgbaImage>::with_capacity(last_mip_levels + 1);
        raw_mips.push(
            image::RgbaImage::from_raw(image.width, image.height, raw_data)
                .expect("Failed to create image"),
        );
        for mip_level in 1..=last_mip_levels {
            let perv_mip = &raw_mips[mip_level - 1];
            let down_sampled = image::imageops::resize(
                perv_mip,
                (perv_mip.width() / 2).max(1),
                (perv_mip.height() / 2).max(1),
                image::imageops::FilterType::Lanczos3,
            );
            raw_mips.push(down_sampled);
        }

        // Compress
        let (mips_data, format) = if config.tex_compression {
            match imported_usage {
                ImportedImageUsage::BaseColor => {
                    // NOTE: we dont support alpha for now, so we just drop the alpha channel
                    // NOTE: base color is implicitly sRGB (enforced by glTF spec)
                    let settings = bc7::opaque_basic_settings();
                    let mips_data = raw_mips
                        .drain(..)
                        .map(|mip| {
                            bc7::compress_blocks(
                                &settings,
                                &BlockAlignedImage::from(mip).as_surface(),
                            )
                        })
                        .collect();
                    (mips_data, ImageFormat::BC7Srgb)
                }
                ImportedImageUsage::Normal => {
                    // NOTE: normal only in xyz, linear (unorm)
                    let settings = bc7::opaque_basic_settings();
                    let mips_data = raw_mips
                        .drain(..)
                        .map(|mip| {
                            bc7::compress_blocks(
                                &settings,
                                &BlockAlignedImage::from(mip).as_surface(),
                            )
                        })
                        .collect();
                    (mips_data, ImageFormat::BC7Unorm)
                }
                ImportedImageUsage::MetalRough => {
                    // NOTE: metalic roughness texture only in xy, linear (unorm)
                    let mips_data = raw_mips
                        .drain(..)
                        .map(|mip| bc5::compress_blocks(&BlockAlignedImage::from(mip).as_surface()))
                        .collect();
                    (mips_data, ImageFormat::BC5Unorm)
                }
                ImportedImageUsage::Transmission => {
                    //NOTE: transmission only has r channel (linear unorm)
                    let mips_data = raw_mips
                        .drain(..)
                        .map(|mip| bc4::compress_blocks(&BlockAlignedImage::from(mip).as_surface()))
                        .collect();
                    (mips_data, ImageFormat::BC4Unorm)
                }
                _ => (
                    raw_mips
                        .drain(..)
                        .map(image::ImageBuffer::into_raw)
                        .collect(),
                    ImageFormat::R8G8B8A8Unorm,
                ),
            }
        } else {
            let raw_data = raw_mips
                .drain(..)
                .map(image::ImageBuffer::into_raw)
                .collect::<Vec<_>>();
            // Only base color and emmisive is (and must be) sRGB (by glTF spec)
            match imported_usage {
                ImportedImageUsage::BaseColor => (raw_data, ImageFormat::R8G8B8A8Srgb),
                _ => (raw_data, ImageFormat::R8G8B8A8Unorm),
            }
        };

        Image {
            width: image.width,
            height: image.height,
            format,
            mips_data,
        }
    };

    let num_used_images = used_image_indices.len();

    // Input data for multi threading
    let config = Arc::new(config);
    let raw_images = Arc::new(images);
    let image_usages = Arc::new(image_usages);
    let used_image_indices = Arc::new(used_image_indices);

    // Shared memory
    let next_image_index = Arc::new(Mutex::new(0usize));
    let model_images_holder = Arc::new(Mutex::new(vec![Image::empty(); num_used_images]));

    // Multi-thread processing
    let num_threads = thread::available_parallelism().unwrap().get();
    (0..num_threads)
        .map(|_thread_index| {
            let config = config.clone();
            let raw_images = raw_images.clone();
            let image_usages = image_usages.clone();
            let used_image_indices = used_image_indices.clone();

            let next_image_index = next_image_index.clone();
            let model_images_holder = model_images_holder.clone();
            thread::spawn(move || {
                loop {
                    // Get next work
                    let image_index;
                    {
                        let mut m = next_image_index.lock().unwrap();
                        image_index = *m;
                        if image_index >= num_used_images {
                            break;
                        }
                        *m = image_index + 1;
                    };

                    // remap to raw image index
                    let raw_image_index = used_image_indices[image_index];

                    // Work
                    let data = &raw_images[raw_image_index as usize];
                    let image_usage = *image_usages.get(&raw_image_index).unwrap();
                    let image = image_work_func(image_index, data, image_usage, &config);

                    // Fill resullt
                    {
                        let mut m = model_images_holder.lock().unwrap();
                        m[image_index] = image;
                    }
                }
            })
        })
        .collect::<Vec<_>>() // make sure all thread is spawned
        .drain(..)
        .for_each(|handle| {
            handle
                .join()
                .expect("Image processing thread can not complete.");
        });

    assert!(*next_image_index.lock().unwrap() == num_used_images);
    let model_images = Arc::try_unwrap(model_images_holder)
        .ok()
        .unwrap()
        .into_inner()
        .unwrap();

    return Ok(Model {
        instances,
        geometry_groups,
        images: model_images,
        materials: model_materials,
    });
}
