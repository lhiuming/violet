use std::collections::{hash_map::Entry, BTreeMap, HashMap};
use std::ffi::CString;
use std::mem::size_of;

use ash::extensions::khr;
use ash::vk;
use log::{error, trace};
use rspirv_reflect::{self};
use spirq::{self};

use crate::render_device::RenderDevice;

const LOG_REFLECTION_INFO: bool = false;
const LOG_DESCRIPTOR_INFO: bool = false;

// TODO it should be provided by user?
const BINDLESS_SIZE: u32 = 1024;

// BEGIN Handle

#[derive(Eq, Hash)]
pub struct Handle<T> {
    id: u16,
    generation: u16,
    _phantom_data: std::marker::PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(id: u16, generation: u16) -> Self {
        Self {
            id,
            generation,
            _phantom_data: std::marker::PhantomData,
        }
    }

    pub fn null() -> Self {
        Self::new(u16::MAX, u16::MAX)
    }

    pub fn id(&self) -> u16 {
        self.id
    }

    pub fn is_null(&self) -> bool {
        self.id == u16::MAX
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            generation: self.generation,
            _phantom_data: self._phantom_data.clone(),
        }
    }
}

impl<T> Copy for Handle<T> {}

impl<Pipeline> PartialEq for Handle<Pipeline> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.generation == other.generation
    }
}

// END Handle

pub struct PipelineDevice {
    device: ash::Device,
    raytracing_entry: Option<khr::RayTracingPipeline>,
    pipeline_cache: vk::PipelineCache,
}

impl PipelineDevice {
    pub fn new(rd: &RenderDevice) -> PipelineDevice {
        PipelineDevice {
            device: rd.device.clone(),
            raytracing_entry: rd.khr_ray_tracing_pipeline.clone(),
            pipeline_cache: vk::PipelineCache::null(),
        }
    }
}

// AKA shader
pub struct PipelineProgram {
    pub shader_module: vk::ShaderModule,
    pub reflect_module: rspirv_reflect::Reflection,
    pub spirq_module: spirq::EntryPoint,
    pub entry_point_c: CString,
}

pub struct Shaders {
    pipeline_device: PipelineDevice,
    gfx_pipelines_map:
        HashMap<(ShaderDefinition, ShaderDefinition, GraphicsDesc), Handle<Pipeline>>,
    compute_pipelines_map: HashMap<ShaderDefinition, Handle<Pipeline>>,
    // TODO trim down this monsterous 'key' type
    raytracing_pipelines_map: HashMap<
        (
            ShaderDefinition,
            Vec<ShaderDefinition>,
            Option<ShaderDefinition>,
            RayTracingDesc,
        ),
        Handle<Pipeline>,
    >,
    shader_loader: ShaderLoader,

    pipelines: Vec<Pipeline>,
    generation: u16,
}

impl Shaders {
    pub fn new(rd: &RenderDevice) -> Shaders {
        Shaders {
            pipeline_device: PipelineDevice::new(&rd),
            gfx_pipelines_map: HashMap::new(),
            compute_pipelines_map: HashMap::new(),
            raytracing_pipelines_map: HashMap::new(),
            shader_loader: ShaderLoader::new(),
            pipelines: Vec::new(),
            generation: 0,
        }
    }

    pub fn add_path(&mut self, path: String) {
        self.shader_loader.add_path(path);
    }

    pub fn shader_debug(&self) -> bool {
        self.shader_loader.shader_debug
    }

    pub fn set_shader_debug(&mut self, value: bool) {
        if self.shader_loader.shader_debug != value {
            self.shader_loader.shader_debug = value;
            self.reload_all();
        }
    }

    #[inline]
    fn add_pipeline(&mut self, pipeline: Pipeline) -> Handle<Pipeline> {
        let id = self.pipelines.len();
        self.pipelines.push(pipeline);
        Handle::new(id as u16, self.generation)
    }

    pub fn create_gfx_pipeline(
        &mut self,
        vs_def: ShaderDefinition,
        ps_def: ShaderDefinition,
        desc: &GraphicsDesc,
        hack: &ShadersConfig,
    ) -> Option<Handle<Pipeline>> {
        // look from cache
        // If not in cache, create and push into cache
        // TODO currenty need to query twice even if cache can hit (constians_key, get)
        let key = (vs_def, ps_def, *desc);

        if !self.gfx_pipelines_map.contains_key(&key) {
            let vs = self.shader_loader.load(&self.pipeline_device, &vs_def)?;
            let ps = self.shader_loader.load(&self.pipeline_device, &ps_def)?;
            let pipeline_created =
                create_graphics_pipeline(&self.pipeline_device, &vs, &ps, desc, hack);
            if let Some(pipeline) = pipeline_created {
                let handle = self.add_pipeline(pipeline);
                self.gfx_pipelines_map.insert(key, handle.clone());
                return Some(handle);
            }
            return None;
        }

        Some(*self.gfx_pipelines_map.get(&key).unwrap())
    }

    pub fn create_compute_pipeline(
        &mut self,
        cs_def: ShaderDefinition,
        hack: &ShadersConfig,
    ) -> Option<Handle<Pipeline>> {
        let key = cs_def;

        if !self.compute_pipelines_map.contains_key(&key) {
            let cs = self.shader_loader.load(&self.pipeline_device, &cs_def)?;
            let pipeline_created =
                create_compute_pipeline(&self.pipeline_device, &cs_def, &cs, hack);
            if let Some(pipeline) = pipeline_created {
                let handle = self.add_pipeline(pipeline);
                self.compute_pipelines_map.insert(key, handle);
                return Some(handle);
            }
            return None;
        }

        Some(*self.compute_pipelines_map.get(&key).unwrap())
    }

    pub fn create_raytracing_pipeline(
        &mut self,
        ray_gen_def: ShaderDefinition,
        miss_def: &[ShaderDefinition],
        hit_def: Option<ShaderDefinition>,
        desc: &RayTracingDesc,
        hack: &ShadersConfig,
    ) -> Option<Handle<Pipeline>> {
        assert!(miss_def.len() > 0);

        let key = (ray_gen_def, miss_def.to_vec(), hit_def, *desc);

        if !self.raytracing_pipelines_map.contains_key(&key) {
            let raygen_cs = self
                .shader_loader
                .load(&self.pipeline_device, &ray_gen_def)?;
            let miss_cs: Vec<_> = miss_def
                .iter()
                .map(|def| self.shader_loader.load(&self.pipeline_device, def).unwrap())
                .collect();
            let closest_hit = match hit_def {
                Some(def) => Some(self.shader_loader.load(&self.pipeline_device, &def)?),
                None => None,
            };
            let pipeline_created = create_raytracing_pipeline(
                &self.pipeline_device,
                &raygen_cs,
                &miss_cs,
                &closest_hit,
                desc,
                &hack,
            );
            if let Some(pipeline) = pipeline_created {
                let handle = self.add_pipeline(pipeline);
                self.raytracing_pipelines_map.insert(key, handle);
                return Some(handle);
            }
            return None;
        }

        Some(*self.raytracing_pipelines_map.get(&key).unwrap())
    }

    pub fn get_pipeline(&self, handle: Handle<Pipeline>) -> Option<&Pipeline> {
        if handle.is_null() || handle.generation != self.generation {
            None
        } else {
            let id = handle.id();
            Some(&self.pipelines[id as usize])
        }
    }

    pub fn reload_all(&mut self) {
        // TODO reload all shaders by checking file timestamps (and checksum?)
        self.gfx_pipelines_map.clear();
        self.compute_pipelines_map.clear();
        self.raytracing_pipelines_map.clear();
        self.pipelines.clear();

        self.generation += 1;
    }
}

fn create_pipeline_program(
    device: &PipelineDevice,
    binary: &[u8],
    shader_def: &ShaderDefinition,
) -> Option<PipelineProgram> {
    let device = &device.device;

    // Create shader module
    let shader_module = {
        assert!(binary.len() & 0x3 == 0);
        let binary_u32 =
            unsafe { std::slice::from_raw_parts(binary.as_ptr() as *const u32, binary.len() / 4) };
        let create_info = vk::ShaderModuleCreateInfo::builder().code(binary_u32);
        unsafe { device.create_shader_module(&create_info, None) }.ok()?
    };

    // Get reflect info
    let reflect_module = match rspirv_reflect::Reflection::new_from_spirv(binary) {
        Ok(refl) => Some(refl),
        Err(refl_err) => {
            error!("Failed to reflect shader module: {:?}", refl_err);
            None
        }
    }?;

    // Get spirq info (alternative reflection)
    let spirq_module = {
        let binary_u32 =
            unsafe { std::slice::from_raw_parts(binary.as_ptr() as *const u32, binary.len() / 4) };

        let entry_points = spirq::ReflectConfig::new()
            .spv(binary_u32)
            .reflect()
            .unwrap();
        //assert!(entry_points.len() == 1);
        entry_points[0].to_owned()
    };

    // Debug: print the reflect content
    if LOG_REFLECTION_INFO {
        trace!(
            "Reflection(shader: {}, entry_point: {})",
            shader_def.virtual_path,
            shader_def.entry_point
        );
        trace!(
            "\tdesciptor_sets: {:?}",
            reflect_module.get_descriptor_sets()
        );
        if let Some(pc) = reflect_module.get_push_constant_range().unwrap_or_default() {
            trace!("\tpush_consants: offset {}, size{}", pc.offset, pc.size);
        }
    }

    let entry_point_c = CString::new(shader_def.entry_point).unwrap();

    Some(PipelineProgram {
        shader_module,
        reflect_module,
        spirq_module,
        entry_point_c,
    })
}

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

trait ToCString {
    fn to_cstring(&self) -> CString;
}

impl ToCString for String {
    fn to_cstring(&self) -> CString {
        CString::new(self.clone()).unwrap()
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum ShaderStage {
    Compute,
    Vert,
    Frag,
    RayGen,
    Miss,
    ClosestHit,
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct ShaderDefinition {
    pub virtual_path: &'static str,
    pub entry_point: &'static str,
    pub stage: ShaderStage,
}

impl ShaderDefinition {
    pub fn new(
        virtual_path: &'static str,
        entry_point: &'static str,
        stage: ShaderStage,
    ) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage,
        }
    }

    pub fn compute(virtual_path: &'static str, entry_point: &'static str) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage: ShaderStage::Compute,
        }
    }

    pub fn vert(virtual_path: &'static str, entry_point: &'static str) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage: ShaderStage::Vert,
        }
    }

    pub fn frag(virtual_path: &'static str, entry_point: &'static str) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage: ShaderStage::Frag,
        }
    }

    pub fn raygen(virtual_path: &'static str, entry_point: &'static str) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage: ShaderStage::RayGen,
        }
    }

    pub fn miss(virtual_path: &'static str, entry_point: &'static str) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path,
            entry_point,
            stage: ShaderStage::Miss,
        }
    }

    // TODO use macro to raplace boiler plate?
    pub fn closesthit(virtual_path: &'static str, entry_point: &'static str) -> Self {
        Self {
            virtual_path,
            entry_point,
            stage: ShaderStage::ClosestHit,
        }
    }
}

pub struct ShadersConfig {
    pub set_layout_override: HashMap<u32, vk::DescriptorSetLayout>,
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct BlendDesc {
    pub src_color_blend_factor: vk::BlendFactor,
    pub dst_color_blend_factor: vk::BlendFactor,
    pub color_blend_op: vk::BlendOp,
    pub src_alpha_blend_factor: vk::BlendFactor,
    pub dst_alpha_blend_factor: vk::BlendFactor,
    pub alpha_blend_op: vk::BlendOp,
}

impl BlendDesc {
    /// Alpha blending with premultiplied alpha output,
    /// e.g. to achieve Porter and Duff 'over' operator, typically used in UI.
    pub fn premultiplied_alpha() -> Self {
        Self {
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum Blend {
    Disable,
    Enable(BlendDesc),
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct GraphicsDesc {
    pub color_attachment_count: u8,
    pub color_attachments: [vk::Format; 4],
    pub depth_attachment: Option<vk::Format>,
    pub stencil_attachment: Option<vk::Format>,
    pub blend: Blend,
}

impl GraphicsDesc {
    pub fn blend_enabled(&self) -> bool {
        return self.blend != Blend::Disable;
    }
}

impl Default for GraphicsDesc {
    fn default() -> Self {
        Self {
            color_attachment_count: 0,
            color_attachments: [vk::Format::UNDEFINED; 4],
            depth_attachment: None,
            stencil_attachment: None,
            blend: Blend::Disable,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct RayTracingDesc {
    pub ray_recursiion_depth: u32,
}

impl Default for RayTracingDesc {
    fn default() -> Self {
        Self {
            ray_recursiion_depth: 32,
        }
    }
}

impl Default for ShadersConfig {
    fn default() -> Self {
        Self {
            set_layout_override: Default::default(),
        }
    }
}

pub struct CompiledShader {
    //pub artifact: shaderc::CompilationArtifact,
    pub program: PipelineProgram,
}

struct IncludeHandler {
    include_root: PathBuf,
}

impl hassle_rs::DxcIncludeHandler for IncludeHandler {
    fn load_source(&self, filename: String) -> Option<String> {
        let path = self.include_root.join(filename);
        match std::fs::File::open(path) {
            Ok(mut f) => {
                let mut content = String::new();
                f.read_to_string(&mut content).unwrap();
                Some(content)
            }
            Err(_) => None,
        }
    }
}

// Context for shader compilation
struct ShaderLoader {
    compiler: hassle_rs::DxcCompiler,
    library: hassle_rs::DxcLibrary,
    _dxc: hassle_rs::Dxc, // delacare this last to drop it after {compiler, library}

    shader_debug: bool,
    search_pathes: Vec<String>,
    longest_path_len: usize,
}

impl ShaderLoader {
    pub fn new() -> ShaderLoader {
        let dxc = hassle_rs::Dxc::new(None).unwrap();
        let compiler = dxc.create_compiler().unwrap();
        let library = dxc.create_library().unwrap();
        let default_path = "./shader/".to_string();
        ShaderLoader {
            compiler,
            library,
            _dxc: dxc,
            shader_debug: false,
            longest_path_len: default_path.len(),
            search_pathes: vec![default_path],
        }
    }

    pub fn add_path(&mut self, path: String) {
        self.longest_path_len = self.longest_path_len.max(path.len());
        self.search_pathes.push(path);
    }

    fn find_shader_file(&self, virtual_path: &str) -> Option<PathBuf> {
        let len_hint = self.longest_path_len + virtual_path.len();
        let mut path_buf = PathBuf::with_capacity(len_hint);

        for dir in &self.search_pathes {
            path_buf.push(dir);
            path_buf.push(&virtual_path);
            if path_buf.is_file() && path_buf.exists() {
                return Some(path_buf);
            }

            path_buf.clear();
        }

        error!("Shaders: failed to find shader file: {}", virtual_path);
        None
    }

    // Load shader with retry
    pub fn load(
        &self,
        device: &PipelineDevice,
        shader_def: &ShaderDefinition,
    ) -> Option<CompiledShader> {
        loop {
            let shader = self.load_shader_once(device, shader_def);

            // Breakpoint to let user fix shader
            if shader.is_none() {
                #[cfg(feature = "nightly")]
                unsafe {
                    std::intrinsics::breakpoint();
                }

                #[cfg(not(feature = "nightly"))]
                return None;
            } else {
                return shader;
            }
        }
    }

    pub fn load_shader_once(
        &self,
        device: &PipelineDevice,
        shader_def: &ShaderDefinition,
    ) -> Option<CompiledShader> {
        // Seach file from pathes
        // todo map v_path to actuall pathes
        let path = self.find_shader_file(shader_def.virtual_path)?;

        let display = path.display();

        // Read file content
        let mut file = match File::open(&path) {
            Err(why) => panic!("Coundn't open shader path {}: {}", display, why),
            Ok(file) => file,
        };
        let mut text = String::new();
        match file.read_to_string(&mut text) {
            Err(why) => panic!("Couldn't read file {}: {}", display, why),
            Ok(_) => (),
        };

        // Compile the shader
        let file_name_os = path.file_name().unwrap();
        let file_dir = path.parent().unwrap().to_owned();
        let file_name = file_name_os.to_str().unwrap();
        let compiled_binary = self.compile_hlsl(
            file_name,
            &text,
            &shader_def.entry_point,
            shader_def.stage,
            file_dir,
        )?;

        let program = match create_pipeline_program(device, &compiled_binary, &shader_def) {
            Some(program) => program,
            None => return None,
        };

        Some(CompiledShader { program: program })
    }

    fn compile_hlsl(
        &self,
        source_name: &str,
        shader_text: &str,
        entry_point: &str,
        stage: ShaderStage,
        source_dir: PathBuf,
    ) -> Option<Vec<u8>> {
        let blob = self
            .library
            .create_blob_with_encoding_from_str(shader_text)
            .ok()?;

        // ref(raytracing): https://learn.microsoft.com/en-us/windows/win32/direct3d12/direct3d-12-raytracing-hlsl-shaders
        let target_profile = match stage {
            ShaderStage::Compute => "cs_5_0",
            ShaderStage::Vert => "vs_5_0",
            ShaderStage::Frag => "ps_5_0",
            ShaderStage::RayGen => "lib_6_3",
            ShaderStage::Miss => "lib_6_3",
            ShaderStage::ClosestHit => "lib_6_3",
        };

        // NOTE: -fspv-debug=vulkan-with-source requires extended instruction set support form the reflector
        let mut args = vec![
            // output spirv
            "-spirv",
            // Warning as Error
            "-WX",
            // Embed debug info (symbols for debugging and profiling)
            "-Zi",
            // NOTE: requires Google extention in vulkan
            //"-fspv-reflect",
            // Enable Raytracing ("Vulkan 1.1 with SPIR-V 1.4 is required for Raytracing" is printed by DXC)
            "-fspv-target-env=vulkan1.1spirv1.4",
        ];
        if self.shader_debug {
            // Disable Optimizations
            args.push("-Od");
        } else {
            // All Optimizations
            args.push("-O3");
            // Ignore NaN and +-Infs in arithemethic
            args.push("-ffinite-math-only");
        }

        let result = self.compiler.compile(
            &blob,
            source_name,
            entry_point,
            target_profile,
            &args,
            Some(Box::new(IncludeHandler {
                include_root: source_dir,
            })),
            &[],
        );

        match result {
            Err((result, _result_hr)) => {
                let error_blob = result.get_error_buffer().ok()?;
                let error_string = self.library.get_blob_as_string(&error_blob);
                error!("Failed to compile shader: {}", error_string);
                None
            }
            Ok(result) => {
                let result_blob = result.get_result().ok()?;
                Some(result_blob.to_vec())
            }
        }
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)] // TODO
pub struct PipelineDescriptorInfo {
    pub set_index: u32,
    pub binding_index: u32,
    pub descriptor_type: vk::DescriptorType,
    // TODO type? array len?
}

fn create_merged_descriptor_set_layouts(
    device: &ash::Device,
    stages_info: &[(vk::ShaderStageFlags, &rspirv_reflect::Reflection)],
    used_set: &mut HashMap<u32, ()>,
    property_map: &mut HashMap<String, PipelineDescriptorInfo>,
    hack: &ShadersConfig,
) -> Vec<vk::DescriptorSetLayout> {
    // NOTE: set layout is not allowed to be null (except for pipeline library?)
    let fill_empty_set = true;

    // Merge descriptor sets info
    let mut merged_descritpr_sets: HashMap<u32, _> = HashMap::new();
    for (stage, relfection) in stages_info {
        for (set_index, set_bindings) in relfection.get_descriptor_sets().unwrap() {
            // Get or create a merged set record
            let merged_set: &mut BTreeMap<u32, vk::DescriptorSetLayoutBinding> =
                merged_descritpr_sets
                    .entry(set_index)
                    .or_insert_with(|| BTreeMap::new());
            // Merge bindings from this stage
            for (binding_index, descriptor_info) in set_bindings {
                // From relfection info to vk struct
                // NOTE: is this safe?
                let descriptor_type = vk::DescriptorType::from_raw(descriptor_info.ty.0 as i32);
                let count = match descriptor_info.binding_count {
                    rspirv_reflect::BindingCount::One => 1,
                    rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                    rspirv_reflect::BindingCount::Unbounded => BINDLESS_SIZE,
                };
                // New or merge binding record (stage flags)
                match merged_set.entry(binding_index) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        // TODO add extension flags for bindless textures
                        let binding = vk::DescriptorSetLayoutBinding::builder()
                            .binding(binding_index)
                            .descriptor_type(descriptor_type)
                            .descriptor_count(count)
                            .stage_flags(*stage);
                        entry.insert(binding.build());
                    }
                    std::collections::btree_map::Entry::Occupied(entry) => {
                        let binding = entry.into_mut();
                        assert!(descriptor_type == binding.descriptor_type);
                        assert!(count == binding.descriptor_count);
                        binding.stage_flags |= *stage;
                    }
                }

                // Collect property for later use
                let prop_info = PipelineDescriptorInfo {
                    set_index,
                    binding_index,
                    descriptor_type,
                };
                match property_map.entry(descriptor_info.name.clone()) {
                    Entry::Occupied(entry) => {
                        // sanity check
                        assert_eq!(entry.get(), &prop_info);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(prop_info);
                    }
                }
            }

            // Collect set index
            used_set.insert(set_index, ());
        }
    }

    // Make vk struct array
    let mut set_layouts = Vec::new();
    if merged_descritpr_sets.len() > 0 {
        let last_set_index = merged_descritpr_sets.keys().max().unwrap();
        set_layouts.resize(
            (last_set_index + 1) as usize,
            vk::DescriptorSetLayout::null(),
        );

        merged_descritpr_sets
            .drain()
            .for_each(|(set_index, bindings)| {
                // Check override
                if hack.set_layout_override.contains_key(&set_index) {
                    set_layouts[set_index as usize] = hack.set_layout_override[&set_index];
                    if LOG_DESCRIPTOR_INFO {
                        trace!("Overriding set layout for set {}", set_index);
                    }
                    return;
                }
                // Create set layout
                let set_index = set_index as usize;
                let bindings = bindings
                    .into_values()
                    .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
                let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
                assert!(set_layouts.len() > set_index);
                match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                    Ok(set_layout) => {
                        set_layouts[set_index] = set_layout;
                    }
                    Err(e) => {
                        error!(
                            "Failed to create descriptor set layout: {:?}, from bindings {:?}",
                            e, bindings
                        );
                    }
                }
            });
    }

    // Fill empty set layouts
    if fill_empty_set {
        for set_layout in &mut set_layouts {
            if *set_layout == vk::DescriptorSetLayout::null() {
                let create_info = vk::DescriptorSetLayoutCreateInfo::builder();
                *set_layout =
                    unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();
            }
        }
    }

    set_layouts
}

// Create all push constant ranges use in all stages
fn make_merged_push_constant_ranges(
    stages_info: &[(vk::ShaderStageFlags, &rspirv_reflect::Reflection)],
) -> Vec<vk::PushConstantRange> {
    let mut push_constant_ranges = Vec::<vk::PushConstantRange>::new();
    for (stage_flag, reflection) in stages_info {
        let stage_flag = *stage_flag;
        if let Ok(Some(info)) = reflection.get_push_constant_range() {
            if info.size == 0 {
                continue;
            }

            // merge to prev range entry
            let mut merged = false;
            for range in &mut push_constant_ranges {
                if (range.offset == info.offset) && (range.size == info.size) {
                    range.stage_flags |= stage_flag;
                    merged = true;
                    break;
                } else {
                    // sanity check: ranges are not overlaped
                    assert!(
                        (range.offset + range.size) <= info.offset
                            || (info.offset + info.size) <= range.offset
                    );
                }
            }
            // else, this is a new range
            if !merged {
                push_constant_ranges.push(vk::PushConstantRange {
                    stage_flags: stage_flag,
                    offset: info.offset,
                    size: info.size,
                });
            }
        }
    }

    push_constant_ranges
}

// TODO should be allow copy, but required to allow Handle copy
pub struct Pipeline {
    pub handle: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub layout: vk::PipelineLayout,

    // reflection info
    pub used_set: HashMap<u32, ()>,
    pub property_map: HashMap<String, PipelineDescriptorInfo>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub fn create_compute_pipeline(
    device: &PipelineDevice,
    shader_def: &ShaderDefinition,
    compiled: &CompiledShader,
    hack: &ShadersConfig,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let device = &device.device;

    let program = &compiled.program;
    //    let reflect_module = &program.reflect_module;
    let stage_info = [(vk::ShaderStageFlags::COMPUTE, &program.reflect_module)];

    // Reflection info to be collected
    let mut property_map = HashMap::new();
    let mut used_set = HashMap::new();

    // Create all set layouts used
    let set_layouts = create_merged_descriptor_set_layouts(
        device,
        &stage_info,
        &mut used_set,
        &mut property_map,
        hack,
    );

    // Create all push constant range use in shader
    let push_constant_ranges = make_merged_push_constant_ranges(&stage_info);

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Create pipeline object
    let pipeline = {
        let entry_point_c = CString::new(shader_def.entry_point)
            .expect(&format!("Bad entry point name: {}", shader_def.entry_point));
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(program.shader_module)
            .name(&entry_point_c);
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(layout);
        let create_infos = [create_info.build()];
        let result =
            unsafe { device.create_compute_pipelines(pipeline_cache, &create_infos, None) }.ok()?;
        result[0]
    };

    Some(Pipeline {
        handle: pipeline,
        set_layouts,
        layout,
        used_set,
        property_map,
        push_constant_ranges,
    })
}

pub fn create_raytracing_pipeline(
    device: &PipelineDevice,
    raygen: &CompiledShader,
    miss: &[CompiledShader],
    closest_hit: &Option<CompiledShader>,
    desc: &RayTracingDesc,
    hack: &ShadersConfig,
) -> Option<Pipeline> {
    assert!(miss.len() > 0);

    let raytracing_entry = device.raytracing_entry.as_ref()?;

    let mut stages_info = vec![(
        vk::ShaderStageFlags::RAYGEN_KHR,
        &raygen.program.reflect_module,
    )];
    stages_info.extend(miss.iter().map(|shader| {
        (
            vk::ShaderStageFlags::MISS_KHR,
            &shader.program.reflect_module,
        )
    }));
    if let Some(chit) = closest_hit {
        stages_info.push((
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            &chit.program.reflect_module,
        ));
    }

    // Reflection info to be collect
    let mut property_map = HashMap::new();
    let mut used_set = HashMap::new();

    // Set layout for all stages
    let set_layouts = create_merged_descriptor_set_layouts(
        &device.device,
        &stages_info,
        &mut used_set,
        &mut property_map,
        hack,
    );

    // Push constant for all stages
    let push_constant_ranges = make_merged_push_constant_ranges(&stages_info);

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        unsafe { device.device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    let raygen_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::RAYGEN_KHR)
        .module(raygen.program.shader_module)
        .name(&raygen.program.entry_point_c);

    let mut stages = vec![*raygen_info];

    stages.extend(miss.iter().map(|shader| {
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::MISS_KHR)
            .module(shader.program.shader_module)
            .name(&shader.program.entry_point_c)
            .build()
    }));

    if let Some(chit) = closest_hit {
        let info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
            .module(chit.program.shader_module)
            .name(&chit.program.entry_point_c);
        stages.push(*info);
    }

    // raygen shader as first group
    let raygen_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(0)
        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
        .any_hit_shader(vk::SHADER_UNUSED_KHR)
        .intersection_shader(vk::SHADER_UNUSED_KHR)
        .build();
    let mut groups = vec![raygen_group];
    // miss shaders are placed after the raygen shader
    groups.extend((0..miss.len() as u32).map(|miss_index| {
        vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(1 + miss_index) // always one raygen before
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build()
    }));
    // closest hit shader place after miss shaders
    if closest_hit.is_some() {
        let chit_index = 1 + miss.len() as u32; // one raygen and few miss shaders before
        let hit_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR) // NOTE: device lost if left as zero
            .closest_hit_shader(chit_index)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)
            .build();
        groups.push(hit_group);
    }

    let mut flags = vk::PipelineCreateFlags::RAY_TRACING_NO_NULL_MISS_SHADERS_KHR;
    if closest_hit.is_some() {
        flags |= vk::PipelineCreateFlags::RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR;
    }
    let create_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .flags(flags)
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(desc.ray_recursiion_depth)
        //.library_info(todo!())
        //.dynamic_state(dynamic_state)
        .layout(layout);

    let pipeline = unsafe {
        raytracing_entry
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(), // not using deferred creation
                device.pipeline_cache,
                &[*create_info],
                None,
            )
            .unwrap()[0]
    };

    Some(Pipeline {
        handle: pipeline,
        set_layouts,
        layout,
        used_set,
        property_map,
        push_constant_ranges,
    })
}

pub fn create_graphics_pipeline(
    device: &PipelineDevice,
    vs: &CompiledShader,
    ps: &CompiledShader,
    desc: &GraphicsDesc,
    hack: &ShadersConfig,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let device = &device.device;

    let stages_info = [
        (vk::ShaderStageFlags::VERTEX, &vs.program.reflect_module),
        (vk::ShaderStageFlags::FRAGMENT, &ps.program.reflect_module),
    ];

    // Reflection info to be collected
    let mut property_map = HashMap::new();
    let mut used_set = HashMap::new();

    // Create all set layouts used in all stages
    let set_layouts = create_merged_descriptor_set_layouts(
        device,
        &stages_info,
        &mut used_set,
        &mut property_map,
        hack,
    );

    // Create all push constant ranges use in all stages
    let push_constant_ranges = make_merged_push_constant_ranges(&stages_info);

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Stages
    let vs_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vs.program.shader_module)
        .name(&vs.program.entry_point_c);
    let ps_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(ps.program.shader_module)
        .name(&ps.program.entry_point_c);
    let stage_infos = [vs_info.build(), ps_info.build()];

    // States
    let vertex_info = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    //let tess = ();
    let viewport = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1) // actual state is dynamic
        .scissor_count(1); // actual dynamic
    let raster = vk::PipelineRasterizationStateCreateInfo::builder().line_width(1.0) // default
    ;
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil =
        vk::PipelineDepthStencilStateCreateInfo::builder().depth_compare_op(vk::CompareOp::GREATER);
    // [dynamic_rendering]: blend state attachment count must be equal to color attachment count
    let blend_state_attachments = (0..desc.color_attachment_count)
        .map(|_| {
            let mut attachment = vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(desc.blend != Blend::Disable)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .build();
            if let Blend::Enable(blend) = desc.blend {
                // NOTE: currently the only use case if pre-multiplied alpha (UI)
                attachment.src_color_blend_factor = blend.src_color_blend_factor;
                attachment.dst_color_blend_factor = blend.dst_color_blend_factor;
                attachment.color_blend_op = blend.color_blend_op;
                attachment.src_alpha_blend_factor = blend.src_alpha_blend_factor;
                attachment.dst_alpha_blend_factor = blend.dst_alpha_blend_factor;
                attachment.alpha_blend_op = blend.alpha_blend_op;
            }
            attachment
        })
        .collect::<Vec<_>>();
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_state_attachments);
    let dynamic_states = [
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        //vk::DynamicState::STENCIL_WRITE_MASK,
        //vk::DynamicState::STENCIL_COMPARE_MASK,
        //vk::DynamicState::STENCIL_REFERENCE,
        vk::DynamicState::DEPTH_TEST_ENABLE,
        vk::DynamicState::DEPTH_WRITE_ENABLE,
        vk::DynamicState::STENCIL_TEST_ENABLE,
        vk::DynamicState::STENCIL_WRITE_MASK,
        vk::DynamicState::STENCIL_COMPARE_MASK,
        vk::DynamicState::STENCIL_OP,
        vk::DynamicState::STENCIL_REFERENCE,
        //vk::DynamicState::COLOR_BLEND_ENABLE_EXT,
        //vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
    ];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Extention: PipelineRendering (dynamic_rendering)
    // NOTE: color_attachment_count here must be the same with that in PipelineColorBlendStateCreateInfo, if has fragment output
    // NOTE: even with dynamic rendering, color_attachment_formats must be equal those specified in RenderingInfo when calling CmdBeginRendering.
    let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::builder()
        .view_mask(0) // not using multi-view
        .color_attachment_formats(&desc.color_attachments[0..desc.color_attachment_count as usize])
        .depth_attachment_format(desc.depth_attachment.unwrap_or(vk::Format::UNDEFINED))
        .stencil_attachment_format(desc.stencil_attachment.unwrap_or(vk::Format::UNDEFINED));

    let create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stage_infos)
        .vertex_input_state(&vertex_info)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .dynamic_state(&dynamic_state)
        .push_next(&mut pipeline_rendering);
    let create_infos = [create_info.build()];
    let result =
        unsafe { device.create_graphics_pipelines(pipeline_cache, &create_infos, None) }.ok()?;
    let pipeline = result[0];

    Some(Pipeline {
        handle: pipeline,
        set_layouts,
        layout,
        used_set,
        property_map,
        push_constant_ranges,
    })
}

type NamedVec<'a, T> = Vec<(&'a str, T)>;

pub struct DescriptorSetWriteBuilder<'a> {
    buffer_views: NamedVec<'a, vk::BufferView>,
    buffer_infos: NamedVec<'a, vk::DescriptorBufferInfo>,
    image_infos: NamedVec<'a, vk::DescriptorImageInfo>,
    //samplers: NamedVec<'a, vk::Sampler>,
    accel_structs: NamedVec<'a, vk::AccelerationStructureKHR>,

    write_accel_strusts: Vec<vk::WriteDescriptorSetAccelerationStructureKHR>,
    writes: Vec<vk::WriteDescriptorSet>,
}

impl Default for DescriptorSetWriteBuilder<'_> {
    fn default() -> Self {
        Self {
            buffer_views: Default::default(),
            buffer_infos: Default::default(),
            image_infos: Default::default(),
            //samplers: Default::default(),
            accel_structs: Default::default(),
            write_accel_strusts: Default::default(),
            writes: Default::default(),
        }
    }
}

impl<'a> DescriptorSetWriteBuilder<'a> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn build<F>(
        &mut self,
        pipeline: &Pipeline,
        set: vk::DescriptorSet,
        set_index: u32, // for sanity check
        fn_unused: F,
    ) -> &[vk::WriteDescriptorSet]
    where
        F: Fn(&str, &str),
    {
        self.writes.clear();
        self.write_accel_strusts.clear();

        for (name, buffer_view) in &self.buffer_views {
            if let Some(info) = pipeline.property_map.get(*name) {
                assert!(info.set_index == set_index);
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .texel_buffer_view(std::slice::from_ref(buffer_view));
                self.writes.push(*write);
            } else {
                fn_unused(name, "buffer_view");
            }
        }

        for (name, buffer_info) in &self.buffer_infos {
            if let Some(info) = pipeline.property_map.get(*name) {
                assert!(info.set_index == set_index);
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .buffer_info(std::slice::from_ref(buffer_info));
                self.writes.push(*write);
            } else {
                fn_unused(name, "buffer_info");
            }
        }

        for (name, image_info) in &self.image_infos {
            if let Some(info) = pipeline.property_map.get(*name) {
                assert!(info.set_index == set_index);
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .image_info(std::slice::from_ref(image_info));
                self.writes.push(*write);
            } else {
                fn_unused(name, "image_info");
            }
        }

        self.write_accel_strusts.reserve(self.accel_structs.len());
        for (name, accel_struct) in &self.accel_structs {
            if let Some(info) = pipeline.property_map.get(*name) {
                assert!(info.set_index == set_index);
                self.write_accel_strusts.push(
                    vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                        .acceleration_structures(std::slice::from_ref(accel_struct))
                        .build(),
                );
                let write_accel_structure = self.write_accel_strusts.last_mut().unwrap();

                let mut write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .push_next(write_accel_structure)
                    .build();
                write.descriptor_count = write_accel_structure.acceleration_structure_count;
                self.writes.push(write);
            } else {
                fn_unused(name, "accel_struct");
            }
        }

        return &self.writes;
    }

    #[inline]
    pub fn texel_buffer(&mut self, name: &'a str, buffer_view: vk::BufferView) -> &mut Self {
        self.buffer_views.push((name, buffer_view));
        self
    }

    #[inline]
    pub fn buffer(&mut self, name: &'a str, buffer: vk::Buffer, offset: u64) -> &mut Self {
        self.buffer_infos.push((
            name,
            vk::DescriptorBufferInfo {
                buffer,
                offset,
                range: vk::WHOLE_SIZE,
            },
        ));
        self
    }

    #[inline]
    pub fn image(
        &mut self,
        name: &'a str,
        image_view: vk::ImageView,
        image_layout: vk::ImageLayout,
    ) -> &mut Self {
        self.image_infos.push((
            name,
            vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: image_view,
                image_layout,
            },
        ));
        self
    }

    #[inline]
    pub fn sampler(&mut self, name: &'a str, sampler: vk::Sampler) -> &mut Self {
        self.image_infos.push((
            name,
            vk::DescriptorImageInfo {
                sampler: sampler,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            },
        ));
        self
    }

    #[inline]
    pub fn accel_struct(
        &mut self,
        name: &'a str,
        accel_struct: vk::AccelerationStructureKHR,
    ) -> &mut Self {
        self.accel_structs.push((name, accel_struct));
        self
    }
}

pub struct PushConstantsBuilder {
    data: Vec<u8>,
}

impl PushConstantsBuilder {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push_inplace<T>(&mut self, value: &T)
    where
        T: Copy,
    {
        // Push constant size must be a multiple of 4; we assume each push is a complete field.
        assert!(size_of::<T>() % 4 == 0);

        let size = std::mem::size_of::<T>();
        let offset = self.data.len();
        self.data.resize(offset + size, 0);
        self.data[offset..offset + size].copy_from_slice(unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, size)
        });
    }

    pub fn push<T>(mut self, value: &T) -> Self
    where
        T: Copy,
    {
        self.push_inplace(value);
        self
    }

    pub fn push_slice<T>(mut self, values: &[T]) -> Self
    where
        T: Copy,
    {
        // Push constant size must be a multiple of 4; we assume each push is a complete field.
        assert!(size_of::<T>() % 4 == 0);

        let size = std::mem::size_of::<T>() * values.len();
        let offset = self.data.len();
        self.data.resize(offset + size, 0);
        self.data[offset..offset + size].copy_from_slice(unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, size)
        });

        self
    }

    pub fn build(&self) -> &[u8] {
        &self.data
    }
}
