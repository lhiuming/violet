use std::collections::{hash_map::Entry, BTreeMap, HashMap};
use std::ffi::CString;

use ash::extensions::khr;
use ash::vk;
use rspirv_reflect::{self};
use spirq::{self};

use crate::render_device::RenderDevice;

// BEGIN Handle

#[derive(PartialEq, Eq, Hash)]
pub struct Handle<T> {
    id: usize,
    _phantom_data: std::marker::PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            _phantom_data: std::marker::PhantomData,
        }
    }

    pub fn null() -> Self {
        Self::new(usize::MAX)
    }

    pub fn id(&self) -> usize {
        self.id
    }

    #[allow(dead_code)]
    pub fn is_null(&self) -> bool {
        self.id == usize::MAX
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _phantom_data: self._phantom_data.clone(),
        }
    }
}

impl<T> Copy for Handle<T> {}

// END Handle

pub struct PipelineDevice {
    device: ash::Device,
    raytracing_entry: khr::RayTracingPipeline,
    pipeline_cache: vk::PipelineCache,
}

impl PipelineDevice {
    pub fn new(rd: &RenderDevice) -> PipelineDevice {
        PipelineDevice {
            device: rd.device_entry.clone(),
            raytracing_entry: rd.raytracing_pipeline_entry.clone(),
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
    gfx_pipelines_map: HashMap<(ShaderDefinition, ShaderDefinition), Handle<Pipeline>>,
    compute_pipelines_map: HashMap<ShaderDefinition, Handle<Pipeline>>,
    raytracing_pipelines_map: HashMap<ShaderDefinition, Handle<Pipeline>>,
    shader_loader: ShaderLoader,

    pipelines: Vec<Pipeline>,
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
        }
    }

    #[inline]
    fn add_pipeline(&mut self, pipeline: Pipeline) -> Handle<Pipeline> {
        let id = self.pipelines.len();
        self.pipelines.push(pipeline);
        Handle::new(id)
    }

    pub fn create_gfx_pipeline(
        &mut self,
        vs_def: ShaderDefinition,
        ps_def: ShaderDefinition,
        hack: &HackStuff,
    ) -> Option<Handle<Pipeline>> {
        // look from cache
        // If not in cache, create and push into cache
        // TODO currenty need to query twice even if cache can hit (constians_key, get)
        let key = (vs_def, ps_def);

        if !self.gfx_pipelines_map.contains_key(&key) {
            let vs = self.shader_loader.load(&self.pipeline_device, &vs_def)?;
            let ps = self.shader_loader.load(&self.pipeline_device, &ps_def)?;
            let pipeline_created = create_graphics_pipeline(&self.pipeline_device, &vs, &ps, &hack);
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
        hack: &HackStuff,
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
        hack: &HackStuff,
    ) -> Option<Handle<Pipeline>> {
        let key = ray_gen_def;

        if !self.raytracing_pipelines_map.contains_key(&key) {
            let cs = self
                .shader_loader
                .load(&self.pipeline_device, &ray_gen_def)?;
            let pipeline_created = create_raytracing_pipeline(&self.pipeline_device, &cs, &hack);
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
        if handle.is_null() {
            None
        } else {
            let id = handle.id();
            Some(&self.pipelines[id])
        }
    }

    pub fn reload_all(&mut self) {
        // TODO reload all shaders by checking file timestamps (and checksum?)
        self.gfx_pipelines_map.clear();
        self.compute_pipelines_map.clear();
        self.raytracing_pipelines_map.clear();
        self.pipelines.clear();
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
            println!("Error: Failed to reflect shader module: {:?}", refl_err);
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
        assert!(entry_points.len() == 1);
        entry_points[0].to_owned()
    };

    // Debug: print the reflect content
    {
        println!(
            "Reflection(shader: {}, entry_point: {})",
            shader_def.virtual_path, shader_def.entry_point
        );
        println!(
            "\tdesciptor_sets: {:?}",
            reflect_module.get_descriptor_sets()
        );
        if let Some(pc) = reflect_module.get_push_constant_range().unwrap_or_default() {
            println!("\tpush_consants: offset {}, size{}", pc.offset, pc.size);
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
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub struct ShaderDefinition {
    pub virtual_path: &'static str,
    pub entry_point: &'static str,
    pub stage: ShaderStage,
}

impl ShaderDefinition {
    #[allow(dead_code)]
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
}

pub struct HackStuff {
    pub bindless_size: u32, // Used to create descriptor layout
    pub set_layout_override: HashMap<u32, vk::DescriptorSetLayout>,
    pub ray_recursiion_depth: u32,
}

pub struct CompiledShader {
    //pub artifact: shaderc::CompilationArtifact,
    pub program: PipelineProgram,
}

struct IncludeHandler {}

impl hassle_rs::DxcIncludeHandler for IncludeHandler {
    fn load_source(&self, filename: String) -> Option<String> {
        let path = PathBuf::new().join("./shader").join(filename);
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
}

impl ShaderLoader {
    pub fn new() -> ShaderLoader {
        let dxc = hassle_rs::Dxc::new(None).unwrap();
        let compiler = dxc.create_compiler().unwrap();
        let library = dxc.create_library().unwrap();
        ShaderLoader {
            compiler,
            library,
            _dxc: dxc,
        }
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
                unsafe {
                    std::intrinsics::breakpoint();
                }
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
        // todo map v_path to actuall pathes
        let mut path = PathBuf::new();
        path.push("./shader/");
        path.push(&shader_def.virtual_path);
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
        let file_name = file_name_os.to_str().unwrap();
        let compiled_binary =
            self.compile_hlsl(file_name, &text, &shader_def.entry_point, shader_def.stage)?;

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
        };

        // NOTE: -fspv-debug=vulkan-with-source requires extended instruction set support form the reflector
        let args = [
            // output spirv
            "-spirv",
            // no optimization
            "-Zi",
            // NOTE: requires Google extention in vulkan
            //"-fspv-reflect",
            // Enable Raytracing ("Vulkan 1.1 with SPIR-V 1.4 is required for Raytracing" is printed by DXC)
            "-fspv-target-env=vulkan1.1spirv1.4",
        ];
        let result = self.compiler.compile(
            &blob,
            source_name,
            entry_point,
            target_profile,
            &args,
            Some(Box::new(IncludeHandler {})),
            &[],
        );

        match result {
            Err(result) => {
                let error_blob = result.0.get_error_buffer().ok()?;
                let error_string = self.library.get_blob_as_string(&error_blob);
                println!("Error: Failed to compile shader: {}", error_string);
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

// TODO file empty layouts?
fn create_merged_descriptor_set_layouts(
    device: &ash::Device,
    stages_info: &[(vk::ShaderStageFlags, &rspirv_reflect::Reflection)],
    property_map: &mut HashMap<String, PipelineDescriptorInfo>,
    hack: &HackStuff,
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
                    rspirv_reflect::BindingCount::Unbounded => hack.bindless_size,
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
                    println!("Overriding set layout for set {}", set_index);
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
                        println!(
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
    pub property_map: HashMap<String, PipelineDescriptorInfo>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub fn create_compute_pipeline(
    device: &PipelineDevice,
    shader_def: &ShaderDefinition,
    compiled: &CompiledShader,
    hack: &HackStuff,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let device = &device.device;

    let program = &compiled.program;
    //    let reflect_module = &program.reflect_module;
    let stage_info = [(vk::ShaderStageFlags::COMPUTE, &program.reflect_module)];

    // Reflection info to be collected
    let mut property_map = HashMap::new();

    // Create all set layouts used
    let set_layouts =
        create_merged_descriptor_set_layouts(device, &stage_info, &mut property_map, hack);

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
        let entry_point_c = CString::new(shader_def.entry_point.clone())
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
        property_map,
        push_constant_ranges,
    })
}

pub fn create_raytracing_pipeline(
    device: &PipelineDevice,
    ray_gen: &CompiledShader,
    hack: &HackStuff,
) -> Option<Pipeline> {
    let stages_info = [(
        vk::ShaderStageFlags::RAYGEN_KHR,
        &ray_gen.program.reflect_module,
    )];

    // Reflection info to be collect
    let mut property_map = HashMap::new();

    // Set layout for all stages
    let set_layouts =
        create_merged_descriptor_set_layouts(&device.device, &stages_info, &mut property_map, hack);

    // Push constant for all stages
    let push_constant_ranges = make_merged_push_constant_ranges(&stages_info);

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        unsafe { device.device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    let raygen = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::RAYGEN_KHR)
        .module(ray_gen.program.shader_module)
        .name(&ray_gen.program.entry_point_c);
    //let any_hit = vk::PipelineShaderStageCreateInfo::builder();
    //let miss = vk::PipelineShaderStageCreateInfo::builder();
    let stages = [*raygen];

    let raygen_group = vk::RayTracingShaderGroupCreateInfoKHR::builder()
        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
        .general_shader(0)
        .closest_hit_shader(vk::SHADER_UNUSED_KHR) // TODO
        .any_hit_shader(vk::SHADER_UNUSED_KHR) // TODO
        .intersection_shader(vk::SHADER_UNUSED_KHR)
        .build();
    let groups = [raygen_group];

    // TODO check ray tracing related flags
    let flags = vk::PipelineCreateFlags::empty();
    let create_info = vk::RayTracingPipelineCreateInfoKHR::builder()
        .flags(flags)
        .stages(&stages)
        .groups(&groups)
        .max_pipeline_ray_recursion_depth(hack.ray_recursiion_depth) // TODO rt depth
        //.library_info(todo!())
        //.dynamic_state(dynamic_state)
        .layout(layout);

    let pipeline = unsafe {
        device
            .raytracing_entry
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
        property_map,
        push_constant_ranges,
    })
}

pub fn create_graphics_pipeline(
    device: &PipelineDevice,
    vs: &CompiledShader,
    ps: &CompiledShader,
    hack: &HackStuff,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let device = &device.device;

    let stages_info = [
        (vk::ShaderStageFlags::VERTEX, &vs.program.reflect_module),
        (vk::ShaderStageFlags::FRAGMENT, &ps.program.reflect_module),
    ];

    // Reflection info to be collected
    let mut property_map = HashMap::new();

    // Create all set layouts used in all stages
    let set_layouts =
        create_merged_descriptor_set_layouts(device, &stages_info, &mut property_map, hack);

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
    let raster = vk::PipelineRasterizationStateCreateInfo::builder();
    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil =
        vk::PipelineDepthStencilStateCreateInfo::builder().depth_compare_op(vk::CompareOp::GREATER);
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::from_raw(0xFFFFFFFF));
    let attachments = [attachment.build()];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);
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
    ];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Extention: PipelineRendering
    let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::builder()
    .depth_attachment_format(vk::Format::D24_UNORM_S8_UINT)
    .stencil_attachment_format(vk::Format::D24_UNORM_S8_UINT)
    //.color_attachment_formats(&[vk::Format::R8G8B8A8_UNORM])
    ;

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
        property_map,
        push_constant_ranges,
    })
}

type NamedVec<'a, T> = Vec<(&'a str, T)>;

#[allow(dead_code)]
pub struct DescriptorSetWriteBuilder<'a> {
    buffer_views: NamedVec<'a, vk::BufferView>,
    buffer_infos: NamedVec<'a, vk::DescriptorBufferInfo>,
    image_infos: NamedVec<'a, vk::DescriptorImageInfo>,
    samplers: NamedVec<'a, vk::Sampler>,
    accel_strusts: NamedVec<'a, vk::AccelerationStructureKHR>,

    writes: Vec<vk::WriteDescriptorSet>,
}

impl Default for DescriptorSetWriteBuilder<'_> {
    fn default() -> Self {
        Self {
            buffer_views: Default::default(),
            buffer_infos: Default::default(),
            image_infos: Default::default(),
            samplers: Default::default(),
            accel_strusts: Default::default(),
            writes: Default::default(),
        }
    }
}

#[allow(dead_code)]
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

        for (name, accel_struct) in &self.accel_strusts {
            if let Some(info) = pipeline.property_map.get(*name) {
                assert!(info.set_index == set_index);
                let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                    .acceleration_structures(std::slice::from_ref(accel_struct))
                    .build();

                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .push_next(&mut as_write);
                self.writes.push(*write);
            } else {
                fn_unused(name, "accel_struct");
            }
        }

        return &self.writes;
    }

    #[inline]
    pub fn buffer(&'a mut self, name: &'a str, buffer_view: vk::BufferView) -> &mut Self {
        self.buffer_views.push((name, buffer_view));
        self
    }

    #[inline]
    pub fn constant_buffer(&mut self, name: &'a str, buffer: vk::Buffer) -> &mut Self {
        self.buffer_infos.push((
            name,
            vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
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
        self.accel_strusts.push((name, accel_struct));
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

    pub fn push<T>(mut self, value: &T) -> Self
    where
        T: Copy,
    {
        let size = std::mem::size_of::<T>();
        let offset = self.data.len();
        self.data.resize(offset + size, 0);
        self.data[offset..offset + size].copy_from_slice(unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, size)
        });
        self
    }

    pub fn pushv<T>(self, value: T) -> Self
    where
        T: Copy,
    {
        self.push(&value)
    }

    pub fn build(&self) -> &[u8] {
        &self.data
    }
}
