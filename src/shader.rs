use std::collections::{hash_map::Entry, HashMap};
use std::ffi::CString;

use ash::vk::{self, PushConstantRange};
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
    pipeline_cache: vk::PipelineCache,
}

impl PipelineDevice {
    pub fn new(rd: &RenderDevice) -> PipelineDevice {
        PipelineDevice {
            device: rd.device.clone(),
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
    shader_loader: ShaderLoader,

    pipelines: Vec<Pipeline>,
}

impl Shaders {
    pub fn new(rd: &RenderDevice) -> Shaders {
        Shaders {
            pipeline_device: PipelineDevice::new(&rd),
            gfx_pipelines_map: HashMap::new(),
            compute_pipelines_map: HashMap::new(),
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
    ) -> Option<Handle<Pipeline>> {
        let key = cs_def;

        if !self.compute_pipelines_map.contains_key(&key) {
            let cs = self.shader_loader.load(&self.pipeline_device, &cs_def)?;
            let pipeline_created = create_compute_pipeline(&self.pipeline_device, &cs_def, &cs);
            if let Some(pipeline) = pipeline_created {
                let handle = self.add_pipeline(pipeline);
                self.compute_pipelines_map.insert(key, handle);
                return Some(handle);
            }
            return None;
        }

        Some(*self.compute_pipelines_map.get(&key).unwrap())
    }

    pub fn get_pipeline(&self, handle: Handle<Pipeline>) -> Option<&Pipeline> {
        let id = handle.id();
        Some(&self.pipelines[id])
    }

    pub fn reload_all(&mut self) {
        // TODO reload all shaders by checking file timestamps (and checksum?)
        self.gfx_pipelines_map.clear();
        self.compute_pipelines_map.clear();
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

        let target_profile = match stage {
            ShaderStage::Compute => "cs_5_0",
            ShaderStage::Vert => "vs_5_0",
            ShaderStage::Frag => "ps_5_0",
        };

        // NOTE: -fspv-debug=vulkan-with-source requires extended instruction set support form the reflector
        let args = [
            // output spirv
            "-spirv",
            // no optimization
            "-Zi",
            // NOTE: requires Google extention in vulkan
            //"-fspv-reflect",
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

// TODO should be allow copy, but required to allow Handle copy
pub struct Pipeline {
    pub handle: vk::Pipeline,
    pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub layout: vk::PipelineLayout,

    // reflection info
    pub descriptor_infos: HashMap<String, PipelineDescriptorInfo>,
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub fn create_compute_pipeline(
    device: &PipelineDevice,
    shader_def: &ShaderDefinition,
    compiled: &CompiledShader,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let device = &device.device;

    let program = &compiled.program;
    let reflect_module = &program.reflect_module;

    // Create all set layouts used in compute
    let mut set_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
    let reflected_descriptor_sets = reflect_module.get_descriptor_sets().unwrap(); // todo
    let last_set = reflected_descriptor_sets
        .keys()
        .reduce(|last_set, set| if set > last_set { set } else { last_set })
        .map(|arg| *arg);
    if let Some(last_set) = last_set {
        set_layouts.resize((last_set + 1) as usize, vk::DescriptorSetLayout::null());
    }
    reflected_descriptor_sets
        .iter()
        .for_each(|(set, bindings)| {
            let bindings_info = bindings
                .iter()
                .map(
                    |(binding, descriptor_info)| vk::DescriptorSetLayoutBinding {
                        binding: *binding,
                        descriptor_type: vk::DescriptorType::from_raw(descriptor_info.ty.0 as i32),
                        descriptor_count: match descriptor_info.binding_count {
                            rspirv_reflect::BindingCount::One => 1,
                            rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                            rspirv_reflect::BindingCount::Unbounded => todo!(),
                        },
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        p_immutable_samplers: std::ptr::null(),
                    },
                )
                .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings_info);
            match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                Ok(set_layout) => {
                    assert!(set_layouts.len() as u32 > *set);
                    set_layouts[*set as usize] = set_layout;
                }
                Err(_) => todo!(),
            }
        });

    // Create all push constant range use in shader
    let push_constant_range = {
        let pc_range = reflect_module.get_push_constant_range().unwrap();
        pc_range.map(|info| vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: info.offset,
            size: info.size,
        })
    };
    let mut push_constant_ranges = Vec::<PushConstantRange>::new();
    if push_constant_range.is_some() {
        push_constant_ranges.push(push_constant_range.unwrap());
    }

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

    // Reflection info
    let descriptor_infos = HashMap::new();

    Some(Pipeline {
        handle: pipeline,
        set_layouts,
        layout,
        descriptor_infos,
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

    let shaders = [
        (vk::ShaderStageFlags::VERTEX, vs),
        (vk::ShaderStageFlags::FRAGMENT, ps),
    ];

    // Reflection info for descriptors
    let mut descriptor_infos = HashMap::new();

    // Collect and merge descriptor set from all stages
    type MergedSet = HashMap<u32, vk::DescriptorSetLayoutBinding>;
    type MergedLayout = HashMap<u32, MergedSet>;
    let mut merged_layout = MergedLayout::new();
    let mut last_set = 0;
    for shader in shaders {
        let stage = shader.0;
        let reflect = &shader.1.program.reflect_module;
        for (set, set_bindings) in reflect.get_descriptor_sets().unwrap().iter() {
            last_set = if *set > last_set { *set } else { last_set };
            let merged_set = match merged_layout.entry(*set) {
                Entry::Occupied(o) => o.into_mut(),
                Entry::Vacant(v) => v.insert(MergedSet::new()),
            };
            for (binding, descriptor_info) in set_bindings {
                let descriptor_type = vk::DescriptorType::from_raw(descriptor_info.ty.0 as i32);
                let count = match descriptor_info.binding_count {
                    rspirv_reflect::BindingCount::One => 1,
                    rspirv_reflect::BindingCount::StaticSized(size) => size as u32,
                    rspirv_reflect::BindingCount::Unbounded => hack.bindless_size,
                };
                if let Some(binding_info) = merged_set.get_mut(binding) {
                    assert!(descriptor_type == binding_info.descriptor_type);
                    assert!(count == binding_info.descriptor_count); // really?
                    binding_info.stage_flags |= stage;
                } else {
                    // TODO add extension flags for bindless textures
                    let binding_info = vk::DescriptorSetLayoutBinding::builder()
                        .binding(*binding)
                        .descriptor_type(descriptor_type)
                        .descriptor_count(count)
                        .stage_flags(stage);
                    merged_set.insert(*binding, binding_info.build());
                }

                // Collect properties for later use
                let prop_info = PipelineDescriptorInfo {
                    set_index: *set,
                    binding_index: *binding,
                    descriptor_type,
                };
                match descriptor_infos.entry(descriptor_info.name.clone()) {
                    Entry::Occupied(entry) => {
                        assert_eq!(entry.get(), &prop_info);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(prop_info);
                    }
                }
            }
        }
    }

    let mut set_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
    if merged_layout.len() > 0 {
        set_layouts.resize((last_set + 1) as usize, vk::DescriptorSetLayout::null());
        merged_layout.drain().for_each(|(set, bindings)| {
            if hack.set_layout_override.contains_key(&set) {
                set_layouts[set as usize] = hack.set_layout_override[&set];
                println!("Overriding set layout for set {}", set);
                return;
            }

            let bindings = bindings
                .into_values()
                .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            assert!(set_layouts.len() as u32 > set);
            match unsafe { device.create_descriptor_set_layout(&create_info, None) } {
                Ok(set_layout) => {
                    set_layouts[set as usize] = set_layout;
                }
                Err(_) => todo!(),
            }
        });
    }

    // Fill empty set layouts
    for set_layout in &mut set_layouts {
        if *set_layout == vk::DescriptorSetLayout::null() {
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder();
            *set_layout =
                unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();
        }
    }

    // Create all push constant ranges use in all stages
    let mut push_constant_ranges = Vec::<PushConstantRange>::new();
    for (stage_flag, shader) in shaders {
        if let Ok(Some(info)) = shader.program.reflect_module.get_push_constant_range() {
            // merge to prev range
            let mut merged = false;
            for range in &mut push_constant_ranges {
                if (range.offset == info.offset) && (range.size == info.size) {
                    range.stage_flags |= stage_flag;
                    merged = true;
                    break;
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
        descriptor_infos,
        push_constant_ranges,
    })
}

type NamedVec<'a, T> = Vec<(&'a str, T)>;

pub struct DescriptorSetWriteBuilder<'a> {
    buffer_views: NamedVec<'a, vk::BufferView>,
    buffer_infos: NamedVec<'a, vk::DescriptorBufferInfo>,
    image_infos: NamedVec<'a, vk::DescriptorImageInfo>,
    images: NamedVec<'a, vk::Image>,
    samplers: NamedVec<'a, vk::Sampler>,

    writes: Vec<vk::WriteDescriptorSet>,
}

impl<'a> DescriptorSetWriteBuilder<'a> {
    pub fn new() -> Self {
        Self {
            buffer_views: Vec::new(),
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub fn build(
        &mut self,
        pipeline: &Pipeline,
        set: vk::DescriptorSet,
    ) -> &[vk::WriteDescriptorSet] {
        for (name, buffer_view) in &self.buffer_views {
            if let Some(info) = pipeline.descriptor_infos.get(*name) {
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .texel_buffer_view(std::slice::from_ref(buffer_view));
                self.writes.push(*write);
            }
        }

        for (name, buffer_info) in &self.buffer_infos {
            if let Some(info) = pipeline.descriptor_infos.get(*name) {
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .buffer_info(std::slice::from_ref(buffer_info));
                self.writes.push(*write);
            }
        }

        for (name, image_info) in &self.image_infos {
            if let Some(info) = pipeline.descriptor_infos.get(*name) {
                let write = vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(info.binding_index)
                    .dst_array_element(0)
                    .descriptor_type(info.descriptor_type)
                    .image_info(std::slice::from_ref(image_info));
                self.writes.push(*write);
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
