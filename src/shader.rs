use std::ffi::CString;

use ash::vk::{self, PushConstantRange};
use rspirv_reflect::{self};

use crate::render_device::RenderDevice;

pub struct PipelineDevice {
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
    pipeline_cache: vk::PipelineCache,
}

impl PipelineDevice {
    pub fn new(rd: &RenderDevice) -> PipelineDevice {
        PipelineDevice {
            device: rd.device.clone(),
            pipeline_cache: vk::PipelineCache::null(),
            descriptor_pool: rd.descriptor_pool,
        }
    }
}

// AKA shader
pub struct PipelineProgram {
    pub shader_module: vk::ShaderModule,
    pub reflect_module: rspirv_reflect::Reflection,
    pub entry_point_c: CString,
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
        Ok(refl) => { Some(refl) },
        Err(refl_err) => {
            println!("Error: Failed to reflect shader module: {:?}", refl_err);
            None
        },
    }?;

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

    let entry_point_c = shader_def.entry_point.to_cstring();

    Some(PipelineProgram {
        shader_module,
        reflect_module,
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

pub enum ShaderStage {
    Compute,
    Vert,
    Frag,
}

pub struct ShaderDefinition {
    pub virtual_path: String,
    pub entry_point: String,
    pub stage: ShaderStage,
}

impl ShaderDefinition {
    pub fn new(virtual_path: &str, entry_point: &str, stage: ShaderStage) -> ShaderDefinition {
        ShaderDefinition {
            virtual_path: virtual_path.to_string(),
            entry_point: entry_point.to_string(),
            stage,
        }
    }
}

pub struct CompiledShader {
    //pub artifact: shaderc::CompilationArtifact,
    pub program: PipelineProgram,
}

pub fn load_shader(
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
    //options.set_auto_bind_uniforms(true);
    let target_profile = match shader_def.stage {
        ShaderStage::Compute => "cs_5_0",
        ShaderStage::Vert => "vs_5_0",
        ShaderStage::Frag => "ps_5_0",
    };
    // NOTE: -fspv-debug=vulkan-with-source requires extended instruction set support form the reflector
    // NOTE: -fspv-reflect requires Google extention in vulkan 
    let compile_result = hassle_rs::compile_hlsl(
        file_name,
        &text,
        &shader_def.entry_point,
        target_profile,
        //&["-spirv"],
        //&["-spirv", "-Zi", "-fspv-reflect"], 
        &["-spirv", "-Zi"], 
        &[],
    );
    
    let compiled_binary = match compile_result {
        Ok(bin) => bin,
        Err(reason) => {
            println!("Shaer compiled binay is not valid: {}", reason);
            return None;
        }
    };

    let program = match create_pipeline_program(device, &compiled_binary, &shader_def) {
        Some(program) => program,
        None => return None,
    };

    Some(CompiledShader { program: program })
}

pub struct Pipeline {
    pub handle: vk::Pipeline,
    //pub set_layouts: Vec<vk::DescriptorSetLayout>,
    pub layout: vk::PipelineLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

pub fn create_compute_pipeline(
    device: &PipelineDevice,
    shader_def: &ShaderDefinition,
    compiled: &CompiledShader,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let descriptor_pool = device.descriptor_pool;
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
        pc_range.map(|info| { 
            vk::PushConstantRange{ stage_flags: vk::ShaderStageFlags::COMPUTE, offset: info.offset, size: info.size }
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

    // Create all used descriptor sets
    let descriptor_sets = {
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        unsafe { device.allocate_descriptor_sets(&create_info) }.ok()?
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
        //set_layouts,
        layout,
        descriptor_sets,
    })
}

pub fn create_graphics_pipeline(
    device: &PipelineDevice,
    vs: &CompiledShader,
    ps: &CompiledShader,
) -> Option<Pipeline> {
    let pipeline_cache = device.pipeline_cache;
    let descriptor_pool = device.descriptor_pool;
    let device = &device.device;

    let shaders = [
        (vk::ShaderStageFlags::VERTEX, vs),
        (vk::ShaderStageFlags::FRAGMENT, ps),
    ];

    // Collect and merge descriptor set from all stages
    use std::collections::hash_map::{Entry, HashMap};
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
                    rspirv_reflect::BindingCount::Unbounded => todo!(),
                };
                if let Some(binding_info) = merged_set.get_mut(binding) {
                    assert!(descriptor_type == binding_info.descriptor_type);
                    assert!(count == binding_info.descriptor_count); // really?
                    binding_info.stage_flags |= stage;
                } else {
                    let binding_info = vk::DescriptorSetLayoutBinding::builder()
                        .binding(*binding)
                        .descriptor_type(descriptor_type)
                        .descriptor_count(count)
                        .stage_flags(stage);
                    merged_set.insert(*binding, binding_info.build());
                }
            }
        }
    }

    let mut set_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
    if merged_layout.len() > 0 {
        set_layouts.resize((last_set + 1) as usize, vk::DescriptorSetLayout::null());
        merged_layout.drain().for_each(|(set, bindings)| {
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

    // Create all push constant ranges use in all stages 
    let mut push_constant_ranges = Vec::<PushConstantRange>::new();
    for (stage_flag, shader) in shaders {
        if let Ok(Some(info)) = shader.program.reflect_module.get_push_constant_range() {
            push_constant_ranges.push( 
                vk::PushConstantRange{
                    stage_flags: stage_flag, offset: info.offset, size: info.size
                }
            );
        }
    };

    // Create pipeline layout
    let layout = {
        let create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
        unsafe { device.create_pipeline_layout(&create_info, None) }.ok()?
    };

    // Create all used descriptor sets
    let descriptor_sets = {
        if set_layouts.len() == 0 {
            Vec::new()
        } else {
            let create_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
            unsafe { device.allocate_descriptor_sets(&create_info) }.ok()?
        }
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
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
    .depth_write_enable(true)
    .depth_test_enable(true)
    .depth_compare_op(vk::CompareOp::GREATER);
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::from_raw(0xFFFFFFFF));
    let attachments = [attachment.build()];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);


    // Extention: PipelineRendering 
    let mut pipeline_rendering = vk::PipelineRenderingCreateInfo::builder()
    .depth_attachment_format(vk::Format::D16_UNORM)
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
        .push_next(&mut pipeline_rendering)
        ;
    let create_infos = [create_info.build()];
    let result =
        unsafe { device.create_graphics_pipelines(pipeline_cache, &create_infos, None) }.ok()?;
    let pipeline = result[0];

    Some(Pipeline {
        handle: pipeline,
        //set_layouts,
        layout,
        descriptor_sets,
    })
}
