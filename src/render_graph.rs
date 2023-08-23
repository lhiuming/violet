use std::collections::HashMap;
use std::hash::Hash;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

use ash::vk;
use glam::UVec3;

use crate::command_buffer::CommandBuffer;
use crate::render_device::{
    AccelerationStructure, Buffer, BufferDesc, RenderDevice, ShaderBindingTableFiller, Texture,
    TextureDesc, TextureView, TextureViewDesc,
};
use crate::shader::{
    self, Handle, Pipeline, PushConstantsBuilder, ShaderDefinition, Shaders, ShadersConfig,
};

pub struct RGHandle<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGHandle<T> {
    fn new(id: usize) -> Self {
        RGHandle {
            id,
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Clone for RGHandle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Copy for RGHandle<T> {}

impl<T> PartialEq for RGHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for RGHandle<T> {}

impl<T> Hash for RGHandle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

pub struct RGTemporal<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGTemporal<T> {
    fn new(id: usize) -> Self {
        RGTemporal {
            id,
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Clone for RGTemporal<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _phantom: PhantomData::default(),
        }
    }
}

impl<T> Copy for RGTemporal<T> {}

#[derive(PartialEq, Eq)]
pub enum RenderPassType {
    Graphics,
    Compute,
    RayTracing,
    Copy,
    Present,
}

impl RenderPassType {
    pub fn to_bind_point(&self) -> Option<vk::PipelineBindPoint> {
        let bind_point = match self {
            RenderPassType::Graphics => vk::PipelineBindPoint::GRAPHICS,
            RenderPassType::Compute => vk::PipelineBindPoint::COMPUTE,
            RenderPassType::RayTracing => vk::PipelineBindPoint::RAY_TRACING_KHR,
            RenderPassType::Copy => return None,
            RenderPassType::Present => return None,
        };
        Some(bind_point)
    }
}

#[derive(Clone, Copy)]
pub enum ColorLoadOp {
    Load,
    Clear(vk::ClearColorValue),
    DontCare,
}

#[derive(Clone, Copy)]
pub struct ColorTarget {
    pub view: RGHandle<TextureView>,
    pub load_op: ColorLoadOp,
}

#[derive(Clone, Copy)]
pub enum DepthLoadOp {
    Load,
    Clear(vk::ClearDepthStencilValue),
}

#[derive(Clone, Copy)]
pub struct DepthStencilTarget {
    pub view: RGHandle<TextureView>,
    pub load_op: DepthLoadOp,
    pub store_op: vk::AttachmentStoreOp,
}

struct RaytracingPassData {
    raygen_shader: Option<ShaderDefinition>,
    miss_shader: Option<ShaderDefinition>,
    chit_shader: Option<ShaderDefinition>,
    dimension: UVec3,
}

pub struct RenderPass<'a> {
    //params: dyn Paramter,
    //read_textures: Vec<RGHandle<Texture>>,
    //input_buffers: RGHandle<>
    //mutable_textures: Vec<RGHandle<Texture>>,
    name: String,
    ty: RenderPassType,
    pipeline: Handle<Pipeline>,
    descriptor_set_index: u32,
    textures: Vec<(&'static str, RGHandle<TextureView>)>,
    buffers: Vec<(&'static str, RGHandle<Buffer>)>,
    rw_buffers: Vec<(&'static str, RGHandle<Buffer>)>,
    accel_structs: Vec<(&'static str, RGHandle<AccelerationStructure>)>,
    color_targets: Vec<ColorTarget>,
    depth_stencil: Option<DepthStencilTarget>,
    rw_textures: Vec<(&'static str, RGHandle<TextureView>)>,
    push_constants: PushConstantsBuilder,
    copy_src: Option<RGHandle<TextureView>>,
    copy_dst: Option<RGHandle<TextureView>>,
    present_texture: Option<RGHandle<TextureView>>,
    render: Option<Box<dyn 'a + FnOnce(&CommandBuffer, &Shaders, &RenderPass)>>,

    // Varying data
    raytracing: Option<RaytracingPassData>,
}

impl RenderPass<'_> {
    pub fn new(name: &str, ty: RenderPassType) -> Self {
        RenderPass {
            name: String::from(name),
            ty,
            pipeline: Handle::null(),
            descriptor_set_index: 0,
            textures: Vec::new(),
            buffers: Vec::new(),
            rw_buffers: Vec::new(),
            accel_structs: Vec::new(),
            color_targets: Vec::new(),
            depth_stencil: None,
            rw_textures: Vec::new(),
            push_constants: PushConstantsBuilder::new(),
            copy_src: None,
            copy_dst: None,
            present_texture: None,
            render: None,

            raytracing: None,
        }
    }

    pub fn get_texture(&self, name: &str) -> RGHandle<TextureView> {
        for (n, t) in &self.textures {
            if n == &name {
                return *t;
            }
        }
        panic!("Cannot find texture with name: {}", name);
    }
}

pub trait PassBuilderTrait<'a> {
    fn inner_opt(&mut self) -> &mut Option<RenderPass<'a>>;

    fn render_graph(&mut self) -> &mut RenderGraphBuilder<'a>;

    fn inner(&mut self) -> &mut RenderPass<'a> {
        self.inner_opt().as_mut().unwrap()
    }

    fn done(&mut self) {
        if self.inner_opt().is_none() {
            panic!("RenderPassBuilder::done is called multiple times!");
        }
        let inner = self.inner_opt().take();
        self.render_graph().add_pass(inner.unwrap());
    }

    // Binding texture to per-pass descriptor set
    fn texture(&mut self, name: &'static str, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().textures.push((name, texture));
        self
    }

    // Binding rw texture to per-pass descriptor set
    fn rw_texture(&mut self, name: &'static str, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().rw_textures.push((name, texture));
        self
    }

    fn buffer(&mut self, name: &'static str, buffer: RGHandle<Buffer>) -> &mut Self {
        self.inner().buffers.push((name, buffer));
        self
    }

    fn rw_buffer(&mut self, name: &'static str, buffer: RGHandle<Buffer>) -> &mut Self {
        self.inner().rw_buffers.push((name, buffer));
        self
    }

    // Binding acceleration structure to per-pass descriptor set
    fn accel_struct(
        &mut self,
        name: &'static str,
        accel_struct: RGHandle<AccelerationStructure>,
    ) -> &mut Self {
        self.inner().accel_structs.push((name, accel_struct));
        self
    }

    fn push_constant<T>(&mut self, value: &T) -> &mut Self
    where
        T: Copy,
    {
        self.inner().push_constants.push_inplace::<T>(value);
        self
    }
}

pub struct RenderPassBuilder<'a, 'b> {
    inner: Option<RenderPass<'a>>,
    render_graph: &'b mut RenderGraphBuilder<'a>,
}

impl<'a, 'b> PassBuilderTrait<'a> for RenderPassBuilder<'a, 'b> {
    fn inner_opt(&mut self) -> &mut Option<RenderPass<'a>> {
        &mut self.inner
    }

    fn render_graph(&mut self) -> &mut RenderGraphBuilder<'a> {
        self.render_graph
    }
}

// TODO generate setter with macro?
impl<'a, 'b> RenderPassBuilder<'a, 'b> {
    pub fn new(
        render_graph: &'b mut RenderGraphBuilder<'a>,
        name: &str,
        ty: RenderPassType,
    ) -> Self {
        RenderPassBuilder {
            inner: Some(RenderPass::new(name, ty)),
            render_graph,
        }
    }

    /*
    fn inner(&mut self) -> &mut RenderPass<'a> {
        self.inner.as_mut().unwrap()
    }

    fn done(&mut self) {
        if self.inner.is_none() {
            panic!("RenderPassBuilder::done is called multiple times!");
        }
        let inner = self.inner.take();
        self.render_graph.add_pass(inner.unwrap());
    }
    */

    pub fn pipeline(&mut self, pipeline: Handle<Pipeline>) -> &mut Self {
        self.inner().pipeline = pipeline;
        self
    }

    // Binding an external descriptor set
    pub fn descritpro_set(&mut self, _: u32, _: vk::DescriptorSet) -> &mut Self {
        println!("Warning: deprecated API!");
        self
    }

    pub fn descritpro_sets(&mut self, _: &[(u32, vk::DescriptorSet)]) -> &mut Self {
        println!("Warning: deprecated API!");
        self
    }

    pub fn color_targets(&mut self, rts: &[ColorTarget]) -> &mut Self {
        self.inner().color_targets.clear();
        self.inner().color_targets.extend_from_slice(rts);
        self
    }

    pub fn depth_stencil(&mut self, ds: DepthStencilTarget) -> &mut Self {
        self.inner().depth_stencil = Some(ds);
        self
    }

    // Index for the per-pass descriptor set
    pub fn descriptor_set_index(&mut self, index: u32) -> &mut Self {
        self.inner().descriptor_set_index = index;
        self
    }

    /*

    pub fn push_constant<T>(&mut self, value: &T) -> &mut Self
    where
        T: Copy,
    {
        self.inner().push_constants.push_inplace::<T>(value);
        self
    }

    // Binding texture to per-pass descriptor set
    pub fn texture(mut self, name: &'a str, texture: RGHandle<TextureView>) -> Self {
        self.inner().textures.push((name, texture));
        self
    }

    // Binding rw texture to per-pass descriptor set
    pub fn rw_texture(mut self, name: &'a str, texture: RGHandle<TextureView>) -> Self {
        self.inner().rw_textures.push((name, texture));
        self
    }

    pub fn buffer(mut self, name: &'a str, buffer: RGHandle<Buffer>) -> Self {
        self.inner().buffers.push((name, buffer));
        self
    }

    pub fn rw_buffer(mut self, name: &'a str, buffer: RGHandle<Buffer>) -> Self {
        self.inner().rw_buffers.push((name, buffer));
        self
    }

    // Binding acceleration structure to per-pass descriptor set
    pub fn accel_struct(
        mut self,
        name: &'a str,
        accel_struct: RGHandle<AccelerationStructure>,
    ) -> Self {
        self.inner().accel_structs.push((name, accel_struct));
        self
    }

    */

    pub fn copy_src(&mut self, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().copy_src = Some(texture);
        self
    }

    pub fn copy_dst(&mut self, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().copy_dst = Some(texture);
        self
    }

    pub fn present_texture(&mut self, texture: RGHandle<TextureView>) -> &mut Self {
        self.inner().present_texture = Some(texture);
        self
    }

    pub fn render<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&CommandBuffer, &Shaders, &RenderPass) + 'a,
    {
        self.inner().render = Some(Box::new(f));
        self
    }
}

impl Drop for RenderPassBuilder<'_, '_> {
    fn drop(&mut self) {
        // Call Self::done automatically if not called mannually
        if self.inner.is_some() {
            //println!("RenderPassBuilder for \"{}\" is dropped without calling done!",self.inner().name);
            self.done();
        }
    }
}

pub struct RaytracingPassBuilder<'a, 'b> {
    inner: Option<RenderPass<'a>>,
    render_graph: &'b mut RenderGraphBuilder<'a>,
}

impl Drop for RaytracingPassBuilder<'_, '_> {
    fn drop(&mut self) {
        // Call Self::done automatically if not called mannually
        if self.inner.is_some() {
            self.done();
        }
    }
}

impl<'a, 'b> PassBuilderTrait<'a> for RaytracingPassBuilder<'a, 'b> {
    fn inner_opt(&mut self) -> &mut Option<RenderPass<'a>> {
        &mut self.inner
    }

    fn render_graph(&mut self) -> &mut RenderGraphBuilder<'a> {
        self.render_graph
    }
}

impl<'a, 'b> RaytracingPassBuilder<'a, 'b> {
    pub fn new(render_graph: &'b mut RenderGraphBuilder<'a>, name: &str) -> Self {
        let mut pass = RenderPass::new(name, RenderPassType::RayTracing);
        pass.raytracing = Some(RaytracingPassData {
            raygen_shader: None,
            miss_shader: None,
            chit_shader: None,
            dimension: UVec3::new(0, 0, 0),
        });
        Self {
            inner: Some(pass),
            render_graph,
        }
    }

    fn inner(&mut self) -> &mut RaytracingPassData {
        self.inner.as_mut().unwrap().raytracing.as_mut().unwrap()
    }

    pub fn raygen_shader(&mut self, path: &'static str) -> &mut Self {
        self.inner().raygen_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::RayGen,
        });
        self
    }

    pub fn miss_shader(&mut self, path: &'static str) -> &mut Self {
        self.inner().miss_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::Miss,
        });
        self
    }

    pub fn closest_hit_shader(&mut self, path: &'static str) -> &mut Self {
        self.inner().chit_shader.replace(ShaderDefinition {
            virtual_path: path,
            entry_point: "main",
            stage: shader::ShaderStage::ClosestHit,
        });
        self
    }

    pub fn dimension(&mut self, width: u32, height: u32, depth: u32) -> &mut Self {
        self.inner().dimension = UVec3::new(width, height, depth);
        self
    }
}

struct ResPool<K, V>
where
    K: Eq + Hash,
{
    map: HashMap<K, Vec<V>>,
}

impl<K, V> ResPool<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn push(&mut self, key: K, v: V) {
        self.map.entry(key).or_insert_with(Vec::new).push(v);
    }

    pub fn pop(&mut self, key: &K) -> Option<V> {
        let pooled = self.map.get_mut(key);
        if let Some(pooled) = pooled {
            pooled.pop()
        } else {
            None
        }
    }
}

// An internal resource type for ray tracing passes.
// This kind of passes have just a few shander handles, and no extra data in shader binding tables.
struct PassShaderBindingTable {
    sbt: Buffer,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl PassShaderBindingTable {
    fn new(rd: &RenderDevice) -> Self {
        let handle_size = rd.physical_device.shader_group_handle_size() as u64;
        let stride = std::cmp::max(
            handle_size,
            rd.physical_device.shader_group_base_alignment() as u64,
        );

        let sbt = rd
            .create_buffer(BufferDesc {
                size: handle_size + stride * 2,
                usage: vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                memory_property: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
            })
            .unwrap();

        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap(),
            stride: handle_size,
            size: handle_size,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + stride,
            stride: handle_size,
            size: handle_size,
        };

        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt.device_address.unwrap() + stride * 2,
            stride: handle_size,
            size: handle_size,
        };

        Self {
            sbt,
            raygen_region,
            miss_region,
            hit_region,
        }
    }

    fn update_shader_group_handles(
        &mut self,
        rd: &RenderDevice,
        pipeline: &Pipeline,
        has_hit: bool,
    ) {
        // TODO check shader change
        let group_count = if has_hit { 3 } else { 2 };
        let handle_data = rd.get_ray_tracing_shader_group_handles(pipeline.handle, 0, group_count);
        let mut filler = ShaderBindingTableFiller::new(&rd.physical_device, self.sbt.data);
        filler.write_handles(&handle_data, 0, 1);
        filler.start_group();
        filler.write_handles(&handle_data, 1, 1);
        if has_hit {
            filler.start_group();
            filler.write_handles(&handle_data, 2, 1);
        }
    }
}

const RG_MAX_SET: u32 = 1024;

pub struct RenderGraphCache {
    // Resused VK stuffs
    vk_descriptor_pool: vk::DescriptorPool,

    // buffered VK objects
    free_vk_descriptor_sets: Vec<vk::DescriptorSet>,

    // Temporal Resources
    temporal_textures: HashMap<usize, Texture>,
    temporal_buffers: HashMap<usize, Buffer>,
    next_temporal_id: usize,

    // Resource pool
    texture_pool: ResPool<TextureDesc, Texture>,
    texture_view_pool: ResPool<(Texture, TextureViewDesc), TextureView>,
    buffer_pool: ResPool<BufferDesc, Buffer>,
    sbt_pool: ResPool<u32, PassShaderBindingTable>,
}

impl RenderGraphCache {
    pub fn new(rd: &RenderDevice) -> Self {
        let vk_descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            RG_MAX_SET,
            &[vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1024,
            }],
        );
        Self {
            vk_descriptor_pool,
            free_vk_descriptor_sets: Vec::new(),
            temporal_textures: HashMap::new(),
            temporal_buffers: HashMap::new(),
            next_temporal_id: 0,
            texture_pool: ResPool::new(),
            texture_view_pool: ResPool::new(),
            buffer_pool: ResPool::new(),
            sbt_pool: ResPool::new(),
        }
    }

    fn allocate_dessriptor_set(
        &mut self,
        rd: &RenderDevice,
        set_layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let layouts = [set_layout];
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.vk_descriptor_pool)
            .set_layouts(&layouts);
        unsafe {
            match rd.device_entry.allocate_descriptor_sets(&create_info) {
                Ok(sets) => sets[0],
                Err(e) => {
                    panic!("Failed to allocate descriptor set: {:?}", e);
                }
            }
        }
    }

    fn release_descriptor_set(&mut self, rd: &RenderDevice, descriptor_set: vk::DescriptorSet) {
        self.free_vk_descriptor_sets.push(descriptor_set);

        // TODO free after command buffer is done (fence)
        let buffer_limit = (RG_MAX_SET / 2) as usize;
        let release_heuristic = buffer_limit / 2;
        if self.free_vk_descriptor_sets.len() > buffer_limit {
            let old_sets = &self.free_vk_descriptor_sets[0..release_heuristic];
            unsafe {
                rd.device_entry
                    .free_descriptor_sets(self.vk_descriptor_pool, &old_sets)
                    .expect("Failed to free descriptor set");
            }
            self.free_vk_descriptor_sets.drain(0..release_heuristic);
        }
    }
}

struct RenderGraphExecuteContext {
    // Created Resources
    pub textures: Vec<Option<Texture>>,          // by handle::id
    pub texture_views: Vec<Option<TextureView>>, // by handle::id
    pub buffers: Vec<Option<Buffer>>,            // by handle::id

    // Per-pass descriptor set
    pub descriptor_sets: Vec<vk::DescriptorSet>, // by pass index

    // Per-pass shader binding tables
    pub shader_binding_tables: Vec<Option<PassShaderBindingTable>>, // by pass index
}

impl RenderGraphExecuteContext {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            texture_views: Vec::new(),
            buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            shader_binding_tables: Vec::new(),
        }
    }
}

enum RenderResource<V, E> {
    Virtual(V),      // a.k.a. Transient Resources
    Temporal(usize), // Index to the cache
    External(E),
}

struct VirtualTextureView {
    pub texture: RGHandle<Texture>,
    pub desc: Option<TextureViewDesc>, // if none, use ::auto::(texture::desc)
}

pub enum ResTypeEnum {
    Texture,
    Buffer,
}

pub trait ResType {
    fn get_enum() -> ResTypeEnum;
}

impl ResType for Texture {
    fn get_enum() -> ResTypeEnum {
        ResTypeEnum::Texture
    }
}

impl ResType for Buffer {
    fn get_enum() -> ResTypeEnum {
        ResTypeEnum::Buffer
    }
}

// A Render Graph to handle resource transitions automatically
pub struct RenderGraphBuilder<'a> {
    passes: Vec<RenderPass<'a>>,

    shader_config: shader::ShadersConfig,

    // Descriptor sets that would be bound for all passes
    // Exist for ergonomics reason; descriptors set like per-frame stuffs can be specify as this.
    global_descriptor_sets: Vec<(u32, vk::DescriptorSet)>,

    transient_to_temporal_textures: HashMap<RGHandle<Texture>, RGTemporal<Texture>>,
    transient_to_temporal_buffers: HashMap<RGHandle<Buffer>, RGTemporal<Buffer>>,

    // Array indexed by RGHandle
    textures: Vec<RenderResource<TextureDesc, Texture>>,
    texture_views: Vec<RenderResource<VirtualTextureView, TextureView>>,
    buffers: Vec<RenderResource<BufferDesc, Buffer>>,
    accel_structs: Vec<RenderResource<(), AccelerationStructure>>,

    hack_frame_index: u32,
}

// Private stuff
impl<'a> RenderGraphBuilder<'a> {
    fn is_virtual<T>(&self, handle: RGHandle<T>) -> bool
    where
        T: ResType,
    {
        match T::get_enum() {
            ResTypeEnum::Texture => match self.textures.get(handle.id) {
                Some(RenderResource::Virtual(_)) => true,
                _ => false,
            },
            ResTypeEnum::Buffer => match self.buffers.get(handle.id) {
                Some(RenderResource::Virtual(_)) => true,
                _ => false,
            },
        }
    }

    fn find_converted_temporal<T>(&self, temporal_handle: RGTemporal<T>) -> Option<RGHandle<T>>
    where
        T: ResType,
    {
        let temporal_id = temporal_handle.id;

        match T::get_enum() {
            ResTypeEnum::Texture => {
                for i in 0..self.textures.len() {
                    if let RenderResource::Temporal(id) = &self.textures[i] {
                        if *id == temporal_id {
                            return Some(RGHandle::new(i));
                        }
                    }
                }
            }
            ResTypeEnum::Buffer => {
                for i in 0..self.buffers.len() {
                    if let RenderResource::Temporal(id) = &self.buffers[i] {
                        if *id == temporal_id {
                            return Some(RGHandle::new(i));
                        }
                    }
                }
            }
        };

        None
    }
}

// Interface
impl<'a> RenderGraphBuilder<'a> {
    pub fn new() -> Self {
        Self::new_with_shader_config(shader::ShadersConfig::default())
    }

    pub fn new_with_shader_config(shader_config: ShadersConfig) -> Self {
        Self {
            passes: Vec::new(),

            shader_config,

            global_descriptor_sets: Vec::new(),

            transient_to_temporal_textures: HashMap::new(),
            transient_to_temporal_buffers: HashMap::new(),

            textures: Vec::new(),
            texture_views: Vec::new(),
            buffers: Vec::new(),
            accel_structs: Vec::new(),

            hack_frame_index: 0,
        }
    }

    pub fn set_frame_index(&mut self, frame_index: u32) {
        self.hack_frame_index = frame_index;
    }

    pub fn create_texutre(&mut self, desc: TextureDesc) -> RGHandle<Texture> {
        let id = self.textures.len();
        self.textures.push(RenderResource::Virtual(desc));
        RGHandle::new(id)
    }

    pub fn create_texture_view(
        &mut self,
        texture: RGHandle<Texture>,
        desc: Option<TextureViewDesc>,
    ) -> RGHandle<TextureView> {
        let id = self.texture_views.len();
        self.texture_views
            .push(RenderResource::Virtual(VirtualTextureView {
                texture,
                desc,
            }));
        RGHandle::new(id)
    }

    pub fn register_texture(&mut self, texture: Texture) -> RGHandle<Texture> {
        // Check if already registered
        // TOOD make this routine part of self.textures
        for handle_id in 0..self.textures.len() {
            let texture_resource = &self.textures[handle_id];
            if let RenderResource::External(prev_tex) = texture_resource {
                if *prev_tex == texture {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the texture
        let id = self.textures.len();
        self.textures.push(RenderResource::External(texture));
        RGHandle::new(id)
    }

    pub fn register_texture_view(&mut self, texture_view: TextureView) -> RGHandle<TextureView> {
        // Check is already registered
        for handle_id in 0..self.texture_views.len() {
            let view_resource = &self.texture_views[handle_id];
            if let RenderResource::External(prev_texture_view) = view_resource {
                if *prev_texture_view == texture_view {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register underlying texture (for dependecy tracking)
        self.register_texture(texture_view.texture);

        // Register the view
        let id = self.texture_views.len();
        self.texture_views
            .push(RenderResource::External(texture_view));
        RGHandle::new(id)
    }

    pub fn create_buffer(&mut self, desc: BufferDesc) -> RGHandle<Buffer> {
        let id = self.buffers.len();
        self.buffers.push(RenderResource::Virtual(desc));
        RGHandle::new(id)
    }

    pub fn register_buffer(&mut self, buffer: Buffer) -> RGHandle<Buffer> {
        // Check if already registered
        for handle_id in 0..self.buffers.len() {
            let res = &self.buffers[handle_id];
            if let RenderResource::External(prev_buf) = res {
                if *prev_buf == buffer {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the buffer
        let id = self.buffers.len();
        self.buffers.push(RenderResource::External(buffer));
        RGHandle::new(id)
    }

    pub fn register_accel_struct(
        &mut self,
        accel_struct: AccelerationStructure,
    ) -> RGHandle<AccelerationStructure> {
        // Check is already registered
        for handle_id in 0..self.accel_structs.len() {
            let view_resource = &self.accel_structs[handle_id];
            if let RenderResource::External(prev) = view_resource {
                if *prev == accel_struct {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the underlying buffer
        // TODO

        // Register the accel struct
        let id = self.accel_structs.len();
        self.accel_structs
            .push(RenderResource::External(accel_struct));
        RGHandle::new(id)
    }

    // Convert a transient resource to a temporal one (content is kept through frames)
    pub fn convert_to_temporal<T>(
        &mut self,
        cache: &mut RenderGraphCache,
        handle: RGHandle<T>,
    ) -> RGTemporal<T>
    where
        T: ResType,
    {
        // Validate
        assert!(self.is_virtual(handle));

        // all type using same temporal id scope... :)
        let temporal_id = cache.next_temporal_id;
        cache.next_temporal_id += 1;

        // record for later caching
        match T::get_enum() {
            ResTypeEnum::Texture => {
                self.transient_to_temporal_textures
                    .insert(RGHandle::new(handle.id), RGTemporal::new(temporal_id));
            }
            ResTypeEnum::Buffer => {
                self.transient_to_temporal_buffers
                    .insert(RGHandle::new(handle.id), RGTemporal::new(temporal_id));
            }
        }

        RGTemporal::<T>::new(temporal_id)
    }

    // Convert a temporal resource to a transient one (content is discarded after last usage in this frame)
    pub fn convert_to_transient<T>(&mut self, temporal_handle: RGTemporal<T>) -> RGHandle<T>
    where
        T: ResType,
    {
        // Check if already registered
        if let Some(handle) = self.find_converted_temporal(temporal_handle) {
            return handle;
        }

        // Register the texture
        let id = match T::get_enum() {
            ResTypeEnum::Texture => {
                let id = self.textures.len();
                self.textures
                    .push(RenderResource::Temporal(temporal_handle.id));
                assert!(!self
                    .transient_to_temporal_textures
                    .contains_key(&RGHandle::new(id)));
                id
            }
            ResTypeEnum::Buffer => {
                let id = self.buffers.len();
                self.buffers
                    .push(RenderResource::Temporal(temporal_handle.id));
                assert!(!self
                    .transient_to_temporal_buffers
                    .contains_key(&RGHandle::new(id)));
                id
            }
        };

        RGHandle::new(id)
    }

    pub fn get_texture_desc(&self, texture: RGHandle<Texture>) -> &TextureDesc {
        match &self.textures[texture.id] {
            RenderResource::Virtual(desc) => desc,
            RenderResource::Temporal(_) => todo!("Temporal resources are in cache"),
            RenderResource::External(texture) => &texture.desc,
        }
    }

    pub fn get_texture_desc_from_view(&self, texture_view: RGHandle<TextureView>) -> &TextureDesc {
        match &self.texture_views[texture_view.id] {
            RenderResource::Virtual(virtual_view) => self.get_texture_desc(virtual_view.texture),
            RenderResource::Temporal(_) => panic!("Temporal resources are in cache"),
            RenderResource::External(texture_view) => &texture_view.texture.desc,
        }
    }

    pub fn get_buffer_desc(&self, buffer: RGHandle<Buffer>) -> &BufferDesc {
        match &self.buffers[buffer.id] {
            RenderResource::Virtual(desc) => desc,
            RenderResource::Temporal(_) => todo!("Temporal resources are in cache"),
            RenderResource::External(buffer) => &buffer.desc,
        }
    }

    pub fn add_global_descriptor_sets(&mut self, sets: &[(u32, vk::DescriptorSet)]) {
        for (index, _set) in sets {
            if self
                .global_descriptor_sets
                .iter()
                .find(|(i, _)| i == index)
                .is_some()
            {
                println!("Warning: global descriptor set {} is overrided", index);
                continue;
            }
        }

        self.global_descriptor_sets.extend_from_slice(sets);
    }

    pub fn add_pass(&mut self, pass: RenderPass<'a>) {
        self.passes.push(pass);
    }

    pub fn new_pass<'b>(&'b mut self, name: &str, ty: RenderPassType) -> RenderPassBuilder<'a, 'b> {
        //println!("Adding new pass: {}", name);
        RenderPassBuilder::new(self, name, ty)
    }

    pub fn new_raytracing<'b>(&'b mut self, name: &str) -> RaytracingPassBuilder<'a, 'b> {
        RaytracingPassBuilder::new(self, name)
    }
}

// Internal Methods
impl RenderGraphBuilder<'_> {
    // Assume execute context is already populated
    #[inline]
    fn get_texture(&self, ctx: &RenderGraphExecuteContext, handle: RGHandle<Texture>) -> Texture {
        let resource = &self.textures[handle.id];
        match resource {
            RenderResource::Virtual(_) => ctx.textures[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.textures[handle.id].unwrap(),
            RenderResource::External(texture) => *texture,
        }
    }

    // Assume execute context is already populated
    #[inline]
    fn get_texture_view(
        &self,
        ctx: &RenderGraphExecuteContext,
        handle: RGHandle<TextureView>,
    ) -> TextureView {
        let resource = &self.texture_views[handle.id];
        match resource {
            RenderResource::Virtual(_) => ctx.texture_views[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.texture_views[handle.id].unwrap(),
            RenderResource::External(view) => *view,
        }
    }

    #[inline]
    fn get_buffer(&self, ctx: &RenderGraphExecuteContext, handle: RGHandle<Buffer>) -> Buffer {
        match &self.buffers[handle.id] {
            RenderResource::Virtual(_) => ctx.buffers[handle.id].unwrap(),
            RenderResource::Temporal(_) => ctx.buffers[handle.id].unwrap(),
            RenderResource::External(buffer) => *buffer,
        }
    }

    #[inline]
    fn get_accel_struct(&self, handle: RGHandle<AccelerationStructure>) -> AccelerationStructure {
        let accel_struct = &self.accel_structs[handle.id];
        match accel_struct {
            RenderResource::Virtual(_) => unimplemented!(),
            RenderResource::Temporal(_) => unimplemented!(),
            RenderResource::External(accel_struct) => *accel_struct,
        }
    }

    fn begin_graphics(
        &self,
        ctx: &RenderGraphExecuteContext,
        command_buffer: &CommandBuffer,
        pass: &RenderPass,
    ) {
        let size = if pass.color_targets.len() > 0 {
            self.get_texture_desc_from_view(pass.color_targets[0].view)
                .size_2d()
        } else if let Some(handle) = pass.depth_stencil {
            self.get_texture_desc_from_view(handle.view).size_2d()
        } else {
            panic!();
        };

        let color_attachments: Vec<_> = pass
            .color_targets
            .iter()
            .map(|&target| {
                let image_view = self.get_texture_view(ctx, target.view).image_view;

                let mut builder = vk::RenderingAttachmentInfo::builder()
                    .image_view(image_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) // TODO read it
                    .store_op(vk::AttachmentStoreOp::STORE);

                builder = match target.load_op {
                    ColorLoadOp::Load => builder.load_op(vk::AttachmentLoadOp::LOAD),
                    ColorLoadOp::Clear(clear_value) => builder
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .clear_value(vk::ClearValue { color: clear_value }),
                    ColorLoadOp::DontCare => builder.load_op(vk::AttachmentLoadOp::DONT_CARE),
                };

                builder.build()
            })
            .collect();

        let depth_attachment = pass.depth_stencil.map(|target| {
            let image_view = self.get_texture_view(ctx, target.view).image_view;

            let mut builder = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

            builder = match target.load_op {
                DepthLoadOp::Load => builder.load_op(vk::AttachmentLoadOp::LOAD),
                DepthLoadOp::Clear(clear_value) => builder
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        depth_stencil: clear_value,
                    }),
            };

            builder.build()
        });

        let mut rendering_info = vk::RenderingInfo::builder()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: size,
            })
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&color_attachments);
        if depth_attachment.is_some() {
            rendering_info = rendering_info.depth_attachment(depth_attachment.as_ref().unwrap());
        };

        command_buffer.begin_rendering(&rendering_info);
    }

    fn end_graphics(&mut self, command_buffer: &CommandBuffer) {
        command_buffer.end_rendering();
    }

    fn bind_resources(
        &self,
        ctx: &RenderGraphExecuteContext,
        rd: &RenderDevice,
        shaders: &Shaders,
        command_buffer: &CommandBuffer,
        pass: &RenderPass,
        internal_set: vk::DescriptorSet,
    ) -> Option<()> {
        let pipeline_bind_point = pass.ty.to_bind_point()?;

        // Check for any resources
        let has_internal_set = internal_set != vk::DescriptorSet::null();
        let has_global_set = self.global_descriptor_sets.len() > 0;
        let any_resource = has_internal_set || has_global_set;
        if !any_resource {
            return None;
        }

        let pipeline = shaders.get_pipeline(pass.pipeline);
        if pipeline.is_none() {
            println!(
                "Warning[RenderGraph::{}]: bind_resources failed: pipeline is none.",
                pass.name
            );
            return None;
        }
        let pipeline = pipeline.unwrap();

        // Update descriptor set
        // TODO this struct can be reused (a lot vec)
        if has_internal_set {
            let mut builder = shader::DescriptorSetWriteBuilder::new();

            let builder: &mut shader::DescriptorSetWriteBuilder<'_> = &mut builder;
            for (name, handle) in &pass.textures {
                let view = self.get_texture_view(ctx, *handle);
                builder.image(
                    name,
                    view.image_view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            for (name, handle) in &pass.rw_textures {
                let view = self.get_texture_view(ctx, *handle);
                builder.image(name, view.image_view, vk::ImageLayout::GENERAL);
            }
            for (name, handle) in &pass.buffers {
                let buffer = self.get_buffer(ctx, *handle);
                builder.buffer(name, buffer.handle);
            }
            for (name, handle) in &pass.rw_buffers {
                let buffer = self.get_buffer(ctx, *handle);
                builder.buffer(name, buffer.handle);
            }
            for (name, handle) in &pass.accel_structs {
                let accel_struct = self.get_accel_struct(*handle);
                builder.accel_struct(name, accel_struct.handle);
            }
            let writes = builder.build(
                pipeline,
                internal_set,
                pass.descriptor_set_index,
                |prop_name, ty_name| {
                    println!(
                        "Warning[RenderGraph::{}]: property {}:{} is not found.",
                        pass.name, prop_name, ty_name
                    );
                },
            );
            if !writes.is_empty() {
                unsafe {
                    rd.device_entry.update_descriptor_sets(&writes, &[]);
                }
            }

            // Bind set
            command_buffer.bind_descriptor_set(
                pipeline_bind_point,
                pipeline.layout,
                pass.descriptor_set_index,
                internal_set,
                None,
            );
        }

        // Set globally bound sets
        // TODO should be be set evey pass? Can't it be set once per queue/command-buffer??
        for (set_index, set) in &self.global_descriptor_sets {
            if (*set_index == pass.descriptor_set_index)
                && (internal_set != vk::DescriptorSet::null())
            {
                println!("Error: RenderPass {} global set index {} is conflicted with internal set index {}", pass.name, set_index, pass.descriptor_set_index);
                continue;
            }

            if !pipeline.used_set.contains_key(set_index) {
                // Since global sets exist only for ergonomics purpose, it is totally okay if not used
                continue;
            }

            command_buffer.bind_descriptor_set(
                pipeline_bind_point,
                pipeline.layout,
                *set_index,
                *set,
                None,
            );
        }

        Some(())
    }

    fn ad_hoc_transition(
        &self,
        command_buffer: &CommandBuffer,
        ctx: &mut RenderGraphExecuteContext,
        pass_index: u32,
    ) {
        // helper
        let map_stage_mask = |pass: &RenderPass<'_>| match pass.ty {
            RenderPassType::Graphics => vk::PipelineStageFlags::ALL_GRAPHICS, // TODO fragment access? vertex access? color output?
            RenderPassType::Compute => vk::PipelineStageFlags::COMPUTE_SHADER,
            RenderPassType::RayTracing => vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            RenderPassType::Copy => vk::PipelineStageFlags::TRANSFER,
            RenderPassType::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        };

        let get_last_access = |image: vk::Image,
                               aspect: vk::ImageAspectFlags,
                               end_pass_index: u32|
         -> Option<(u32, vk::ImageLayout)> {
            for pass_index in (0..end_pass_index).rev() {
                let pass = &self.passes[pass_index as usize];
                // Check all mutating view
                for rt in &pass.color_targets {
                    let rt_view = self.get_texture_view(ctx, rt.view);
                    if (rt_view.texture.image == image) && (rt_view.desc.aspect == aspect) {
                        return Some((pass_index, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
                    }
                }
                if let Some(rt) = pass.depth_stencil {
                    let rt_view = self.get_texture_view(ctx, rt.view);
                    if (rt_view.texture.image == image) && (rt_view.desc.aspect == aspect) {
                        return Some((
                            pass_index,
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        ));
                    }
                }
                for rw in &pass.rw_textures {
                    let rw_view = self.get_texture_view(ctx, rw.1);
                    if (rw_view.texture.image == image) && (rw_view.desc.aspect == aspect) {
                        return Some((pass_index, vk::ImageLayout::GENERAL));
                    }
                }
                if let Some(copy_dst) = pass.copy_dst {
                    let dst = self.get_texture_view(ctx, copy_dst);
                    if (dst.texture.image == image) && (dst.desc.aspect == aspect) {
                        return Some((pass_index, vk::ImageLayout::TRANSFER_DST_OPTIMAL));
                    }
                }
                // Check all sampling view
                for tex_view in &pass.textures {
                    let tex_view = self.get_texture_view(ctx, tex_view.1);
                    if (tex_view.texture.image == image) && (tex_view.desc.aspect == aspect) {
                        return Some((pass_index, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
                    }
                }
                if let Some(copy_src) = pass.copy_src {
                    let src = self.get_texture_view(ctx, copy_src);
                    if (src.texture.image == image) && (src.desc.aspect == aspect) {
                        return Some((pass_index, vk::ImageLayout::TRANSFER_SRC_OPTIMAL));
                    }
                }
            }
            None
        };

        let pass = &self.passes[pass_index as usize];

        let transition_to =
            |image: vk::Image, range: vk::ImageSubresourceRange, new_layout: vk::ImageLayout| {
                if let Some((last_pass_index, last_layout)) =
                    get_last_access(image, vk::ImageAspectFlags::COLOR, pass_index)
                {
                    if last_layout != new_layout {
                        command_buffer.transition_image_layout(
                            image,
                            map_stage_mask(&self.passes[last_pass_index as usize]),
                            map_stage_mask(pass),
                            last_layout,
                            new_layout,
                            range,
                        )
                    }
                } else {
                    // TODO support temporal resource
                    command_buffer.transition_image_layout(
                        image,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        map_stage_mask(pass),
                        vk::ImageLayout::UNDEFINED,
                        new_layout,
                        range,
                    )
                };
            };

        let transition_view_to = |handle: RGHandle<TextureView>, layout: vk::ImageLayout| {
            let view = self.get_texture_view(ctx, handle);
            transition_to(
                view.texture.image,
                view.desc.make_subresource_range(),
                layout,
            );
        };

        for (_name, handle) in &pass.textures {
            transition_view_to(*handle, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }

        for (_name, handle) in &pass.rw_textures {
            transition_view_to(*handle, vk::ImageLayout::GENERAL);
        }

        for (_rt_index, rt) in pass.color_targets.iter().enumerate() {
            transition_view_to(rt.view, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }

        if let Some(ds) = pass.depth_stencil {
            transition_view_to(
                ds.view,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, // TODO finer grain for depth-stencil access
            );
        }

        if let Some(copy_src) = pass.copy_src {
            transition_view_to(copy_src, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        }

        if let Some(copy_dst) = pass.copy_dst {
            transition_view_to(copy_dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        }

        // Special: check present, transition after last access
        if let Some(present_tex) = pass.present_texture {
            transition_view_to(present_tex, vk::ImageLayout::PRESENT_SRC_KHR);
        }

        // Buffers

        let get_last_access =
            |vk_buffer: vk::Buffer, end_pass_index: u32| -> Option<(u32, vk::AccessFlags)> {
                for pass_index in (0..end_pass_index).rev() {
                    let pass = &self.passes[pass_index as usize];

                    // Check all mutating
                    for (_name, handle) in &pass.rw_buffers {
                        let buffer = self.get_buffer(ctx, *handle);
                        if buffer.handle == vk_buffer {
                            return Some((pass_index, vk::AccessFlags::SHADER_WRITE));
                        }
                    }

                    // Check all sampling
                    for (_name, handle) in &pass.buffers {
                        let buffer = self.get_buffer(ctx, *handle);
                        if buffer.handle == vk_buffer {
                            return Some((pass_index, vk::AccessFlags::SHADER_READ));
                        }
                    }
                }
                None
            };

        let transition_to = |vk_buffer: vk::Buffer, dst_access_mask: vk::AccessFlags| {
            if let Some((last_pass_index, src_access_mask)) = get_last_access(vk_buffer, pass_index)
            {
                if dst_access_mask != src_access_mask {
                    let barrier = vk::BufferMemoryBarrier::builder()
                        .src_access_mask(src_access_mask)
                        .dst_access_mask(dst_access_mask)
                        //.src_queue_family_index(0) // ?
                        //.dst_queue_family_index(0) // ?
                        .buffer(vk_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);

                    command_buffer.pipeline_barrier(
                        map_stage_mask(&self.passes[last_pass_index as usize]),
                        map_stage_mask(pass),
                        vk::DependencyFlags::empty(), // TODO do it properly
                        &[],
                        std::slice::from_ref(&barrier),
                        &[],
                    );
                }
            } else {
                // Nothing to sync
                // TODO support temporal resource
            };
        };

        for (_name, handle) in &pass.buffers {
            transition_to(
                self.get_buffer(ctx, *handle).handle,
                vk::AccessFlags::SHADER_READ,
            );
        }

        for (_name, handle) in &pass.rw_buffers {
            transition_to(
                self.get_buffer(ctx, *handle).handle,
                vk::AccessFlags::SHADER_WRITE,
            );
        }
    }

    pub fn execute(
        &mut self,
        rd: &RenderDevice,
        command_buffer: &CommandBuffer,
        shaders: &mut Shaders,
        cache: &mut RenderGraphCache,
    ) {
        let mut exec_context = RenderGraphExecuteContext::new();

        // Some pre-processing
        for pass_index in 0..self.passes.len() {
            let pass = &mut self.passes[pass_index];

            if let Some(rt) = &pass.raytracing {
                pass.pipeline = shaders
                    .create_raytracing_pipeline(
                        rt.raygen_shader.unwrap(),
                        rt.miss_shader.unwrap(),
                        rt.chit_shader,
                        &self.shader_config,
                    )
                    .unwrap();
            }
        }

        // Populate textures and views
        // TODO drain self.texture, self.texture_views to context
        // TODO memory aliasing
        for i in 0..self.textures.len() {
            let texture_resource = &self.textures[i];
            let texture = match texture_resource {
                // Create texture
                RenderResource::Virtual(desc) => {
                    let tex = cache
                        .texture_pool
                        .pop(&desc)
                        .unwrap_or_else(|| rd.create_texture(*desc).unwrap());
                    Some(tex)
                }
                RenderResource::Temporal(temporal_id) => {
                    let tex = cache.temporal_textures.remove(temporal_id).unwrap();
                    Some(tex)
                }
                RenderResource::External(_) => None,
            };
            exec_context.textures.push(texture);
        }
        for i in 0..self.texture_views.len() {
            let view_resource = &self.texture_views[i];
            let view = match view_resource {
                // Create texture view
                RenderResource::Virtual(virtual_view) => {
                    let texture = self.get_texture(&exec_context, virtual_view.texture);
                    let view_desc = virtual_view
                        .desc
                        .unwrap_or_else(|| TextureViewDesc::auto(&texture.desc));
                    let view = cache
                        .texture_view_pool
                        .pop(&(texture, view_desc))
                        .unwrap_or_else(|| rd.create_texture_view(texture, view_desc).unwrap());
                    Some(view)
                }
                RenderResource::Temporal(_) => panic!("No temporal texture view"),
                RenderResource::External(_) => None,
            };
            exec_context.texture_views.push(view);
        }
        for i in 0..self.buffers.len() {
            let res = &self.buffers[i];
            let buffer = match res {
                RenderResource::Virtual(desc) => {
                    let buf = cache
                        .buffer_pool
                        .pop(&desc)
                        .unwrap_or_else(|| rd.create_buffer(*desc).unwrap());
                    Some(buf)
                }
                RenderResource::Temporal(temporal_id) => {
                    let buf = cache.temporal_buffers.remove(temporal_id).unwrap();
                    Some(buf)
                }
                RenderResource::External(_) => None,
            };
            exec_context.buffers.push(buffer);
        }

        // Create (temp) descriptor set for each pass
        for pass in &self.passes {
            let pipeline = shaders.get_pipeline(pass.pipeline);
            let set = match pipeline {
                Some(pipeline) => {
                    if pass.descriptor_set_index as usize >= pipeline.set_layouts.len() {
                        /* Totally happen in shader debug
                        println!(
                            "Warning: unused descriptor set {} for pass {}",
                            pass.descriptor_set_index, pass.name
                        );
                        */
                        vk::DescriptorSet::null()
                    } else {
                        let set_layout = pipeline.set_layouts[pass.descriptor_set_index as usize];
                        cache.allocate_dessriptor_set(rd, set_layout)
                    }
                }
                None => {
                    if (pass.ty != RenderPassType::Present) && (pass.ty != RenderPassType::Copy) {
                        println!("Warning[RenderGraph]: pipeline not provided by pass {}; temporal descriptor set is not created.", pass.name);
                    }
                    vk::DescriptorSet::null()
                }
            };

            exec_context.descriptor_sets.push(set);
        }
        assert!(exec_context.descriptor_sets.len() == self.passes.len());

        // Create and update ShaderBindingTable for each pass
        let sbt_frame_index = self.hack_frame_index % 3;
        for pass in &self.passes {
            let mut pass_sbt = None;
            if let Some(rt) = &pass.raytracing {
                let mut sbt = cache
                    .sbt_pool
                    .pop(&sbt_frame_index)
                    .unwrap_or_else(|| PassShaderBindingTable::new(rd));

                // update anyway :)
                // TODO using frame index % 3 to void cpu-write-on-GPU-read; should do it with proper synchronization
                let has_hit = rt.chit_shader.is_some();
                let pipeline = shaders.get_pipeline(pass.pipeline).unwrap();
                sbt.update_shader_group_handles(rd, pipeline, has_hit);

                pass_sbt = Some(sbt);
            }
            exec_context.shader_binding_tables.push(pass_sbt);
        }

        command_buffer.insert_checkpoint();

        for pass_index in 0..self.passes.len() {
            // take the callback before unmutable reference
            let render = self.passes[pass_index].render.take();

            let pass = &self.passes[pass_index];

            // TODO analysis of the DAG and sync properly
            self.ad_hoc_transition(command_buffer, &mut exec_context, pass_index as u32);

            command_buffer.insert_checkpoint();

            // Begin render pass (if graphics)
            let is_graphic = pass.ty == RenderPassType::Graphics;
            if is_graphic {
                self.begin_graphics(&exec_context, command_buffer, pass);
                command_buffer.insert_checkpoint();
            }

            // Bind pipeline (if set)
            if let Some(pipeline) = shaders.get_pipeline(pass.pipeline) {
                let bind_point = match pass.ty {
                    RenderPassType::Graphics => vk::PipelineBindPoint::GRAPHICS,
                    RenderPassType::Compute => vk::PipelineBindPoint::COMPUTE,
                    RenderPassType::RayTracing => vk::PipelineBindPoint::RAY_TRACING_KHR,
                    RenderPassType::Copy => panic!("Copy pass should not have pipeline"),
                    RenderPassType::Present => panic!("Present pass should not have pipeline"),
                };
                command_buffer.bind_pipeline(bind_point, pipeline.handle);
            }

            // Bind resources
            self.bind_resources(
                &exec_context,
                rd,
                shaders,
                command_buffer,
                &pass,
                exec_context.descriptor_sets[pass_index],
            );
            command_buffer.insert_checkpoint();

            // Push Constant (if pushed)
            let pc_data = pass.push_constants.build();
            if let Some(pipeline) = shaders.get_pipeline(pass.pipeline) {
                if (pc_data.len() > 0) && (pipeline.push_constant_ranges.len() > 0) {
                    let range = pipeline.push_constant_ranges[0];
                    command_buffer.push_constants(pipeline.layout, range.stage_flags, 0, pc_data);
                } else if pc_data.len() > 0 {
                    println!(
                        "Warning[RenderGraph]: push constant is not used by pass {}",
                        pass.name
                    );
                } else if pipeline.push_constant_ranges.len() > 0 {
                    // It is okay the render pass set push constant range them self in custom callback
                    /*
                    println!(
                        "Warning[RenderGraph]: push constant is not provided for pass {}",
                        pass.name
                    );
                    */
                }
            } else if pc_data.len() > 0 {
                println!(
                    "Warning[RenderGraph]: pipeline is not provided for pass {}",
                    pass.name
                );
            }

            // Run pass
            if let Some(render) = render {
                render(command_buffer, &shaders, pass);
                command_buffer.insert_checkpoint();
            } else if let Some(rt) = &pass.raytracing {
                // [new passbuilder]
                let has_hit = rt.chit_shader.is_some();
                let dim = rt.dimension;

                let empty_region = vk::StridedDeviceAddressRegionKHR::default();

                let sbt = exec_context.shader_binding_tables[pass_index]
                    .as_ref()
                    .unwrap();
                let hit_region = if has_hit {
                    &sbt.hit_region
                } else {
                    &empty_region
                };

                command_buffer.trace_rays(
                    &sbt.raygen_region,
                    &sbt.miss_region,
                    hit_region,
                    &empty_region,
                    dim.x,
                    dim.y,
                    dim.z,
                );
            } else {
                match pass.ty {
                    RenderPassType::Present => {}
                    RenderPassType::Copy => {
                        let src = self.get_texture_view(&exec_context, pass.copy_src.unwrap());
                        let dst = self.get_texture_view(&exec_context, pass.copy_dst.unwrap());
                        let region = vk::ImageCopy {
                            src_subresource: src.desc.make_subresrouce_layer(),
                            src_offset: vk::Offset3D::default(),
                            dst_subresource: dst.desc.make_subresrouce_layer(),
                            dst_offset: vk::Offset3D::default(),
                            extent: dst.texture.desc.size_3d(),
                        };
                        unsafe {
                            command_buffer.device.cmd_copy_image(
                                command_buffer.command_buffer,
                                src.texture.image,
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                                dst.texture.image,
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                                std::slice::from_ref(&region),
                            );
                        }
                    }
                    _ => println!(
                        "Warning[RenderGraph]: render callback not provided by pass {}!",
                        pass.name
                    ),
                }
            }

            // End render pass (if graphics)
            if is_graphic {
                self.end_graphics(command_buffer);
                command_buffer.insert_checkpoint();
            }
        }

        command_buffer.insert_checkpoint();

        // Process all textures converted to temporals
        for (handle, temporal_handle) in &self.transient_to_temporal_textures {
            let tex = exec_context.textures[handle.id].take().unwrap();
            cache.temporal_textures.insert(temporal_handle.id, tex);
        }
        self.transient_to_temporal_textures.clear();
        for (handle, temporal_handle) in &self.transient_to_temporal_buffers {
            let buf = exec_context.buffers[handle.id].take().unwrap();
            cache.temporal_buffers.insert(temporal_handle.id, buf);
        }

        // Pool back all resource objects
        for view in exec_context
            .texture_views
            .drain(0..exec_context.texture_views.len())
        {
            if let Some(view) = view {
                cache
                    .texture_view_pool
                    .push((view.texture, view.desc), view);
            }
        }
        for texture in exec_context.textures.drain(0..exec_context.textures.len()) {
            if let Some(texture) = texture {
                cache.texture_pool.push(texture.desc, texture);
            }
        }
        for buffer in exec_context.buffers.drain(0..exec_context.buffers.len()) {
            if let Some(buffer) = buffer {
                cache.buffer_pool.push(buffer.desc, buffer);
            }
        }
        for set in exec_context
            .descriptor_sets
            .drain(0..exec_context.descriptor_sets.len())
        {
            if set != vk::DescriptorSet::null() {
                // TODO may be need some frame buffering, because it may be still using?
                cache.release_descriptor_set(rd, set);
            }
        }
        for sbt in exec_context
            .shader_binding_tables
            .drain(0..exec_context.shader_binding_tables.len())
        {
            if let Some(sbt) = sbt {
                cache.sbt_pool.push(sbt_frame_index, sbt);
            }
        }
    }
}
