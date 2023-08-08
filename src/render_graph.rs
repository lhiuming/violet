use std::collections::HashMap;
use std::hash::Hash;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

use ash::vk;

use crate::command_buffer::CommandBuffer;
use crate::render_device::{
    AccelerationStructure, RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc,
};
use crate::shader::{self, Handle, Pipeline, Shaders};

pub struct RGHandle<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGHandle<T> {
    pub fn null() -> Self {
        RGHandle {
            id: usize::MAX,
            _phantom: PhantomData::default(),
        }
    }

    fn new(id: usize) -> Self {
        RGHandle {
            id,
            _phantom: PhantomData::default(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.id == usize::MAX
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
    pub fn null() -> Self {
        RGTemporal {
            id: usize::MAX,
            _phantom: PhantomData::default(),
        }
    }

    fn new(id: usize) -> Self {
        RGTemporal {
            id,
            _phantom: PhantomData::default(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.id == usize::MAX
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

pub struct RenderPass<'a> {
    //params: dyn Paramter,
    //read_textures: Vec<RGHandle<Texture>>,
    //input_buffers: RGHandle<>
    //mutable_textures: Vec<RGHandle<Texture>>,
    name: String,
    ty: RenderPassType,
    pipeline: Handle<Pipeline>,
    external_descritpr_sets: Vec<(u32, vk::DescriptorSet)>,
    descriptor_set_index: u32,
    textures: Vec<(&'a str, RGHandle<TextureView>)>,
    accel_structs: Vec<(&'a str, RGHandle<AccelerationStructure>)>,
    color_targets: Vec<ColorTarget>,
    depth_stencil: Option<DepthStencilTarget>,
    rw_textures: Vec<(&'a str, RGHandle<TextureView>)>,
    copy_src: Option<RGHandle<TextureView>>,
    copy_dst: Option<RGHandle<TextureView>>,
    present_texture: Option<RGHandle<TextureView>>,
    render: Option<Box<dyn 'a + FnOnce(&CommandBuffer, &Shaders, &RenderPass)>>,
}

impl RenderPass<'_> {
    pub fn new(name: &str, ty: RenderPassType) -> Self {
        RenderPass {
            name: String::from(name),
            ty,
            pipeline: Handle::null(),
            external_descritpr_sets: Vec::new(),
            descriptor_set_index: 0,
            textures: Vec::new(),
            accel_structs: Vec::new(),
            color_targets: Vec::new(),
            depth_stencil: None,
            rw_textures: Vec::new(),
            copy_src: None,
            copy_dst: None,
            present_texture: None,
            render: None,
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

pub struct RenderPassBuilder<'a, 'b> {
    inner: Option<RenderPass<'a>>,
    render_graph: &'b mut RenderGraphBuilder<'a>,
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

    pub fn pipeline(mut self, pipeline: Handle<Pipeline>) -> Self {
        self.inner().pipeline = pipeline;
        self
    }

    // Binding an external descriptor set
    pub fn descritpro_set(mut self, set_index: u32, set: vk::DescriptorSet) -> Self {
        self.inner().external_descritpr_sets.push((set_index, set));
        self
    }

    pub fn descritpro_sets(mut self, sets: &[(u32, vk::DescriptorSet)]) -> Self {
        for pair in sets {
            self.inner().external_descritpr_sets.push(*pair)
        }
        self
    }

    pub fn color_targets(mut self, rts: &[ColorTarget]) -> Self {
        self.inner().color_targets.clear();
        self.inner().color_targets.extend_from_slice(rts);
        self
    }

    pub fn depth_stencil(mut self, ds: DepthStencilTarget) -> Self {
        self.inner().depth_stencil = Some(ds);
        self
    }

    // Index for the per-pass descriptor set
    pub fn descriptor_set_index(mut self, index: u32) -> Self {
        self.inner().descriptor_set_index = index;
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

    // Binding acceleration structure to per-pass descriptor set
    pub fn accel_struct(
        mut self,
        name: &'a str,
        accel_struct: RGHandle<AccelerationStructure>,
    ) -> Self {
        self.inner().accel_structs.push((name, accel_struct));
        self
    }

    pub fn copy_src(mut self, texture: RGHandle<TextureView>) -> Self {
        self.inner().copy_src = Some(texture);
        self
    }

    pub fn copy_dst(mut self, texture: RGHandle<TextureView>) -> Self {
        self.inner().copy_dst = Some(texture);
        self
    }

    pub fn present_texture(mut self, texture: RGHandle<TextureView>) -> Self {
        self.inner().present_texture = Some(texture);
        self
    }

    pub fn render<F>(mut self, f: F) -> Self
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

const RG_MAX_SET: u32 = 1024;

pub struct RenderGraphCache {
    // Resused VK stuffs
    vk_descriptor_pool: vk::DescriptorPool,

    // buffered VK objects
    free_vk_descriptor_sets: Vec<vk::DescriptorSet>,

    // Temporal Resources
    temporal_textures: HashMap<usize, Texture>,
    next_temporal_id: usize,

    // Resource pool
    texture_pool: ResPool<TextureDesc, Texture>,
    texture_view_pool: ResPool<(Texture, TextureViewDesc), TextureView>,
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
            next_temporal_id: 0,
            texture_pool: ResPool::new(),
            texture_view_pool: ResPool::new(),
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

    // Per-pass descriptor set
    pub descriptor_sets: Vec<vk::DescriptorSet>, // by pass index
}

impl RenderGraphExecuteContext {
    pub fn new() -> Self {
        Self {
            textures: Vec::new(),
            texture_views: Vec::new(),
            descriptor_sets: Vec::new(),
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

// A Render Graph to handle resource transitions automatically
pub struct RenderGraphBuilder<'a> {
    passes: Vec<RenderPass<'a>>,

    transient_to_temporal_textures: HashMap<RGHandle<Texture>, RGTemporal<Texture>>,

    // Array indexed by RGHandle
    textures: Vec<RenderResource<TextureDesc, Texture>>,
    texture_views: Vec<RenderResource<VirtualTextureView, TextureView>>,
    accel_structs: Vec<RenderResource<(), AccelerationStructure>>,
}

// Interface
impl<'a> RenderGraphBuilder<'a> {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),

            transient_to_temporal_textures: HashMap::new(),

            textures: Vec::new(),
            texture_views: Vec::new(),
            accel_structs: Vec::new(),
        }
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
    pub fn convert_to_temporal(
        &mut self,
        cache: &mut RenderGraphCache,
        handle: RGHandle<Texture>,
    ) -> RGTemporal<Texture> {
        assert!(match self.textures[handle.id] {
            RenderResource::Virtual(_) => true,
            _ => false,
        });

        let temporal_id = cache.next_temporal_id;
        cache.next_temporal_id += 1;
        let temporal_handle = RGTemporal::<Texture>::new(temporal_id);

        self.transient_to_temporal_textures
            .insert(handle, temporal_handle);

        RGTemporal::<Texture>::new(temporal_id)
    }

    // Convert a temporal resource to a transient one (content is discarded after last usage in this frame)
    pub fn convert_to_transient(
        &mut self,
        temporal_handle: RGTemporal<Texture>,
    ) -> RGHandle<Texture> {
        // Check if already registered
        // TOOD make this routine part of self.textures
        for handle_id in 0..self.textures.len() {
            let texture_resource = &self.textures[handle_id];
            if let RenderResource::Temporal(temporal_id) = texture_resource {
                if *temporal_id == temporal_handle.id {
                    return RGHandle::new(handle_id);
                }
            }
        }

        // Register the texture
        let id = self.textures.len();
        self.textures
            .push(RenderResource::Temporal(temporal_handle.id));
        let handle = RGHandle::new(id);

        // Cehck sanity
        assert!(!self.transient_to_temporal_textures.contains_key(&handle));

        handle
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

    pub fn add_pass(&mut self, pass: RenderPass<'a>) {
        self.passes.push(pass);
    }

    pub fn new_pass<'b>(&'b mut self, name: &str, ty: RenderPassType) -> RenderPassBuilder<'a, 'b> {
        //println!("Adding new pass: {}", name);
        RenderPassBuilder::new(self, name, ty)
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
        let has_external_set = !pass.external_descritpr_sets.is_empty();
        let any_resource = has_internal_set || has_external_set;
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

            let builder = &mut builder;
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

        // Set external set
        for (set_index, set) in &pass.external_descritpr_sets {
            if (*set_index == pass.descriptor_set_index)
                && (internal_set != vk::DescriptorSet::null())
            {
                println!("Error: RenderPass {} external set index {} is conflicted with internal set index {}", pass.name, set_index, pass.descriptor_set_index);
                continue;
            }

            if !pipeline.used_set.contains_key(set_index) {
                println!(
                    "Warning[RenderGraph: {}]: set index {} is not used",
                    pass.name, set_index
                );
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
    }

    pub fn execute(
        &mut self,
        rd: &RenderDevice,
        command_buffer: &CommandBuffer,
        shaders: &Shaders,
        cache: &mut RenderGraphCache,
    ) {
        let mut exec_context = RenderGraphExecuteContext::new();

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

            // Run pass
            if let Some(render) = render {
                render(command_buffer, &shaders, pass);
                command_buffer.insert_checkpoint();
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
        for set in exec_context
            .descriptor_sets
            .drain(0..exec_context.descriptor_sets.len())
        {
            if set != vk::DescriptorSet::null() {
                // TODO may be need some frame buffering, because it may be still using?
                cache.release_descriptor_set(rd, set);
            }
        }
    }
}
