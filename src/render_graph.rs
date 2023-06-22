use std::collections::HashMap;
use std::hash::Hash;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

use ash::vk;

use crate::command_buffer::CommandBuffer;
use crate::render_device::{RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc};
use crate::shader::{self, Handle, Pipeline, Shaders};

pub struct RGHandle<T> {
    id: usize,
    _phantom: PhantomData<T>,
}

impl<T> RGHandle<T> {
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

// TODO better naming
#[derive(Clone, Copy)]
pub enum HandleEnum<T> {
    Inetrnal(RGHandle<T>),
    External(T),
}

impl<T> From<RGHandle<T>> for HandleEnum<T> {
    fn from(handle: RGHandle<T>) -> Self {
        HandleEnum::Inetrnal(handle)
    }
}

impl<T> From<HandleEnum<T>> for RGHandle<T> {
    fn from(handle: HandleEnum<T>) -> Self {
        match handle {
            HandleEnum::Inetrnal(h) => h,
            _ => panic!("Cannot convert external handle to internal handle!"),
        }
    }
}

impl<T> From<T> for HandleEnum<T> {
    fn from(t: T) -> Self {
        HandleEnum::External(t)
    }
}

// TODO make this a macro
impl From<HandleEnum<Texture>> for Texture {
    fn from(handle: HandleEnum<Texture>) -> Self {
        match handle {
            HandleEnum::External(t) => t,
            _ => panic!("Cannot convert internal handle to external handle!"),
        }
    }
}

impl From<HandleEnum<TextureView>> for TextureView {
    fn from(handle: HandleEnum<TextureView>) -> Self {
        match handle {
            HandleEnum::External(t) => t,
            _ => panic!("Cannot convert internal handle to external handle!"),
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum RenderPassType {
    Graphics,
    Compute,
}

#[derive(Clone, Copy)]
pub enum ColorLoadOp {
    Load,
    Clear(vk::ClearColorValue),
    DontCare,
}

#[derive(Clone, Copy)]
pub struct ColorTarget {
    pub view: HandleEnum<TextureView>,
    pub load_op: ColorLoadOp,
}

#[derive(Clone, Copy)]
pub enum DepthLoadOp {
    Load,
    Clear(vk::ClearDepthStencilValue),
}

#[derive(Clone, Copy)]
pub struct DepthStencilTarget {
    pub view: HandleEnum<TextureView>,
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
    descriptor_set_index: u32,
    textures: Vec<(&'a str, HandleEnum<TextureView>)>,
    color_targets: Vec<ColorTarget>,
    depth_stencil: Option<DepthStencilTarget>,
    rw_textures: Vec<(&'a str, HandleEnum<TextureView>)>,
    mannual_transition: Option<Box<dyn 'a + FnOnce(&TransitionInterface, &RenderPass)>>, // Hack
    render: Option<Box<dyn 'a + FnOnce(&CommandBuffer, &Shaders, RenderPass)>>,
}

impl RenderPass<'_> {
    pub fn new(name: &str, ty: RenderPassType) -> Self {
        RenderPass {
            name: String::from(name),
            ty,
            pipeline: Handle::null(),
            descriptor_set_index: 0,
            textures: Vec::new(),
            color_targets: Vec::new(),
            depth_stencil: None,
            rw_textures: Vec::new(),
            mannual_transition: None,
            render: None,
        }
    }

    pub fn get_texture(&self, name: &str) -> HandleEnum<TextureView> {
        for (n, t) in &self.textures {
            if n == &name {
                return *t;
            }
        }
        panic!("Cannot find texture with name: {}", name);
    }

    pub fn get_color_targets(&self) -> &[ColorTarget] {
        &self.color_targets
    }
}

pub struct RenderPassBuilder<'a, 'b> {
    inner: Option<RenderPass<'a>>,
    render_graph: &'b mut RenderGraph<'a>,
}

// TODO generate setter with macro?
impl<'a, 'b> RenderPassBuilder<'a, 'b> {
    pub fn new(render_graph: &'b mut RenderGraph<'a>, name: &str, ty: RenderPassType) -> Self {
        RenderPassBuilder {
            inner: Some(RenderPass::new(name, ty)),
            render_graph,
        }
    }

    fn inner(&mut self) -> &mut RenderPass<'a> {
        self.inner.as_mut().unwrap()
    }

    pub fn done(&mut self) {
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

    pub fn descriptor_set_index(mut self, index: u32) -> Self {
        self.inner().descriptor_set_index = index;
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

    pub fn texture(mut self, name: &'a str, texture: HandleEnum<TextureView>) -> Self {
        self.inner().textures.push((name, texture));
        self
    }

    pub fn rw_texture(mut self, name: &'a str, texture: HandleEnum<TextureView>) -> Self {
        self.inner().rw_textures.push((name, texture));
        self
    }

    pub fn mannual_transition<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&TransitionInterface, &RenderPass) + 'a,
    {
        self.inner().mannual_transition = Some(Box::new(f));
        self
    }

    pub fn render<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&CommandBuffer, &Shaders, RenderPass) + 'a,
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

pub struct RenderGraphCache {
    // Resused VK stuffs
    descriptor_pool: vk::DescriptorPool,

    // Resource pool
    texture_pool: ResPool<TextureDesc, Texture>,
    texture_view_pool: ResPool<TextureViewDesc, TextureView>,
}

impl RenderGraphCache {
    pub fn new(rd: &RenderDevice) -> Self {
        let descriptor_pool = rd.create_descriptor_pool(
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorPoolCreateFlags::empty(),
        );
        Self {
            descriptor_pool,
            texture_pool: ResPool::new(),
            texture_view_pool: ResPool::new(),
        }
    }

    fn get_dessriptor_set(
        &mut self,
        rd: &RenderDevice,
        set_layout: vk::DescriptorSetLayout,
    ) -> vk::DescriptorSet {
        let layouts = [set_layout];
        let create_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        unsafe {
            rd.device
                .allocate_descriptor_sets(&create_info)
                .expect("Failed to create descriptor set for pass")[0]
        }
    }

    fn return_descriptor_set(&mut self, rd: &RenderDevice, descriptor_set: vk::DescriptorSet) {
        unsafe {
            rd.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                .expect("Failed to free descriptor set");
        }
    }
}

pub struct TransitionInterface<'a> {
    render_graph: &'a RenderGraph<'a>, // TODO should only access the resouce part
    pub cmd_buf: &'a CommandBuffer,
}

impl TransitionInterface<'_> {
    pub fn get_image(&self, handle: HandleEnum<TextureView>) -> vk::Image {
        match handle {
            HandleEnum::Inetrnal(handle) => {
                let view = self.render_graph.texture_views[handle.id];
                view.texture.image
            }
            HandleEnum::External(view) => view.texture.image,
        }
    }
}

// A Render Graph to handle resource transitions automatically
pub struct RenderGraph<'a> {
    passes: Vec<RenderPass<'a>>,
    cache: &'a mut RenderGraphCache,

    // Array indexed by RGHandle
    texture_descs: Vec<TextureDesc>,
    texture_view_descs: Vec<TextureViewDesc>,
    texture_view_textures: Vec<RGHandle<Texture>>,

    // Textures
    textures: Vec<Texture>,          // by handle::id
    texture_views: Vec<TextureView>, // by handle::id

    descriptor_sets: Vec<vk::DescriptorSet>, // by pass
}

impl<'a> RenderGraph<'a> {
    pub fn new(cache: &'a mut RenderGraphCache) -> Self {
        Self {
            passes: Vec::new(),
            cache,
            texture_descs: Vec::new(),
            texture_view_descs: Vec::new(),
            texture_view_textures: Vec::new(),
            textures: Vec::new(),
            texture_views: Vec::new(),
            descriptor_sets: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.passes.clear();
        self.texture_descs.clear();
        self.texture_view_descs.clear();

        assert!(self.textures.is_empty());
        assert!(self.texture_views.is_empty());
    }

    pub fn create_texutre(&mut self, desc: TextureDesc) -> RGHandle<Texture> {
        let id = self.texture_descs.len();
        self.texture_descs.push(desc);
        RGHandle::new(id)
    }

    pub fn create_texture_view(
        &mut self,
        texture: RGHandle<Texture>,
        desc: TextureViewDesc,
    ) -> RGHandle<TextureView> {
        let id = self.texture_view_descs.len();
        self.texture_view_descs.push(desc);
        self.texture_view_textures.push(texture);
        RGHandle::new(id)
    }

    /*
    pub fn register_texture(&mut self, texture: Handle<Texture>) -> RGHandle<Handle<Texture>> {
        let id = self.texture_descs.len();
        RGHandle::new(id).external(true)
    }

    pub fn register_texture_view(
        &mut self,
        texture_view: Handle<TextureView>,
    ) -> RGHandle<TextureView> {
        unimplemented!()
    }
    */

    #[allow(dead_code)]
    pub fn add_pass(&mut self, pass: RenderPass<'a>) {
        self.passes.push(pass);
    }

    #[allow(dead_code)]
    pub fn new_pass<'b>(&'b mut self, name: &str, ty: RenderPassType) -> RenderPassBuilder<'a, 'b> {
        //println!("Adding new pass: {}", name);
        RenderPassBuilder::new(self, name, ty)
    }

    fn get_texture_desc_from_view(&self, handle_enum: HandleEnum<TextureView>) -> TextureDesc {
        match handle_enum {
            HandleEnum::Inetrnal(handle) => {
                let texture_handle = self.texture_view_textures.get(handle.id).unwrap();
                self.texture_descs[texture_handle.id]
            }
            HandleEnum::External(view) => view.texture.desc,
        }
    }

    fn get_texture_view(&self, handle_enum: HandleEnum<TextureView>) -> TextureView {
        match handle_enum {
            HandleEnum::Inetrnal(handle) => self.texture_views[handle.id],
            HandleEnum::External(view) => view,
        }
    }

    fn begin_graphics(&mut self, command_buffer: &CommandBuffer, pass: &RenderPass) {
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
                let image_view = match target.view {
                    HandleEnum::Inetrnal(handle) => self.texture_views[handle.id].image_view,
                    HandleEnum::External(view) => view.image_view,
                };

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
            let image_view = match target.view {
                HandleEnum::Inetrnal(handle) => self.texture_views[handle.id].image_view,
                HandleEnum::External(view) => view.image_view,
            };

            let mut builder = vk::RenderingAttachmentInfoKHR::builder()
                .image_view(image_view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });

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
        &mut self,
        rd: &RenderDevice,
        shaders: &Shaders,
        command_buffer: &CommandBuffer,
        pass: &RenderPass,
        set: vk::DescriptorSet,
    ) -> Option<()> {
        let pipeline = shaders.get_pipeline(pass.pipeline)?;

        // Update descriptor set
        // TODO this struct can be reused (a lot vec)
        let mut builder = shader::DescriptorSetWriteBuilder::new();
        {
            let builder = &mut builder;
            for (name, handle) in &pass.textures {
                let view = self.get_texture_view(*handle);
                builder.image(
                    name,
                    view.image_view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
            for (name, handle) in &pass.rw_textures {
                let view = self.get_texture_view(*handle);
                builder.image(name, view.image_view, vk::ImageLayout::GENERAL);
            }
            let writes = builder.build(pipeline, set);
            unsafe {
                rd.device.update_descriptor_sets(&writes, &[]);
            }
        }

        // Bind set
        let pipeline_bind_point = match pass.ty {
            RenderPassType::Graphics => vk::PipelineBindPoint::GRAPHICS,
            RenderPassType::Compute => vk::PipelineBindPoint::COMPUTE,
        };
        command_buffer.bind_descriptor_set(
            pipeline_bind_point,
            pipeline.layout,
            pass.descriptor_set_index,
            set,
            None,
        );

        Some(())
    }

    pub fn execute(
        &mut self,
        rd: &RenderDevice,
        command_buffer: &CommandBuffer,
        shaders: &Shaders,
    ) {
        // Populate textures and views
        // TODO memory aliasing
        for i in 0..self.texture_descs.len() {
            let desc = self.texture_descs[i];
            let texture = self
                .cache
                .texture_pool
                .pop(&desc)
                .unwrap_or_else(|| rd.create_texture(desc).unwrap());
            self.textures.push(texture);
        }
        for i in 0..self.texture_view_descs.len() {
            let desc = self.texture_view_descs[i];
            let texture_handle = self.texture_view_textures[i];
            let texture = self.textures[texture_handle.id];
            let view = self
                .cache
                .texture_view_pool
                .pop(&desc)
                .unwrap_or_else(|| rd.create_texture_view(texture, desc).unwrap());
            self.texture_views.push(view);
        }
        for pass in &self.passes {
            let pipeline = shaders.get_pipeline(pass.pipeline);
            let set = match pipeline {
                Some(pipeline) => {
                    let set_layout = pipeline.set_layouts[pass.descriptor_set_index as usize];
                    self.cache.get_dessriptor_set(rd, set_layout)
                }
                None => vk::DescriptorSet::null(),
            };

            self.descriptor_sets.push(set);
        }

        let passes = self.passes.drain(0..self.passes.len()).collect::<Vec<_>>();
        for (pass_index, mut pass) in passes.into_iter().enumerate() {
            // TODO analysis of the DAG and sync properly
            if let Some(mannual_transition) = pass.mannual_transition.take() {
                let transition_interface = TransitionInterface {
                    render_graph: self,
                    cmd_buf: command_buffer,
                };
                mannual_transition(&transition_interface, &mut pass);
            }

            // Begin render pass (if graphics)
            let is_graphic = pass.ty == RenderPassType::Graphics;
            if is_graphic {
                self.begin_graphics(command_buffer, &pass);
            }

            // Bind resources
            let set = self.descriptor_sets[pass_index];
            if set != vk::DescriptorSet::null() {
                self.bind_resources(rd, shaders, command_buffer, &pass, set);
            }

            // Run pass
            let render = pass.render.take().unwrap();
            render(command_buffer, &shaders, pass);

            // End render pass (if graphics)
            if is_graphic {
                self.end_graphics(command_buffer);
            }
        }

        // Pool back all resource objects
        for view in self.texture_views.drain(0..self.texture_views.len()) {
            self.cache.texture_view_pool.push(view.desc, view);
        }
        for texture in self.textures.drain(0..self.textures.len()) {
            self.cache.texture_pool.push(texture.desc, texture);
        }
        for set in self.descriptor_sets.drain(0..self.descriptor_sets.len()) {
            if set != vk::DescriptorSet::null() {
                // TODO may be need some frame buffering, because it may be still using?
                self.cache.return_descriptor_set(rd, set);
            }
        }
    }
}
