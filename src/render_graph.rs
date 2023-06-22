use std::collections::HashMap;
use std::hash::Hash;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

use ash::vk;

use crate::command_buffer::CommandBuffer;
use crate::render_device::{RenderDevice, Texture, TextureDesc, TextureView, TextureViewDesc};
use crate::shader::Shaders;

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
    color_targets: Vec<ColorTarget>,
    depth_stencil: Option<DepthStencilTarget>,
    mannual_transition: Option<Box<dyn 'a + FnOnce(&CommandBuffer, &RenderPass)>>, // Hack
    logic: Option<Box<dyn 'a + FnOnce(&CommandBuffer, &Shaders, RenderPass)>>,
}

impl RenderPass<'_> {
    pub fn new(name: &str, ty: RenderPassType) -> Self {
        RenderPass {
            name: String::from(name),
            ty,
            color_targets: Vec::new(),
            depth_stencil: None,
            mannual_transition: None,
            logic: None,
        }
    }

    pub fn get_color_targets(&self) -> &[ColorTarget] {
        &self.color_targets
    }
}

pub struct RenderPassBuilder<'a, 'b> {
    inner: Option<RenderPass<'a>>,
    render_graph: &'b mut RenderGraph<'a>,
}

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

    pub fn color_targets(mut self, rts: &[ColorTarget]) -> Self {
        self.inner().color_targets.clear();
        self.inner().color_targets.extend_from_slice(rts);
        self
    }

    pub fn depth_stencil(mut self, ds: DepthStencilTarget) -> Self {
        self.inner().depth_stencil = Some(ds);
        self
    }

    pub fn mannual_transition<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&CommandBuffer, &RenderPass) + 'a,
    {
        self.inner().mannual_transition = Some(Box::new(f));
        self
    }

    pub fn logic<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&CommandBuffer, &Shaders, RenderPass) + 'a,
    {
        self.inner().logic = Some(Box::new(f));
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

// A Render Graph to handle resource transitions automatically
pub struct RenderGraph<'a> {
    passes: Vec<RenderPass<'a>>,

    // Array indexed by RGHandle
    texture_descs: Vec<TextureDesc>,
    texture_view_descs: Vec<TextureViewDesc>,
    texture_view_textures: Vec<RGHandle<Texture>>,

    // Textures
    textures: Vec<Texture>,          // by texture id
    texture_views: Vec<TextureView>, // by texture id

    // Resource pool
    texture_pool: ResPool<TextureDesc, Texture>,
    texture_view_pool: ResPool<TextureViewDesc, TextureView>,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            texture_descs: Vec::new(),
            texture_view_descs: Vec::new(),
            texture_view_textures: Vec::new(),
            textures: Vec::new(),
            texture_views: Vec::new(),
            texture_pool: ResPool::new(),
            texture_view_pool: ResPool::new(),
        }
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

    // TOOD actually build a DAG, and properly sync before read
    pub fn execute(
        &mut self,
        rd: &RenderDevice,
        command_buffer: &CommandBuffer,
        shaders: &Shaders,
    ) {
        // Populate textures and views
        // TODO manage lifetime by DAG and do memory aliasing
        for i in 0..self.texture_descs.len() {
            let desc = self.texture_descs[i];
            let texture = self
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
                .texture_view_pool
                .pop(&desc)
                .unwrap_or_else(|| rd.create_texture_view(texture, desc).unwrap());
            self.texture_views.push(view);
        }

        let passes = self.passes.drain(0..self.passes.len()).collect::<Vec<_>>();
        for mut pass in passes {
            // Just-in-time transition, assuming all dependent lass is executed
            // TODO analysis of the DAG and do it properly
            //self.just_in_time_transition(command_buffer, &pass);

            if let Some(mannual_transition) = pass.mannual_transition.take() {
                mannual_transition(command_buffer, &mut pass);
            }

            // Begin render pass (if graphics)
            let is_graphic = pass.ty == RenderPassType::Graphics;
            if is_graphic {
                self.begin_graphics(command_buffer, &pass);
            }

            // Run pass
            let logic = pass.logic.take().unwrap();
            logic(command_buffer, &shaders, pass);

            // End render pass (if graphics)
            if is_graphic {
                self.end_graphics(command_buffer);
            }
        }

        // Pool back all resource objects
        for view in self.texture_views.drain(0..self.texture_views.len()) {
            self.texture_view_pool.push(view.desc, view);
        }
        for texture in self.textures.drain(0..self.textures.len()) {
            self.texture_pool.push(texture.desc, texture);
        }
    }
}
