use crate::command_buffer::CommandBuffer;
use crate::render_device::{Handle, Texture, TextureDesc, TextureView, TextureViewDesc};
use crate::shader::Shaders;
use std::marker::{Copy, PhantomData};
use std::ops::FnOnce;

pub struct RGHandle<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

impl<T> RGHandle<T> {
    pub fn null() -> Self {
        RGHandle {
            id: 0,
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

pub struct RenderPass<'a> {
    //params: dyn Paramter,
    //read_textures: Vec<RGHandle<Texture>>,
    //input_buffers: RGHandle<>
    //mutable_textures: Vec<RGHandle<Texture>>,
    name: String,
    color_targets: Vec<RGHandle<TextureView>>,
    depth_stencil: RGHandle<TextureView>,
    logic: Box<dyn 'a + FnOnce(CommandBuffer, Shaders, RenderPass)>,
}

impl<'a> RenderPass<'a> {
    pub fn new(name: &str) -> Self {
        RenderPass {
            name: String::from(name),
            color_targets: Vec::new(),
            depth_stencil: RGHandle::null(),
            logic: Box::new(Self::empty_logic),
        }
    }

    fn empty_logic(_cb: CommandBuffer, _shaders: Shaders, _rp: RenderPass) {
        println!("You must provide a render pass logic!");
    }

    pub fn color_targets(&mut self, rts: &[RGHandle<TextureView>]) -> &mut Self {
        self.color_targets.clear();
        //self.color_targets.copy_from_slice(rts); // WHY can I USE this?
        for rt in rts {
            self.color_targets.push(*rt);
        }
        self
    }

    pub fn depth_stencil(&mut self, ds: RGHandle<TextureView>) -> &mut Self {
        self.depth_stencil = ds;
        self
    }

    pub fn logic<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(CommandBuffer, Shaders, RenderPass) + 'a,
    {
        self.logic = Box::new(f);
        self
    }
}

// A Render Graph to handle resource transitions automatically
pub struct RenderGraph<'a> {
    // todo
    passes: Vec<RenderPass<'a>>,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn create_texutre(&mut self, desc: TextureDesc) -> RGHandle<Texture> {
        unimplemented!()
    }

    pub fn create_texture_view(
        &mut self,
        texture: RGHandle<Texture>,
        desc: TextureViewDesc,
    ) -> RGHandle<TextureView> {
        unimplemented!()
    }

    pub fn register_texture(&mut self, texture: Handle<Texture>) -> RGHandle<Texture> {
        unimplemented!()
    }

    pub fn register_texture_view(
        &mut self,
        texture_view: Handle<TextureView>,
    ) -> RGHandle<TextureView> {
        unimplemented!()
    }

    pub fn add_pass(&mut self, pass: RenderPass<'a>) {
        self.passes.push(pass);
    }

    pub fn new_pass(&mut self, name: &str) -> &'a mut RenderPass {
        self.passes.push(RenderPass::new(name));
        self.passes.last_mut().unwrap()
    }

    pub fn execute(&mut self) {}
}
