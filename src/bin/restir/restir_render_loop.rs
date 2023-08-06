use violet::{
    render_device::RenderDevice,
    render_loop::{RenderLoop, ViewInfo},
    render_scene::RenderScene,
    shader::Shaders,
};

pub struct RestirRenderLoop {}

impl RenderLoop for RestirRenderLoop {
    fn new(rd: &RenderDevice) -> Self {
        // TODO
        Self {}
    }

    fn render(
        &mut self,
        rd: &mut RenderDevice,
        shaders: &mut Shaders,
        scene: &RenderScene,
        view_info: &ViewInfo,
    ) {
        // TODO
    }
}
