mod denoising;
mod reference_path_tracer;
mod restir_render_loop;
mod restir_renderer;

use restir_render_loop::RestirRenderLoop;
use violet_app::app;

fn main() {
    app::run_with_renderloop::<RestirRenderLoop>();
}
