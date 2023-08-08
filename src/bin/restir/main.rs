mod restir_render_loop;
use restir_render_loop::RestirRenderLoop;

use violet::app;

fn main() {
    app::run_with_renderloop::<RestirRenderLoop>();
}
