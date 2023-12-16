// If nightly is allowed,
// enable core_intrinsics to use:
//   - std::intrinsics::breakpoint
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]

pub mod app;
pub mod command_buffer;
pub mod gpu_profiling;
pub mod imgui;
pub mod model;
pub mod render_device;
pub mod render_graph;
pub mod render_loop;
pub mod render_scene;
pub mod renderdoc;
pub mod shader;
pub mod window;
