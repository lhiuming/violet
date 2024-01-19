// If nightly is allowed,
// enable core_intrinsics to use:
//   - std::intrinsics::breakpoint
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]

pub mod app;
pub mod imgui;
pub mod render_loop;
pub mod renderdoc;
pub mod window;
