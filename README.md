# `🌈 violet`

[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/lhiuming/violet/blob/main/LICENSE)
[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/lhiuming/violet/main)](https://github.com/lhiuming/violet/commits/main)

`violet` is a hobby project for experimenting new rendering techniques with 🦀Rust and 🌋Vulkan. Currently it features:

- A ReSTIR-based hybrid-renderer, with real-time ray-traced GI
- SVGF/ReLAX style denoiser, filtering indirect diffuse and specular separately
- Spatial Hash Grid world radiance cache (scratch)
- Reference Path tracer
- glTF model loading
- Render graph (scratch)

![Sponza rendered with restir](https://github.com/lhiuming/violet/assets/2281376/527df52f-1130-43c2-a38f-8a2f1467d43a)
(*Sponza* rendererd with `violet-restir`)

## Dig into the Codebase

- [restir_renderer.rs](crates/violet-restir/src/restir_renderer.rs): render passes for ReSTIR-based indirect lighting.
- [shaders](shader)

## Building and Running

Firstly, you need to install [Rust](https://www.rust-lang.org/tools/install) and, if necessary, the Vulkan driver for your graphics card (e.g. [nvdia](https://developer.nvidia.com/vulkan-driver)).

Then, you can build and run the `restir` app to view the "Sponza" model:

```
cargo run --bin restir --release -- "./assets/Sponza/Sponza.gltf"
```

## Acknowledgments

This project benefits from a bunch of wonderful open-source projects, including: 

- [ash](https://github.com/ash-rs/ash): thin and simple, should be your go-to rust binding for Vulkan!
- [egui](https://github.com/emilk/egui): lovely ImGUI written in pure rust.
- [puffin](https://github.com/EmbarkStudios/puffin): easy to use profile for Rust (it also comes with a integration with `egui`👏)
- [rspirv-reflect](https://github.com/Traverse-Research/rspirv-reflect): minimalism SPIR-V reflection libray.
- [MinimalAtmosphere](https://github.com/Fewes/MinimalAtmosphere): single-file atmospheric scattering implementation from Felix Westin.

And special thanks to @h3r2tic for his brilliant [💡kajiya](https://github.com/EmbarkStudios/kajiya), which inspired this entire journey into Rust and Vulkan rendering.
