# 🌈violet

`violet` is my hobby project for experimenting new rendering stuffs, written in 🦀Rust and 🎨HLSL. Currently it features:

- Vulkan backend
- glTF model loading
- A Reference Path Tracer
- ReSTIR-based [hybrid-renderer](https://github.com/lhiuming/violet/tree/main/src/bin/restir)
- SVGF/ReLAX style denoiser

<img width="961" alt="violet_restir" src="https://github.com/lhiuming/violet/assets/2281376/bf378afc-29d4-470d-9c29-04f25434ff72">

## Building and Running

Firstly, you need to install [Rust](https://www.rust-lang.org/tools/install), and possibly install an Vulkan driver for you graphics card (e.g. [nvdia](https://developer.nvidia.com/vulkan-driver)).

Then you can build, e.g., the `restir` app, and view the "Sponza" model:

```
cargo run --bin restir --release -- "./assets/Sponza/Sponza.gltf"
```

## Acknowledgments

This project benefits from a bunch of wonderful open-source projects, especially: 

- [ash](https://github.com/ash-rs/ash): thin and simple, should be your go-to rust binding for Vulkan!
- [egui](https://github.com/emilk/egui): lovely ImGUI in pure rust.
- [puffin](https://github.com/EmbarkStudios/puffin): easy to use profile for Rust (it also comes with a integration with egui👏)
- [rspirv-reflect](https://github.com/Traverse-Research/rspirv-reflect): minimalism SPIR-V reflection libray.
- [MinimalAtmosphere](https://github.com/Fewes/MinimalAtmosphere): single-file atmospheric scattering implementation from Felix Westin.
