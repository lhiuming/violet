use std::env;
use std::mem;

use ash::vk;

mod window;
use window::Window;

mod render_device;
use render_device::RenderDevice;

mod shader;
use shader::{
    create_compute_pipeline, create_graphics_pipeline, load_shader, PipelineDevice,
    ShaderDefinition, ShaderStage,
};

mod gltf_asset;

mod render_loop;
use render_loop::{RednerLoop, RenderScene};

use crate::gltf_asset::UploadContext;
use crate::render_device::TextureDesc;
use crate::render_device::TextureViewDesc;
use crate::render_loop::AllocBuffer;
use crate::render_loop::AllocTexture2D;

#[repr(C)]
#[derive(Debug)]
pub struct float3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl float3 {
    pub fn new(x: f32, y: f32, z: f32) -> float3 {
        float3 { x, y, z }
    }

    pub fn add_vector(a: &float3, b: &float3) -> float3 {
        float3::new(a.x + b.x, a.y + b.y, a.z + b.z)
    }

    pub fn cross(a: &float3, b: &float3) -> float3 {
        float3 {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }

    pub fn dot(a: &float3, b: &float3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn mul_scalar(a: &float3, b: &f32) -> float3 {
        float3::new(a.x * b, a.y * b, a.z * b)
    }

    pub fn mul_vector(a: &float3, b: &float3) -> float3 {
        float3::new(a.x * b.x, a.y * b.y, a.z * b.z)
    }

    pub fn normalize(a: &float3) -> float3 {
        float3::mul_scalar(a, &(1.0 / a.len()))
    }

    pub fn len(&self) -> f32 {
        float3::dot(self, self).sqrt()
    }
}

// TODO use something like impl_ops
impl std::ops::AddAssign<float3> for float3 {
    fn add_assign(&mut self, rhs: float3) {
        *self = float3::add_vector(self, &rhs)
    }
}
impl std::ops::Add<float3> for float3 {
    type Output = float3;

    fn add(self, rhs: float3) -> Self::Output {
        float3::add_vector(&self, &rhs)
    }
}
impl std::ops::Mul<&float3> for f32 {
    type Output = float3;

    fn mul(self, rhs: &float3) -> Self::Output {
        float3::mul_scalar(rhs, &self)
    }
}

#[repr(C)]
pub struct float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl float4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> float4 {
        float4 { x, y, z, w }
    }
    pub fn from3(xyz: &float3, w: f32) -> float4 {
        float4 {
            x: xyz.x,
            y: xyz.y,
            z: xyz.z,
            w: w,
        }
    }
}

// Matrix 4x4 type, row major
#[repr(C)]
pub struct float4x4 {
    pub rows: [float4; 4],
}

impl float4x4 {
    pub fn from_rows(rows: [float4; 4]) -> float4x4 {
        float4x4 { rows }
    }

    pub fn from_data(data: &[f32; 16]) -> float4x4 {
        unsafe {
            let mut ret: float4x4 = mem::MaybeUninit::uninit().assume_init();
            let src = data.as_ptr();
            let dst = ret.rows.as_mut_ptr() as *mut f32;
            dst.copy_from_nonoverlapping(src, 16);
            ret
        }
    }

    pub fn mul(lhs: &float4x4, rhs: &float4x4) -> float4x4 {
        unsafe {
            let mut ret: float4x4 = mem::MaybeUninit::uninit().assume_init();
            let dst_ptr = ret.rows.as_mut_ptr() as *mut f32;
            let dst = std::slice::from_raw_parts_mut(dst_ptr, 16);
            let rhs_ptr = rhs.rows.as_ptr() as *const f32;
            let rhs = std::slice::from_raw_parts(rhs_ptr, 16);
            let mut dst_ind = 0;
            for i in 0..4 {
                let lhs_row = &lhs.rows[i];
                for j in 0..4 {
                    // lhr.row[i] dot rhs.col[j]
                    dst[dst_ind] = lhs_row.x * rhs[j + 0]
                        + lhs_row.y * rhs[j + 4]
                        + lhs_row.z * rhs[j + 8]
                        + lhs_row.w * rhs[j + 12];
                    dst_ind += 1;
                }
            }
            ret
        }
    }
}

// Assumming positive Z; mapping near-plane to 1, far-plane to 0 (reversed Z).
// Never flip y (or x).
fn perspective_projection(
    near_plane: f32,
    far_plane: f32,
    fov_horizontal_radian: f32,
    width_by_height: f32,
) -> float4x4 {
    let ran = (fov_horizontal_radian / 2.0).tan();
    let width = near_plane * ran;
    let m00 = near_plane / width;
    let m11 = near_plane * width_by_height / width;
    let m22 = -near_plane / (far_plane - near_plane);
    //let m23 = far_plane * near_plane / (far_plane - near_plane);
    // NOTE: this allow far_plane -> infinite
    let m23 = near_plane / (1.0 - near_plane / far_plane);
    float4x4::from_data(&[
        m00, 0.0, 0.0, 0.0, //
        0.0, m11, 0.0, 0.0, //
        0.0, 0.0, m22, m23, //
        0.0, 0.0, 1.0, 0.0, //
    ])
}

fn main() {
    println!("Hello, rusty world!");

    // Create a system window
    // TODO implement Drop for Window
    let window = Window::new(1280, 720, "Rusty Violet");

    let rd = RenderDevice::create(Window::system_handle_for_module(), window.system_handle());
    let swapchain = &rd.swapchain;

    // Initialize shaders
    let pipeline_device = PipelineDevice::new(&rd);
    let mesh_cs_def = ShaderDefinition::new("MeshCS.hlsl", "main", ShaderStage::Compute);
    let mesh_cs_pipeline = {
        let shader = load_shader(&pipeline_device, &mesh_cs_def).unwrap();
        create_compute_pipeline(&pipeline_device, &mesh_cs_def, &shader)
    };
    let mesh_vs_def = ShaderDefinition::new("MeshVSPS.hlsl", "vs_main", ShaderStage::Vert);
    let mesh_ps_def = ShaderDefinition::new("MeshVSPS.hlsl", "ps_main", ShaderStage::Frag);
    let mesh_gfx_pipeline = {
        let vs = load_shader(&pipeline_device, &mesh_vs_def).unwrap();
        let ps = load_shader(&pipeline_device, &mesh_ps_def).unwrap();
        create_graphics_pipeline(&pipeline_device, &vs, &ps)
    };

    // Buffer for whole scene
    let ib_size = 4 * 1024 * 1024;
    let vb_size = 4 * 1024 * 1024;
    let mut index_buffer = AllocBuffer::new(rd.create_buffer(
        ib_size,
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::Format::UNDEFINED,
    )
    .unwrap());
    let mut vertex_buffer = AllocBuffer::new(rd.create_buffer(
        vb_size,
        vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
        vk::Format::R32_UINT,
    )
    .unwrap());

    // Texture for whole scene
    let tex_width = 2048;
    let tex_height = 2048;
    let tex_array_len = 5;
    let mut material_texture = {
        let texture = rd.create_texture(TextureDesc::new_2d_array(
            tex_width,
            tex_height,
            tex_array_len,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )).unwrap();
        let texture_view = rd.create_texture_view(&texture, TextureViewDesc::default(&texture)).unwrap();
        AllocTexture2D::new(texture, texture_view)
    };

    let args: Vec<String> = env::args().collect();

    let mut upload_context = UploadContext::new(&rd);

    // Read the gltf model
    let gltf = if args.len() > 1 {
        gltf_asset::load(&args[1], &rd, &mut upload_context, &mut index_buffer, &mut vertex_buffer, &mut material_texture)
    } else {
        None
    };

    let render_scene = RenderScene {
        vertex_buffer,
        index_buffer,
        material_texture,
        mesh_gfx_pipeline,
        mesh_cs_pipeline,
        gltf,
    };

    let render_loop = RednerLoop::new(&rd);

    // Init camera
    // NOTE:
    //   - Using a right-hand coordinate in both view and world space;
    //   - View/camera space: x-axis is right, y-axis is down (, z-axis is forward);
    //   - World/background space: camera yz-plane is kept paralled to world z-axis (up/anti-gravity direction);
    let up_dir = float3::new(0.0, 0.0, 1.0);
    let mut camera_dir = float3::new(0.0, 1.0, 0.0);
    let mut camera_right = float3::new(1.0, 0.0, 0.0); // derived
    let mut camera_down = float3::new(0.0, 0.0, -1.0); // derived
    let mut camera_pos = float3::new(0.0, -5.0, 2.0);
    let mut prev_time = std::time::Instant::now();

    while !window.should_close() {
        window.poll_events();

        // Time udpate
        let delta_seconds;
        {
            let curr_time = std::time::Instant::now();
            let delta_time = curr_time.duration_since(prev_time);
            prev_time = curr_time;
            delta_seconds = delta_time.as_secs_f32();
            //println!("Violet: frame delta time: {}ms", delta_seconds / 1000.0);
        }

        // Update camera
        let view_proj: float4x4;
        {
            let fov = (90.0f32).to_radians(); // horizontal

            // Camera navigation
            {
                // Move (by WASD+EQ)
                let (forward, right, up) = window.nav_dir();
                let speed = 2.0; // meter per secs
                let mov = speed * delta_seconds;
                camera_pos += (forward * mov) * &camera_dir;
                camera_pos += (right * mov) * &camera_right;
                camera_pos += (-up * mov) * &camera_down;

                // Rotate (by mouse darg with right button pressed)
                if let Some((beg_x, beg_y, end_x, end_y)) = window.effective_darg() {
                    let w = swapchain.extent.width as f32;
                    let h = swapchain.extent.height as f32;
                    let right_x = (fov * 0.5).tan() * 1.0;
                    let down_y = right_x * h / w;
                    let screen_pos_to_world_dir = |x: i16, y: i16| -> float3 {
                        let x = right_x * ((x as f32) / w * 2.0 - 1.0);
                        let y = down_y * ((y as f32) / h * 2.0 - 1.0);
                        let dir = float3::normalize(&float3::new(x, y, 1.0)); // in camera space
                        dir.x * &camera_right + dir.y * &camera_down + dir.z * &camera_dir
                    };

                    let from_dir = screen_pos_to_world_dir(beg_x, beg_y);
                    let to_dir = screen_pos_to_world_dir(end_x, end_y);
                    let rot_axis = float3::cross(&from_dir, &to_dir);
                    if rot_axis.len() > 0.0f32 {
                        let rot_axis = float3::normalize(&rot_axis);
                        let rot_cos = float3::dot(&from_dir, &to_dir);
                        let rot_sin = (1.0 - rot_cos * rot_cos).sqrt();
                        camera_dir = rot_cos * &camera_dir
                            + rot_sin * &float3::cross(&rot_axis, &camera_dir)
                            + (1.0 - rot_cos) * float3::dot(&rot_axis, &camera_dir) * &rot_axis;
                        camera_dir = float3::normalize(&camera_dir); // avoid precision loss
                        camera_right = float3::normalize(&float3::cross(&camera_dir, &up_dir));
                        camera_down = float3::normalize(&float3::cross(&camera_dir, &camera_right));
                    }
                }
            }

            // World-to-view transform
            let pos_comp = float3 {
                x: -float3::dot(&camera_right, &camera_pos),
                y: -float3::dot(&camera_down, &camera_pos),
                z: -float3::dot(&camera_dir, &camera_pos),
            };
            let view = float4x4::from_rows([
                float4::from3(&camera_right, pos_comp.x),
                float4::from3(&camera_down, pos_comp.y),
                float4::from3(&camera_dir, pos_comp.z),
                float4::new(0.0, 0.0, 0.0, 1.0),
            ]);

            // Perspective proj
            let width_by_height =
                (swapchain.extent.width as f32) / (swapchain.extent.height as f32);
            let proj = perspective_projection(0.05, 102400.0, fov, width_by_height);

            view_proj = float4x4::mul(&proj, &view);
        }

        render_loop.render(&rd, &render_scene, view_proj);
    }
}
