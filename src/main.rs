#![feature(core_intrinsics)]
use std::env;
use std::path::Path;

use glam::{Mat4, Vec3, Vec4};

mod window;
use window::Window;

mod render_device;
use render_device::RenderDevice;

mod shader;
use shader::Shaders;

mod model;

mod render_loop;
use render_loop::{RednerLoop, RenderScene};

// Assumming positive Z; mapping near-plane to 1, far-plane to 0 (reversed Z).
// Never flip y (or x).
fn perspective_projection(
    near_plane: f32,
    far_plane: f32,
    fov_horizontal_radian: f32,
    width_by_height: f32,
) -> Mat4 {
    let ran = (fov_horizontal_radian / 2.0).tan();
    let width = near_plane * ran;
    let m00 = near_plane / width;
    let m11 = near_plane * width_by_height / width;
    let m22 = -near_plane / (far_plane - near_plane);
    //let m23 = far_plane * near_plane / (far_plane - near_plane);
    // NOTE: this allow far_plane -> infinite
    let m23 = near_plane / (1.0 - near_plane / far_plane);

    /* col 0    1    2    3
         m00, 0.0, 0.0, 0.0,
         0.0, m11, 0.0, 0.0,
         0.0, 0.0, m22, m23,
         0.0, 0.0, 1.0, 0.0,
    */
    Mat4::from_cols_array_2d(&[
        [m00, 0.0, 0.0, 0.0],
        [0.0, m11, 0.0, 0.0],
        [0.0, 0.0, m22, 1.0],
        [0.0, 0.0, m23, 0.0],
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
    let mut shaders = Shaders::new(&rd);

    // Set up the scene
    let mut render_scene = RenderScene::new(&rd);

    let args: Vec<String> = env::args().collect();
    let model = model::load(Path::new(&args[1]));
    if let Ok(model) = model {
        render_scene.add(&rd, &model);
    } else {
        println!(
            "Failed to load model ({}): {:?}",
            &args[1],
            model.err().unwrap()
        );
    }

    let render_loop = RednerLoop::new(&rd);

    // Init camera
    // NOTE:
    //   - Using a right-hand coordinate in both view and world space;
    //   - View/camera space: x-axis is right, y-axis is down (, z-axis is forward);
    //   - World/background space: camera yz-plane is kept paralled to world z-axis (up/anti-gravity direction);
    let up_dir = Vec3::new(0.0, 0.0, 1.0);
    let mut camera_dir = Vec3::new(0.0, 1.0, 0.0);
    let mut camera_right = Vec3::new(1.0, 0.0, 0.0); // derived
    let mut camera_down = Vec3::new(0.0, 0.0, -1.0); // derived
    let mut camera_pos = Vec3::new(0.0, -5.0, 2.0);
    let mut prev_time = std::time::Instant::now();

    while !window.should_close() {
        window.poll_events();

        // Reload shaders
        if window.click_R() {
            shaders.reload_all();
        }

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
        let view_proj: Mat4;
        {
            let fov = (90.0f32).to_radians(); // horizontal

            // Camera navigation
            {
                // Move (by WASD+EQ)
                let (forward, right, up) = window.nav_dir();
                let speed = 2.0; // meter per secs
                let mov = speed * delta_seconds;
                camera_pos += (forward * mov) * camera_dir;
                camera_pos += (right * mov) * camera_right;
                camera_pos += (-up * mov) * camera_down;

                // Rotate (by mouse darg with right button pressed)
                if let Some((beg_x, beg_y, end_x, end_y)) = window.effective_darg() {
                    let w = swapchain.extent.width as f32;
                    let h = swapchain.extent.height as f32;
                    let right_x = (fov * 0.5).tan() * 1.0;
                    let down_y = right_x * h / w;
                    let screen_pos_to_world_dir = |x: i16, y: i16| -> Vec3 {
                        let x = right_x * ((x as f32) / w * 2.0 - 1.0);
                        let y = down_y * ((y as f32) / h * 2.0 - 1.0);
                        let dir = Vec3::new(x, y, 1.0).normalize(); // in camera space
                        dir.x * camera_right + dir.y * camera_down + dir.z * camera_dir
                    };

                    let from_dir = screen_pos_to_world_dir(beg_x, beg_y);
                    let to_dir = screen_pos_to_world_dir(end_x, end_y);
                    let rot_axis = from_dir.cross(to_dir);
                    if let Some(rot_axis) = rot_axis.try_normalize() {
                        let rot_cos = from_dir.dot(to_dir);
                        let rot_sin = (1.0 - rot_cos * rot_cos).sqrt();
                        camera_dir = rot_cos * camera_dir
                            + rot_sin * rot_axis.cross(camera_dir)
                            + (1.0 - rot_cos) * rot_axis.dot(camera_dir) * rot_axis;
                        camera_dir = camera_dir.normalize(); // avoid precision loss
                        camera_right = camera_dir.cross(up_dir).normalize();
                        camera_down = camera_dir.cross(camera_right).normalize();
                    }
                }
            }

            // World-to-view transform
            let pos_comp = Vec3 {
                x: -camera_right.dot(camera_pos),
                y: -camera_down.dot(camera_pos),
                z: -camera_dir.dot(camera_pos),
            };
            let view = Mat4::from_cols(
                camera_right.extend(pos_comp.x),
                camera_down.extend(pos_comp.y),
                camera_dir.extend(pos_comp.z),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            )
            .transpose();

            // Perspective proj
            let width_by_height =
                (swapchain.extent.width as f32) / (swapchain.extent.height as f32);
            let proj = perspective_projection(0.05, 102400.0, fov, width_by_height);

            view_proj = proj * view;
        }

        render_loop.render(&rd, &mut shaders, &render_scene, view_proj);
    }
}
