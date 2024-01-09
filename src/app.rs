use std::any::type_name;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::path::Path;
use std::time;

use clap::Parser;
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};

use crate::{
    imgui, model,
    render_device::{DeviceConfig, RenderDevice},
    render_loop::{RenderLoop, ViewInfo},
    render_scene::RenderScene,
    renderdoc,
    shader::Shaders,
    window::Window,
};

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

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Relative Path for the glTF model to load
    path: Option<String>,

    /// Enable V-Sync or not
    #[arg(long)]
    vsync: bool,

    /// Load RenderDoc or not
    #[arg(long)]
    renderdoc: bool,

    /// Enable in-game Profiler or not
    #[arg(long)]
    profiler: bool,

    /// Force re-import assets, even if cache is available
    #[arg(long)]
    reimport_assets: bool,

    /// Force to disable texture compression
    #[arg(long)]
    disable_tex_compress: bool,

    /// Camera parameter presets (position Vec3 and angle Vec2)
    #[arg(long, num_args = 0..=45, value_delimiter = ',')]
    camera_preset: Vec<f32>,
}

// NOTE:
//   - Using a right-hand coordinate in both view and world space;
//   - View/camera space: x-axis is right, y-axis is down (, z-axis is forward);
//   - World/background space: camera yz-plane is kept paralled to world z-axis (up/anti-gravity direction, so called Z-Up);
struct Camera {
    pos: Vec3,
    right: Vec3,   // X
    down: Vec3,    // Y
    forward: Vec3, // Z
}

static CAMERA_UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

impl Camera {
    pub fn from_pos_angle(pos: Vec3, yaw_pitch: Vec2) -> Camera {
        // default:
        //   forward = Vec3::new(0.0, 1.0, 0.0);
        //   right   = Vec3::new(1.0, 0.0, 0.0);
        // apply yaw and pitch
        let yaw = yaw_pitch.x.to_radians().clamp(-PI, PI);
        let pitch = yaw_pitch.y.to_radians().clamp(-PI * 0.5, PI * 0.5);
        let right = Vec3::new(yaw.cos(), -yaw.sin(), 0.0);
        let forward = Vec3::new(
            yaw.sin() * pitch.cos(),
            yaw.cos() * pitch.cos(),
            pitch.sin(),
        );
        let down = forward.cross(right).normalize();
        Camera {
            pos,
            right,
            down,
            forward,
        }
    }

    pub fn get_angle(&self) -> Vec2 {
        let yaw = (-self.right.y).atan2(self.right.x).to_degrees();
        let pitch = self.forward.z.asin().to_degrees();
        Vec2::new(yaw, pitch)
    }
}

pub struct Config {
    pub optional_shader_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            optional_shader_path: None,
        }
    }
}

pub fn run_with_renderloop<T>()
where
    T: RenderLoop,
{
    run_with_config::<T>(Config::default());
}

pub fn run_with_config<T>(config: Config)
where
    T: RenderLoop,
{
    let args = Args::parse();

    println!("Hello, rusty world!");

    // Load RenderDoc
    let rdoc = if args.renderdoc {
        renderdoc::RenderDoc::new()
    } else {
        None
    };

    // Enable in-game Profiler
    puffin::set_scopes_on(args.profiler);

    // Create a system window
    // TODO implement Drop for Window
    let window_size = UVec2::new(1920, 1080);
    let mut window = Window::new(window_size, "Rusty Violet");

    let mut rd = {
        let config = DeviceConfig {
            app_handle: Window::system_handle_for_module(),
            window_handle: window.system_handle(),
            vsync: args.vsync,
        };
        RenderDevice::create(config).unwrap()
    };

    // Initialize shaders
    let mut shaders = Shaders::new(&rd);
    if let Some(path) = config.optional_shader_path {
        shaders.add_path(path);
    }

    // Set up the scene
    let mut render_scene = RenderScene::new(&rd);

    if let Some(path) = &args.path {
        println!("Loading model: {}", path);
        let config = model::LoadConfig {
            force_reimport: args.reimport_assets,
            tex_compression: !args.disable_tex_compress,
        };
        let model = model::load(Path::new(&path), config);
        if let Ok(model) = model {
            println!("Uploading to GPU ...");
            render_scene.add(&rd, &model);
            render_scene.rebuild_top_level_accel_struct(&rd);
        } else {
            println!(
                "Failed to load model ({}): {:?}",
                path,
                model.err().unwrap()
            );
        }
    }

    // Create render loop
    let mut render_loop = match T::new(&mut rd) {
        Some(rl) => rl,
        None => {
            panic!("Renderloop is not supported.");
        }
    };

    // Add ImGUI
    let mut imgui = imgui::ImGUI::new();
    let mut show_gui = true;
    let mut full_stat = false;
    let mut stat_filter_ms = 0.05;

    // Init Camera
    static CAMERA_INIT_POS: Vec3 = Vec3::new(0.0, -4.0, 2.0);
    let mut camera = Camera::from_pos_angle(CAMERA_INIT_POS, Vec2::ZERO);
    let move_speed = 20.0f32; // uint per second

    // Init sun
    let mut sun_dir_theta = -0.271f32;
    let mut sun_dir_phi = 0.524f32;

    // Init time
    let init_time = time::Instant::now();
    let mut prev_time = time::Instant::now();
    let mut frame_durations = VecDeque::<time::Duration>::new();
    let avg_delta_time_ms = |durations: &VecDeque<time::Duration>| {
        let delta_time_sum: time::Duration = durations.iter().sum();
        delta_time_sum.as_secs_f64() * 1000.0 / durations.len().max(1) as f64
    };

    // Render loop
    println!("Start RenderLoop: {:?}", type_name::<T>());
    while !window.should_close() {
        puffin::profile_scope!("MainLoop");

        window.poll_events();

        // Reload shaders
        if window.clicked('r') {
            shaders.reload_all();
            println!("App:: shader reloaded.")
        }

        // Reset camera
        if window.clicked('g') {
            camera = Camera::from_pos_angle(CAMERA_INIT_POS, Vec2::ZERO);
        }

        // Switch to one of camera preset (mapping to key 1~9)
        let num_preset = (args.camera_preset.len() / 5).min(9);
        for index in 0..num_preset {
            let c = ('1' as u8 + index as u8) as char;
            if window.clicked(c) {
                let params = &args.camera_preset[(index * 5)..((index + 1) * 5)];
                let pos = Vec3::new(params[0], params[1], params[2]);
                let angle = Vec2::new(params[3], params[4]);
                camera = Camera::from_pos_angle(pos, angle);
            }
        }

        // Time udpate
        let delta_seconds;
        {
            let curr_time = time::Instant::now();
            let delta_time = curr_time.duration_since(prev_time);
            prev_time = curr_time;
            delta_seconds = delta_time.as_secs_f32();
            if frame_durations.len() >= 32 {
                frame_durations.resize(31, time::Duration::ZERO);
            };
            frame_durations.push_front(delta_time);
        }

        // Update camera
        let view_info: ViewInfo;
        {
            let fov = (90.0f32).to_radians(); // horizontal

            // Camera navigation
            let mut moved;
            {
                // Move (by WASD+EQ)
                let (forward, right, up) = window.nav_dir();
                let mov = move_speed * delta_seconds;
                camera.pos += (forward * mov) * camera.forward;
                camera.pos += (right * mov) * camera.right;
                camera.pos += (-up * mov) * camera.down;
                moved = (forward != 0.0) || (right != 0.0) || (up != 0.0);

                // Rotate (by mouse darg with right button pressed)
                let (drag_x, drag_y) = window.effective_darg_delta();
                if (drag_x != 0) || (drag_y != 0) {
                    let drag = Vec2::new(drag_x as f32, drag_y as f32);

                    // Estimate the target direction
                    let target_dir_unnorm = {
                        let w = rd.swapchain.extent.width as f32;
                        let h = rd.swapchain.extent.height as f32;
                        let right_x = (fov * 0.5).tan() * 1.0;
                        let down_y = right_x * h / w;

                        let x = right_x * (drag.x / w * 2.0);
                        let y = down_y * (drag.y / h * 2.0);
                        x * camera.right + y * camera.down + camera.forward
                    };

                    // Use estimated target direction to get rotate axis
                    let rot_axis_unnorm = camera.forward.cross(target_dir_unnorm);

                    if let Some(rot_axis) = rot_axis_unnorm.try_normalize() {
                        const RAD_PER_PIXEL: f32 = 0.001;
                        let angle = RAD_PER_PIXEL * drag.length();
                        let (rot_sin, rot_cos) = angle.sin_cos();

                        // Rotate forward
                        camera.forward =
                            rot_cos * camera.forward + rot_sin * rot_axis.cross(camera.forward);
                        camera.forward = camera.forward.normalize(); // avoid precision loss

                        // Re-derive right and down direction
                        camera.right = camera.forward.cross(CAMERA_UP).normalize();
                        camera.down = camera.forward.cross(camera.right).normalize();

                        moved = true;
                    }
                }
            }

            // World-to-view transform
            let pos_comp = Vec3 {
                x: -camera.right.dot(camera.pos),
                y: -camera.down.dot(camera.pos),
                z: -camera.forward.dot(camera.pos),
            };
            let view = Mat4::from_cols(
                camera.right.extend(pos_comp.x),
                camera.down.extend(pos_comp.y),
                camera.forward.extend(pos_comp.z),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            )
            .transpose();

            // Perspective proj
            let near_z = 0.05f32;
            let width_by_height =
                (rd.swapchain.extent.width as f32) / (rd.swapchain.extent.height as f32);
            let proj = perspective_projection(near_z, 102400.0, fov, width_by_height);

            view_info = ViewInfo {
                view_position: camera.pos,
                view_transform: view,
                projection: proj,
                near_z,
                moved,
            };
        }

        // Update sun light
        {
            let key_pair_to_delta = |key1: char, key2: char| -> f32 {
                let mut delta = if window.pushed(key1) { 1.0f32 } else { 0.0f32 };
                delta += if window.pushed(key2) { -1.0f32 } else { 0.0f32 };
                delta
            };

            let angle_vel = 1.0;
            sun_dir_theta += key_pair_to_delta('j', 'l') * angle_vel * delta_seconds;
            sun_dir_phi += key_pair_to_delta('k', 'i') * angle_vel * delta_seconds;

            sun_dir_theta = sun_dir_theta % (PI * 2.0);
            sun_dir_phi = sun_dir_phi.clamp(-0.5 * PI, 0.5 * PI);

            render_scene.sun_dir = Vec3::new(
                sun_dir_theta.cos() * sun_dir_phi.sin(),
                sun_dir_theta.sin() * sun_dir_phi.sin(),
                sun_dir_phi.cos(),
            );
        }

        if window.clicked('c') {
            if let Some(rdoc) = &rdoc {
                rdoc.trigger_capture();
            }
        }

        // Show GUI
        if window.clicked('u') {
            show_gui = !show_gui;
        }
        let imgui_output = if show_gui {
            puffin::profile_scope!("ImGUI");

            let ui_time = (time::Instant::now() - init_time).as_secs_f64();
            Some(imgui.run(window_size, &window, Some(ui_time), |ctx| {
                // Render Loop
                imgui::Window::new("Render Loop")
                    .default_open(false)
                    .show(ctx, |ui| {
                        render_loop.ui(ui);
                    });

                // Puffin CPU Profiler
                if args.profiler {
                    imgui::Window::new("Puffin")
                        .default_open(false)
                        .show(ctx, puffin_egui::profiler_ui);
                }

                // Internal GPU Profiler
                imgui::Window::new("GPU")
                    .default_open(false)
                    .vscroll(true)
                    .show(ctx, |ui| {
                        // shader debug
                        let mut shader_debug = shaders.shader_debug();
                        ui.toggle_value(&mut shader_debug, "shader_debug");
                        shaders.set_shader_debug(shader_debug);

                        // view info
                        ui.label(format!(
                            "View Pos: x:{:.2} y:{:.2} z:{:.2}",
                            camera.pos.x, camera.pos.y, camera.pos.z
                        ));
                        let angle = camera.get_angle();
                        ui.label(format!(
                            "View Dir: yaw:{:4.1} pitch:{:4.1}",
                            angle.x, angle.y
                        ));

                        // stat
                        if let Some(stat) = render_loop.gpu_stat() {
                            ui.horizontal(|ui| {
                                ui.toggle_value(&mut full_stat, "full_stat");
                                ui.set_enabled(!full_stat);
                                ui.add(
                                    egui::Slider::new(&mut stat_filter_ms, 0.01..=0.5)
                                        .text("threshold"),
                                )
                            });

                            let hline_stroke =
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(160, 64, 224));

                            let latest_frame = stat.latest_ready_frame();
                            let entries = stat.entries(false);
                            let max_time_ns = entries
                                .max_by_key(|e| e.avg(latest_frame).0 as u64)
                                .map(|e| e.avg(latest_frame).0);

                            for entry in stat.entries(false) {
                                let (avg_time_ns, avg_freq) = entry.avg(latest_frame);
                                let avg_time_ms = avg_time_ns / 1_000_000.0;
                                // skip
                                if !full_stat {
                                    if (avg_time_ms < stat_filter_ms) || (avg_freq <= 0.0) {
                                        continue;
                                    }
                                }
                                // Text
                                ui.label(format!(
                                    "{:>5.2} {} ({:.2})",
                                    avg_time_ms,
                                    entry.name(),
                                    avg_freq
                                ));
                                // Under line (similar to Seperator)
                                let percent = (avg_time_ns / max_time_ns.unwrap()) as f32;
                                let available_space = ui.available_size_before_wrap();
                                let size = egui::vec2(available_space.x * percent, 1.0);
                                let (rect, response) =
                                    ui.allocate_at_least(size, egui::Sense::hover());
                                if ui.is_rect_visible(response.rect) {
                                    let painter = ui.painter();
                                    painter.hline(
                                        rect.left()..=rect.right(),
                                        painter.round_to_pixel(rect.center().y),
                                        hline_stroke,
                                    );
                                }
                            }
                        }
                    });

                // Mini Stat Indicators
                imgui::Window::new("STAT")
                    .anchor(egui::Align2::RIGHT_BOTTOM, egui::Vec2::ZERO)
                    .title_bar(false)
                    .frame(
                        egui::Frame::none()
                            .inner_margin(egui::Margin::same(4.0))
                            .rounding(egui::Rounding::from(4.0))
                            .fill(egui::Color32::from_black_alpha(64)),
                    )
                    .show(ctx, |ui| {
                        let avg_ms = avg_delta_time_ms(&frame_durations);
                        ui.colored_label(egui::Color32::WHITE, format!("{:3.1}ms", avg_ms));
                    });
            }))
        } else {
            None
        };

        // Render if swapchain is not changed; save a lot troubles :)
        if !window.minimized() {
            render_loop.render(
                &mut rd,
                &mut shaders,
                &render_scene,
                &view_info,
                imgui_output.as_ref(),
            );
        }

        if window.clicked('p') {
            let avg_delta_time_ms = avg_delta_time_ms(&frame_durations);
            println!("MainLoop: {:5.2}ms", avg_delta_time_ms);

            if let Some(stat) = render_loop.gpu_stat() {
                stat.print();
            }
        }

        // End Profiler Frame
        if args.profiler {
            puffin::GlobalProfiler::lock().new_frame()
        }
    }
}
