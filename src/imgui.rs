/*
Integration of egui.
*/

use egui::{Event, Key, *};
use glam::UVec2;

use crate::window;
use crate::window::Message;

pub use egui::{TopBottomPanel, Ui, Window};

pub struct ImGUI {
    pub egui_ctx: egui::Context,
    curr_pixels_per_point: f32,
}

// Things to be render/updated on GPU.
pub struct ImGUIOuput {
    //pub repaint_after: std::time::Duration,
    pub textures_delta: egui::epaint::textures::TexturesDelta,
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
    pub pixels_per_point: f32,
}

fn transmute_key(discriminant: u8) -> Key {
    let key: Key = unsafe { std::mem::transmute(discriminant) };
    assert!(key as u8 == discriminant);
    key
}

fn char_to_egui_key(char: u8) -> Option<Key> {
    let discriminant = if (b'a' <= char) && (char <= b'z') {
        Key::A as u8 + (char - b'a') as u8
    } else if (b'0' <= char) && (char <= b'9') {
        Key::Num0 as u8 + (char - b'0') as u8
    } else {
        return None;
    };
    Some(transmute_key(discriminant))
}

// Limited integration of egui -- platform output is ignored.
impl ImGUI {
    pub fn new() -> Self {
        let egui_ctx = egui::Context::default();
        // Setup style
        let mut style = egui::Style::default();
        style
            .override_text_style
            .replace(egui::TextStyle::Monospace);
        style.visuals.window_shadow = epaint::Shadow::NONE;
        egui_ctx.set_style(style);
        Self {
            egui_ctx,
            curr_pixels_per_point: 1.0,
        }
    }

    // Gather input (mouse, touches, keyboard, screen size, etc):
    pub fn gather_input(
        &self,
        window_size: UVec2,
        window: &window::Window,
        time: Option<f64>,
    ) -> egui::RawInput {
        let pixels_per_point = window.pixels_per_point();
        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect {
                min: egui::Pos2 { x: 0.0, y: 0.0 },
                max: egui::Pos2 {
                    x: window_size.x as f32 / pixels_per_point,
                    y: window_size.y as f32 / pixels_per_point,
                },
            }),
            pixels_per_point: Some(pixels_per_point),
            time,
            ..Default::default()
        };

        let to_pos2 =
            |x: &i16, y: &i16| pos2(*x as f32 / pixels_per_point, *y as f32 / pixels_per_point);

        raw_input.events = window
            .msg_stream()
            .iter()
            .map(|msg| match msg {
                Message::Key { char, pressed } => Event::Key {
                    key: char_to_egui_key(*char).unwrap(),
                    pressed: *pressed,
                    repeat: false,
                    modifiers: Default::default(),
                },
                Message::MouseMove { x, y } => Event::PointerMoved(to_pos2(x, y)),
                Message::MouseButton {
                    x,
                    y,
                    left,
                    pressed,
                } => {
                    let button = if *left {
                        PointerButton::Primary
                    } else {
                        PointerButton::Secondary
                    };

                    Event::PointerButton {
                        pos: to_pos2(x, y),
                        button,
                        pressed: *pressed,
                        modifiers: Default::default(),
                    }
                }
            })
            .collect();

        raw_input
    }

    // Before adding ui elements
    pub fn begin(&mut self, raw_input: egui::RawInput) {
        if let Some(pixels_per_point) = raw_input.pixels_per_point {
            self.curr_pixels_per_point = pixels_per_point;
        }
        self.egui_ctx.begin_frame(raw_input);
    }

    // After adding all ui elements
    pub fn end(&mut self) -> ImGUIOuput {
        let full_output = self.egui_ctx.end_frame();
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes); // creates triangles to paint
        ImGUIOuput {
            textures_delta: full_output.textures_delta,
            clipped_primitives,
            pixels_per_point: self.curr_pixels_per_point,
        }
    }

    // Instead of gather_input, begin, end, just call this.
    #[inline]
    pub fn run(
        &mut self,
        window_size: UVec2,
        window: &window::Window,
        time_secs: Option<f64>,
        add_ui: impl FnOnce(&egui::Context),
    ) -> ImGUIOuput {
        self.begin(self.gather_input(window_size, window, time_secs));
        add_ui(&self.egui_ctx);
        self.end()
    }
}
