/*
Integration of egui.
*/

use egui::{Event, Key, *};
use glam::UVec2;

use crate::window::{Message, Window};

pub use egui::Ui;

pub struct ImGUI {
    pub egui_ctx: egui::Context,
}

// Things to be render/updated on GPU.
pub struct ImGUIOuput {
    //pub repaint_after: std::time::Duration,
    pub textures_delta: egui::epaint::textures::TexturesDelta,
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
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
        Self { egui_ctx }
    }

    // Gather input (mouse, touches, keyboard, screen size, etc):
    pub fn gather_input(&self, window_size: UVec2, window: &Window) -> egui::RawInput {
        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect {
                min: egui::Pos2 { x: 0.0, y: 0.0 },
                max: egui::Pos2 {
                    x: window_size.x as f32,
                    y: window_size.y as f32,
                },
            }),
            ..Default::default()
        };

        // NOTE: no scaling; logical pixel == pixel
        let to_pos2 = |x: &i16, y: &i16| pos2(*x as f32, *y as f32);

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
        self.egui_ctx.begin_frame(raw_input);
    }

    // After adding all ui elements
    pub fn end(&mut self) -> ImGUIOuput {
        let full_output = self.egui_ctx.end_frame();
        let clipped_primitives = self.egui_ctx.tessellate(full_output.shapes); // creates triangles to paint
        ImGUIOuput {
            textures_delta: full_output.textures_delta,
            clipped_primitives,
        }
    }
}
