/*
Integration of egui.
*/

use egui;

pub struct ImGUI {
    pub egui_ctx: egui::Context,
}

// Things to be render/updated on GPU.
pub struct ImGUIOuput {
    //pub repaint_after: std::time::Duration,
    pub textures_delta: egui::epaint::textures::TexturesDelta,
    pub clipped_primitives: Vec<egui::ClippedPrimitive>,
}

// Limited integration of egui -- platform output is ignored.
impl ImGUI {
    pub fn new() -> Self {
        let egui_ctx = egui::Context::default();
        Self { egui_ctx }
    }

    // Gather input (mouse, touches, keyboard, screen size, etc):
    pub fn gather_input(&self) -> egui::RawInput {
        let raw_input = egui::RawInput {
            // pixels_per_point: Some(1.0),
            ..Default::default()
        };
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
