mod window;

use window::Window;

fn main() {
    println!("Hello, rusty world!");

    let window = Window::new(1280, 720, "Violet Rusty");
}
