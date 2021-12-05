mod window;
use window::Window;

fn main() {
    println!("Hello, rusty world!");

    let window = Window::new(1280, 720, "Rusty Violet");

    loop {
        if window.should_close() {
            break;
        }
        window.poll_events();

        // todo render things
    }
}
