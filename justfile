set windows-shell := ["powershell", "-NoProfile", "-Command"]

# Build CLI + GUI
build:
    cargo build

# Type-check
check:
    cargo check

# Run all tests
test:
    cargo test

gui:
    cargo run --bin heightmap_gui --features gui

# Live gallery of the Brickadia egui theme (palette, buttons, Font Awesome icons)
sandbox:
    cargo run --example theme_sandbox --features gui

# Render an image as TextDisplay bricks
text input output="out.brz":
    cargo run --bin heightmap -- "{{input}}" --text -o "{{output}}"

dist:
    cargo build --release --bin heightmap
    cargo build --release --bin heightmap_gui --features gui