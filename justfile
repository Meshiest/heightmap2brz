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

# Release builds of both binaries, exactly as CI builds them
dist:
    # --no-default-features drops `gui` (eframe/rfd/rodio), so the CLI needs
    # no GTK, ALSA or X11 packages to build on Linux.
    cargo build --release --no-default-features --bin heightmap
    cargo build --release --bin heightmap_gui --features gui

# `bash` on Windows is the WSL shim, which sees different paths and a different
# git. Git for Windows ships its own; set GIT_BASH for a non-default install.
BASH := if os() == "windows" { '& "' + env_var_or_default("GIT_BASH", "C:/Program Files/Git/bin/bash.exe") + '"' } else { "bash" }

REMOTE := env_var_or_default("REMOTE", "origin")

# Remotes `just tag-all` publishes to
REMOTES := "origin community"

# Tag the version in Cargo.toml and push it, refusing to move an existing tag
tag remote=REMOTE:
    {{ BASH }} tools/tag.sh {{ remote }}

# Tag once and push it to every remote, so each publishes its own release
tag-all:
    {{ BASH }} tools/tag.sh {{ REMOTES }}

# Preview the release notes CI would publish (default: Cargo.toml's version)
notes version="":
    {{ BASH }} tools/release-notes.sh "{{ version }}"