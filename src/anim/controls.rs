//! Pre-wired physical control buttons for a render's clock.
//!
//! Every video and audio renderer drives one shared clock
//! ([`super::clock::build_clock`]) whose `Pause`/`Restart`/`Resume` pins are
//! surfaced as microchip inputs for a builder to wire by hand. This module
//! wires them automatically instead: three coloured `B_Button` bricks in a row
//! on the world's main grid, each carrying a `Component_TextDisplay` label on
//! its top face and its `bHeld` output fed straight into the matching pin -- so
//! a freshly generated build is pausable, restartable and resumable out of the
//! box with no manual wiring.
//!
//! Factored once and called from all five renderers (both audio builders and
//! the three video renderers), rather than copied into each: the assembly
//! depends only on the three pin ids and a clear main-grid anchor.
//!
//! The button carries the animated-button component
//! ([`BUTTON`]), whose `PromptCustomLabel` is a bare `str` schema field, so its
//! literal is baked as a plain [`String`], not a `WireVariant`. The visible
//! label is a second component on the SAME brick, not a separate cube.
//!
//! # Three things this cannot verify -- the owner must confirm on a test render
//!
//! 1. That `Button.bHeld -> pin` actually drives the Timer's Pause/Restart/Resume.
//! 2. That `PromptCustomLabel` shows the control's name on look.
//! 3. That a `Component_TextDisplay` riding on the same brick as the animated
//!    button renders (the label and the button coexisting on one `B_Button`).
use super::chip::pin_target;
use crate::text::{
    ANCHOR_CENTRE, DEFAULT_OUTLINE_WIDTH, FACE_Z_POSITIVE, FontPreset, OUTLINE_OUTLINED,
    text_label_component,
};
use brdb::{
    AsBrdbValue, Brick, BrickType, Color, IntVector, Position, WirePort, World,
    assets::LiteralComponent,
};

/// The animated interactive button component every control brick carries: the
/// `B_Button` that visibly depresses when held (as opposed to the plain
/// `Component_Button` on a round plate). Its ports match the plain button's.
pub const BUTTON: &str = "Component_Internal_AnimatedButton";
/// The bool output that is true while the button is held down, wired straight
/// into the clock pin.
const BUTTON_HELD: &str = "bHeld";
/// The input port carrying the on-look interaction prompt (a bare `str`).
const BUTTON_PROMPT_LABEL: &str = "PromptCustomLabel";
/// The animated push-button brick (depresses when held).
const BUTTON_BRICK: &str = "B_Button";

/// The three controls, in row order, each with its button colour.
pub const CONTROLS: [(&str, Color); 3] = [
    ("Pause", Color { r: 255, g: 0, b: 0 }),
    ("Restart", Color { r: 0, g: 0, b: 255 }),
    ("Resume", Color { r: 0, g: 255, b: 0 }),
];

/// Main-grid bricks a full control assembly adds: one button per control, each
/// carrying its own label component (no separate cube). The cost estimators
/// count this (see [`super::cost`] and [`crate::audio::cost`]) so the readout
/// still describes the render.
pub const CONTROL_BRICKS: usize = CONTROLS.len();
/// Wires it adds: one per control, `bHeld -> pin`.
pub const CONTROL_WIRES: usize = CONTROLS.len();

/// The label's component `LineHeight` (font size).
const LABEL_LINE_HEIGHT: f32 = 1.0;
/// Clearance past the farthest main-grid face before the button row starts.
const ANCHOR_GAP: i32 = 5;

/// The half-extent `layout::assert_bricks_dont_overlap` will measure for `asset`,
/// via the same `Brick::local_bounds` it uses on every brick.
///
/// Derived at runtime, never hardcoded: a wrong hardcoded size would space the
/// label to overlap the button's real box and the game would silently drop one
/// -- the trap `speakers::speaker_half` documents.
fn measured_half(asset: &'static str) -> IntVector {
    let (min, max) = Brick { asset: BrickType::from(asset), ..Default::default() }.local_bounds();
    IntVector { x: (max.x - min.x) / 2, y: (max.y - min.y) / 2, z: (max.z - min.z) / 2 }
}

/// A clear spot on the main grid, one gate cell past everything already placed
/// there, with every coordinate non-negative.
///
/// Derived from the bricks already on the main grid (the display screen or the
/// speaker cluster, the microchip shell, and any subtitle anchor) rather than a
/// constant tuned for one configuration: the buttons sit entirely beyond the
/// greatest x face in play, so they clear every other main-grid brick on the x
/// axis alone -- for any screen size, any speaker count, and the speakers-in-chip
/// case where the main grid holds only the shell. `.max(0)` keeps the anchor
/// non-negative even then (the shell sits at a negative x), for the same
/// chunk-encoding reason inner-grid coordinates must be non-negative.
///
/// Must be called after every other main-grid brick is placed (display/speaker
/// bricks, chip shell, subtitles) and before [`add_control_buttons`], so the
/// extent it reads is the final one.
pub fn control_anchor(world: &World) -> Position {
    let max_x = world
        .bricks
        .iter()
        .map(|b| b.local_bounds().1.x)
        .max()
        .unwrap_or(0);
    Position { x: (max_x + ANCHOR_GAP).max(0), y: 0, z: 0 }
}

/// Build the three control-button assemblies and wire them into the clock's
/// `Pause`/`Restart`/`Resume` pins.
///
/// The three pin ids are the chip inputs [`super::clock::build_clock`] returns
/// (`Clock::pause_pin` etc.). `anchor` is a clear, non-negative main-grid corner
/// -- pass [`control_anchor`]. Each control is a single coloured `B_Button` (red
/// pause, blue restart, green resume) carrying two components: the animated
/// [`BUTTON`] with its `PromptCustomLabel` set to the name, and a
/// `Component_TextDisplay` label centred on the button's top face (drawn Z+,
/// readable from above).
///
/// The buttons sit side by side along +y, 10 units apart, beyond the main grid's
/// x extent so the row collides with nothing.
///
/// Each button's `bHeld` output wires to [`pin_target`]`(pin, true)` -- the pin's
/// external `RER_Input`, not `pin_source`. The button is on the main grid and the
/// pin on the chip's inner grid, so `brdb` resolves this to a remote wire at
/// write time.
///
/// Call before `chip::finish` (so its main-grid overlap check sees these bricks)
/// and before `World::register_used_components`.
pub fn add_control_buttons(
    world: &mut World,
    pause_pin: usize,
    restart_pin: usize,
    resume_pin: usize,
    anchor: Position,
) {
    debug_assert!(
        anchor.x >= 0 && anchor.y >= 0 && anchor.z >= 0,
        "control-button anchor must be non-negative, got {anchor:?}"
    );
    // Plain-text label geometry: a single readable line, no image encoding.
    let label_opts = FontPreset::MonaspaceArgon.options(1.0);

    let half = measured_half(BUTTON_BRICK);
    // Side by side along +y, 10 units apart -- or flush by the button's own
    // width if that is wider, so they never overlap whatever `brdb` reports.
    let slot_y = (2 * half.y).max(10);

    let pins = [pause_pin, restart_pin, resume_pin];
    for (i, (&(name, color), &pin)) in CONTROLS.iter().zip(&pins).enumerate() {
        let base_y = anchor.y + i as i32 * slot_y;
        let button_pos = Position {
            x: anchor.x + half.x,
            y: base_y + half.y,
            z: anchor.z + half.z,
        };

        // The visible label, centred on the button's top face and drawn Z+, as a
        // second component on the button brick itself -- no separate cube.
        let label = text_label_component(
            world,
            name.to_string(),
            LABEL_LINE_HEIGHT,
            FACE_Z_POSITIVE,
            ANCHOR_CENTRE,
            OUTLINE_OUTLINED,
            DEFAULT_OUTLINE_WIDTH,
            &label_opts,
        );

        // The coloured animated button carrying both components. Its
        // `PromptCustomLabel` (the on-look prompt) is a bare String because the
        // schema declares that field as `str`.
        let (button, button_id) = Brick {
            asset: BrickType::from(BUTTON_BRICK),
            position: button_pos,
            color,
            ..Default::default()
        }
        .with_component(LiteralComponent::new(BUTTON).with_data([(
            BUTTON_PROMPT_LABEL,
            Box::new(name.to_string()) as Box<dyn AsBrdbValue>,
        )]))
        .with_component(label)
        .with_id_split();
        world.add_brick(button);

        // The button's held state drives the pin's external input directly.
        world.add_wire_connection(
            WirePort::new(button_id, BUTTON, BUTTON_HELD),
            pin_target(pin, true),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::chip::{MICROCHIP_INPUT, add_input_pin, new_chip};
    use crate::anim::layout::assert_bricks_dont_overlap;
    use brdb::{IntVector, Vector3f};

    /// A world with a microchip carrying three input pins, standing in for the
    /// clock's Pause/Restart/Resume. Returns the world and the three pin ids.
    fn world_with_pins() -> (World, [usize; 3]) {
        let mut world = World::new();
        let mut chip = new_chip(
            &mut world,
            Position { x: -20, y: 0, z: 2 },
            Vector3f { x: -20.0, y: 0.0, z: 40.0 },
            IntVector { x: 5, y: 5, z: 5 },
        );
        let pause = add_input_pin(&mut chip, "Pause", Position { x: 5, y: 5, z: 6 });
        let restart = add_input_pin(&mut chip, "Restart", Position { x: 5, y: 15, z: 6 });
        let resume = add_input_pin(&mut chip, "Resume", Position { x: 5, y: 25, z: 6 });
        crate::anim::chip::finish(&mut world, chip).expect("finish");
        (world, [pause, restart, resume])
    }

    #[test]
    fn builds_three_buttons_three_labels_and_three_wires() {
        let (mut world, [pause, restart, resume]) = world_with_pins();
        let bricks_before = world.bricks.len();
        let wires_before = world.wires.len();
        let anchor = control_anchor(&world);
        add_control_buttons(&mut world, pause, restart, resume, anchor);

        assert_eq!(
            world.bricks.len() - bricks_before,
            CONTROL_BRICKS,
            "3 buttons, each carrying its own label component"
        );
        assert_eq!(world.wires.len() - wires_before, CONTROL_WIRES, "1 wire per control");

        let count = |needle: &str| {
            world
                .bricks
                .iter()
                .filter(|b| {
                    b.components.iter().any(|c| {
                        c.component_type().is_some_and(|t| t.to_string() == needle)
                    })
                })
                .count()
        };
        assert_eq!(count(BUTTON), 3, "one animated button per control");
        assert_eq!(count("Component_TextDisplay"), 3, "one label per control");
    }

    /// Each button carries its control's colour: red pause, blue restart,
    /// green resume.
    #[test]
    fn each_button_is_coloured_by_its_control() {
        let (mut world, [pause, restart, resume]) = world_with_pins();
        let anchor = control_anchor(&world);
        add_control_buttons(&mut world, pause, restart, resume, anchor);

        for (name, color) in CONTROLS {
            let found = world.bricks.iter().any(|b| {
                b.color == color
                    && b.components.iter().any(|c| {
                        c.component_type().is_some_and(|t| t.to_string() == BUTTON)
                    })
            });
            assert!(found, "{name} button must carry colour {color:?}");
        }
    }

    /// Every button's `bHeld` must resolve to a distinct pin's external input --
    /// pause, restart and resume, one each.
    #[test]
    fn each_button_targets_a_distinct_control_pin() {
        let (mut world, pins) = world_with_pins();
        let [pause, restart, resume] = pins;
        let anchor = control_anchor(&world);
        add_control_buttons(&mut world, pause, restart, resume, anchor);

        let mut targeted: Vec<usize> = world
            .wires
            .iter()
            .filter(|w| {
                w.source.component_type.to_string() == BUTTON
                    && w.source.port_name.to_string() == BUTTON_HELD
            })
            .map(|w| {
                assert_eq!(
                    w.target.component_type.to_string(),
                    MICROCHIP_INPUT,
                    "a button must drive a microchip input pin"
                );
                assert_eq!(
                    w.target.port_name.to_string(),
                    "RER_Input",
                    "it must feed the pin's external input, not its internal source"
                );
                w.target.brick_id
            })
            .collect();
        targeted.sort_unstable();
        let mut want = pins.to_vec();
        want.sort_unstable();
        assert_eq!(targeted, want, "the three buttons must cover exactly the three pins");
    }

    /// The whole assembly must be collision-free against the chip shell and
    /// against itself -- `chip::finish` runs this same check on a real render.
    #[test]
    fn the_assembly_never_overlaps() {
        let (mut world, [pause, restart, resume]) = world_with_pins();
        let anchor = control_anchor(&world);
        add_control_buttons(&mut world, pause, restart, resume, anchor);
        assert_bricks_dont_overlap(&world.bricks).expect("control buttons must not overlap");
    }

    /// Every authored brick coordinate the assembly places must be non-negative,
    /// for the same chunk-encoding reason the rest of the crate keeps them so.
    #[test]
    fn every_button_brick_coordinate_is_non_negative() {
        let (mut world, [pause, restart, resume]) = world_with_pins();
        let before = world.bricks.len();
        let anchor = control_anchor(&world);
        add_control_buttons(&mut world, pause, restart, resume, anchor);
        for b in &world.bricks[before..] {
            let (min, _) = b.local_bounds();
            assert!(
                min.x >= 0 && min.y >= 0 && min.z >= 0,
                "control brick at {:?} reaches negative {min:?}",
                b.position
            );
        }
    }
}
