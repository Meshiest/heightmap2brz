//! Timer -> frame index. Four gates, all shared by the whole screen.
use super::chip::{Chip, pin_source, pin_target};
use super::layout::{CELL, GATE_HALF};
use brdb::{
    Direction, IntVector, Rotation,
    AsBrdbValue, BString, Brick, BrickType, Position, WirePort, World, assets::LiteralComponent,
    schema::WireVariant,
};
use std::collections::HashMap;
use std::sync::Arc;

pub const TIMER: &str = "BrickComponentType_WireGraphPseudo_Timer";
pub const MULTIPLY: &str = "BrickComponentType_WireGraph_Expr_MathMultiply";
pub const BITWISE_OR: &str = "BrickComponentType_WireGraph_Expr_BitwiseOR";
pub const MODULO: &str = "BrickComponentType_WireGraph_Expr_MathModuloFloored";

pub struct Clock {
    /// Source port carrying the wrapped integer frame index.
    pub frame_index: WirePort,
    pub pause_pin: usize,
    pub restart_pin: usize,
    pub resume_pin: usize,
    /// Playback-rate input. Overrides the baked fps when wired; the baked
    /// value stands when it is left unconnected.
    pub rate_pin: usize,
    /// Exec output carrying the timer's `Expired` pulse. Silent while
    /// `Limit` is 0 (free-running) — see `build_clock`.
    pub done_pin: usize,
}

/// Add a gate brick carrying `class`, with `data` inlined on its ports.
/// Inlining a constant as component data costs no extra brick — this is what
/// keeps the per-pixel gate count at 2 instead of 4. Only valid for ports
/// that appear in the gate's `inputs` list; wiring a data-only field produces
/// a file the game rejects.
///
/// `asset`, `class` and the `data` keys are `&'static str`: every caller
/// passes a string literal or a top-level `const`, and `BrickType`/`BString`
/// only implement `Into<BString>` for `&'static str` (not for an arbitrary
/// borrowed `&str`), so a shorter-lived string could never satisfy this
/// signature anyway.
///
/// Goes through `Chip::add_brick`, the single write path that keeps the
/// brick list and the overlap-check bounds in sync. Never push to the chip's
/// internals directly — they are private for exactly this reason.
pub fn gate(
    chip: &mut Chip,
    asset: &'static str,
    class: &'static str,
    pos: Position,
    data: Vec<(&'static str, Box<dyn AsBrdbValue>)>,
) -> usize {
    gate_oriented(
        chip,
        asset,
        class,
        pos,
        data,
        Direction::default(),
        Rotation::default(),
        GATE_HALF,
    )
}

/// [`gate`], but with an explicit facing, roll, and post-rotation half-extent.
///
/// `half` MUST describe the volume the brick occupies once `direction` and
/// `rotation` are applied — it is not merely cosmetic. `Brick::local_bounds`
/// reports the extent as *authored* and never applies rotation, so a rotated
/// gate registered with its unrotated `GATE_HALF` leaves the overlap check
/// measuring a box the brick no longer fills. That shipped once: standing the
/// pixel gates on end made them 10 units tall while the checker still
/// believed they were 4, so real collisions passed silently and the stage
/// offsets sat 3 units too shallow, burying a stage behind the plane.
pub fn gate_oriented(
    chip: &mut Chip,
    asset: &'static str,
    class: &'static str,
    pos: Position,
    data: Vec<(&'static str, Box<dyn AsBrdbValue>)>,
    direction: Direction,
    rotation: Rotation,
    half: IntVector,
) -> usize {
    let mut map: HashMap<BString, Box<dyn AsBrdbValue>> = HashMap::new();
    for (k, v) in data {
        map.insert(k.into(), v);
    }
    let brick = Brick {
        asset: BrickType::from(asset),
        position: pos,
        direction,
        rotation,
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(class, Arc::new(map)));
    chip.add_brick(brick, half)
}

/// Build the shared clock subgraph: `Timer.Time -> Multiply(x fps) ->
/// BitwiseOR(|0) -> ModuloFloored(% frame_count)`, plus three control pins
/// (`Pause`, `Restart`, `Resume`) wired straight into the timer.
///
/// `BitwiseOR`'s `InputB` is inlined `0`; its ports are typed `int`, so
/// routing the multiplied float through it truncates to an integer frame
/// number for free. Looping goes through `ModuloFloored` rather than wiring
/// `Timer.Expired` back to `Timer.Restart` — that would be a cycle from the
/// timer's own output back to its own input, and every wire-graph cycle must
/// cross a tick barrier (a buffer gate) or the game rejects the graph;
/// `ModuloFloored` needs no such barrier because it never feeds back into the
/// timer. `Timer.Limit` is explicitly set to `0.0` for free-running (see the
/// note at its call site below — leaving it unset does *not* default to 0).
pub fn build_clock(
    world: &mut World,
    chip: &mut Chip,
    fps: f32,
    frame_count: usize,
    origin: Position,
) -> Clock {
    let at = |i: i32| Position { x: origin.x, y: origin.y + i * CELL, z: origin.z };

    // `Limit` must be set explicitly to 0.0, not left unset: `Limit` (a plain
    // `f64` field on `BrickComponentData_WireGraphPseudo_Timer`, like
    // `BitwiseOR`'s `InputB` below) falls back to the schema's registered
    // struct default when omitted, and that default is `1.0`
    // (`STRUCT_DEFAULTS` in `component_db.rs`) — not the free-running `0`
    // this clock needs. Leaving it unset would silently cap the timer at one
    // second instead of letting it run free.
    let timer = gate(chip, "B_1x1_Gate_Pseudo_Timer", TIMER, at(0), vec![(
        "Limit",
        Box::new(0.0f64) as Box<dyn AsBrdbValue>,
    )]);
    let mul = gate(chip, "B_1x1_Gate_Expr_MathMultiply", MULTIPLY, at(1), vec![(
        "InputB",
        Box::new(WireVariant::Number(fps as f64)) as Box<dyn AsBrdbValue>,
    )]);
    // `BitwiseOR`'s data struct (`BrickComponentData_WireGraph_Expr_IntInt_Int`)
    // declares `InputA`/`InputB` as plain `i64` fields, not the
    // `WireGraphPrimMathVariant` tagged union `Multiply`/`ModuloFloored` use
    // below — so the literal here must be a bare `i64`, not `WireVariant::Int`
    // (which only implements the wire-variant cast, not the scalar-int one
    // the schema writer needs for a plain `i64` field). It is still the
    // deliberate float->int truncation described above: this port is typed
    // `int`, so wiring `Multiply`'s float `Output` into `InputA` coerces it.
    let trunc = gate(chip, "B_1x1_Gate_Expr_BitwiseOR", BITWISE_OR, at(2), vec![(
        "InputB",
        Box::new(0i64) as Box<dyn AsBrdbValue>,
    )]);
    // The frame count is a whole number of frames, so `InputB` is an Int
    // variant rather than a Number: the wrap point is an index, not a
    // measurement, and feeding it as a float leaves the gate doing float
    // modulo on a value that can only ever be integral.
    let wrap = gate(chip, "B_1x1_Gate_Expr_MathModuloFloored", MODULO, at(3), vec![(
        "InputB",
        Box::new(WireVariant::Int(frame_count as i64)) as Box<dyn AsBrdbValue>,
    )]);

    world.add_wire_connection(
        WirePort::new(timer, TIMER, "Time"),
        WirePort::new(mul, MULTIPLY, "InputA"),
    );
    world.add_wire_connection(
        WirePort::new(mul, MULTIPLY, "Output"),
        WirePort::new(trunc, BITWISE_OR, "InputA"),
    );
    world.add_wire_connection(
        WirePort::new(trunc, BITWISE_OR, "Output"),
        WirePort::new(wrap, MODULO, "InputA"),
    );

    // Control pins, placed in their own row clear of the clock chain.
    let pin_at = |i: i32| Position { x: origin.x + CELL, y: origin.y + i * CELL, z: origin.z };
    let pause_pin = super::chip::add_input_pin(chip, "Pause", pin_at(0));
    let restart_pin = super::chip::add_input_pin(chip, "Restart", pin_at(1));
    let resume_pin = super::chip::add_input_pin(chip, "Resume", pin_at(2));
    for (pin, port) in [
        (pause_pin, "Pause"),
        (restart_pin, "Restart"),
        (resume_pin, "Resume"),
    ] {
        world.add_wire_connection(pin_source(pin, true), WirePort::new(timer, TIMER, port));
    }

    // Playback rate: drives the same `Multiply.InputB` that carries the baked
    // fps. The inlined `fps` above stays as the value the gate reads when this
    // pin is left unwired, so a pasted chip plays at its authored speed with
    // nothing attached; wiring the pin overrides it live. Rate is a plain
    // multiplier on elapsed time, so 2.0 is double speed and 0.0 freezes.
    let rate_pin = super::chip::add_input_pin(chip, "Rate", pin_at(3));
    world.add_wire_connection(
        pin_source(rate_pin, true),
        WirePort::new(mul, MULTIPLY, "InputB"),
    );

    // Exec-on-done: the timer's `Expired` pulse, surfaced so a builder can
    // chain something to the end of a run.
    //
    // NOTE: `Limit` is 0 here (free-running), and a timer that never reaches a
    // limit never expires — so this port exists but stays silent unless a
    // caller sets a non-zero `Limit`. It is wired now so the chip's outward
    // shape is settled; making it actually fire is a separate decision about
    // whether the clock should be bounded by the clip's duration.
    let done_pin = super::chip::add_output_pin(chip, "Done", pin_at(4));
    world.add_wire_connection(
        WirePort::new(timer, TIMER, "Expired"),
        pin_target(done_pin, false),
    );

    Clock {
        frame_index: WirePort::new(wrap, MODULO, "Output"),
        pause_pin,
        restart_pin,
        resume_pin,
        rate_pin,
        done_pin,
    }
}
