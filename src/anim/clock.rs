//! Timer -> frame index. Six gates, all shared by the whole screen (four for
//! the index chain, two for the length and progress status taps).
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
    /// The timer's raw `Time` output: continuous seconds, advancing every tick
    /// while the clock runs and frozen while it is paused.
    ///
    /// The same port the fps multiply already reads (`Timer.Time ->
    /// Multiply.InputA`, wired in `build_clock`), surfaced here so a
    /// consumer can tap it without disturbing that math -- reading it again
    /// is dataflow fan-out, which the graph allows freely. `audio::speakers::scaffold`
    /// uses it to tell a running clock from a frozen one.
    pub time: WirePort,
    pub pause_pin: usize,
    pub restart_pin: usize,
    pub resume_pin: usize,
    /// Playback-rate input. Overrides the baked fps when wired; the baked
    /// value stands when it is left unconnected.
    pub rate_pin: usize,
    /// Exec output carrying the timer's `Expired` pulse.
    ///
    /// Silent on a LOOPING clock, whose `Limit` is 0 (free-running): a timer
    /// that never reaches a limit never expires. On a non-looping clock it
    /// fires once, at [`stop_limit`] -- see `build_clock`.
    pub done_pin: usize,
    /// Output pin carrying playback progress in percent (0..100), a float.
    ///
    /// The wrapped frame index scaled so frame 0 reads 0 and the last frame
    /// reads 100; a looping clock sweeps 0..100 and wraps with the index.
    /// Quantized to whole frames (the render's own granularity) and
    /// rate-independent. Always wired, inert until a builder taps it.
    pub progress_pin: usize,
    /// Output pin carrying the clip's intrinsic length in seconds, a float
    /// constant (`frame_count / fps`).
    ///
    /// Reports the authored length regardless of the `Rate` pin: rate changes
    /// how fast the clip plays, not how long the content is. Always wired,
    /// inert until a builder taps it.
    pub length_pin: usize,
}

/// The `Timer.Limit`, in seconds, for a clock that must stop at the end of
/// its content instead of looping it.
///
/// The frame index this clock produces is `floor(Time * fps) % frame_count`,
/// so the limit decides which frame is on screen when the timer expires. The
/// exact duration, `frame_count / fps`, must not be used: `floor(frame_count)
/// % frame_count == 0`, so a render told to stop at the end would snap back
/// to frame 0. Subtracting half a frame lands the limit inside the last
/// frame's slot instead:
///
/// ```text
///   limit = (frame_count - 0.5) / fps
///   floor(limit * fps) % frame_count = frame_count - 1
/// ```
///
/// Half a frame, not an epsilon: it is the largest offset that still lands
/// inside the final slot, so it survives float rounding at any fps and frame
/// count. Always strictly positive (`frame_count >= 1`, `fps > 0`), so it can
/// never collide with the `0.0` free-running sentinel.
///
/// Whether the game's `Timer` holds `Time` at `Limit` once `Expired` fires
/// (rather than keeps counting, in which case the modulo just keeps wrapping
/// and playback loops anyway) is unverified in game. The `Rate` pin
/// ([`Clock::rate_pin`]) overrides the baked fps at runtime while this limit
/// stays fixed in seconds against the fps baked at build time, so driving
/// `Rate` on a non-looping clock moves where playback stops.
pub fn stop_limit(fps: f32, frame_count: usize) -> f64 {
    (frame_count as f64 - 0.5) / fps as f64
}

/// The `Timer.Limit` a LOOPING clock carries: free-running, so the timer never
/// expires and `ModuloFloored` wraps the frame index forever.
///
/// Named rather than spelled `0.0` at the call site because it is a sentinel,
/// not a measurement -- and because `Limit`'s registered struct default is
/// `1.0`, so "no limit" is the one value that must always be written out
/// explicitly. See `build_clock`.
pub const FREE_RUNNING_LIMIT: f64 = 0.0;

/// Add a gate brick carrying `class`, with `data` inlined on its ports.
/// Inlining a constant as component data costs no extra brick -- this is what
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
/// internals directly -- they are private for exactly this reason.
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
/// `half` must describe the volume the brick occupies once `direction` and
/// `rotation` are applied, not the unrotated authored size: `Brick::local_bounds`
/// reports the extent as authored and never applies rotation, so a rotated
/// gate registered with its unrotated half leaves the overlap check measuring
/// a box the brick no longer fills.
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
/// `Timer.Expired` back to `Timer.Restart` -- that would be a cycle from the
/// timer's own output back to its own input, and every wire-graph cycle must
/// cross a tick barrier (a buffer gate) or the game rejects the graph;
/// `ModuloFloored` needs no such barrier because it never feeds back into the
/// timer.
///
/// # `loop_playback`
///
/// The only thing this flag changes is the value inlined on `Timer.Limit`;
/// the graph is the same four gates and eight wires either way.
///
/// * `true` (the default) writes [`FREE_RUNNING_LIMIT`]. The timer never
///   expires, `ModuloFloored` wraps the frame index, and playback repeats.
/// * `false` writes [`stop_limit`], which expires halfway through the last
///   frame's slot so the index sits at `frame_count - 1` rather than
///   wrapping to 0. Whether the picture really holds there is unverified in
///   game -- see that function's doc.
///
/// Either way `Limit` is written explicitly: it falls back to the schema's
/// registered struct default (`1.0`, not `0`) when omitted, which would
/// silently cap the timer at one second.
pub fn build_clock(
    world: &mut World,
    chip: &mut Chip,
    fps: f32,
    frame_count: usize,
    loop_playback: bool,
    origin: Position,
) -> Clock {
    let at = |i: i32| Position { x: origin.x, y: origin.y + i * CELL, z: origin.z };

    // See this function's doc for both values and for why neither may be
    // left to the struct default.
    let limit = if loop_playback {
        FREE_RUNNING_LIMIT
    } else {
        stop_limit(fps, frame_count)
    };
    let timer = gate(chip, "B_1x1_Gate_Pseudo_Timer", TIMER, at(0), vec![(
        "Limit",
        Box::new(limit) as Box<dyn AsBrdbValue>,
    )]);
    let mul = gate(chip, "B_1x1_Gate_Expr_MathMultiply", MULTIPLY, at(1), vec![(
        "InputB",
        Box::new(WireVariant::Number(fps as f64)) as Box<dyn AsBrdbValue>,
    )]);
    // `BitwiseOR`'s data struct declares `InputA`/`InputB` as plain `i64`
    // fields, not the tagged-union `WireGraphPrimMathVariant` `Multiply`/
    // `ModuloFloored` use -- so this literal must be a bare `i64`, not
    // `WireVariant::Int`. Also the deliberate float->int truncation: this
    // port is typed `int`, so wiring `Multiply`'s float `Output` in coerces it.
    let trunc = gate(chip, "B_1x1_Gate_Expr_BitwiseOR", BITWISE_OR, at(2), vec![(
        "InputB",
        Box::new(0i64) as Box<dyn AsBrdbValue>,
    )]);
    // Frame count is a whole number of frames: `InputB` is an Int variant,
    // not Number, since the wrap point is an index, not a measurement.
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

    // Playback rate: drives the same `Multiply.InputB` that carries the
    // baked fps, so a pasted chip plays at its authored speed unless this
    // pin is wired. Rate is a plain multiplier (2.0 = double speed, 0.0 freezes).
    let rate_pin = super::chip::add_input_pin(chip, "Rate", pin_at(3));
    world.add_wire_connection(
        pin_source(rate_pin, true),
        WirePort::new(mul, MULTIPLY, "InputB"),
    );

    // Exec-on-done: the timer's `Expired` pulse, surfaced so a builder can
    // chain something to the end of a run. Wired unconditionally (even when
    // looping) so the chip's outward shape and cost readout don't depend on
    // the flag; on a looping clock `Limit` is 0, so this port stays silent.
    let done_pin = super::chip::add_output_pin(chip, "Done", pin_at(4));
    world.add_wire_connection(
        WirePort::new(timer, TIMER, "Expired"),
        pin_target(done_pin, false),
    );

    // --- Status taps: length and progress ----------------------------------
    // Two always-on output pins a builder can wire to a readout, inert until
    // tapped (like `Done` and `Rate`). Both are floats, and both extend the
    // clock's own two columns downward into service rows every caller leaves
    // free (its other service gates sit at negative rows).
    //
    // Length is the clip's intrinsic duration, `frame_count / fps` -- a
    // build-time constant. It is emitted by a `Multiply` with BOTH inputs
    // inlined (`length * 1.0`): with nothing wired to either port the gate
    // evaluates its inlined defaults every tick, the same mechanism that lets
    // `Multiply.InputB` above carry the baked fps. Reports the authored length
    // regardless of the `Rate` pin -- rate changes playback speed, not the
    // content's length.
    let length_secs = frame_count as f64 / fps as f64;
    let length_gate = gate(chip, "B_1x1_Gate_Expr_MathMultiply", MULTIPLY, at(4), vec![
        ("InputA", Box::new(WireVariant::Number(length_secs)) as Box<dyn AsBrdbValue>),
        ("InputB", Box::new(WireVariant::Number(1.0)) as Box<dyn AsBrdbValue>),
    ]);
    let length_pin = super::chip::add_output_pin(chip, "Length", pin_at(5));
    world.add_wire_connection(
        WirePort::new(length_gate, MULTIPLY, "Output"),
        pin_target(length_pin, false),
    );

    // Progress is the wrapped frame index scaled to percent: `frame_index *
    // 100 / (frame_count - 1)`, so frame 0 reads 0 and the last frame reads
    // 100. Reading `wrap.Output` again is dataflow fan-out (the caller also
    // reads it as the frame index), which the graph allows freely.
    // `(frame_count - 1).max(1)` guards a single-frame clip, which then reads
    // a constant 0.
    let progress_scale = 100.0 / frame_count.saturating_sub(1).max(1) as f64;
    let progress_gate = gate(chip, "B_1x1_Gate_Expr_MathMultiply", MULTIPLY, at(5), vec![(
        "InputB",
        Box::new(WireVariant::Number(progress_scale)) as Box<dyn AsBrdbValue>,
    )]);
    world.add_wire_connection(
        WirePort::new(wrap, MODULO, "Output"),
        WirePort::new(progress_gate, MULTIPLY, "InputA"),
    );
    let progress_pin = super::chip::add_output_pin(chip, "Progress", pin_at(6));
    world.add_wire_connection(
        WirePort::new(progress_gate, MULTIPLY, "Output"),
        pin_target(progress_pin, false),
    );

    Clock {
        frame_index: WirePort::new(wrap, MODULO, "Output"),
        // The timer's raw stopwatch, exposed for tapping. Not a new wire:
        // the `Timer.Time -> Multiply.InputA` connection above is the only
        // wire off this port `build_clock` itself emits.
        time: WirePort::new(timer, TIMER, "Time"),
        pause_pin,
        restart_pin,
        resume_pin,
        rate_pin,
        done_pin,
        progress_pin,
        length_pin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame index the built graph produces for a given `Time`:
    /// `Multiply(x fps) -> BitwiseOR(|0) -> ModuloFloored(% frame_count)`.
    /// Written out rather than asserted against a magic number, since it is
    /// the arithmetic `stop_limit` has to survive.
    fn frame_index_at(time: f64, fps: f32, frame_count: usize) -> usize {
        let truncated = (time * fps as f64).floor() as i64;
        truncated.rem_euclid(frame_count as i64) as usize
    }

    /// At the moment a non-looping clock expires, the frame on screen must
    /// be the last one. Swept across fps and frame counts that do not
    /// divide evenly, since a formula that only works at 1 fps or a power
    /// of two would pass a single-case test.
    #[test]
    fn the_stop_limit_lands_on_the_last_frame_never_the_first() {
        for &fps in &[1.0f32, 2.0, 10.0, 12.0, 15.0, 23.976, 30.0, 60.0] {
            for &frames in &[1usize, 2, 3, 7, 20, 90, 1000, 65_535] {
                let limit = stop_limit(fps, frames);
                assert_eq!(
                    frame_index_at(limit, fps, frames),
                    frames - 1,
                    "at {fps} fps over {frames} frames the clock must expire on the last \
                     frame, not wrap"
                );
            }
        }
    }

    /// The trap this formula exists to avoid, stated as a test so it can never
    /// be "simplified" back in: the clip's exact duration puts the index at 0.
    #[test]
    fn the_exact_duration_would_snap_back_to_frame_zero() {
        for &fps in &[10.0f32, 30.0] {
            for &frames in &[3usize, 20, 90] {
                let naive = frames as f64 / fps as f64;
                assert_eq!(
                    frame_index_at(naive, fps, frames),
                    0,
                    "frame_count % frame_count is 0 -- this is exactly why stop_limit \
                     subtracts half a frame"
                );
            }
        }
    }

    /// A stop limit must never collide with the free-running sentinel, or a
    /// render asked to stop would loop instead. The smallest case is one frame.
    #[test]
    fn a_stop_limit_is_always_a_positive_time_never_the_free_running_sentinel() {
        for &fps in &[1.0f32, 30.0, 240.0] {
            for &frames in &[1usize, 2, 65_535] {
                let limit = stop_limit(fps, frames);
                assert!(limit > 0.0, "{limit} at {fps} fps over {frames} frames must be > 0");
                assert_ne!(limit, FREE_RUNNING_LIMIT);
            }
        }
    }

    /// Half a frame before the end, expressed the other way round: the limit
    /// sits strictly inside the last frame's slot, between its start and the
    /// clip's full duration.
    #[test]
    fn the_stop_limit_sits_inside_the_last_frames_slot() {
        let (fps, frames) = (30.0f32, 90usize);
        let limit = stop_limit(fps, frames);
        let last_slot_start = (frames - 1) as f64 / fps as f64;
        let clip_duration = frames as f64 / fps as f64;
        assert!(limit > last_slot_start, "{limit} must be past the last frame's start");
        assert!(limit < clip_duration, "{limit} must be short of the clip's full duration");
    }

    /// A looping clock's limit is the free-running sentinel, which is what
    /// makes `ModuloFloored` wrap forever. Stated so the constant cannot drift
    /// to the schema's `1.0` struct default unnoticed.
    #[test]
    fn the_looping_limit_is_free_running_zero() {
        assert_eq!(FREE_RUNNING_LIMIT, 0.0);
    }

    // --- build_clock itself --------------------------------------------------
    //
    // Four of the five tests above are about `stop_limit`, a one-line pure
    // function. `build_clock` -- the thing that puts gates and wires in a
    // save, and whose wire count the hex cost estimate was wrong about for its
    // whole life -- had none at all.

    /// A chip to build a clock into, with the world that owns its wires.
    fn a_chip() -> (World, super::Chip) {
        let mut world = World::new();
        let chip = super::super::chip::new_chip(
            &mut world,
            Position { x: 0, y: 0, z: 2 },
            brdb::Vector3f { x: 0.0, y: 0.0, z: 40.0 },
            IntVector { x: 5, y: 5, z: 5 },
        );
        (world, chip)
    }

    /// Every wire as `(source component/port -> target component/port)`, which
    /// is what makes the assertion below readable when it fails.
    fn wire_shapes(world: &World) -> Vec<(String, String)> {
        world
            .wires
            .iter()
            .map(|w| {
                (
                    format!("{}.{}", w.source.component_type, w.source.port_name),
                    format!("{}.{}", w.target.component_type, w.target.port_name),
                )
            })
            .collect()
    }

    /// The eleven wires, named. `cost::estimate` once counted six of them for
    /// hex mode at every screen size -- the `Rate` and `Done` pins were missing
    /// from the formula -- so listing them by port rather than counting makes a
    /// wire that moves as visible as one that disappears.
    #[test]
    fn build_clock_emits_exactly_eleven_wires_and_they_are_these_eleven() {
        let (mut world, mut chip) = a_chip();
        build_clock(&mut world, &mut chip, 10.0, 90, true, Position { x: 0, y: 0, z: 6 });

        let mut got = wire_shapes(&world);
        got.sort();
        let mut want = vec![
            // The chain.
            (format!("{TIMER}.Time"), format!("{MULTIPLY}.InputA")),
            (format!("{MULTIPLY}.Output"), format!("{BITWISE_OR}.InputA")),
            (format!("{BITWISE_OR}.Output"), format!("{MODULO}.InputA")),
            // The three control pins, each straight into the timer.
            (
                format!("{}.RER_Output", super::super::chip::MICROCHIP_INPUT),
                format!("{TIMER}.Pause"),
            ),
            (
                format!("{}.RER_Output", super::super::chip::MICROCHIP_INPUT),
                format!("{TIMER}.Restart"),
            ),
            (
                format!("{}.RER_Output", super::super::chip::MICROCHIP_INPUT),
                format!("{TIMER}.Resume"),
            ),
            // Rate into the fps multiply, and the Done exec out.
            (
                format!("{}.RER_Output", super::super::chip::MICROCHIP_INPUT),
                format!("{MULTIPLY}.InputB"),
            ),
            (
                format!("{TIMER}.Expired"),
                format!("{}.RER_Input", super::super::chip::MICROCHIP_OUTPUT),
            ),
            // The two status taps: length (a constant Multiply, out to a pin)
            // and progress (the wrapped index into a scaling Multiply, out to a
            // pin). Reading `Modulo.Output` again is dataflow fan-out.
            (
                format!("{MULTIPLY}.Output"),
                format!("{}.RER_Input", super::super::chip::MICROCHIP_OUTPUT),
            ),
            (format!("{MODULO}.Output"), format!("{MULTIPLY}.InputA")),
            (
                format!("{MULTIPLY}.Output"),
                format!("{}.RER_Input", super::super::chip::MICROCHIP_OUTPUT),
            ),
        ];
        want.sort();
        assert_eq!(got, want, "the clock's wiring is not what the estimators count");
        assert_eq!(
            world.wires.len(),
            11,
            "11 is the clock's wire count every `cost` estimator now uses: 3 chain, \
             3 control pins, Rate, Done, and the length + progress taps (2 pin \
             writes + the shared frame-index read)"
        );
    }

    /// Six gates and seven pins, and the `frame_index` the whole render hangs
    /// off is the `ModuloFloored`'s output -- not the multiply's, and not the
    /// timer's raw `Time`.
    #[test]
    fn build_clock_emits_six_gates_seven_pins_and_a_wrapped_frame_index() {
        let (mut world, mut chip) = a_chip();
        let clock = build_clock(&mut world, &mut chip, 30.0, 90, true, Position { x: 0, y: 0, z: 6 });

        assert_eq!(
            chip.placed().len(),
            13,
            "6 gates (timer, fps multiply, bitwise-or, modulo, length, progress) + 7 \
             pins (Pause, Restart, Resume, Rate, Done, Length, Progress)"
        );
        assert_eq!(clock.frame_index.port_name.to_string(), "Output");
        assert_eq!(
            clock.frame_index.component_type.to_string(),
            MODULO,
            "the index must be the wrapped one -- an unwrapped index runs off the end of \
             every array in the render"
        );
        // Every pin is a distinct brick, or two of them would be the same
        // physical pin under two names.
        let pins = [
            clock.pause_pin,
            clock.restart_pin,
            clock.resume_pin,
            clock.rate_pin,
            clock.done_pin,
            clock.progress_pin,
            clock.length_pin,
        ];
        let unique: std::collections::HashSet<usize> = pins.iter().copied().collect();
        assert_eq!(unique.len(), 7, "seven pins must be seven bricks");
    }

    /// The loop toggle changes one inlined value and nothing else. Every
    /// cost estimate counts a flat 6 clock gates and 11 clock wires without
    /// seeing this flag, so that stays true only while both settings emit
    /// identical structure. (What the flag writes into `Timer.Limit` is
    /// checked against a real save in
    /// `tests/anim_world.rs::a_non_looping_clock_writes_a_limit_landing_on_the_last_frame`.)
    #[test]
    fn the_loop_toggle_changes_no_gate_no_pin_and_no_wire() {
        let build = |looping: bool| {
            let (mut world, mut chip) = a_chip();
            build_clock(&mut world, &mut chip, 15.0, 90, looping, Position { x: 0, y: 0, z: 6 });
            let placed: Vec<_> = chip.placed().to_vec();
            let mut wires = wire_shapes(&world);
            wires.sort();
            (placed, wires)
        };
        let (loop_placed, loop_wires) = build(true);
        let (stop_placed, stop_wires) = build(false);
        assert_eq!(loop_placed, stop_placed, "same gates and pins, in the same places");
        assert_eq!(loop_wires, stop_wires, "same wiring");
        // And the two limits really do differ, or this test is comparing a
        // change that never happened.
        assert_ne!(stop_limit(15.0, 90), FREE_RUNNING_LIMIT);
    }

    /// The clock's own bricks must not collide with each other. `chip::finish`
    /// checks it for a whole render, but a clock built into an empty chip is
    /// the smallest case and the one a layout change would break first.
    #[test]
    fn the_clocks_own_bricks_never_overlap() {
        for origin in [Position { x: 0, y: 0, z: 6 }, Position { x: 40, y: 20, z: 6 }] {
            let (mut world, mut chip) = a_chip();
            build_clock(&mut world, &mut chip, 10.0, 5, true, origin);
            super::super::layout::assert_no_overlap(chip.placed())
                .unwrap_or_else(|e| panic!("clock at {origin:?}: {e}"));
        }
    }
}
