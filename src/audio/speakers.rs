//! The speaker cluster and its wiring.
//!
//! Two components per band per bank: an `ArrayVar` of `f64` volumes and an
//! `ArrayVar_Get` that reads it into that speaker's `VolumeMultiplier` -- no
//! `Component_BrickPropertyChanger`, since volume is a real wire input.
//! Pitch and the audio asset are build-time component data: no gates, no wires.
//!
//! Four more inputs are microchip pins: `Inner Radius`, `Max Distance` and
//! `Directional` fan out to every speaker's matching emitter port, `Volume`
//! scales every band through a per-band multiply. Each keeps a baked value
//! underneath for when its pin is unwired.
use super::bands::{BandKind, PITCH_MAX, PITCH_MIN};
use super::track::{AudioOptions, VoiceTrack};
use super::voices::VoiceStreams;
use crate::anim::bricks::{
    ARRAY_GET, ARRAY_VAR, BRANCH, CHANGE_DETECTOR, COMPARE_GE, SELECT, SUBTRACT,
};
use crate::anim::chip::{Chip, add_input_pin, finish, new_chip, pin_source};
use crate::anim::clock::{MULTIPLY, build_clock, gate};
use crate::anim::layout::{GATE_HALF, STAGE_BASE_Z, STAGE_PITCH, lattice_pos_staged};
use brdb::{
    AsBrdbValue, Brick, BrickType, IntVector, Position, Vector3f, WirePort, World,
    assets::{
        LiteralComponent,
        external::{ASSET_TYPES, BA_SYNTH_NOISE_PINK, BA_SYNTH_NOISE_WHITE},
    },
    schema::{BrdbValue, WireArrayVariant, WireVariant},
};

/// The emitter component's type name: a plain `Component_*` (no
/// `BrickComponentType_` prefix), resolved by unvalidated string lookup, so a
/// wrong name encodes fine and does nothing in game.
pub const AUDIO_EMITTER: &str = "Component_AudioEmitter";
pub const SPEAKER_BRICK: &str = "B_1x1F_Speaker";

/// One-tick delay gate: its `Output` is whatever reached its `Input` one game
/// tick ago. Used by [`scaffold`] to compare the clock's `Time` against its own
/// previous value and so detect whether the clock is advancing.
pub const BUFFER_TICKS: &str = "BrickComponentType_WireGraphPseudo_BufferTicks";
/// `!=` comparator: its `bOutput` is true while `InputA` differs from `InputB`.
/// [`scaffold`] feeds it `Time` and last tick's `Time`, so `bOutput` is "the
/// clock is moving".
pub const COMPARE_NE: &str = "BrickComponentType_WireGraph_Expr_CompareNotEqual";
/// The external asset type every synth descriptor lives under.
pub const AUDIO_ASSET_TYPE: &str = "BrickAudioDescriptor";
/// `B_1x1F_Speaker`'s true half-extents, from the game's brick catalog. Not
/// what the build-time overlap check measures -- see [`speaker_half`]. Only
/// ever used as a floor under the measured value.
const SPEAKER_HALF: IntVector = IntVector { x: 5, y: 5, z: 2 };

/// Lattice "height" fed to [`lattice_pos_staged`] for the service rows: 1, so
/// row 0 is the front row and every used row index maps to x >= 5.
const LATTICE_HEIGHT: i32 = 1;

/// Default baked `InnerRadius` / `MaxDistance` per speaker, game units (10 = 1
/// brick). Written explicitly, never left to brdb's `STRUCT_DEFAULTS`
/// (regenerated per build). 15 sits inside the cluster's diagonal (see
/// [`cluster_dims`]), so the bank attenuates within itself; `--inner-radius` /
/// `--max-distance` raise it for a flat field. `bSpatialization = false` does
/// NOT disable distance attenuation.
pub const DEFAULT_INNER_RADIUS: f32 = 15.0;
pub const DEFAULT_MAX_DISTANCE: f32 = 400.0;

/// Baked `bSpatialization`, overridden by the `Directional` pin. Must stay
/// `false`: spatialized emitters at 32 distinct positions comb-filter against
/// each other as the listener moves. Game default is `true`, so omitting this
/// field turns spatialization ON. Does NOT disable distance attenuation.
const SPATIALIZATION: bool = false;

/// Baked `InputB` on the per-band volume multiply, overridden by the `Volume`
/// pin. Must stay `1.0`: it's the gain applied while the pin is unwired, so
/// any other value rescales every render that never touches the pin.
const VOLUME_SCALE: f64 = 1.0;

/// The speaker brick's half-size as `layout::assert_bricks_dont_overlap`
/// measures it (from `local_bounds`), floored at the true size
/// ([`SPEAKER_HALF`]). brdb 0.9.1 returns a flat (5, 5, 6) guess for every
/// basic brick; spacing on the guess is always safe (bigger, never smaller),
/// while a hardcoded true height reads as a collision and is rejected.
pub fn speaker_half() -> IntVector {
    let (min, max) = Brick {
        asset: BrickType::from(SPEAKER_BRICK),
        ..Default::default()
    }
    .local_bounds();
    IntVector {
        x: ((max.x - min.x) / 2).max(SPEAKER_HALF.x),
        y: ((max.y - min.y) / 2).max(SPEAKER_HALF.y),
        z: ((max.z - min.z) / 2).max(SPEAKER_HALF.z),
    }
}

/// Slot counts (nx,ny,nz) for a compact near-cubic cluster of n speakers,
/// nx>=ny>=nz. Compact because bSpatialization=false leaves distance
/// attenuation on, so a spread bank is heard as a slice of the spectrum.
/// Integer-only: cbrt(27.0) can be 3.0000000000000004 and ceil to 4. Pure
/// function of n (two renders of one file must place band k identically).
pub fn cluster_dims(n: usize) -> (usize, usize, usize) {
    let n = n.max(1);
    let mut nx = 1usize;
    while nx * nx * nx < n {
        nx += 1;
    }
    let mut ny = 1usize;
    while nx * ny * ny < n {
        ny += 1;
    }
    let nz = n.div_ceil(nx * ny);
    (nx, ny, nz)
}

/// Where speaker `index` of `total` sits on the main grid: x fastest, then y,
/// then z, over the slots [`cluster_dims`] hands back. Each index gets its own
/// slot -- an overlap makes the game silently delete one of the two bricks.
/// Coordinates stay >= 0 so the chip can sit at a fixed negative x and clear
/// the cluster for any band count.
pub fn speaker_position(index: usize, total: usize) -> Position {
    let (nx, ny, _) = cluster_dims(total);
    let half = speaker_half();
    let ix = (index % nx) as i32;
    let iy = ((index / nx) % ny) as i32;
    let iz = (index / (nx * ny)) as i32;
    Position {
        x: half.x + ix * half.x * 2,
        y: half.y + iy * half.y * 2,
        z: half.z + iz * half.z * 2,
    }
}

/// Low-x face of the in-chip speaker block, in inner-grid units. Service gates
/// occupy rows 0..-10, mapped by [`service`] onto x in `[0, 110]`; starting the
/// block at 120 leaves a full gate cell of clearance. Growing the band count
/// lengthens the service rows along y, never x, so the margin holds for any render.
const SPEAKER_BLOCK_X: i32 = 120;

/// Where in-chip speaker `index` of `total` sits on the chip's inner grid, for
/// [`AudioOptions::speakers_in_chip`]. Same [`cluster_dims`] packing as
/// [`speaker_position`], offset past the service rows ([`SPEAKER_BLOCK_X`]) and
/// lifted onto [`STAGE_BASE_Z`]. Coordinates stay non-negative: negative
/// inner-grid coordinates delete bricks in-game. An emitter on the inner grid
/// plays from the chip's world position, so this shape buys no near-field
/// geometry -- it only needs to be non-overlapping.
pub fn speaker_inner_position(index: usize, total: usize) -> Position {
    let (nx, ny, _) = cluster_dims(total);
    let half = speaker_half();
    let ix = (index % nx) as i32;
    let iy = ((index / nx) % ny) as i32;
    let iz = (index / (nx * ny)) as i32;
    Position {
        x: SPEAKER_BLOCK_X + half.x + ix * half.x * 2,
        y: half.y + iy * half.y * 2,
        z: STAGE_BASE_Z + half.z + iz * half.z * 2,
    }
}

/// Where a service gate at lattice (`col`, `row`) sits inside the chip. Rows
/// count backward from the clock (0) through the select cascade (-9) and the
/// master-volume multiplies (-10). `row` is an index, not a coordinate: it maps
/// to `x = (LATTICE_HEIGHT - 1 - row) * CELL + half.x`, so a more negative row
/// lands further along positive x and every brick stays in the non-negative
/// octant (negative inner-grid coordinates delete bricks in-game).
fn service(col: i32, row: i32) -> Position {
    lattice_pos_staged(col, row, 0, LATTICE_HEIGHT, GATE_HALF, STAGE_PITCH)
}

/// Reject an attenuation setting that would encode perfectly and be audible
/// nowhere: a NaN, infinity or zero radius encodes fine and is silent only in
/// game. Shared by the flag parser, the GUI and [`cost::check`](crate::audio::cost::check),
/// so the refusal lands before the analysis runs, not after.
pub(crate) fn check_attenuation(opts: &AudioOptions) -> Result<(), String> {
    for (flag, value) in [
        ("--inner-radius", opts.inner_radius),
        ("--max-distance", opts.max_distance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "{flag} must be a positive finite number of units, got {value}"
            ));
        }
    }
    if opts.inner_radius > opts.max_distance {
        return Err(format!(
            "--inner-radius ({}) must not exceed --max-distance ({}): the inner radius \
             is the zone of NO attenuation and the max distance is where the sound \
             stops, so the first can never be the larger",
            opts.inner_radius, opts.max_distance
        ));
    }
    Ok(())
}

/// Build one speaker carrying a baked `Component_AudioEmitter`, mint its brick
/// id (for wiring its ports), and place it: on the main grid when `deferred`
/// is `None`, or pushed into `deferred` for [`AudioOptions::speakers_in_chip`]
/// to hand to `Chip::add_brick` once the chip exists.
///
/// `position` is [`speaker_position`] for the main-grid cluster or
/// [`speaker_inner_position`] for the in-chip block. `pitch` is the starting
/// pitch: the band's fixed pitch in the bank (never rewritten), or frame 0's
/// value in voice mode (overwritten every frame by a wire) -- either way it's
/// what a paused or unwired chip plays.
fn add_emitter(
    world: &mut World,
    asset_name: &str,
    pitch: f32,
    position: Position,
    opts: &AudioOptions,
    deferred: Option<&mut Vec<Brick>>,
) -> Result<usize, String> {
    let mut data: std::collections::HashMap<brdb::BString, Box<dyn AsBrdbValue>> =
        std::collections::HashMap::new();
    // Bare `f32`, not a `WireVariant`: the schema declares PitchMultiplier and
    // VolumeMultiplier as plain f32 fields, not the tagged wire-graph union
    // gate ports use, so a WireVariant here compiles and fails at encode with
    // UnimplementedCast("f32", WireVariant).
    data.insert(
        "PitchMultiplier".into(),
        Box::new(pitch) as Box<dyn AsBrdbValue>,
    );
    // 2D sound: emitters at distinct positions would otherwise comb-filter
    // against each other as the listener moves. Also what `Directional`
    // overrides while unwired.
    data.insert(
        "bSpatialization".into(),
        Box::new(SPATIALIZATION) as Box<dyn AsBrdbValue>,
    );
    // Live in a default render: bSpatialization=false disables panning only,
    // not distance attenuation. Also bare f32, same trap as PitchMultiplier.
    data.insert(
        "InnerRadius".into(),
        Box::new(opts.inner_radius) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "MaxDistance".into(),
        Box::new(opts.max_distance) as Box<dyn AsBrdbValue>,
    );
    data.insert("bEnabled".into(), Box::new(true) as Box<dyn AsBrdbValue>);
    // Starts silent; the wire drives it from frame 0.
    data.insert(
        "VolumeMultiplier".into(),
        Box::new(0.0f32) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "AudioDescriptor".into(),
        audio_descriptor_value(world, asset_name)?,
    );

    let (brick, id) = Brick {
        asset: BrickType::from(SPEAKER_BRICK),
        position,
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(
        AUDIO_EMITTER,
        std::sync::Arc::new(data),
    ))
    .with_id_split();
    // Main grid now, or deferred for the chip's inner grid. Wires reference
    // `id`, not the brick's location, and brdb resolves each to a local or
    // remote source at write time from where its bricks land.
    match deferred {
        Some(v) => v.push(brick),
        None => world.add_brick(brick),
    }
    Ok(id)
}

/// The chip, its clock, and the four attenuation/volume input pins -- the
/// scaffolding both render modes put around their speakers.
struct Scaffold {
    chip: Chip,
    /// The clock's `Pause`/`Restart`/`Resume` control pin ids, for the
    /// pre-wired control buttons. Audio always builds the clock, so these
    /// always exist.
    control_pins: (usize, usize, usize),
    /// Source port carrying the wrapped integer frame index.
    frame_index: WirePort,
    /// The gated master-volume source every per-speaker multiply reads: the
    /// pause-mute `Select`'s output, not the raw `Volume` pin. Passes the
    /// master volume through while the clock advances, emits 0 while frozen
    /// (paused, an ended no-loop track, or a stalled external clock).
    master_volume: WirePort,
}

/// Build the microchip, its clock, and the four input pins, and fan three of
/// them out to every speaker.
///
/// An unwired input pin drives NOTHING: the target keeps its baked value (same
/// contract as clock.rs's rate_pin). So each of the four pins keeps a real
/// baked value underneath it; the inverse (unwired pin forcing 0) would
/// silence the build in game with a save that looks perfect.
fn scaffold(
    world: &mut World,
    speaker_ids: &[usize],
    fps: f32,
    frame_count: usize,
    loop_playback: bool,
) -> Scaffold {
    // Beside the cluster on x, never inside it: an overlap on the main grid
    // silently drops one of the two bricks. The cluster grows from the origin
    // along positive x/y/z, so any chip_x that keeps the shell negative clears
    // it for every speaker count.
    let chip_x = -(speaker_half().x * 4);
    let mut chip = new_chip(
        world,
        Position {
            x: chip_x,
            y: 0,
            z: 2,
        },
        Vector3f {
            x: chip_x as f32,
            y: 0.0,
            z: 40.0,
        },
        IntVector {
            x: 10,
            y: 10,
            z: 2,
        },
    );

    // Timer -> multiply by fps -> truncate -> modulo frame count, plus
    // Pause/Restart/Resume/Rate pins. `loop_playback` rides along so a track
    // and a clip stop the same way.
    let clock = build_clock(world, &mut chip, fps, frame_count, loop_playback, service(0, 0));
    let frame_index = clock.frame_index.clone();
    // The clock's control pins, for the pre-wired control buttons.
    let control_pins = (clock.pause_pin, clock.restart_pin, clock.resume_pin);
    // The timer's raw stopwatch, tapped for the pause-mute detector below.
    let time = clock.time.clone();

    // The clock occupies the positive rows from 0 (its six gates and seven
    // pins grow downward from `service(0, 0)`); these speaker input pins sit at
    // row -2 and the pause-mute detector at -4, both clear of it. `Directional`
    // is the user-facing name for `bSpatialization`, the only directional
    // property this component has.
    let inner_radius_pin = add_input_pin(&mut chip, "Inner Radius", service(0, -2));
    let max_distance_pin = add_input_pin(&mut chip, "Max Distance", service(1, -2));
    let directional_pin = add_input_pin(&mut chip, "Directional", service(2, -2));
    let volume_pin = add_input_pin(&mut chip, "Volume", service(3, -2));

    // Three fan straight out to every speaker's own emitter port -- one pin,
    // `speaker_ids.len()` wires -- so a pin wired to only some of the cluster
    // never happens. `pin_source(pin, true)` is the input pin's `RER_Output`,
    // the port that carries its value onward into the chip's graph.
    for (pin, port) in [
        (inner_radius_pin, "InnerRadius"),
        (max_distance_pin, "MaxDistance"),
        (directional_pin, "bSpatialization"),
    ] {
        for &speaker in speaker_ids {
            world.add_wire_connection(
                pin_source(pin, true),
                WirePort::new(speaker, AUDIO_EMITTER, port),
            );
        }
    }

    // --- Pause-mute detector: silence while the clock is frozen -------------
    // Three shared gates. Gates on whether the clock is actually advancing,
    // not the Pause exec, so any stall (pause, stalled external clock, ended
    // no-loop track) silences the bank. `BufferTicks` holds `Timer.Time` back
    // one tick; `CompareNotEqual` is true while `Time` changed since last
    // tick; `Select` passes the master volume through while true, emits 0
    // while frozen.
    //
    // UNVERIFIED IN GAME: assumes `Time` updates every tick, not just at the
    // fps interval -- if it only refreshes at fps, audio would chop into
    // clicks instead of just silencing on pause. Fallback: key off the Pause
    // exec with an `Exec_Toggle` latch instead of Time-change detection.
    //
    // `TicksToWait` is a bare `i32` (not a `WireVariant`); the `Select`'s
    // `InputA`/`InputB` are `WireVariant::Number`.
    let buffer = gate(
        &mut chip,
        "B_1x1_Gate_Pseudo_BufferTicks",
        BUFFER_TICKS,
        service(0, -3),
        vec![("TicksToWait", Box::new(1i32) as Box<dyn AsBrdbValue>)],
    );
    world.add_wire_connection(time.clone(), WirePort::new(buffer, BUFFER_TICKS, "Input"));

    let playing = gate(
        &mut chip,
        "B_1x1_Gate_Expr_CompareNotEqual",
        COMPARE_NE,
        service(1, -3),
        vec![],
    );
    world.add_wire_connection(time, WirePort::new(playing, COMPARE_NE, "InputA"));
    world.add_wire_connection(
        WirePort::new(buffer, BUFFER_TICKS, "Output"),
        WirePort::new(playing, COMPARE_NE, "InputB"),
    );

    // `bSelectB` true -> InputB (master volume); false -> InputA (baked 0.0).
    // `InputB` also carries the baked [`VOLUME_SCALE`] (1.0) underneath the
    // `Volume` pin, so an unwired pin on a playing render multiplies by 1.0.
    let gated = gate(
        &mut chip,
        "B_1x1_Gate_Expr_Select",
        SELECT,
        service(2, -3),
        vec![
            (
                "InputA",
                Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
            ),
            (
                "InputB",
                Box::new(WireVariant::Number(VOLUME_SCALE)) as Box<dyn AsBrdbValue>,
            ),
        ],
    );
    world.add_wire_connection(
        WirePort::new(playing, COMPARE_NE, "bOutput"),
        WirePort::new(gated, SELECT, "bSelectB"),
    );
    world.add_wire_connection(
        pin_source(volume_pin, true),
        WirePort::new(gated, SELECT, "InputB"),
    );

    Scaffold {
        chip,
        control_pins,
        frame_index,
        master_volume: WirePort::new(gated, SELECT, "Output"),
    }
}

/// Per-speaker master-volume multiply: frame value on InputA, Volume pin on
/// InputB. Separate gate because VolumeMultiplier already has one source;
/// InputB inlines VOLUME_SCALE (1.0) so an untouched chip renders identically.
/// The unwired-Volume-pin unity default lives on the Select's own InputB (see
/// [`scaffold`]); this literal is a fallback if that wire were ever absent.
/// TYPE TRAP: MathMultiply ports are WireVariant::Number, unlike the emitter's
/// bare-f32 PitchMultiplier/VolumeMultiplier. Not interchangeable.
fn volume_multiply(
    world: &mut World,
    chip: &mut Chip,
    slot: usize,
    speaker: usize,
    master_volume: &WirePort,
) -> WirePort {
    let gate_id = gate(
        chip,
        "B_1x1_Gate_Expr_MathMultiply",
        MULTIPLY,
        service(slot as i32, -10),
        vec![(
            "InputB",
            Box::new(WireVariant::Number(VOLUME_SCALE)) as Box<dyn AsBrdbValue>,
        )],
    );
    world.add_wire_connection(
        master_volume.clone(),
        WirePort::new(gate_id, MULTIPLY, "InputB"),
    );
    world.add_wire_connection(
        WirePort::new(gate_id, MULTIPLY, "Output"),
        WirePort::new(speaker, AUDIO_EMITTER, "VolumeMultiplier"),
    );
    WirePort::new(gate_id, MULTIPLY, "InputA")
}

/// One dense per-frame stream of `f64` and the port its value drives.
///
/// This is the unit the banking machinery works in, and it is what lets the
/// band bank and the voice mode share it: the bank hands over one stream per
/// band (its volume), the voice mode two per voice (its pitch and its volume),
/// and neither knows anything about the other's shape.
struct FrameStream<'a> {
    /// `frame_count` values, dense. Short arrays are legal (the last bank is
    /// usually a partial one) but a stream shorter than the track silently
    /// stops updating at its end rather than erroring, so callers check.
    values: &'a [f64],
    target: WirePort,
}

/// Build the change-detector-driven, banked `ArrayVar`/`ArrayVar_Get` cascade
/// that writes one value from each stream on every frame.
///
/// Two gates per stream per bank: an `ArrayVar` holding that bank's slice and
/// an `ArrayVar_Get` that reads it at the frame index. Where there is more than
/// one bank, a `Branch` cascade at the front picks whose exec chain runs and a
/// `Select` cascade at the back picks whose value reaches the target.
///
/// Reused verbatim by both render modes. It is the only part of either that
/// scales with the length of the track, so a second copy of it would be the
/// second thing to get wrong about bank boundaries.
fn build_stream_cascade(
    world: &mut World,
    chip: &mut Chip,
    frame_index: &WirePort,
    frame_count: usize,
    bank_size: usize,
    streams: &[FrameStream<'_>],
) {
    let bank_size = bank_size.max(1);
    let n_banks = frame_count.div_ceil(bank_size).max(1);

    let detector = gate(
        chip,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, -4),
        vec![],
    );
    world.add_wire_connection(
        frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    let mut index_of_bank = Vec::with_capacity(n_banks);
    index_of_bank.push(frame_index.clone());
    for k in 1..n_banks {
        let sub = gate(
            chip,
            "B_1x1_Gate_Expr_MathSubtract",
            SUBTRACT,
            service(k as i32, -6),
            vec![(
                "InputB",
                Box::new(WireVariant::Number((k * bank_size) as f64)) as Box<dyn AsBrdbValue>,
            )],
        );
        world.add_wire_connection(frame_index.clone(), WirePort::new(sub, SUBTRACT, "InputA"));
        index_of_bank.push(WirePort::new(sub, SUBTRACT, "Output"));
    }

    let mut ge = Vec::with_capacity(n_banks.saturating_sub(1));
    for k in 1..n_banks {
        let cmp = gate(
            chip,
            "B_1x1_Gate_Expr_CompareGreaterOrEqual",
            COMPARE_GE,
            service(k as i32, -7),
            vec![(
                "InputB",
                Box::new(WireVariant::Int((k * bank_size) as i64)) as Box<dyn AsBrdbValue>,
            )],
        );
        world.add_wire_connection(frame_index.clone(), WirePort::new(cmp, COMPARE_GE, "InputA"));
        ge.push(WirePort::new(cmp, COMPARE_GE, "bOutput"));
    }

    // get_of[bank][stream]
    let n_streams = streams.len();
    let mut get_of: Vec<Vec<usize>> = vec![Vec::with_capacity(n_streams); n_banks];
    for (s, stream) in streams.iter().enumerate() {
        for bi in 0..n_banks {
            let lo = bi * bank_size;
            let hi = ((bi + 1) * bank_size).min(stream.values.len());
            let slice = if lo < hi {
                stream.values[lo..hi].to_vec()
            } else {
                Vec::new()
            };
            let col = ((s * n_banks + bi) * 2) as i32;
            let array = gate(
                chip,
                "B_1x1_Gate_Variable_Array",
                ARRAY_VAR,
                service(col, -5),
                vec![(
                    "Value",
                    Box::new(WireArrayVariant::DoubleArray(slice)) as Box<dyn AsBrdbValue>,
                )],
            );
            let get = gate(
                chip,
                "B_1x1_Gate_Exec_ArrayVar_Get",
                ARRAY_GET,
                service(col + 1, -5),
                vec![],
            );
            world.add_wire_connection(
                WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
                WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
            );
            world.add_wire_connection(
                index_of_bank[bi].clone(),
                WirePort::new(get, ARRAY_GET, "Index"),
            );
            get_of[bi].push(get);
        }
    }

    // Exec: branches cascade at the front so exactly one bank's chain runs.
    // A truthy `bCond` takes `ExecOutA` -- easy to invert by mistake.
    let mut exec_src = WirePort::new(detector, CHANGE_DETECTOR, "OnChanged");
    for bi in 0..n_banks {
        let entry = if bi + 1 < n_banks {
            let br = gate(
                chip,
                "B_1x1_Gate_Exec_Branch",
                BRANCH,
                service(bi as i32, -8),
                vec![],
            );
            world.add_wire_connection(ge[bi].clone(), WirePort::new(br, BRANCH, "bCond"));
            world.add_wire_connection(exec_src, WirePort::new(br, BRANCH, "Exec"));
            exec_src = WirePort::new(br, BRANCH, "ExecOutA");
            WirePort::new(br, BRANCH, "ExecOutB")
        } else {
            exec_src.clone()
        };
        let mut prev = entry;
        for &get in &get_of[bi] {
            world.add_wire_connection(prev, WirePort::new(get, ARRAY_GET, "Exec"));
            prev = WirePort::new(get, ARRAY_GET, "ExecOut");
        }
    }

    // Value: one select per stream per boundary, cascading, then into the
    // stream's target.
    for (s, stream) in streams.iter().enumerate() {
        let mut value = WirePort::new(get_of[0][s], ARRAY_GET, "Value");
        for bi in 1..n_banks {
            let sel = gate(
                chip,
                "B_1x1_Gate_Expr_Select",
                SELECT,
                service((s * n_banks + bi) as i32, -9),
                vec![],
            );
            world.add_wire_connection(ge[bi - 1].clone(), WirePort::new(sel, SELECT, "bSelectB"));
            world.add_wire_connection(value, WirePort::new(sel, SELECT, "InputA"));
            world.add_wire_connection(
                WirePort::new(get_of[bi][s], ARRAY_GET, "Value"),
                WirePort::new(sel, SELECT, "InputB"),
            );
            value = WirePort::new(sel, SELECT, "Output");
        }
        world.add_wire_connection(value, stream.target.clone());
    }
}

/// Build a world whose speaker cluster plays `track` back as a fixed band bank.
pub fn build_speaker_world(track: &VoiceTrack, opts: &AudioOptions) -> Result<World, String> {
    if track.frame_count == 0 {
        return Err("audio track has 0 frames -- nothing to render".to_string());
    }
    if track.volumes.len() != track.plan.len() {
        return Err(format!(
            "track has {} volume arrays but {} bands",
            track.volumes.len(),
            track.plan.len()
        ));
    }
    check_attenuation(opts)?;

    let mut world = World::new();
    world.meta.bundle.description =
        "Audio spectrum playback generated from an audio file".to_string();

    // --- 1. The speaker cluster on the main grid ----------------------------
    // Position is audible: bSpatialization=false stops panning, not distance
    // attenuation, so speakers go into the tightest 3D packing (cluster_dims),
    // not a column.
    let n_speakers = track.plan.len();
    let in_chip = opts.speakers_in_chip;
    let mut speaker_ids = Vec::with_capacity(n_speakers);
    // Holds emitter bricks until the chip exists, when --speakers-in-chip is
    // set; empty otherwise (placement loop below is then a no-op).
    let mut in_chip_speakers: Vec<Brick> = Vec::new();
    for (b, kind) in track.plan.kinds.iter().enumerate() {
        let asset_name = match kind {
            // Tonal bands play the user-selected waveform (sine by default);
            // the noise bands are a different asset and untouched by the choice.
            BandKind::Tonal => opts.tonal_synth.asset(),
            BandKind::WhiteNoise => BA_SYNTH_NOISE_WHITE,
            BandKind::PinkNoise => BA_SYNTH_NOISE_PINK,
        };
        let position = if in_chip {
            speaker_inner_position(b, n_speakers)
        } else {
            speaker_position(b, n_speakers)
        };
        speaker_ids.push(add_emitter(
            &mut world,
            asset_name.as_ref(),
            track.plan.pitches[b],
            position,
            opts,
            in_chip.then_some(&mut in_chip_speakers),
        )?);
    }

    // --- 2. Chip, clock and the four input pins -----------------------------
    let mut sc = scaffold(
        &mut world,
        &speaker_ids,
        track.fps,
        track.frame_count,
        opts.loop_playback,
    );

    // With --speakers-in-chip, place the deferred speakers on the chip's inner
    // grid now it exists; empty (a no-op) for the default layout.
    for brick in in_chip_speakers {
        sc.chip.add_brick(brick, speaker_half());
    }

    // --- 3. One master-volume multiply per band -----------------------------
    let targets: Vec<WirePort> = speaker_ids
        .iter()
        .enumerate()
        .map(|(b, &speaker)| {
            volume_multiply(&mut world, &mut sc.chip, b, speaker, &sc.master_volume)
        })
        .collect();

    // --- 4. One stream per band: its volume ---------------------------------
    // Pitch is build-time data here, never rewritten, so a band fading in and
    // out carries no retrigger risk -- the property voice mode gives up.
    let streams: Vec<FrameStream<'_>> = track
        .volumes
        .iter()
        .zip(targets)
        .map(|(values, target)| FrameStream { values, target })
        .collect();
    build_stream_cascade(
        &mut world,
        &mut sc.chip,
        &sc.frame_index,
        track.frame_count,
        opts.bank_size,
        &streams,
    );

    // --- 5. Control buttons -------------------------------------------------
    // Default-on: three labelled buttons wired into the clock's control pins,
    // placed beyond the cluster by `control_anchor`. Before `finish` so its
    // overlap check sees them.
    if opts.control_buttons {
        let (pause, restart, resume) = sc.control_pins;
        let anchor = crate::anim::controls::control_anchor(&world);
        crate::anim::controls::add_control_buttons(&mut world, pause, restart, resume, anchor);
    }

    finish(&mut world, sc.chip)?;
    // Must be last: register_used_components only sees registered gates after
    // finish publishes the chip's inner grid, else to_brz_vec fails with
    // UnregisteredComponentType.
    world.register_used_components();
    Ok(world)
}

/// Build a world whose speakers track spectral peaks: both pitch and volume
/// written every frame. Same cluster, chip, clock, pins and banking cascade as
/// [`build_speaker_world`], but `voice_count` speakers (all tonal, no
/// `BandKind`/noise -- cymbals, sibilance and sub-bass are simply absent, see
/// `voices::min_hz`) each contributing two streams instead of one.
///
/// UNVERIFIED IN GAME: `PitchMultiplier` is wired here, not baked, and it is
/// unconfirmed whether a per-frame pitch write retunes a running voice or
/// retriggers it (`diag_5_pitch_ramp.brz` in `examples/audio_diagnostics.rs`
/// settles it). The baked value is frame 0's, so a paused chip holds the
/// first note rather than an arbitrary one.
pub fn build_voice_world(streams: &VoiceStreams, opts: &AudioOptions) -> Result<World, String> {
    if streams.frame_count == 0 {
        return Err("audio track has 0 frames -- nothing to render".to_string());
    }
    let n_voices = streams.voice_count();
    if n_voices == 0 {
        return Err("a voice-mode track needs at least one voice".to_string());
    }
    if streams.volumes.len() != n_voices {
        return Err(format!(
            "track has {} pitch arrays but {} volume arrays",
            n_voices,
            streams.volumes.len()
        ));
    }
    // Both arrays of every voice are read at the same frame index, so a short
    // one would leave that voice reading a stale value (or nothing) from the
    // point it ran out -- silently, and only for part of the track.
    for (v, (p, vol)) in streams.pitches.iter().zip(&streams.volumes).enumerate() {
        if p.len() != streams.frame_count || vol.len() != streams.frame_count {
            return Err(format!(
                "voice {v} has {} pitch and {} volume values, expected {} of each",
                p.len(),
                vol.len(),
                streams.frame_count
            ));
        }
    }
    // A pitch outside the emitter's legal range is clamped in game, turning a
    // wrong number into a wrong note rather than silence. analyze_voices
    // already clamps; this guards any other caller.
    for (v, row) in streams.pitches.iter().enumerate() {
        for (f, &p) in row.iter().enumerate() {
            if !p.is_finite() || p < PITCH_MIN as f64 || p > PITCH_MAX as f64 {
                return Err(format!(
                    "voice {v} frame {f} has PitchMultiplier {p}, outside the emitter's \
                     legal {PITCH_MIN}..{PITCH_MAX} range -- the game would clamp it \
                     and play a wrong note"
                ));
            }
        }
    }
    check_attenuation(opts)?;

    let mut world = World::new();
    world.meta.bundle.description =
        "Audio peak-tracking playback generated from an audio file".to_string();

    // --- 1. The speaker cluster (main grid, or the chip's inner grid) -------
    let in_chip = opts.speakers_in_chip;
    let mut speaker_ids = Vec::with_capacity(n_voices);
    // Deferred until the chip exists; empty (loop a no-op) unless
    // `--speakers-in-chip`. Same mechanism as the bank builder.
    let mut in_chip_speakers: Vec<Brick> = Vec::new();
    for v in 0..n_voices {
        let position = if in_chip {
            speaker_inner_position(v, n_voices)
        } else {
            speaker_position(v, n_voices)
        };
        speaker_ids.push(add_emitter(
            &mut world,
            // Voice mode is every-speaker tonal (peak-tracking, no noise
            // bands), so the whole cluster plays the selected waveform.
            opts.tonal_synth.asset().as_ref(),
            streams.pitches[v][0] as f32,
            position,
            opts,
            in_chip.then_some(&mut in_chip_speakers),
        )?);
    }

    // --- 2. Chip, clock and the four input pins -----------------------------
    let mut sc = scaffold(
        &mut world,
        &speaker_ids,
        streams.fps,
        streams.frame_count,
        opts.loop_playback,
    );

    // Place the deferred in-chip speakers now the chip exists (a no-op for the
    // default beside-the-chip layout). See `build_speaker_world`.
    for brick in in_chip_speakers {
        sc.chip.add_brick(brick, speaker_half());
    }

    // --- 3. One master-volume multiply per voice ----------------------------
    let volume_targets: Vec<WirePort> = speaker_ids
        .iter()
        .enumerate()
        .map(|(v, &speaker)| {
            volume_multiply(&mut world, &mut sc.chip, v, speaker, &sc.master_volume)
        })
        .collect();

    // --- 4. Two streams per voice: its pitch and its volume -----------------
    // Interleaved per voice so the two arrays sit next to each other in the
    // chip. Pitch goes straight into the emitter with no multiply in between:
    // scaling it by Volume would transpose the render whenever volume changed.
    let mut frame_streams: Vec<FrameStream<'_>> = Vec::with_capacity(n_voices * 2);
    for v in 0..n_voices {
        frame_streams.push(FrameStream {
            values: &streams.pitches[v],
            target: WirePort::new(speaker_ids[v], AUDIO_EMITTER, "PitchMultiplier"),
        });
        frame_streams.push(FrameStream {
            values: &streams.volumes[v],
            target: volume_targets[v].clone(),
        });
    }
    build_stream_cascade(
        &mut world,
        &mut sc.chip,
        &sc.frame_index,
        streams.frame_count,
        opts.bank_size,
        &frame_streams,
    );

    // --- 5. Control buttons -------------------------------------------------
    // Same as build_speaker_world: default-on buttons wired into the clock's
    // control pins, before finish so its overlap check sees them.
    if opts.control_buttons {
        let (pause, restart, resume) = sc.control_pins;
        let anchor = crate::anim::controls::control_anchor(&world);
        crate::anim::controls::add_control_buttons(&mut world, pause, restart, resume, anchor);
    }

    finish(&mut world, sc.chip)?;
    // Must be last, after finish publishes the chip's inner grid -- see
    // [`build_speaker_world`].
    world.register_used_components();
    Ok(world)
}

/// Resolve a synth asset name to the value `AudioDescriptor` expects.
///
/// The schema type is a bare `object`, so the value is a [`BrdbValue::Asset`]
/// carrying the reference's index -- not `WireVariant::Object` (the wire-graph
/// union, a different encoding path; both compile).
///
/// The `(type, name)` pair is checked against brdb's `ASSET_TYPES` catalog
/// first: nothing on the write path does that otherwise, so a typo in either
/// string would encode fine and fail only in game.
fn audio_descriptor_value(
    world: &mut World,
    asset_name: &str,
) -> Result<Box<dyn AsBrdbValue>, String> {
    let (_, known) = ASSET_TYPES
        .iter()
        .find(|(ty, _)| *ty == AUDIO_ASSET_TYPE)
        .ok_or_else(|| format!("brdb knows no external asset type {AUDIO_ASSET_TYPE:?}"))?;
    if !known.iter().any(|a| a.as_ref() == asset_name) {
        return Err(format!(
            "{asset_name:?} is not a {AUDIO_ASSET_TYPE} asset in brdb's catalog"
        ));
    }

    // `insert_full` returns the index of an existing entry if the pair is
    // already present, so repeated calls for the same asset (30 tonal bands
    // all share the sine) register it once and reuse the index.
    let (asset_index, _) = world.global_data.external_asset_references.insert_full((
        AUDIO_ASSET_TYPE.to_string(),
        asset_name.to_string(),
    ));
    Ok(Box::new(BrdbValue::Asset(Some(asset_index))) as Box<dyn AsBrdbValue>)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::track::{SynthWave, band_plan};
    use crate::audio::voices::{VoiceStats, VoiceStreams};
    use brdb::{Brz, IntoReader};

    /// A bank-mode track with the plan `o` asks for, so the built world's
    /// speakers and their descriptors describe the render `o` names.
    fn bank_track(o: &AudioOptions, frames: usize) -> VoiceTrack {
        let plan = band_plan(o).expect("valid plan");
        let n = plan.len();
        VoiceTrack {
            plan,
            volumes: vec![vec![0.5; frames]; n],
            fps: o.fps,
            frame_count: frames,
        }
    }

    fn voice_streams(voices: usize, frames: usize, fps: f32) -> VoiceStreams {
        VoiceStreams {
            pitches: vec![vec![1.0; frames]; voices],
            volumes: vec![vec![0.5; frames]; voices],
            fps,
            frame_count: frames,
            stats: VoiceStats::default(),
        }
    }

    /// Write the world to a `.brz` and read it back, returning every
    /// audio-descriptor asset name the saved file carries. A genuine
    /// round-trip, not an in-memory inspection: `.brz` bytes are not
    /// reproducible, so a hash or size match would be meaningless.
    fn written_audio_asset_names(w: &World) -> Vec<String> {
        let bytes = w.to_brz_vec().expect("the world must encode to brz");
        let global = Brz::read_slice(&bytes)
            .expect("the written brz must parse")
            .into_reader()
            .read_global_data()
            .expect("the written brz must carry global data");
        global
            .external_asset_references
            .iter()
            .filter(|(ty, _)| ty.as_str() == AUDIO_ASSET_TYPE)
            .map(|(_, name)| name.clone())
            .collect()
    }

    fn bank_opts(synth: SynthWave, noise_bands: usize) -> AudioOptions {
        AudioOptions {
            bands: Some(4 + noise_bands),
            noise_bands,
            tonal_synth: synth,
            ..Default::default()
        }
    }

    /// `--synth square`: every tonal emitter carries the square descriptor,
    /// noise bands keep their own descriptors, no sine leaks in.
    #[test]
    fn square_tonal_with_noise_bands_writes_square_plus_noise_and_no_sine() {
        let o = bank_opts(SynthWave::Square, 2);
        let world = build_speaker_world(&bank_track(&o, 8), &o).expect("build");
        let names = written_audio_asset_names(&world);

        assert!(names.contains(&"BA_Synth_Basic_Square".to_string()), "{names:?}");
        assert!(names.contains(&"BA_Synth_Noise_White".to_string()), "{names:?}");
        assert!(names.contains(&"BA_Synth_Noise_Pink".to_string()), "{names:?}");
        assert!(
            !names.contains(&"BA_Synth_Basic_Sine".to_string()),
            "no sine may be registered once the tonal wave is square: {names:?}"
        );
        assert_eq!(names.len(), 3, "exactly one tonal + two noise assets: {names:?}");
    }

    /// The default (Sine) render is the same tonal-descriptor layout it was
    /// before the flag: sine tonal, noise unchanged.
    #[test]
    fn default_sine_bank_is_the_pre_flag_layout() {
        let o = bank_opts(SynthWave::Sine, 2);
        assert_eq!(o.tonal_synth, SynthWave::default());
        let world = build_speaker_world(&bank_track(&o, 8), &o).expect("build");
        let names = written_audio_asset_names(&world);
        assert!(names.contains(&"BA_Synth_Basic_Sine".to_string()), "{names:?}");
        assert!(names.contains(&"BA_Synth_Noise_White".to_string()), "{names:?}");
        assert!(names.contains(&"BA_Synth_Noise_Pink".to_string()), "{names:?}");
        assert_eq!(names.len(), 3, "{names:?}");
    }

    /// The noise assets are the same whatever the waveform is -- the choice is
    /// for tonal bands only.
    #[test]
    fn the_waveform_choice_never_touches_the_noise_bands() {
        for w in SynthWave::ALL {
            let o = bank_opts(w, 2);
            let world = build_speaker_world(&bank_track(&o, 8), &o).expect("build");
            let names = written_audio_asset_names(&world);
            assert!(
                names.contains(&"BA_Synth_Noise_White".to_string())
                    && names.contains(&"BA_Synth_Noise_Pink".to_string()),
                "{w:?}: both noise assets must always be present: {names:?}"
            );
            assert!(
                names.contains(&w.asset().as_ref().to_string()),
                "{w:?}: its own tonal asset must be present: {names:?}"
            );
            assert_eq!(names.len(), 3, "{w:?}: one tonal + two noise: {names:?}");
        }
    }

    /// Voice mode is all-tonal, so the whole cluster carries the selected
    /// waveform and nothing else. Confirms `build_voice_world` honours
    /// `tonal_synth` too (its own hardcoded sine is what this replaced).
    #[test]
    fn voice_mode_honours_the_selected_waveform() {
        let o = AudioOptions { max_voices: 5, tonal_synth: SynthWave::Triangle, ..Default::default() };
        let world = build_voice_world(&voice_streams(5, 8, o.fps), &o).expect("build");
        let names = written_audio_asset_names(&world);
        assert_eq!(names, vec!["BA_Synth_Basic_Triangle".to_string()], "{names:?}");
    }
}
