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
use super::track::{AudioOptions, SynthWave, VoiceTrack};
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

/// Z (inner-grid units) of the in-chip speaker block's floor. Every gate this
/// crate places in a chip -- the clock, its pins, the pause-mute detector and
/// the per-speaker MIDI playhead -- sits at lattice stage 0, so the whole gate
/// layer tops out at `STAGE_BASE_Z + 2 * GATE_HALF.z`. The speaker cluster
/// stacks ABOVE that plane, so it clears the gates however far they spread
/// across the grid: unlike the audio bank (whose service rows grow only along
/// y), the MIDI playhead grows the lattice along BOTH x (rows) and y (columns),
/// so a fixed sideways offset could not stay clear of a large score. One full
/// [`STAGE_PITCH`] of lift leaves air between the gate tops and the speakers.
const SPEAKER_BLOCK_Z: i32 = STAGE_BASE_Z + STAGE_PITCH;

/// Where in-chip speaker `index` of `total` sits on the chip's inner grid, for
/// [`AudioOptions::speakers_in_chip`]. Same [`cluster_dims`] packing as
/// [`speaker_position`], stacked on the [`SPEAKER_BLOCK_Z`] plane ABOVE the gate
/// layer so the cluster never overlaps the gates -- whatever their x/y spread.
/// Coordinates stay non-negative: negative inner-grid coordinates delete bricks
/// in-game. An emitter on the inner grid plays from the chip's world position,
/// so this shape buys no near-field geometry -- it only needs to be
/// non-overlapping.
pub fn speaker_inner_position(index: usize, total: usize) -> Position {
    let (nx, ny, _) = cluster_dims(total);
    let half = speaker_half();
    let ix = (index % nx) as i32;
    let iy = ((index / nx) % ny) as i32;
    let iz = (index / (nx * ny)) as i32;
    Position {
        x: half.x + ix * half.x * 2,
        y: half.y + iy * half.y * 2,
        z: SPEAKER_BLOCK_Z + half.z + iz * half.z * 2,
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
    /// The timer's raw `Time` (continuous seconds), for a consumer that reads
    /// the runtime directly rather than the wrapped frame index -- the MIDI
    /// event playhead compares it against note start/end times.
    time: WirePort,
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
        time: clock.time.clone(),
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
    row: i32,
) -> WirePort {
    let gate_id = gate(
        chip,
        "B_1x1_Gate_Expr_MathMultiply",
        MULTIPLY,
        service(slot as i32, row),
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
            // Below the audio select cascade (rows -4..-9), per the service doc.
            volume_multiply(&mut world, &mut sc.chip, b, speaker, &sc.master_volume, -10)
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
    if streams.voice_count() == 0 {
        return Err("a voice-mode track needs at least one voice".to_string());
    }
    // Voice mode is every-speaker tonal (peak-tracking, no noise bands), so the
    // whole cluster plays one selected waveform -- a uniform synth. The MIDI
    // path passes a different synth per speaker through this same builder.
    let synths = vec![opts.tonal_synth; streams.voice_count()];
    let pitches: Vec<&[f64]> = streams.pitches.iter().map(Vec::as_slice).collect();
    let volumes: Vec<&[f64]> = streams.volumes.iter().map(Vec::as_slice).collect();
    build_pitch_volume_world(
        &pitches,
        &volumes,
        &synths,
        streams.fps,
        streams.frame_count,
        opts,
        "Audio peak-tracking playback generated from an audio file",
    )
}

/// Build a world that plays a MIDI file with an EVENT-BASED playback circuit.
///
/// Each speaker stores its notes as ONE quat array -- each element
/// `(start, end, pitch, vol)` -- and a persistent `Var` register indexes into
/// it. Every tick, `ArrayGet(events, idx)` + `SplitQuaternion` reads the current
/// event; the speaker plays `pitch` while `start <= t <= end`; when `t > end`
/// (and the get is in bounds) `Var_Increment` bumps the index; and when the
/// playback time jumps backward (a restart or a loop wrap) `Var_Set` resets it
/// to 0. This is the stateful-`Var` playhead verified in game via the probe:
/// O(1) gates per speaker (no per-note compare chain) and only the events stored.
///
/// Reuses the audio [`scaffold`] for the chip, clock, spatialization pins,
/// pause-mute master volume (a frozen clock is silent) and control buttons; the
/// per-speaker playhead is new and sits on service rows below the scaffold's.
/// Looping feeds the playhead `Time mod duration`; playing once feeds raw `Time`
/// and the index caps at the last event.
pub fn build_midi_event_world(
    score: &crate::midi::MidiScore,
    opts: &crate::midi::MidiOptions,
) -> Result<World, String> {
    const VAR: &str = "BrickComponentType_WireGraphPseudo_Var";
    const VAR_INCREMENT: &str = "BrickComponentType_WireGraph_Exec_Var_Increment";
    const VAR_SET: &str = "BrickComponentType_WireGraph_Exec_Var_Set";
    const SPLIT_QUAT: &str = "BrickComponentType_WireGraph_Expr_SplitQuaternion";
    const COMPARE_GREATER: &str = "BrickComponentType_WireGraph_Expr_CompareGreater";
    const COMPARE_LE: &str = "BrickComponentType_WireGraph_Expr_CompareLessOrEqual";
    const COMPARE_LESS: &str = "BrickComponentType_WireGraph_Expr_CompareLess";
    use crate::anim::clock::MODULO;

    if score.voices.is_empty() && score.percussion_lanes.is_empty() {
        return Err(
            "this MIDI produced no speakers -- every note was outside the playable range"
                .to_string(),
        );
    }
    for (v, voice) in score.voices.iter().enumerate() {
        if voice.notes.is_empty() {
            return Err(format!("speaker {v} has no notes"));
        }
        for (i, note) in voice.notes.iter().enumerate() {
            if !note.pitch.is_finite() || note.pitch < PITCH_MIN as f64 || note.pitch > PITCH_MAX as f64 {
                return Err(format!(
                    "speaker {v} note {i} has PitchMultiplier {}, outside the emitter's legal \
                     {PITCH_MIN}..{PITCH_MAX} range",
                    note.pitch
                ));
            }
        }
    }
    if !score.duration_s.is_finite() || score.duration_s <= 0.0 {
        return Err(format!("MIDI duration must be positive, got {}", score.duration_s));
    }

    let audio_opts = AudioOptions {
        inner_radius: opts.inner_radius,
        max_distance: opts.max_distance,
        ..AudioOptions::default()
    };
    check_attenuation(&audio_opts)?;

    let n_speakers = score.voices.len();
    let mut world = World::new();
    world.meta.bundle.description = "MIDI event playback generated from a MIDI file".to_string();

    // --- 1. The speaker cluster (main grid, or the chip's inner grid) --------
    let in_chip = opts.speakers_in_chip;
    let mut speaker_ids = Vec::with_capacity(n_speakers);
    let mut in_chip_speakers: Vec<Brick> = Vec::new();
    for (v, voice) in score.voices.iter().enumerate() {
        let position = if in_chip {
            speaker_inner_position(v, n_speakers)
        } else {
            speaker_position(v, n_speakers)
        };
        speaker_ids.push(add_emitter(
            &mut world,
            voice.synth.asset().as_ref(),
            voice.notes[0].pitch as f32,
            position,
            &audio_opts,
            in_chip.then_some(&mut in_chip_speakers),
        )?);
    }

    // --- 2. Chip, clock, pins, pause-mute (shared scaffold) ------------------
    // Feed the clock a real frame count at 60 fps times the playback rate, so
    // its `Progress` (0..1) and `Length` (seconds) status output pins track this
    // piece at the chosen speed -- the frame-index chain is otherwise unused
    // here (the playhead reads `Time` directly below). A looping render frees
    // the timer (Time counts up, the frame index wraps); a play-once render
    // stops it at the end.
    let rate = (opts.playback_rate as f64).max(0.01);
    let clock_fps = 60.0f32 * rate as f32;
    let frame_count = (score.duration_s * 60.0).round().max(1.0) as usize;
    let mut sc = scaffold(&mut world, &speaker_ids, clock_fps, frame_count, opts.loop_playback);
    for brick in in_chip_speakers {
        sc.chip.add_brick(brick, speaker_half());
    }

    // Layout below the scaffold (clock 0, pins -2, pause-mute -3) and the
    // per-voice master-volume multiplies (row -4): the SHARED playhead gates
    // (time scaling, the loop wrap and the backward-jump detector) fill one row,
    // then every voice gets its OWN row below. So each voice reads as the same
    // left-to-right strip, stacked one per row -- a clean repeating pattern
    // rather than every voice's gates splooted end-to-end down a serpentine.
    const SHARED_ROW: i32 = -5;
    let mut shared_col = 0i32;
    let mut shared_pos = || {
        let p = service(shared_col, SHARED_ROW);
        shared_col += 1;
        p
    };

    // Playback time in seconds: scale the clock's `Time` by the baked rate
    // (identity at 1.0), then -- for a looping render -- wrap it at the
    // duration. The clock's fps carries the same rate, so its Progress/Length
    // stay in step with these note comparisons.
    let scaled_time = if (rate - 1.0).abs() < 1e-9 {
        sc.time.clone()
    } else {
        let m = gate(&mut sc.chip, "B_1x1_Gate_Expr_MathMultiply", MULTIPLY, shared_pos(), vec![(
            "InputB",
            Box::new(WireVariant::Number(rate)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(sc.time.clone(), WirePort::new(m, MULTIPLY, "InputA"));
        WirePort::new(m, MULTIPLY, "Output")
    };
    let play_time = if opts.loop_playback {
        let m = gate(&mut sc.chip, "B_1x1_Gate_Expr_MathModuloFloored", MODULO, shared_pos(), vec![(
            "InputB",
            Box::new(WireVariant::Number(score.duration_s)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(scaled_time.clone(), WirePort::new(m, MODULO, "InputA"));
        WirePort::new(m, MODULO, "Output")
    } else {
        scaled_time
    };

    // Shared reset signal: last tick's playback time, and whether it jumped
    // backward (a restart or a loop wrap). Fanned out to every speaker's reset.
    let prev = gate(&mut sc.chip, "B_1x1_Gate_Pseudo_BufferTicks", BUFFER_TICKS, shared_pos(), vec![(
        "TicksToWait",
        Box::new(1i32) as Box<dyn AsBrdbValue>,
    )]);
    world.add_wire_connection(play_time.clone(), WirePort::new(prev, BUFFER_TICKS, "Input"));
    let decreased = gate(&mut sc.chip, "B_1x1_Gate_Expr_CompareLess", COMPARE_LESS, shared_pos(), vec![]);
    world.add_wire_connection(play_time.clone(), WirePort::new(decreased, COMPARE_LESS, "InputA"));
    world.add_wire_connection(WirePort::new(prev, BUFFER_TICKS, "Output"), WirePort::new(decreased, COMPARE_LESS, "InputB"));

    // --- 3. Per-speaker playhead ---------------------------------------------
    for (v, voice) in score.voices.iter().enumerate() {
        // This voice's whole playhead lives on its OWN row, in a fixed
        // left-to-right column order -- the repeating strip. Voice 0 sits just
        // below the shared row; each next voice is one row further along, so the
        // voices stack instead of running end-to-end.
        let row = SHARED_ROW - 1 - v as i32;
        let mut col = 0i32;
        let mut vpos = || {
            let p = service(col, row);
            col += 1;
            p
        };
        // The events, packed one quat each: (start, end, pitch, vol) = (X,Y,Z,W).
        let events: Vec<(f64, f64, f64, f64)> = voice
            .notes
            .iter()
            .map(|n| (n.start_s, n.end_s, n.pitch, n.volume))
            .collect();
        let events_arr = gate(&mut sc.chip, "B_1x1_Gate_Variable_Array", ARRAY_VAR, vpos(), vec![(
            "Value",
            Box::new(WireArrayVariant::QuatArray(events)) as Box<dyn AsBrdbValue>,
        )]);
        // The playhead index register (starts 0).
        let idx_var = gate(&mut sc.chip, "B_1x1_Gate_Variable", VAR, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
        )]);
        // This speaker's per-tick exec pulse (own detector, no cross-speaker
        // chaining -- the reset/advance branches would break a shared chain).
        let detector = gate(&mut sc.chip, "B_1x1_Gate_Expr_ChangeDetectorExec", CHANGE_DETECTOR, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(detector, CHANGE_DETECTOR, "Input"));

        // Read the current event and break it into its four floats.
        let get = gate(&mut sc.chip, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(events_arr, ARRAY_VAR, "ArrayVarRef"), WirePort::new(get, ARRAY_GET, "ArrayVarRef"));
        world.add_wire_connection(WirePort::new(idx_var, VAR, "Value"), WirePort::new(get, ARRAY_GET, "Index"));
        world.add_wire_connection(WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"), WirePort::new(get, ARRAY_GET, "Exec"));
        let split = gate(&mut sc.chip, "B_1x1_Gate_Expr_SplitQuaternion", SPLIT_QUAT, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "Value"), WirePort::new(split, SPLIT_QUAT, "Input"));
        let start = WirePort::new(split, SPLIT_QUAT, "X");
        let end = WirePort::new(split, SPLIT_QUAT, "Y");
        let pitch = WirePort::new(split, SPLIT_QUAT, "Z");
        let vol = WirePort::new(split, SPLIT_QUAT, "W");

        // Reset first: if playback time jumped backward, set the index to 0.
        let br_reset = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(decreased, COMPARE_LESS, "bOutput"), WirePort::new(br_reset, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "ExecOut"), WirePort::new(br_reset, BRANCH, "Exec"));
        let var_set = gate(&mut sc.chip, "B_1x1_Gate_Exec_Var_Set", VAR_SET, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(var_set, VAR_SET, "VarRef"));
        world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutA"), WirePort::new(var_set, VAR_SET, "Exec"));

        // Advance: on the no-reset exec, if Time > end and in bounds, idx += 1.
        let gt_end = gate(&mut sc.chip, "B_1x1_Gate_Expr_CompareGreater", COMPARE_GREATER, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(gt_end, COMPARE_GREATER, "InputA"));
        world.add_wire_connection(end.clone(), WirePort::new(gt_end, COMPARE_GREATER, "InputB"));
        let br_time = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(gt_end, COMPARE_GREATER, "bOutput"), WirePort::new(br_time, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutB"), WirePort::new(br_time, BRANCH, "Exec"));
        let br_oob = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "bOutOfBounds"), WirePort::new(br_oob, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(br_time, BRANCH, "ExecOutA"), WirePort::new(br_oob, BRANCH, "Exec"));
        let inc = gate(&mut sc.chip, "B_1x1_Gate_Exec_Var_Increment", VAR_INCREMENT, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(1)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(inc, VAR_INCREMENT, "VarRef"));
        world.add_wire_connection(WirePort::new(br_oob, BRANCH, "ExecOutB"), WirePort::new(inc, VAR_INCREMENT, "Exec"));

        // volume = (t >= start) ? ((t <= end) ? vol : 0) : 0
        let ge_start = gate(&mut sc.chip, "B_1x1_Gate_Expr_CompareGreaterOrEqual", COMPARE_GE, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(ge_start, COMPARE_GE, "InputA"));
        world.add_wire_connection(start, WirePort::new(ge_start, COMPARE_GE, "InputB"));
        let le_end = gate(&mut sc.chip, "B_1x1_Gate_Expr_CompareLessOrEqual", COMPARE_LE, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(le_end, COMPARE_LE, "InputA"));
        world.add_wire_connection(end, WirePort::new(le_end, COMPARE_LE, "InputB"));

        let vol_inner = gate(&mut sc.chip, "B_1x1_Gate_Expr_Select", SELECT, vpos(), vec![(
            "InputA",
            Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(le_end, COMPARE_LE, "bOutput"), WirePort::new(vol_inner, SELECT, "bSelectB"));
        world.add_wire_connection(vol, WirePort::new(vol_inner, SELECT, "InputB"));
        let vol_gated = gate(&mut sc.chip, "B_1x1_Gate_Expr_Select", SELECT, vpos(), vec![(
            "InputA",
            Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(ge_start, COMPARE_GE, "bOutput"), WirePort::new(vol_gated, SELECT, "bSelectB"));
        world.add_wire_connection(WirePort::new(vol_inner, SELECT, "Output"), WirePort::new(vol_gated, SELECT, "InputB"));

        // Pitch straight into the emitter; volume through the master-volume
        // multiply (pause-mute) so a frozen clock silences it.
        world.add_wire_connection(pitch, WirePort::new(speaker_ids[v], AUDIO_EMITTER, "PitchMultiplier"));
        let vol_target = volume_multiply(&mut world, &mut sc.chip, v, speaker_ids[v], &sc.master_volume, -4);
        world.add_wire_connection(WirePort::new(vol_gated, SELECT, "Output"), vol_target);
    }

    // --- 3b. Percussion lanes ------------------------------------------------
    // Each lane is one oneshot emitter driven by its own playhead: a `Var`
    // index steps through the strike times, and on each advance its incrementing
    // value fires the emitter's `Play` exec (any increment triggers the sample).
    // Same reset/advance shape as the melodic playhead, but the compare is
    // "time reached the strike" and the trigger is the index itself, not a
    // pitch/volume gate. Rows continue below the last voice's.
    let perc_base_row = SHARED_ROW - 1 - n_speakers as i32;
    // Emitters sit in a compact near-square block just past the speaker
    // cluster's far +y edge, sharing its x range -- beside the array, not a row.
    let (_, sp_ny, _) = cluster_dims(n_speakers.max(1));
    let ph = speaker_half();
    let perc_cols = ((score.percussion_lanes.len() as f64).sqrt().ceil() as i32).max(1);
    let perc_y0 = sp_ny as i32 * ph.y * 2 + ph.y * 2;
    for (l, lane) in score.percussion_lanes.iter().enumerate() {
        let (ix, iy) = (l as i32 % perc_cols, l as i32 / perc_cols);
        let emitter = crate::audio::percussion::add_oneshot_emitter(
            &mut world,
            &lane.sound,
            Position { x: ph.x + ix * ph.x * 2, y: perc_y0 + ph.y + iy * ph.y * 2, z: 6 },
        );

        let row = perc_base_row - l as i32;
        let mut col = 0i32;
        let mut vpos = || {
            let p = service(col, row);
            col += 1;
            p
        };

        // Strike times, and the playhead index into them (starts 0).
        let hits_arr = gate(&mut sc.chip, "B_1x1_Gate_Variable_Array", ARRAY_VAR, vpos(), vec![(
            "Value",
            Box::new(WireArrayVariant::DoubleArray(lane.hits.clone())) as Box<dyn AsBrdbValue>,
        )]);
        let idx_var = gate(&mut sc.chip, "B_1x1_Gate_Variable", VAR, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
        )]);
        let detector = gate(&mut sc.chip, "B_1x1_Gate_Expr_ChangeDetectorExec", CHANGE_DETECTOR, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(detector, CHANGE_DETECTOR, "Input"));

        // Read the current strike time (bOutOfBounds once the index passes the end).
        let get = gate(&mut sc.chip, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(hits_arr, ARRAY_VAR, "ArrayVarRef"), WirePort::new(get, ARRAY_GET, "ArrayVarRef"));
        world.add_wire_connection(WirePort::new(idx_var, VAR, "Value"), WirePort::new(get, ARRAY_GET, "Index"));
        world.add_wire_connection(WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"), WirePort::new(get, ARRAY_GET, "Exec"));

        // Reset first: if playback time jumped backward (a loop wrap), index = 0.
        // Setting it to 0 does not fire Play (0 is not non-zero).
        let br_reset = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(decreased, COMPARE_LESS, "bOutput"), WirePort::new(br_reset, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "ExecOut"), WirePort::new(br_reset, BRANCH, "Exec"));
        let var_set = gate(&mut sc.chip, "B_1x1_Gate_Exec_Var_Set", VAR_SET, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(var_set, VAR_SET, "VarRef"));
        world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutA"), WirePort::new(var_set, VAR_SET, "Exec"));

        // Advance: on the no-reset exec, if playback time has reached the strike
        // and the index is in bounds, increment it -- which fires the oneshot.
        let reached = gate(&mut sc.chip, "B_1x1_Gate_Expr_CompareGreaterOrEqual", COMPARE_GE, vpos(), vec![]);
        world.add_wire_connection(play_time.clone(), WirePort::new(reached, COMPARE_GE, "InputA"));
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "Value"), WirePort::new(reached, COMPARE_GE, "InputB"));
        let br_reached = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(reached, COMPARE_GE, "bOutput"), WirePort::new(br_reached, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutB"), WirePort::new(br_reached, BRANCH, "Exec"));
        let br_oob = gate(&mut sc.chip, "B_1x1_Gate_Exec_Branch", BRANCH, vpos(), vec![]);
        world.add_wire_connection(WirePort::new(get, ARRAY_GET, "bOutOfBounds"), WirePort::new(br_oob, BRANCH, "bCond"));
        world.add_wire_connection(WirePort::new(br_reached, BRANCH, "ExecOutA"), WirePort::new(br_oob, BRANCH, "Exec"));
        let inc = gate(&mut sc.chip, "B_1x1_Gate_Exec_Var_Increment", VAR_INCREMENT, vpos(), vec![(
            "Value",
            Box::new(WireVariant::Int(1)) as Box<dyn AsBrdbValue>,
        )]);
        world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(inc, VAR_INCREMENT, "VarRef"));
        world.add_wire_connection(WirePort::new(br_oob, BRANCH, "ExecOutB"), WirePort::new(inc, VAR_INCREMENT, "Exec"));

        // The incrementing index fires the oneshot: one strike per advance.
        world.add_wire_connection(
            WirePort::new(idx_var, VAR, "Value"),
            WirePort::new(emitter, crate::audio::percussion::ONESHOT_EMITTER, crate::audio::percussion::PLAY_PORT),
        );
    }

    // --- 4. Control buttons --------------------------------------------------
    if opts.control_buttons {
        let (pause, restart, resume) = sc.control_pins;
        let anchor = crate::anim::controls::control_anchor(&world);
        crate::anim::controls::add_control_buttons(&mut world, pause, restart, resume, anchor);
    }

    finish(&mut world, sc.chip)?;
    world.register_used_components();
    Ok(world)
}

/// A minimal one-speaker EVENT-BASED playback circuit, for in-game
/// verification of the STATEFUL playhead before committing midi2brick to it.
///
/// Events are stored as ONE quat array -- each element `(start, end, pitch,
/// vol)` -- and a persistent `Var` register indexes into it. Each tick:
/// `ArrayGet(events, idx)` reads the current event and `SplitQuaternion` breaks
/// it into its four floats; the speaker plays `pitch` while `start <= Time <=
/// end`; and when `Time > end` (and the get is not out of bounds) a
/// `Var_Increment` bumps the index by one -- an imperative `idx++` on the tick
/// exec, NOT the combinational `BufferTicks` feedback that failed to advance in
/// an earlier probe. Out of bounds is checked so the index stops at the end and
/// the get holds its last value (the game does not update it while OOB).
///
/// This is O(1) gates per speaker (no per-note compare/select chain) AND stores
/// only the events (one quat each) -- the "arrays not gates" encoding. The clock
/// is the shared render clock with Pause/Restart/Resume buttons: press Resume or
/// Restart to start. The output is an ascending 8-note C-major scale, one note
/// per second, then silence. If the scale advances, the stateful `Var` playhead
/// works and the real builder can be rebuilt on it. Writes a `.brz` via
/// `examples/midi_playhead_probe.rs`.
pub fn build_playhead_probe_world() -> Result<World, String> {
    const VAR: &str = "BrickComponentType_WireGraphPseudo_Var";
    const VAR_INCREMENT: &str = "BrickComponentType_WireGraph_Exec_Var_Increment";
    const VAR_SET: &str = "BrickComponentType_WireGraph_Exec_Var_Set";
    const SPLIT_QUAT: &str = "BrickComponentType_WireGraph_Expr_SplitQuaternion";
    const COMPARE_GREATER: &str = "BrickComponentType_WireGraph_Expr_CompareGreater";
    const COMPARE_LE: &str = "BrickComponentType_WireGraph_Expr_CompareLessOrEqual";
    const COMPARE_LESS: &str = "BrickComponentType_WireGraph_Expr_CompareLess";

    // The score: eight one-second events (0.9 s sounding, 0.1 s gap), C-major.
    // Packed as quats (start, end, pitch, vol) = (X, Y, Z, W).
    let notes: [u8; 8] = [60, 62, 64, 65, 67, 69, 71, 72];
    let events: Vec<(f64, f64, f64, f64)> = notes
        .iter()
        .enumerate()
        .map(|(i, &n)| (i as f64, i as f64 + 0.9, 2.0f64.powf((n as f64 - 69.0) / 12.0), 1.0))
        .collect();
    let first_pitch = events[0].2 as f32;

    let mut world = World::new();
    world.meta.bundle.description =
        "MIDI event-playback playhead probe: a Var-indexed quat-event circuit playing a scale"
            .to_string();

    let mut chip = new_chip(
        &mut world,
        Position { x: 0, y: 0, z: 2 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 5, y: 5, z: 5 },
    );

    // Gate positions: a tight grid, 8 per row, gates touching, on the plane.
    let pos = |slot: i32| Position {
        x: GATE_HALF.x + (slot % 8) * (2 * GATE_HALF.x),
        y: GATE_HALF.y + (slot / 8) * (2 * GATE_HALF.y),
        z: GATE_HALF.z,
    };
    let mut n = 0i32;

    // The shared render clock (Time + control pins), placed clear of the grid.
    let clock = build_clock(
        &mut world,
        &mut chip,
        1.0,
        notes.len().max(1),
        true,
        Position { x: GATE_HALF.x, y: 200, z: GATE_HALF.z },
    );
    let time = || clock.time.clone();

    // The event array (one quat per event) and the index register (starts 0).
    let events_arr = gate(&mut chip, "B_1x1_Gate_Variable_Array", ARRAY_VAR, pos(n), vec![(
        "Value",
        Box::new(WireArrayVariant::QuatArray(events)) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;
    let idx_var = gate(&mut chip, "B_1x1_Gate_Variable", VAR, pos(n), vec![(
        "Value",
        Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;

    // Per-tick exec pulse.
    let detector = gate(&mut chip, "B_1x1_Gate_Expr_ChangeDetectorExec", CHANGE_DETECTOR, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(detector, CHANGE_DETECTOR, "Input"));

    // Last tick's Time, for the restart/loop reset and the pause-mute.
    let prev = gate(&mut chip, "B_1x1_Gate_Pseudo_BufferTicks", BUFFER_TICKS, pos(n), vec![(
        "TicksToWait",
        Box::new(1i32) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(prev, BUFFER_TICKS, "Input"));
    let prev_time = || WirePort::new(prev, BUFFER_TICKS, "Output");
    // Time jumped backward: a restart or a loop wrap.
    let decreased = gate(&mut chip, "B_1x1_Gate_Expr_CompareLess", COMPARE_LESS, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(decreased, COMPARE_LESS, "InputA"));
    world.add_wire_connection(prev_time(), WirePort::new(decreased, COMPARE_LESS, "InputB"));
    // Time changed at all: the clock is advancing (not paused).
    let moving = gate(&mut chip, "B_1x1_Gate_Expr_CompareNotEqual", COMPARE_NE, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(moving, COMPARE_NE, "InputA"));
    world.add_wire_connection(prev_time(), WirePort::new(moving, COMPARE_NE, "InputB"));

    // Read the current event: ArrayGet(events, idx) on the pulse.
    let get = gate(&mut chip, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(
        WirePort::new(events_arr, ARRAY_VAR, "ArrayVarRef"),
        WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
    );
    world.add_wire_connection(WirePort::new(idx_var, VAR, "Value"), WirePort::new(get, ARRAY_GET, "Index"));
    world.add_wire_connection(
        WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
        WirePort::new(get, ARRAY_GET, "Exec"),
    );

    // Break the quat into (start, end, pitch, vol) = (X, Y, Z, W).
    let split = gate(&mut chip, "B_1x1_Gate_Expr_SplitQuaternion", SPLIT_QUAT, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(WirePort::new(get, ARRAY_GET, "Value"), WirePort::new(split, SPLIT_QUAT, "Input"));
    let start = || WirePort::new(split, SPLIT_QUAT, "X");
    let end = || WirePort::new(split, SPLIT_QUAT, "Y");
    let pitch = WirePort::new(split, SPLIT_QUAT, "Z");
    let vol = WirePort::new(split, SPLIT_QUAT, "W");

    // --- Reset: when Time jumped backward, set the index to 0 ----------------
    // Runs first on the tick's exec; its no-reset branch continues to advance.
    let br_reset = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(WirePort::new(decreased, COMPARE_LESS, "bOutput"), WirePort::new(br_reset, BRANCH, "bCond"));
    world.add_wire_connection(WirePort::new(get, ARRAY_GET, "ExecOut"), WirePort::new(br_reset, BRANCH, "Exec"));
    let var_set = gate(&mut chip, "B_1x1_Gate_Exec_Var_Set", VAR_SET, pos(n), vec![(
        "Value",
        Box::new(WireVariant::Int(0)) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;
    world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(var_set, VAR_SET, "VarRef"));
    world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutA"), WirePort::new(var_set, VAR_SET, "Exec"));

    // --- Advance: when Time > end AND the get is in bounds, idx += 1 ----------
    let gt_end = gate(&mut chip, "B_1x1_Gate_Expr_CompareGreater", COMPARE_GREATER, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(gt_end, COMPARE_GREATER, "InputA"));
    world.add_wire_connection(end(), WirePort::new(gt_end, COMPARE_GREATER, "InputB"));
    // Branch on Time>end: the true exec continues to the OOB check.
    let br_time = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(WirePort::new(gt_end, COMPARE_GREATER, "bOutput"), WirePort::new(br_time, BRANCH, "bCond"));
    world.add_wire_connection(WirePort::new(br_reset, BRANCH, "ExecOutB"), WirePort::new(br_time, BRANCH, "Exec"));
    // Branch on bOutOfBounds: the FALSE exec (in bounds) does the increment, so
    // the index never runs past the last event.
    let br_oob = gate(&mut chip, "B_1x1_Gate_Exec_Branch", BRANCH, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(WirePort::new(get, ARRAY_GET, "bOutOfBounds"), WirePort::new(br_oob, BRANCH, "bCond"));
    world.add_wire_connection(WirePort::new(br_time, BRANCH, "ExecOutA"), WirePort::new(br_oob, BRANCH, "Exec"));
    let inc = gate(&mut chip, "B_1x1_Gate_Exec_Var_Increment", VAR_INCREMENT, pos(n), vec![(
        "Value",
        Box::new(WireVariant::Int(1)) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;
    world.add_wire_connection(WirePort::new(idx_var, VAR, "VarRef"), WirePort::new(inc, VAR_INCREMENT, "VarRef"));
    world.add_wire_connection(WirePort::new(br_oob, BRANCH, "ExecOutB"), WirePort::new(inc, VAR_INCREMENT, "Exec"));

    // --- Play: volume = (Time >= start) ? ((Time <= end) ? vol : 0) : 0 ------
    let ge_start = gate(&mut chip, "B_1x1_Gate_Expr_CompareGreaterOrEqual", COMPARE_GE, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(ge_start, COMPARE_GE, "InputA"));
    world.add_wire_connection(start(), WirePort::new(ge_start, COMPARE_GE, "InputB"));
    let le_end = gate(&mut chip, "B_1x1_Gate_Expr_CompareLessOrEqual", COMPARE_LE, pos(n), vec![]);
    n += 1;
    world.add_wire_connection(time(), WirePort::new(le_end, COMPARE_LE, "InputA"));
    world.add_wire_connection(end(), WirePort::new(le_end, COMPARE_LE, "InputB"));

    let vol_inner = gate(&mut chip, "B_1x1_Gate_Expr_Select", SELECT, pos(n), vec![(
        "InputA",
        Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
    )]);
    n += 1;
    world.add_wire_connection(WirePort::new(le_end, COMPARE_LE, "bOutput"), WirePort::new(vol_inner, SELECT, "bSelectB"));
    world.add_wire_connection(vol, WirePort::new(vol_inner, SELECT, "InputB"));
    let vol_out = gate(&mut chip, "B_1x1_Gate_Expr_Select", SELECT, pos(n), vec![(
        "InputA",
        Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
    )]);
    world.add_wire_connection(WirePort::new(ge_start, COMPARE_GE, "bOutput"), WirePort::new(vol_out, SELECT, "bSelectB"));
    world.add_wire_connection(WirePort::new(vol_inner, SELECT, "Output"), WirePort::new(vol_out, SELECT, "InputB"));

    // --- The single speaker, driven by the circuit --------------------------
    let opts = AudioOptions::default();
    let emitter = add_emitter(
        &mut world,
        SynthWave::Sine.asset().as_ref(),
        first_pitch,
        Position { x: 200, y: 0, z: GATE_HALF.z },
        &opts,
        None,
    )?;
    // Pause-mute: silence unless the clock advanced this tick.
    n += 1;
    let pause_mute = gate(&mut chip, "B_1x1_Gate_Expr_Select", SELECT, pos(n), vec![(
        "InputA",
        Box::new(WireVariant::Number(0.0)) as Box<dyn AsBrdbValue>,
    )]);
    world.add_wire_connection(WirePort::new(moving, COMPARE_NE, "bOutput"), WirePort::new(pause_mute, SELECT, "bSelectB"));
    world.add_wire_connection(WirePort::new(vol_out, SELECT, "Output"), WirePort::new(pause_mute, SELECT, "InputB"));

    world.add_wire_connection(pitch, WirePort::new(emitter, AUDIO_EMITTER, "PitchMultiplier"));
    world.add_wire_connection(
        WirePort::new(pause_mute, SELECT, "Output"),
        WirePort::new(emitter, AUDIO_EMITTER, "VolumeMultiplier"),
    );

    // Physical Pause/Restart/Resume buttons.
    let anchor = crate::anim::controls::control_anchor(&world);
    crate::anim::controls::add_control_buttons(
        &mut world,
        clock.pause_pin,
        clock.restart_pin,
        clock.resume_pin,
        anchor,
    );

    finish(&mut world, chip)?;
    world.register_used_components();
    Ok(world)
}

/// The shared per-speaker pitch+volume builder behind both [`build_voice_world`]
/// and [`build_midi_world`]. Every speaker gets its own `synths[i]` (voice mode
/// passes the same one for all; MIDI passes one per instrument), its pitch wired
/// straight into `PitchMultiplier` and its volume through a master-volume
/// multiply, all banked by the shared frame cascade. `opts` supplies only the
/// spatialization/playback fields (inner/max radius, speakers-in-chip, loop,
/// control buttons, bank size); `fps` and `frame_count` come from the streams.
fn build_pitch_volume_world(
    pitches: &[&[f64]],
    volumes: &[&[f64]],
    synths: &[SynthWave],
    fps: f32,
    frame_count: usize,
    opts: &AudioOptions,
    description: &str,
) -> Result<World, String> {
    if frame_count == 0 {
        return Err("audio track has 0 frames -- nothing to render".to_string());
    }
    let n = pitches.len();
    if n == 0 {
        return Err("a track needs at least one speaker".to_string());
    }
    if volumes.len() != n || synths.len() != n {
        return Err(format!(
            "track has {n} pitch arrays but {} volume arrays and {} synths",
            volumes.len(),
            synths.len()
        ));
    }
    // Both arrays of every speaker are read at the same frame index, so a short
    // one would leave that speaker reading a stale value (or nothing) from the
    // point it ran out -- silently, and only for part of the track.
    for (v, (p, vol)) in pitches.iter().zip(volumes).enumerate() {
        if p.len() != frame_count || vol.len() != frame_count {
            return Err(format!(
                "speaker {v} has {} pitch and {} volume values, expected {frame_count} of each",
                p.len(),
                vol.len(),
            ));
        }
    }
    // A pitch outside the emitter's legal range is clamped in game, turning a
    // wrong number into a wrong note rather than silence. The analyzers already
    // clamp/drop; this guards any other caller.
    for (v, row) in pitches.iter().enumerate() {
        for (f, &p) in row.iter().enumerate() {
            if !p.is_finite() || p < PITCH_MIN as f64 || p > PITCH_MAX as f64 {
                return Err(format!(
                    "speaker {v} frame {f} has PitchMultiplier {p}, outside the emitter's \
                     legal {PITCH_MIN}..{PITCH_MAX} range -- the game would clamp it \
                     and play a wrong note"
                ));
            }
        }
    }
    check_attenuation(opts)?;

    let mut world = World::new();
    world.meta.bundle.description = description.to_string();

    // --- 1. The speaker cluster (main grid, or the chip's inner grid) -------
    let in_chip = opts.speakers_in_chip;
    let mut speaker_ids = Vec::with_capacity(n);
    // Deferred until the chip exists; empty (loop a no-op) unless
    // `--speakers-in-chip`. Same mechanism as the bank builder.
    let mut in_chip_speakers: Vec<Brick> = Vec::new();
    for v in 0..n {
        let position = if in_chip {
            speaker_inner_position(v, n)
        } else {
            speaker_position(v, n)
        };
        speaker_ids.push(add_emitter(
            &mut world,
            synths[v].asset().as_ref(),
            pitches[v][0] as f32,
            position,
            opts,
            in_chip.then_some(&mut in_chip_speakers),
        )?);
    }

    // --- 2. Chip, clock and the four input pins -----------------------------
    let mut sc = scaffold(&mut world, &speaker_ids, fps, frame_count, opts.loop_playback);

    // Place the deferred in-chip speakers now the chip exists (a no-op for the
    // default beside-the-chip layout). See `build_speaker_world`.
    for brick in in_chip_speakers {
        sc.chip.add_brick(brick, speaker_half());
    }

    // --- 3. One master-volume multiply per speaker --------------------------
    let volume_targets: Vec<WirePort> = speaker_ids
        .iter()
        .enumerate()
        .map(|(v, &speaker)| {
            volume_multiply(&mut world, &mut sc.chip, v, speaker, &sc.master_volume, -10)
        })
        .collect();

    // --- 4. Two streams per speaker: its pitch and its volume ---------------
    // Interleaved per speaker so the two arrays sit next to each other in the
    // chip. Pitch goes straight into the emitter with no multiply in between:
    // scaling it by Volume would transpose the render whenever volume changed.
    let mut frame_streams: Vec<FrameStream<'_>> = Vec::with_capacity(n * 2);
    for v in 0..n {
        frame_streams.push(FrameStream {
            values: pitches[v],
            target: WirePort::new(speaker_ids[v], AUDIO_EMITTER, "PitchMultiplier"),
        });
        frame_streams.push(FrameStream {
            values: volumes[v],
            target: volume_targets[v].clone(),
        });
    }
    build_stream_cascade(
        &mut world,
        &mut sc.chip,
        &sc.frame_index,
        frame_count,
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

    /// A score of `voices` speakers, each a couple of in-range notes -- enough
    /// to build a real world through [`build_midi_event_world`].
    fn midi_score(voices: usize) -> crate::midi::MidiScore {
        use crate::midi::{NoteSpan, SpeakerVoice};
        let notes = vec![
            NoteSpan { start_s: 0.0, end_s: 0.5, pitch: 1.0, volume: 1.0 },
            NoteSpan { start_s: 0.5, end_s: 1.0, pitch: 2.0, volume: 0.8 },
        ];
        crate::midi::MidiScore {
            voices: (0..voices)
                .map(|i| SpeakerVoice { notes: notes.clone(), synth: SynthWave::Sine, instrument_idx: i })
                .collect(),
            percussion_lanes: vec![],
            duration_s: 1.0,
        }
    }

    /// The in-chip speaker cluster sits entirely above the gate layer, whatever
    /// the speaker count. Every gate this crate places in a chip lives at
    /// lattice stage 0, so its top is fixed; keeping every speaker's floor at or
    /// above that top is the invariant that lets the playhead lattice grow as
    /// wide as a large score needs without ever colliding with the speakers.
    #[test]
    fn in_chip_speakers_clear_the_gate_layer() {
        let gate_top = STAGE_BASE_Z + 2 * GATE_HALF.z;
        let half = speaker_half();
        for total in [1usize, 8, 32, 200] {
            for i in 0..total {
                let bottom = speaker_inner_position(i, total).z - half.z;
                assert!(
                    bottom >= gate_top,
                    "speaker {i}/{total} floor {bottom} must sit at or above the gate top {gate_top}"
                );
            }
        }
    }

    /// A large MIDI placed IN the microchip builds without an overlap error.
    /// The playhead grows the inner gate lattice along BOTH x (rows) and y
    /// (columns), so a wide enough score used to push the gates into the old
    /// sideways-offset speaker block; stacking the speakers onto their own z
    /// plane above the gates keeps them clear. `finish` runs the inner-grid
    /// overlap check, so an overlap surfaces as an `Err` here.
    #[test]
    fn a_large_in_chip_midi_places_without_overlap() {
        let score = midi_score(24);
        let opts = crate::midi::MidiOptions { speakers_in_chip: true, ..Default::default() };
        build_midi_event_world(&score, &opts).expect("a large in-chip MIDI must not overlap");
        // ...and the same score beside the chip (the default) still builds too.
        let opts = crate::midi::MidiOptions { speakers_in_chip: false, ..Default::default() };
        build_midi_event_world(&score, &opts).expect("the beside-the-chip layout must build");
    }
}
