//! The speaker cluster and its wiring.
//!
//! Two components per band per bank: an `ArrayVar` of `f64` volumes and an
//! `ArrayVar_Get` that reads it. The get's `Value` wires DIRECTLY into that
//! speaker's `VolumeMultiplier` -- no `Component_BrickPropertyChanger`,
//! because unlike brick colour, volume is a real wire input.
//!
//! Pitch and the audio asset are build-time component data, costing no gates
//! and no wires at all.
//!
//! Four further inputs are surfaced as microchip pins (see
//! [`build_speaker_world`]'s attenuation section): `Inner Radius`,
//! `Max Distance` and `Directional` fan straight out to every speaker's
//! matching emitter port, and `Volume` scales every band through a per-band
//! multiply. Every one of them keeps a baked component value underneath, which
//! is what the speaker reads while the pin is unwired.
use super::bands::{BandKind, PITCH_MAX, PITCH_MIN};
use super::track::{AudioOptions, VoiceTrack};
use super::voices::VoiceStreams;
use crate::anim::bricks::{
    ARRAY_GET, ARRAY_VAR, BRANCH, CHANGE_DETECTOR, COMPARE_GE, SELECT, SUBTRACT,
};
use crate::anim::chip::{Chip, add_input_pin, finish, new_chip, pin_source};
use crate::anim::clock::{MULTIPLY, build_clock, gate};
use crate::anim::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use brdb::{
    AsBrdbValue, Brick, BrickType, IntVector, Position, Vector3f, WirePort, World,
    assets::{
        LiteralComponent,
        external::{ASSET_TYPES, BA_SYNTH_BASIC_SINE, BA_SYNTH_NOISE_PINK, BA_SYNTH_NOISE_WHITE},
    },
    schema::{BrdbValue, WireArrayVariant, WireVariant},
};

/// The emitter component's type name.
///
/// **No `BrickComponentType_` prefix.** That prefix belongs to the wire-graph
/// gate components only; this one is a plain `Component_*`, and the name is
/// resolved by string lookup in global data with no validation anywhere, so a
/// "corrected" name would encode fine and do nothing in game. Verified against
/// `COMPONENT_TYPE_STRUCT_PAIRS` in brdb 0.9.1 (see
/// `docs/superpowers/notes/2026-07-28-audio-feasibility.md`).
pub const AUDIO_EMITTER: &str = "Component_AudioEmitter";
pub const SPEAKER_BRICK: &str = "B_1x1F_Speaker";
/// The external asset type every synth descriptor lives under.
pub const AUDIO_ASSET_TYPE: &str = "BrickAudioDescriptor";
/// `B_1x1F_Speaker`'s TRUE half-extents, from the game's brick catalog.
///
/// NOTE: this is *not* what the build-time overlap check measures -- see
/// [`speaker_half`]. Only ever used as a FLOOR under the measured value.
const SPEAKER_HALF: IntVector = IntVector { x: 5, y: 5, z: 2 };

/// Lattice "height" fed to [`lattice_pos_staged`] for the service rows.
///
/// The video renderer passes the screen's height there because its service
/// gates sit behind a real pixel lattice. This chip has no pixel lattice at
/// all, so row 0 is the front row and 1 is the smallest height that puts it
/// there: `x = (height - 1 - row) * CELL + half.x`, i.e. `-row * CELL + 5`.
/// Every row index used below is <= 0, so every x is >= 5 -- which is the
/// whole point, see [`build_speaker_world`]'s `service`.
const LATTICE_HEIGHT: i32 = 1;

/// Default baked `InnerRadius` / `MaxDistance` for every speaker, in game units
/// (10 units = 1 brick). The game's own single-prop values; `--inner-radius` /
/// `--max-distance` raise them (e.g. 400 / 4000) for one flat equal-level field
/// across a large build.
///
/// 15 is below the cluster's ~44-unit diagonal (see [`cluster_dims`]), so the
/// bank attenuates within itself -- near-field and local, not a flat field.
/// `bSpatialization = false` stops panning but NOT distance attenuation, so
/// this applies regardless.
///
/// Written explicitly, not left to brdb's `STRUCT_DEFAULTS`: that table is
/// regenerated per build, so a retuned game default would silently move every
/// render. 0.0 is not a safe "unset" -- a zero `MaxDistance` is audible nowhere.
pub const DEFAULT_INNER_RADIUS: f32 = 15.0;
pub const DEFAULT_MAX_DISTANCE: f32 = 400.0;

/// Baked `bSpatialization`, i.e. what the `Directional` pin overrides.
///
/// MUST stay `false`. The whole bank design rests on the speakers summing as
/// 2D sources: spatialized emitters at 32 distinct positions comb-filter
/// against each other and the mix changes as the listener walks. Note the
/// game's own default here is `true`, so omitting this field does not merely
/// leave the value unspecified -- it turns spatialization ON.
///
/// It does NOT switch off distance attenuation, which is why the radii above
/// matter even in a default render.
const SPATIALIZATION: bool = false;

/// Baked `InputB` on the per-band volume multiply, i.e. what the `Volume` pin
/// overrides.
///
/// MUST stay `1.0`: this is the gain the multiply applies while nothing is
/// attached to the pin, so any other value rescales every render that never
/// touches the pin at all. `0.0` would mute the entire bank, in game only.
const VOLUME_SCALE: f64 = 1.0;

/// The speaker brick's half-size AS THE BUILD-TIME OVERLAP CHECK MEASURES IT,
/// floored at the brick's true size.
///
/// Derived from `local_bounds()` rather than hardcoded, because `chip::finish`
/// runs `layout::assert_bricks_dont_overlap` and that measures every brick the
/// same way: a spacing derived from anything else can disagree with the
/// checker and the whole render is rejected. brdb 0.9.1 knows no basic brick's
/// real size and returns a flat (5, 5, 6) guess for every one, this speaker
/// included, whose true half-extent is [`SPEAKER_HALF`] (5, 5, 2). Spacing on
/// the guess is always safe -- it is bigger than the brick, never smaller, so
/// the only in-game consequence is a little air between the speakers. A
/// hardcoded 4-unit pitch (the true height) reads as a collision and is
/// rejected before it can be written; that is not hypothetical, it happened.
///
/// The `.max` floor covers the opposite case: a build whose catalog reports
/// something SMALLER than the brick really is.
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

/// Slot counts `(nx, ny, nz)` for a compact, near-cubic cluster of `n`
/// speakers.
///
/// WHY A CLUSTER AND NOT A COLUMN. `bSpatialization = false` turns off panning
/// but NOT distance attenuation, so where a speaker is decides how loud it is.
/// A column of 32 is 372 units end to end -- comparable to the whole audible
/// range -- so a listener standing anywhere hears the near bands and not the
/// far ones: the bank is spatially filtered into a slice of the spectrum that
/// changes as they walk. Packing the same 32 into 4x3x3 makes the greatest
/// separation between any two speakers about 43 units, a tenth of the inner
/// radius, so every band reaches the listener at the same level. That
/// equality, not compactness for its own sake, is the point.
///
/// THE SHAPE. `nx` is the smallest integer with `nx^3 >= n`, `ny` the smallest
/// with `nx*ny^2 >= n`, and `nz = ceil(n / (nx*ny))` -- so `nx >= ny >= nz`,
/// the box is within one slot of cubic for every `n`, and there are always at
/// least `n` slots. Integer arithmetic throughout, deliberately: `cbrt(27.0)`
/// is `3.0000000000000004` on some targets and `ceil` would make it 4.
///
/// The largest count lands on x and the smallest on z because the measured
/// cell (see [`speaker_half`]) is 10 x 10 x 12 -- taller than it is wide -- so
/// the fewest layers belong on the long axis. For 32 that is 4x3x3 = 40 x 30 x
/// 24 units of centres, whose diagonal is the ~43 units above.
///
/// DETERMINISM. A pure function of `n` alone. Band `k` must land in the same
/// slot on every run, or two renders of the same file differ and a listener
/// comparing them is comparing geometry, not audio.
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

/// Where speaker `index` of `total` sits on the main grid.
///
/// x fastest, then y, then z, over the slots [`cluster_dims`] hands back. Each
/// index below `total` gets its OWN slot: two speakers sharing a position
/// overlap, and an overlap makes the game silently delete one of the two
/// bricks -- a band that vanishes with nothing anywhere to say so.
///
/// Every coordinate is >= 0. The main grid tolerates negatives (only the
/// chip's INNER grid deletes bricks at negative coordinates), but keeping the
/// cluster in the positive octant is what lets the chip sit at a fixed
/// negative x and clear it for ANY band count -- see [`build_speaker_world`].
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

/// Where a service gate at lattice (`col`, `row`) sits inside the chip.
///
/// Rows are numbered BACKWARD from the clock (0) through the select cascade
/// (-9) and the master-volume multiplies (-10): the negative row index is this
/// crate's established way of saying "further back in the service lattice", and
/// it is what the video renderer uses too.
///
/// It is only a row INDEX, never a coordinate. This turns it into
/// `x = (LATTICE_HEIGHT - 1 - row) * CELL + half.x`, so a more negative row
/// lands further along POSITIVE x, and every brick in the chip stays in the
/// non-negative octant. That matters twice over: negative inner-grid
/// coordinates delete bricks in-game (see `chip::finish`'s doc comment for how
/// that was found), and `Chip::recompute_plane_extent` carries a
/// `debug_assert` against them. Writing the coordinate straight from the row
/// (`y: row * CELL`) puts every service gate at a negative y and trips both.
fn service(col: i32, row: i32) -> Position {
    lattice_pos_staged(col, row, 0, LATTICE_HEIGHT, GATE_HALF, STAGE_PITCH)
}

/// Reject an attenuation setting that would encode perfectly and be audible
/// nowhere.
///
/// Both radii are baked as bare `f32`s into every emitter, and nothing
/// downstream looks at them again: a NaN, an infinity or a zero encodes
/// perfectly and produces a bank that is silent, or audible nowhere, only in
/// game. Owned here rather than by the flag parser, so the GUI and any other
/// caller is covered by the same guard -- and called by
/// [`cost::check`](crate::audio::cost::check) as well as by both builders, so
/// the refusal reaches the front end BEFORE the analysis instead of after it.
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

/// Add one speaker to the main grid at cluster slot `index` of `total`,
/// carrying a baked `Component_AudioEmitter`, and return its brick id.
///
/// Shared by both render modes: the emitter's settings are audible (see
/// [`DEFAULT_INNER_RADIUS`] and [`SPATIALIZATION`]), so a second copy of this
/// for the voice mode would be a second instrument that only sounds the same
/// until one of them is retuned.
///
/// `pitch` is the STARTING pitch. In the band bank it is the band's fixed pitch
/// and nothing ever writes it again; in voice mode it is frame 0's value and a
/// wire overwrites it every frame. Either way it is what a paused or unwired
/// chip plays, so it must be a real note.
fn add_emitter(
    world: &mut World,
    asset_name: &str,
    pitch: f32,
    index: usize,
    total: usize,
    opts: &AudioOptions,
) -> Result<usize, String> {
    let mut data: std::collections::HashMap<brdb::BString, Box<dyn AsBrdbValue>> =
        std::collections::HashMap::new();
    // Bare `f32`, NOT a `WireVariant`. `BrickComponentData_AudioEmitter`
    // declares `PitchMultiplier` and `VolumeMultiplier` as plain `f32`
    // fields (see `BRSavedComponentChunkSoA_max.schema`), not the tagged
    // wire-graph union that gate ports use -- a `WireVariant::Number`
    // here compiles and then fails at encode time with
    // `UnimplementedCast("f32", WireVariant)`. Same trap as the clock's
    // `BitwiseOR.InputB`.
    data.insert(
        "PitchMultiplier".into(),
        Box::new(pitch) as Box<dyn AsBrdbValue>,
    );
    // 2D sound. Without this, a cluster of emitters at distinct positions
    // comb-filters against itself and the mix changes as the listener
    // moves -- the single thing that would make a swarm unusable.
    //
    // This is ALSO the value the `Directional` pin overrides, and it is
    // what stands while that pin is unwired -- so it must keep describing
    // the default (2D) build, not the pin's existence. See [`SPATIALIZATION`].
    data.insert(
        "bSpatialization".into(),
        Box::new(SPATIALIZATION) as Box<dyn AsBrdbValue>,
    );
    // The two attenuation radii the `Inner Radius` / `Max Distance` pins
    // override. These are LIVE in a default render, not inert:
    // `bSpatialization = false` disables panning only, and distance
    // attenuation applies regardless -- which is exactly how the bank came
    // to be spatially filtered when they were left at the game's own
    // 15/400. Defaults and reasoning: [`DEFAULT_INNER_RADIUS`].
    //
    // Bare `f32` -- the schema declares both as plain f32, same as
    // `PitchMultiplier` above, so a `WireVariant` compiles and dies at
    // encode. Note the two are the same type: swapping them is invisible
    // to everything but a read-back of the written save.
    data.insert(
        "InnerRadius".into(),
        Box::new(opts.inner_radius) as Box<dyn AsBrdbValue>,
    );
    data.insert(
        "MaxDistance".into(),
        Box::new(opts.max_distance) as Box<dyn AsBrdbValue>,
    );
    data.insert("bEnabled".into(), Box::new(true) as Box<dyn AsBrdbValue>);
    // Starts silent; the wire drives it from frame 0. Bare `f32` again,
    // for the same reason as `PitchMultiplier` above.
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
        position: speaker_position(index, total),
        ..Default::default()
    }
    .with_component(LiteralComponent::new_from_data(
        AUDIO_EMITTER,
        std::sync::Arc::new(data),
    ))
    .with_id_split();
    world.add_brick(brick);
    Ok(id)
}

/// The chip, its clock, and the four attenuation/volume input pins -- the
/// scaffolding both render modes put around their speakers.
struct Scaffold {
    chip: Chip,
    /// Source port carrying the wrapped integer frame index.
    frame_index: WirePort,
    /// The `Volume` pin's source port, which every per-speaker master-volume
    /// multiply reads.
    volume_pin: WirePort,
}

/// Build the microchip, its clock, and the four input pins, and fan three of
/// them out to every speaker.
///
/// # The pin contract, and the one thing that must not be got wrong
///
/// An input pin with nothing attached to it drives NOTHING. It does not
/// deliver zero or false to its target -- the target simply keeps whatever
/// value is baked into it. This is exactly the contract `clock.rs`'s
/// `rate_pin` relies on (an inlined fps on `Multiply.InputB`, overridden live
/// only when the pin is wired), and it is why each of these four keeps a real
/// baked value underneath it: [`SPATIALIZATION`] and `opts.inner_radius` /
/// `opts.max_distance` on the emitters, and [`VOLUME_SCALE`] on the per-speaker
/// multiplies. Were it the other way round -- an unwired pin forcing its target
/// -- this would hand every speaker `MaxDistance = 0` and silence the whole
/// build, in game only, with a save that builds and looks perfect.
fn scaffold(
    world: &mut World,
    speaker_ids: &[usize],
    fps: f32,
    frame_count: usize,
    loop_playback: bool,
) -> Scaffold {
    // Beside the cluster on X, never inside it: an overlap on the main grid
    // makes the game silently DROP one of the two bricks.
    //
    // The cluster grows along POSITIVE x, y and z from a low corner at exactly
    // (0, 0, 0) -- slot 0's centre is `half`, so its low face is the origin
    // (see [`speaker_position`]). So any chip_x whose shell stays at negative
    // x clears the cluster for EVERY speaker count; the shell's own
    // half-extent is 5 and this leaves 15 units of air on top of that.
    // Growing the count moves the cluster's far corner, never this face.
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

    // Reused unchanged from the video path: timer -> multiply by fps ->
    // truncate -> modulo frame count, plus Pause/Restart/Resume/Rate pins.
    // `loop_playback` rides along for the same reason -- one clock, one
    // meaning, so a track and a clip stop the same way.
    let clock = build_clock(world, &mut chip, fps, frame_count, loop_playback, service(0, 0));
    let frame_index = clock.frame_index.clone();

    // Row -2 is chosen because it is empty: the clock's own gates sit at row
    // 0 and `build_clock` puts its five control pins one cell further along x,
    // i.e. row -1, while the detector starts again at -4. These are row
    // INDICES, not coordinates -- `service` maps them onto positive x.
    // `chip::finish` collision-checks the whole inner grid, so a clash here is
    // a build error rather than a brick the game drops.
    //
    // `Directional` is the user-facing name for `bSpatialization`, the only
    // directional property this component has.
    let inner_radius_pin = add_input_pin(&mut chip, "Inner Radius", service(0, -2));
    let max_distance_pin = add_input_pin(&mut chip, "Max Distance", service(1, -2));
    let directional_pin = add_input_pin(&mut chip, "Directional", service(2, -2));
    let volume_pin = add_input_pin(&mut chip, "Volume", service(3, -2));

    // Three of them fan straight out to EVERY speaker's own emitter port --
    // one pin, `speaker_ids.len()` wires. A pin wired to only some of the
    // cluster leaves the rest on their baked values, so the build would split
    // into two groups the moment the pin moved.
    //
    // These are real wire inputs on `Component_AudioEmitter` (its `inputs`
    // list in brdb's component catalog carries `InnerRadius`, `MaxDistance`
    // and `bSpatialization`), so this is wiring, not data -- wiring a
    // data-only field is what produces a save the game rejects.
    //
    // `pin_source(pin, true)` is the INPUT pin's `RER_Output`: the port that
    // carries an input pin's value onward into the chip's graph. `RER_Input`
    // (`pin_target`) is the far side, fed from outside the chip; sourcing a
    // wire from it instead points the wire backwards.
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

    Scaffold {
        chip,
        frame_index,
        volume_pin: pin_source(volume_pin, true),
    }
}

/// One per-speaker master-volume multiply, and the wire from its output into
/// that speaker's `VolumeMultiplier`. Returns the gate's `InputA` port, which
/// is where the frame data goes.
///
/// The `Volume` pin CANNOT wire into `VolumeMultiplier` directly -- that input
/// is already driven by this speaker's own value, and an input with two sources
/// is precisely what this graph never does. So the scaling goes in a gate the
/// pin can own outright: frame value on `InputA`, pin on `InputB`.
///
/// `InputB` carries an inlined [`VOLUME_SCALE`] (1.0) as well as the pin. That
/// is not redundancy -- it is the whole mechanism, copied from `build_clock`'s
/// `rate_pin`: the literal is what the gate multiplies by while the pin is
/// unwired, so a chip nobody has touched renders bit-identically to one with no
/// `Volume` pin at all. Baking 0.0 here would mute every speaker, in game only.
///
/// TYPE TRAP: `MathMultiply`'s ports are the `WireGraphPrimMathVariant` tagged
/// union, so this literal must be a `WireVariant::Number` -- unlike
/// `PitchMultiplier`/`VolumeMultiplier` on the emitter, which are bare `f32`
/// fields where a `WireVariant` compiles and then dies at encode with
/// `UnimplementedCast("f32", WireVariant)`. Both conventions live in this one
/// module; they are not interchangeable.
fn volume_multiply(
    world: &mut World,
    chip: &mut Chip,
    slot: usize,
    speaker: usize,
    volume_pin: &WirePort,
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
        volume_pin.clone(),
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

    // Exec: branches cascade at the FRONT so exactly one bank's chain runs.
    // POLARITY: a truthy `bCond` takes `ExecOutA`. This was inverted in an
    // earlier draft of the video path and would have made every multi-bank
    // render silent -- do not "fix" it without re-reading `Exec_Branch`.
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
    // POSITION IS AUDIBLE. `bSpatialization = false` stops the panning, not
    // the distance attenuation, so a bank spread over hundreds of units is a
    // bank the listener hears a slice of. Every speaker therefore goes into
    // the tightest legal 3D packing instead of a column -- see
    // [`cluster_dims`] for the arithmetic and [`speaker_position`] for the
    // slot each band takes.
    let n_speakers = track.plan.len();
    let mut speaker_ids = Vec::with_capacity(n_speakers);
    for (b, kind) in track.plan.kinds.iter().enumerate() {
        let asset_name = match kind {
            BandKind::Tonal => BA_SYNTH_BASIC_SINE,
            BandKind::WhiteNoise => BA_SYNTH_NOISE_WHITE,
            BandKind::PinkNoise => BA_SYNTH_NOISE_PINK,
        };
        speaker_ids.push(add_emitter(
            &mut world,
            asset_name.as_ref(),
            track.plan.pitches[b],
            b,
            n_speakers,
            opts,
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

    // --- 3. One master-volume multiply per band -----------------------------
    let targets: Vec<WirePort> = speaker_ids
        .iter()
        .enumerate()
        .map(|(b, &speaker)| volume_multiply(&mut world, &mut sc.chip, b, speaker, &sc.volume_pin))
        .collect();

    // --- 4. One stream per band: its volume ---------------------------------
    // Pitch is build-time component data here and is never written, so a band
    // fading in and out carries no retrigger risk. That is the property the
    // voice mode gives up in exchange for having no grid at all.
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

    finish(&mut world, sc.chip)?;
    // Must be LAST, after `finish` has published the chip's inner grid.
    // `register_used_components` walks `world.bricks` and `world.grids` and
    // registers only the component types it can see; running it before
    // `finish` leaves every gate inside the chip unregistered and
    // `to_brz_vec()` then fails with `UnregisteredComponentType`. brdb's own
    // doc on that method says the same ("Call this AFTER all bricks/grids
    // have been added"), and the video renderer orders it the same way.
    world.register_used_components();
    Ok(world)
}

/// Build a world whose speakers TRACK spectral peaks: both pitch and volume
/// written every frame.
///
/// The same cluster, the same chip, the same clock, the same four pins and the
/// same banking cascade as [`build_speaker_world`] -- the only differences are
/// that there are `voice_count` speakers instead of one per band, that they all
/// play the sine synth, and that each contributes **two** streams instead of
/// one.
///
/// # Every speaker is a sine
///
/// There is no `BandKind` here and no noise speaker. A voice is a moving
/// sinusoid tracking one partial; white and pink noise have no frequency to
/// track, so a noise voice would be a speaker whose pitch stream meant nothing.
/// The consequence -- cymbals, sibilance and sub-bass are simply absent rather
/// than folded onto a noise bed -- is a real cost of this mode and is documented
/// on `voices::min_hz`.
///
/// # `PitchMultiplier` is wired, not baked
///
/// This is the one assumption this mode adds over the bank's, and it is
/// UNVERIFIED IN GAME at the time of writing: `diag_5_pitch_ramp.brz` (see
/// `examples/audio_diagnostics.rs`) is the save that settles whether a
/// per-frame pitch write retunes a running voice or retriggers it. The baked
/// `PitchMultiplier` is frame 0's value, so a paused chip holds the first note
/// rather than an arbitrary one.
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
    // A pitch outside the emitter's legal range is CLAMPED in game, turning a
    // wrong number into a wrong note rather than into silence. `analyze_voices`
    // clamps already; this is the guard for any other caller, and for a stream
    // built by hand.
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

    // --- 1. The speaker cluster on the main grid ----------------------------
    let mut speaker_ids = Vec::with_capacity(n_voices);
    for v in 0..n_voices {
        speaker_ids.push(add_emitter(
            &mut world,
            BA_SYNTH_BASIC_SINE.as_ref(),
            streams.pitches[v][0] as f32,
            v,
            n_voices,
            opts,
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

    // --- 3. One master-volume multiply per voice ----------------------------
    let volume_targets: Vec<WirePort> = speaker_ids
        .iter()
        .enumerate()
        .map(|(v, &speaker)| volume_multiply(&mut world, &mut sc.chip, v, speaker, &sc.volume_pin))
        .collect();

    // --- 4. Two streams per voice: its pitch and its volume -----------------
    // Interleaved per voice rather than all pitches then all volumes, so a
    // voice's two arrays sit next to each other in the chip and the layout can
    // be read by eye.
    //
    // The pitch stream goes STRAIGHT into the emitter, with no multiply in
    // between: the `Volume` pin scales levels, and scaling a pitch by it would
    // transpose the whole render whenever a builder touched the volume.
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

    finish(&mut world, sc.chip)?;
    // Must be LAST, after `finish` has published the chip's inner grid -- see
    // [`build_speaker_world`].
    world.register_used_components();
    Ok(world)
}

/// Resolve a synth asset name to the value `AudioDescriptor` expects.
///
/// Verified against brdb 0.9.1 by Task 1 -- a `.brz` was written and read
/// back, and `AudioDescriptor` decoded as `Asset(Some(0))` with the
/// `("BrickAudioDescriptor", "BA_Synth_Basic_Sine")` pair intact in global
/// data. See `docs/superpowers/notes/2026-07-28-audio-feasibility.md`.
///
/// Factored out so the whole design has ONE place depending on this.
///
/// `AudioDescriptor`'s schema type is a bare `object`, so the value is a
/// [`BrdbValue::Asset`] carrying the reference's index -- NOT a
/// `WireVariant::Object`, which is the wire-graph value union and a different
/// encoding path entirely. Both compile.
///
/// The `(type, name)` pair is checked against brdb's own `ASSET_TYPES`
/// catalog first. Nothing on the write path does that -- `write_brdb`'s
/// `"class" | "object"` arm writes the raw index without ever comparing the
/// stored asset type to the schema's -- so a typo in either string encodes
/// perfectly and fails only in game, with nothing to see at build time. This
/// lookup is the only thing standing between a typo and a silent dud save.
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
