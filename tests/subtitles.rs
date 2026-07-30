//! CLI-level coverage for `--subtitles`, `--subtitle-track` and
//! `--subtitle-scale`.
//!
//! These drive the real binary rather than calling into the library, for the
//! same reason `tests/cli_text_mode.rs` does: the thing under test is the
//! argument wiring itself -- the code path that turns typed flags into a parsed
//! `Subtitles` track and an `AnimOptions`. A library-level test cannot see a
//! flag parsed into the wrong field, one never read at all, or (the bug this
//! file's `subtitles_are_timed_against_the_clip_start_offset` exists for) a
//! value hard-coded at the one call site that consumes it.
//!
//! No media path ever appears here. Every input is authored into a temp
//! directory by the test itself.
use brdb::IntoReader;
use std::path::{Path, PathBuf};
use std::process::Command;

fn heightmap() -> Command {
    Command::new(env!("CARGO_BIN_EXE_heightmap"))
}

/// A private temp directory for one test. Named after the test rather than
/// shared, so the whole file can run in parallel.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("h2b_subs_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// `frames` numbered PNGs, each a flat colour that changes per frame so no
/// pixel is culled and every frame differs. Returns the directory, which the
/// CLI accepts as a frame sequence.
fn write_frames(dir: &Path, frames: usize, w: u32, h: u32) -> PathBuf {
    let seq = dir.join("frames");
    std::fs::create_dir_all(&seq).expect("create frame dir");
    for i in 0..frames {
        let img = image::RgbaImage::from_pixel(
            w,
            h,
            image::Rgba([(20 + i * 7) as u8, 90, (200 - i * 3) as u8, 0xFF]),
        );
        // Zero padded so the plain lexical sort and the natural-key sort
        // `decode_sequence` applies agree, whichever one ends up ordering them.
        img.save(seq.join(format!("f_{i:03}.png"))).expect("write frame");
    }
    seq
}

/// The three-cue track every render here uses, on a 4 fps grid:
/// frames 0-3 are cue one, 8-11 cue two, 16-19 cue three, everything else
/// empty. Deliberately spaced so a render starting 2 s in lands on a
/// DIFFERENT cue at frame 0 than one starting at 0 -- which is the whole
/// point of `subtitles_are_timed_against_the_clip_start_offset`.
const CUE_ONE: &str = "CUE ONE";
const CUE_TWO: &str = "CUE TWO";
const CUE_THREE: &str = "CUE THREE";

fn write_srt(dir: &Path) -> PathBuf {
    let path = dir.join("track.srt");
    std::fs::write(
        &path,
        "1\n00:00:00,000 --> 00:00:01,000\nCUE ONE\n\n\
         2\n00:00:02,000 --> 00:00:03,000\nCUE TWO\n\n\
         3\n00:00:04,000 --> 00:00:05,000\nCUE THREE\n",
    )
    .expect("write srt");
    path
}

struct Run {
    ok: bool,
    log: String,
}

fn run(args: &[&str]) -> Run {
    let out = heightmap().args(args).output().expect("run heightmap");
    Run {
        ok: out.status.success(),
        log: String::from_utf8_lossy(&out.stdout).into_owned()
            + &String::from_utf8_lossy(&out.stderr),
    }
}

/// The gate count from the pre-render cost readout -- the number the user is
/// shown before committing to a render.
///
/// Parsed out of `log_cost`'s line rather than recomputed, because the whole
/// question this file asks of it is whether the DISPLAYED number agrees with
/// what got built.
fn estimated_gates(log: &str) -> usize {
    let line = log
        .lines()
        .find(|l| l.contains("Estimated cost"))
        .unwrap_or_else(|| panic!("no cost readout in:\n{log}"));
    for field in line.split(", ") {
        if let Some(n) = field.strip_suffix(" gate(s)") {
            return n
                .trim()
                .parse()
                .unwrap_or_else(|e| panic!("unparseable gate count {n:?}: {e}"));
        }
    }
    panic!("no gate count in {line:?}");
}

/// `(bricks, wires)` in the microchip grid of a written save -- every gate the
/// render actually built, plus the chip's own I/O pins. The pin count is
/// identical on both sides of every comparison here, so the DELTA is the
/// gate delta.
fn chip_bricks_and_wires(path: &Path) -> (u64, u64) {
    let db = brdb::Brz::open(path).expect("reopen").into_reader();
    let mut gid = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        for e in db.entity_chunk(index).expect("entity chunk") {
            if e.is_microchip_grid() {
                gid = e.id;
            }
        }
    }
    let gid = gid.expect("the renderer must publish exactly one microchip grid");
    let (mut bricks, mut wires) = (0u64, 0u64);
    for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
        bricks += chunk.num_bricks as u64;
        wires += chunk.num_wires as u64;
    }
    (bricks, wires)
}

/// Every string `ArrayVar` persisted in the save's microchip grid.
///
/// Component DATA is unreachable from an in-memory `World` (`BrdbComponent`
/// exposes `component_type()` and nothing else), so this round-trips through
/// the written file exactly as `tests/anim_text.rs::band_strings` does. The
/// bytes are never compared -- only the structure is read back.
fn string_arrays(path: &Path) -> Vec<Vec<String>> {
    use brdb::schema::WireArrayVariant;
    let db = brdb::Brz::open(path).expect("reopen").into_reader();
    let mut gid = None;
    for index in db.entity_chunk_index().expect("entity chunk index") {
        for e in db.entity_chunk(index).expect("entity chunk") {
            if e.is_microchip_grid() {
                gid = e.id;
            }
        }
    }
    let gid = gid.expect("the renderer must publish exactly one microchip grid");
    let mut out = Vec::new();
    for chunk in &db.brick_chunk_index(gid).expect("chunk index") {
        let (_soa, structs) = db.component_chunk(gid, chunk.index).expect("components");
        for s in &structs {
            if s.get_name() == "BrickComponentData_WireGraphPseudo_ArrayVar"
                && let Some(value) = s.get("Value")
            {
                let variant: WireArrayVariant =
                    value.try_into().expect("ArrayVar Value must decode");
                if let WireArrayVariant::StringArray(v) = variant {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// The subtitle track's own array, picked out of every string array in the
/// save by the cue text it carries.
///
/// Text mode's BAND arrays are strings too, so "the only string array" would
/// be wrong there; the cue text is what distinguishes the subtitle's.
fn subtitle_array(path: &Path) -> Vec<String> {
    let arrays = string_arrays(path);
    let mut found: Vec<Vec<String>> = arrays
        .into_iter()
        .filter(|a| a.iter().any(|s| s.starts_with("CUE ")))
        .collect();
    assert_eq!(
        found.len(),
        1,
        "exactly one array must carry the subtitle cues"
    );
    found.pop().unwrap()
}

#[test]
fn both_subtitle_sources_at_once_is_an_error_naming_both() {
    let out = run(&[
        "in.png",
        "--anim-mode",
        "text",
        "--subtitles",
        "a.srt",
        "--subtitle-track",
        "0",
    ]);
    assert!(!out.ok, "must not exit 0: {}", out.log);
    assert!(
        out.log.contains("--subtitles") && out.log.contains("--subtitle-track"),
        "{}",
        out.log
    );
}

#[test]
fn a_missing_subtitle_file_errors_naming_the_path() {
    let out = run(&[
        "in.png",
        "--anim-mode",
        "text",
        "--subtitles",
        "definitely-absent.srt",
    ]);
    assert!(!out.ok, "must not exit 0: {}", out.log);
    assert!(out.log.contains("definitely-absent.srt"), "{}", out.log);
}

/// `--subtitle-scale` with no track to scale is a warn-and-ignore, matching
/// how `--colormap` is handled under `--anim-mode`: the render still runs, but
/// the flag is named out loud rather than silently dropped.
#[test]
fn subtitle_scale_without_a_track_warns_and_still_renders() {
    let dir = scratch("scale_warn");
    let seq = write_frames(&dir, 4, 32, 16);
    let brz = dir.join("out.brz");
    let out = run(&[
        seq.to_str().unwrap(),
        "--anim-mode",
        "text",
        "--fps",
        "4",
        "--subtitle-scale",
        "3.5",
        "-o",
        brz.to_str().unwrap(),
    ]);
    assert!(out.ok, "the render must still run: {}", out.log);
    assert!(brz.exists(), "no save written: {}", out.log);
    assert!(
        out.log.contains("--subtitle-scale"),
        "the ignored flag must be named out loud: {}",
        out.log
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// **The headline number.** Same clip, subtitles on vs off: the delta must be
/// one `ArrayVar` + one `ArrayVar_Get`, in the SAVE and in the READOUT alike.
///
/// The readout half is the regression test for a cost estimator that describes
/// a different graph from the one that gets built -- the exact failure
/// `cost::estimate_text` shipped once already when it hard-coded
/// `char_repeat`.
#[test]
fn a_real_render_with_subtitles_adds_exactly_two_gates() {
    let dir = scratch("two_gates");
    let seq = write_frames(&dir, 24, 32, 16);
    let srt = write_srt(&dir);
    let with_brz = dir.join("with.brz");
    let without_brz = dir.join("without.brz");

    let base = [seq.to_str().unwrap(), "--anim-mode", "text", "--fps", "4"];

    let mut without_args = base.to_vec();
    without_args.extend(["-o", without_brz.to_str().unwrap()]);
    let without = run(&without_args);
    assert!(without.ok, "subtitle-free render must succeed: {}", without.log);

    let mut with_args = base.to_vec();
    with_args.extend(["--subtitles", srt.to_str().unwrap(), "-o", with_brz.to_str().unwrap()]);
    let with = run(&with_args);
    assert!(with.ok, "subtitled render must succeed: {}", with.log);

    let (with_bricks, with_wires) = chip_bricks_and_wires(&with_brz);
    let (without_bricks, without_wires) = chip_bricks_and_wires(&without_brz);
    assert_eq!(
        with_bricks - without_bricks,
        2,
        "a subtitle track is one ArrayVar and one Get, nothing else"
    );
    // ArrayVarRef + Index + Exec into the Get, plus the value into the
    // TextDisplay's Text port -- the last of which leaves the chip grid, so
    // only three of the four are counted here.
    assert_eq!(
        with_wires - without_wires,
        3,
        "the subtitle Get takes exactly ArrayVarRef, Index and Exec"
    );

    assert_eq!(
        estimated_gates(&with.log) - estimated_gates(&without.log),
        2,
        "the READOUT must rise by the same 2 the render does -- with {}, without {}",
        estimated_gates(&with.log),
        estimated_gates(&without.log),
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// **Defect 1's regression test.** A subtitle file is in SOURCE time, so a
/// render that starts 2 s in must show the cue live at t = 2 s in its first
/// frame -- not the one live at t = 0.
///
/// With the start offset hard-coded to `0.0` this test sees `CUE ONE` at frame
/// 0 of the `--start 2` render instead of `CUE TWO`, which is exactly what a
/// human would see in game: every subtitle two seconds early.
#[test]
fn subtitles_are_timed_against_the_clip_start_offset() {
    let dir = scratch("start_offset");
    // 24 frames at 4 fps = 6 s of source, covering all three cues.
    let seq = write_frames(&dir, 24, 32, 16);
    let srt = write_srt(&dir);

    let from_zero = dir.join("from_zero.brz");
    let out = run(&[
        seq.to_str().unwrap(),
        "--anim-mode",
        "text",
        "--fps",
        "4",
        "--subtitles",
        srt.to_str().unwrap(),
        "-o",
        from_zero.to_str().unwrap(),
    ]);
    assert!(out.ok, "{}", out.log);
    let entries = subtitle_array(&from_zero);
    assert_eq!(entries.len(), 24, "one entry per emitted frame");
    assert_eq!(entries[0], CUE_ONE, "frame 0 is at t=0.00s");
    assert_eq!(entries[3], CUE_ONE, "frame 3 is at t=0.75s");
    assert_eq!(entries[4], "", "frame 4 is at t=1.00s, past the first cue");
    assert_eq!(entries[8], CUE_TWO, "frame 8 is at t=2.00s");
    assert_eq!(entries[16], CUE_THREE, "frame 16 is at t=4.00s");

    // The same file, started 2 s in: frame 0 is now t=2.00s.
    let from_two = dir.join("from_two.brz");
    let out = run(&[
        seq.to_str().unwrap(),
        "--anim-mode",
        "text",
        "--fps",
        "4",
        "--start",
        "2",
        "--subtitles",
        srt.to_str().unwrap(),
        "-o",
        from_two.to_str().unwrap(),
    ]);
    assert!(out.ok, "{}", out.log);
    let entries = subtitle_array(&from_two);
    assert_eq!(entries.len(), 16, "16 frames left after skipping 2 s at 4 fps");
    assert_eq!(
        entries[0], CUE_TWO,
        "frame 0 of a --start 2 render is at SOURCE t=2.00s -- a subtitle file is \
         in source time, so hard-coding the offset to 0 puts every cue 2 s early"
    );
    assert_eq!(entries[3], CUE_TWO, "frame 3 is at t=2.75s");
    assert_eq!(entries[4], "", "frame 4 is at t=3.00s, past the second cue");
    assert_eq!(entries[8], CUE_THREE, "frame 8 is at t=4.00s");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Every mode gets the same two gates and the same readout, not just the text
/// mode the feature was designed against.
#[test]
fn every_anim_mode_reports_and_builds_the_same_two_extra_gates() {
    for (mode, encoding) in [
        ("brick", Some("hex")),
        ("brick", Some("color-array")),
        ("text", None),
    ] {
        let tag = format!("modes_{mode}_{}", encoding.unwrap_or("none"));
        let dir = scratch(&tag);
        let seq = write_frames(&dir, 8, 16, 8);
        let srt = write_srt(&dir);
        let with_brz = dir.join("with.brz");
        let without_brz = dir.join("without.brz");

        let mut base = vec![seq.to_str().unwrap(), "--anim-mode", mode, "--fps", "4"];
        if let Some(enc) = encoding {
            base.extend(["--anim-encoding", enc]);
        }

        let mut without_args = base.clone();
        without_args.extend(["-o", without_brz.to_str().unwrap()]);
        let without = run(&without_args);
        assert!(without.ok, "{mode}/{encoding:?} without subs: {}", without.log);

        let mut with_args = base.clone();
        with_args.extend(["--subtitles", srt.to_str().unwrap(), "-o", with_brz.to_str().unwrap()]);
        let with = run(&with_args);
        assert!(with.ok, "{mode}/{encoding:?} with subs: {}", with.log);

        let (with_bricks, _) = chip_bricks_and_wires(&with_brz);
        let (without_bricks, _) = chip_bricks_and_wires(&without_brz);
        assert_eq!(
            with_bricks - without_bricks,
            2,
            "{mode}/{encoding:?}: a subtitle track is 2 gates in every mode"
        );
        assert_eq!(
            estimated_gates(&with.log) - estimated_gates(&without.log),
            2,
            "{mode}/{encoding:?}: the readout must rise by the same 2"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
