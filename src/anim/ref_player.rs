//! Video brick mode driven by brick references (`--driver refs`).
//!
//! The wire driver ([`super::bricks`], [`super::color_bricks`]) spends two
//! gates per pixel and one wire into each display brick's
//! `BrickPropertyChanger`. This driver keeps every display brick bare and
//! repaints them from a loop instead:
//!
//! - `pixels`, a `brick[]` array on the main grid, holds a ref to every
//!   display brick (see [`crate::graph::main_grid_brick_refs`]);
//! - every frame is stored as a list of `(pixel, colour)` records: a
//!   *keyframe* lists every pixel, a *delta* only the pixels whose shown
//!   colour changes;
//! - on a frame change, a `CountFor` replays the frames from the last one
//!   shown (or from the target's keyframe after a jump or a wrap) up to the
//!   target, calling the function `apply_frame` for each. Its body loops
//!   over that frame's records and runs `SetBrickProperty(Color)` on
//!   `pixels[i]`.
//!
//! So the gate count is a fixed ~40 whatever the resolution, and the cost
//! moves to loop iterations: one per record replayed, all inside one tick.
//! A dropped tick or a frame skip catches up rather than tearing, and the
//! work per tick is bounded by one keyframe plus the deltas after it.
//!
//! Records come in the two existing encodings:
//! - **hex**: fixed-width text records, `index` in decimal then `RRGGBB`,
//!   packed into strings of whole records. `chunk_records` stores each
//!   string's record count so no gate divides.
//! - **color-array**: parallel `i64` index and `LinearColor` arrays, no
//!   string parsing at all.

use super::bricks::{AnimOptions, CHANGE_DETECTOR, add_display_bricks, new_screen_chip};
use super::chip;
use super::clock;
use super::controls;
use super::layout::{GATE_HALF, STAGE_PITCH, lattice_pos_staged};
use crate::graph::{
    self, ARRAY_GET, ARRAY_VAR, BRICK_PROPERTY_COLOR, EmitPlacement, End, Graph, In, Lit, ROOT,
    SET_BRICK_PROPERTY, VAR, VAR_SET,
};
use crate::progress::{FrameTotal, Progress};
use crate::text::MAX_COMPONENT_CHARS;
use crate::video::stream::FrameSource;
use crate::wire_array::WireArray;
use brdb::{Brick, BrickType, Position, WirePort, World, assets::LiteralComponent};
use image::RgbaImage;
use std::fmt::Write as _;

/// The record encodings, matching [`super::AnimEncoding`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordEncoding {
    Hex,
    ColorArray,
}

/// How frames are split into keyframes and deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordOptions {
    /// A keyframe at least every this many frames. `1` makes every frame a
    /// keyframe, which turns delta encoding off.
    pub keyframe_interval: usize,
    /// A pixel joins a delta only when some channel moved by more than this
    /// from the colour on screen (0: any change). The packer tracks what is
    /// on screen, so skipped changes never accumulate into drift.
    pub delta_threshold: u8,
    /// Hex only: store a keyframe as `RRGGBB` per pixel in `pixels` order,
    /// with no index. Lossless, and about a third smaller on bee-like video;
    /// the colour-array encoding ignores it.
    pub dense_keyframes: bool,
}

impl Default for RecordOptions {
    fn default() -> Self {
        RecordOptions {
            keyframe_interval: 60,
            delta_threshold: 0,
            dense_keyframes: false,
        }
    }
}

/// A pixel's colour as packed: sRGB bytes (or linearised bytes under
/// `--srgb-to-linear`), with alpha 0 for a culled pixel and 255 otherwise.
pub type Rgba = [u8; 4];

const CULLED: Rgba = [0, 0, 0, 0];

/// One frame's records, over the full `width * height` pixel grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRecords {
    pub key: bool,
    pub records: Vec<(u32, Rgba)>,
}

/// Streams frames into keyframes and deltas, tracking what the screen would
/// show after each.
pub struct RecordPacker {
    width: usize,
    height: usize,
    alpha_threshold: u8,
    linearize: Option<&'static [u8; 256]>,
    opts: RecordOptions,
    shown: Vec<Rgba>,
    since_key: usize,
    frames: Vec<FrameRecords>,
    visible: Vec<bool>,
}

impl RecordPacker {
    pub fn new(width: u32, height: u32, alpha_threshold: u8, opts: RecordOptions) -> Self {
        let n = width as usize * height as usize;
        RecordPacker {
            width: width as usize,
            height: height as usize,
            alpha_threshold,
            linearize: None,
            opts,
            shown: vec![CULLED; n],
            since_key: 0,
            frames: Vec::new(),
            visible: vec![false; n],
        }
    }

    /// Run hex colours through the sRGB -> linear table first, as
    /// `pack::Packer::linearize` does. Ignored by the color-array encoding,
    /// which always converts.
    pub fn linearize(mut self, on: bool) -> Self {
        self.linearize = on.then(crate::util::linear_gamma_table);
        self
    }

    pub fn push_frame(&mut self, frame: &RgbaImage) -> Result<(), String> {
        let (fw, fh) = (frame.width() as usize, frame.height() as usize);
        if fw != self.width || fh != self.height {
            return Err(format!(
                "frame {} is {fw}x{fh}, but the source reported {}x{}",
                self.frames.len(),
                self.width,
                self.height
            ));
        }
        let target: Vec<Rgba> = frame
            .as_raw()
            .chunks_exact(4)
            .zip(self.visible.iter_mut())
            .map(|(p, vis)| {
                if p[3] < self.alpha_threshold {
                    return CULLED;
                }
                *vis = true;
                match self.linearize {
                    Some(t) => [t[p[0] as usize], t[p[1] as usize], t[p[2] as usize], 255],
                    None => [p[0], p[1], p[2], 255],
                }
            })
            .collect();
        let threshold = self.opts.delta_threshold;
        let moved = |a: &Rgba, b: &Rgba| {
            a.iter().zip(b).any(|(x, y)| x.abs_diff(*y) > threshold) || (a[3] != b[3])
        };
        let delta: Vec<(u32, Rgba)> = target
            .iter()
            .zip(&self.shown)
            .enumerate()
            .filter(|(_, (t, s))| moved(t, s))
            .map(|(i, (t, _))| (i as u32, *t))
            .collect();
        let interval = self.opts.keyframe_interval.max(1);
        let key = self.frames.is_empty()
            || self.since_key + 1 >= interval
            || delta.len() * 2 > target.len();
        let records = if key {
            self.shown.copy_from_slice(&target);
            self.since_key = 0;
            target
                .iter()
                .enumerate()
                .map(|(i, c)| (i as u32, *c))
                .collect()
        } else {
            for (i, c) in &delta {
                self.shown[*i as usize] = *c;
            }
            self.since_key += 1;
            delta
        };
        self.frames.push(FrameRecords { key, records });
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// The records, plus row-major visibility (opaque in at least one frame).
    pub fn finish(self) -> (Vec<FrameRecords>, Vec<bool>) {
        (self.frames, self.visible)
    }
}

/// The records re-indexed onto the visible pixels only (the order of the
/// `pixels` array) and packed for the game.
///
/// An array holds at most 65,535 entries in game (measured), so the record
/// tables are split into banks of at most `bank_size` entries each, and a
/// frame's entries never straddle two banks. The player reaches a bank
/// through a `component_handle` array of refs to the banks' `ArrayVar`s: an
/// array of arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedRecords {
    pub encoding: RecordEncoding,
    pub pixel_count: usize,
    /// Decimal digits of a hex record's index.
    pub index_width: usize,
    /// Hex keyframes are dense: `RRGGBB` only, pixel `k * dense_per_chunk +
    /// j` for the `j`-th record of the frame's `k`-th string. A dense
    /// string's record count is stored negated.
    pub dense: bool,
    pub dense_per_chunk: usize,
    /// Per frame: the latest keyframe at or before it.
    pub seek_from: Vec<i64>,
    /// Per frame: the bank its entries are in, the first one, and how many.
    pub frame_bank: Vec<i64>,
    pub frame_first: Vec<i64>,
    pub frame_entries: Vec<i64>,
    pub banks: Vec<Bank>,
    /// Records per frame, for the cost readout.
    pub records_per_frame: Vec<usize>,
}

/// One bank of entries, as two parallel arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum Bank {
    /// An entry is a string of whole records and its record count.
    Hex {
        chunks: Vec<String>,
        records: Vec<i64>,
    },
    /// An entry is one record: a pixel index and its linear colour.
    Color {
        index: Vec<i64>,
        color: Vec<(f32, f32, f32, f32)>,
    },
}

impl Bank {
    fn new(encoding: RecordEncoding) -> Self {
        match encoding {
            RecordEncoding::Hex => Bank::Hex {
                chunks: Vec::new(),
                records: Vec::new(),
            },
            RecordEncoding::ColorArray => Bank::Color {
                index: Vec::new(),
                color: Vec::new(),
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Bank::Hex { chunks, .. } => chunks.len(),
            Bank::Color { index, .. } => index.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bank's two arrays: (`a`, `b`).
    pub fn arrays(&self) -> (WireArray, WireArray) {
        match self {
            Bank::Hex { chunks, records } => (
                WireArray::Str(chunks.clone()),
                WireArray::I64(records.clone()),
            ),
            Bank::Color { index, color } => (
                WireArray::I64(index.clone()),
                WireArray::LinearColor(color.clone()),
            ),
        }
    }
}

/// One frame's entries before banking.
enum Entries {
    Hex(Vec<(String, i64)>),
    Color(Vec<(i64, (f32, f32, f32, f32))>),
}

/// Characters per hex record: the index, then `RRGGBB`.
pub fn hex_stride(index_width: usize) -> usize {
    index_width + 6
}

pub fn index_width(pixel_count: usize) -> usize {
    pixel_count.saturating_sub(1).to_string().len()
}

/// `dense` turns on [`RecordOptions::dense_keyframes`]; it only affects hex.
pub fn pack_records(
    frames: &[FrameRecords],
    visible: &[bool],
    encoding: RecordEncoding,
    max_chars: usize,
    bank_size: usize,
    dense: bool,
) -> Result<PackedRecords, String> {
    let bank_size = bank_size.max(1);
    let dense = dense && encoding == RecordEncoding::Hex;
    let dense_per_chunk = (max_chars / 6).max(1);
    // Full-grid index -> position in `pixels`.
    let mut compact = vec![u32::MAX; visible.len()];
    let mut pixel_count = 0usize;
    for (i, v) in visible.iter().enumerate() {
        if *v {
            compact[i] = pixel_count as u32;
            pixel_count += 1;
        }
    }
    let width = index_width(pixel_count);
    let stride = hex_stride(width);
    let per_chunk = (max_chars / stride).max(1);
    let lut = crate::util::srgb_to_linear_f32_table();

    let mut out = PackedRecords {
        encoding,
        pixel_count,
        index_width: width,
        dense,
        dense_per_chunk,
        seek_from: Vec::with_capacity(frames.len()),
        frame_bank: Vec::with_capacity(frames.len()),
        frame_first: Vec::with_capacity(frames.len()),
        frame_entries: Vec::with_capacity(frames.len()),
        banks: vec![Bank::new(encoding)],
        records_per_frame: Vec::with_capacity(frames.len()),
    };
    let mut last_key = 0i64;
    for (f, frame) in frames.iter().enumerate() {
        if frame.key {
            last_key = f as i64;
        }
        out.seek_from.push(last_key);
        // Never-visible pixels have no brick: only a keyframe lists them, and
        // always as culled.
        let records: Vec<(u32, Rgba)> = frame
            .records
            .iter()
            .filter(|(i, _)| compact[*i as usize] != u32::MAX)
            .map(|(i, c)| (compact[*i as usize], *c))
            .collect();
        out.records_per_frame.push(records.len());
        let entries = match encoding {
            // A keyframe lists every visible pixel in `pixels` order, so the
            // index is implied by the record's position.
            RecordEncoding::Hex if dense && frame.key => {
                debug_assert!(
                    records
                        .iter()
                        .enumerate()
                        .all(|(k, (i, _))| *i as usize == k)
                );
                Entries::Hex(
                    records
                        .chunks(dense_per_chunk)
                        .map(|group| {
                            let mut s = String::with_capacity(group.len() * 6);
                            for (_, c) in group {
                                let _ = write!(s, "{:02X}{:02X}{:02X}", c[0], c[1], c[2]);
                            }
                            (s, -(group.len() as i64))
                        })
                        .collect(),
                )
            }
            RecordEncoding::Hex => Entries::Hex(
                records
                    .chunks(per_chunk)
                    .map(|group| {
                        let mut s = String::with_capacity(group.len() * stride);
                        for (i, c) in group {
                            let _ = write!(s, "{i:0width$}{:02X}{:02X}{:02X}", c[0], c[1], c[2]);
                        }
                        (s, group.len() as i64)
                    })
                    .collect(),
            ),
            RecordEncoding::ColorArray => Entries::Color(
                records
                    .iter()
                    .map(|(i, c)| {
                        let color = if c[3] == 0 {
                            (0.0, 0.0, 0.0, 0.0)
                        } else {
                            (
                                lut[c[0] as usize],
                                lut[c[1] as usize],
                                lut[c[2] as usize],
                                1.0,
                            )
                        };
                        (*i as i64, color)
                    })
                    .collect(),
            ),
        };
        let n = match &entries {
            Entries::Hex(e) => e.len(),
            Entries::Color(e) => e.len(),
        };
        if n > bank_size {
            return Err(format!(
                "frame {f} needs {n} table entries, more than one array holds ({bank_size}); \
                 use --driver wires or a smaller picture"
            ));
        }
        if out.banks.last().expect("one bank").len() + n > bank_size {
            out.banks.push(Bank::new(encoding));
        }
        let bank_index = out.banks.len() - 1;
        let bank = out.banks.last_mut().expect("one bank");
        out.frame_bank.push(bank_index as i64);
        out.frame_first.push(bank.len() as i64);
        out.frame_entries.push(n as i64);
        match (bank, entries) {
            (Bank::Hex { chunks, records }, Entries::Hex(e)) => {
                for (s, k) in e {
                    chunks.push(s);
                    records.push(k);
                }
            }
            (Bank::Color { index, color }, Entries::Color(e)) => {
                for (i, c) in e {
                    index.push(i);
                    color.push(c);
                }
            }
            _ => unreachable!("a bank holds its own encoding"),
        }
    }
    Ok(out)
}

impl PackedRecords {
    /// Most records one tick replays: a jump to frame `f` replays from
    /// `seek_from[f]` through `f`.
    pub fn worst_catch_up(&self) -> usize {
        let mut prefix = vec![0usize; self.records_per_frame.len() + 1];
        for (f, n) in self.records_per_frame.iter().enumerate() {
            prefix[f + 1] = prefix[f] + n;
        }
        (0..self.records_per_frame.len())
            .map(|f| prefix[f + 1] - prefix[self.seek_from[f] as usize])
            .max()
            .unwrap_or(0)
    }

    /// Most records a single frame carries (sequential playback, no drops).
    pub fn worst_frame(&self) -> usize {
        self.records_per_frame.iter().copied().max().unwrap_or(0)
    }

    pub fn total_records(&self) -> usize {
        self.records_per_frame.iter().sum()
    }

    /// The longest array outside the banks (which `pack_records` keeps
    /// within the cap): the per-frame tables, `pixels`, and the bank refs.
    pub fn longest_array(&self) -> usize {
        [self.seek_from.len(), self.pixel_count, self.banks.len()]
            .into_iter()
            .max()
            .unwrap_or(0)
    }
}

/// The player's ports into the rest of the screen chip.
pub struct PlayerPorts {
    /// The wrapped frame index.
    pub frame_index: End,
    /// The exec pulse on a frame change.
    pub on_change: End,
    /// `ArrayVarRef` of the `pixels` array on the main grid.
    pub pixels: End,
    /// `ArrayVarRef`s of the two `component_handle` arrays holding refs to
    /// every bank's `a` and `b` arrays, in bank order.
    pub a_banks: End,
    pub b_banks: End,
    /// Where the status text goes: blank until a loop is cut short, then a
    /// warning saying where. `None` builds no status.
    pub status: Option<End>,
}

/// The nodes a test needs to drive and read the player.
pub struct Player {
    pub graph: Graph,
    pub shown: graph::NodeId,
    pub apply_frame: graph::Func,
    /// The status variable, when [`PlayerPorts::status`] asked for one.
    pub status: Option<graph::NodeId>,
}

fn i64_array(g: &mut Graph, grid: graph::GridId, values: &[i64]) -> End {
    let n = g.gate(
        grid,
        ARRAY_VAR,
        vec![("Value", Lit::Array(WireArray::I64(values.to_vec())).into())],
    );
    End::Node(n, "ArrayVarRef")
}

/// `array[index]`, read by an exec `Get` fired by `exec`.
fn get(g: &mut Graph, grid: graph::GridId, exec: End, array: End, index: End) -> graph::NodeId {
    g.exec(
        grid,
        exec,
        ARRAY_GET,
        vec![("ArrayVarRef", array.into()), ("Index", index.into())],
    )
}

/// `LineHeight` of the status text on the chip shell: large enough to read
/// from beside the screen, since the shell is a single 1x1 brick.
const STATUS_LINE_HEIGHT: f32 = 3.0;

/// The status text when a loop is cut short. `{0}` is the iterations it ran,
/// `{1}` how many it had to run and `{2}` the frame.
fn abort_message(what: &str) -> String {
    format!(
        "<color=\"f66\">Loop cut short at frame {{2}}: {{0}} of {{1}} {what}.</>\n\
         Turn Loop Gates off or use a smaller picture,\n\
         or increase Max Work in Callable Chip Limits in the gamemode settings."
    )
}

/// When `lp` completes, set the status variable behind `status` (a `VarRef`)
/// to [`abort_message`] if the server cut it short, and leave it as it is
/// otherwise; then add 1 to the variable behind `cut` if it was cut short.
/// The gates go in `grid`, where the loop's shell is. Returns the exec
/// output the next step chains from.
///
/// A `Select` picks the value rather than a `Branch` picking whether to set
/// it, so no exec path depends on which of a `Branch`'s outputs is true.
#[allow(clippy::too_many_arguments)]
fn report_abort(
    g: &mut Graph,
    grid: graph::GridId,
    lp: &graph::Loop,
    count: End,
    frame: End,
    status: End,
    cut: End,
    what: &str,
) -> End {
    let current = g.exec(
        grid,
        lp.completed(),
        "BrickComponentType_Exec_Var_Get",
        vec![("VarRef", status.clone().into())],
    );
    let message = g.gate(
        grid,
        "BrickComponentType_Expr_String_FormatText",
        vec![
            ("FormatString", Lit::Str(abort_message(what)).into()),
            ("InputA", End::Node(lp.shell, "IterationsExecuted").into()),
            ("InputB", count.into()),
            ("InputC", frame.into()),
        ],
    );
    let value = g.gate(
        grid,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", End::Node(current, "Value").into()),
            ("InputB", End::Node(message, "Output").into()),
            ("bSelectB", End::Node(lp.shell, "bAborted").into()),
        ],
    );
    let set = g.exec(
        grid,
        End::Node(current, "ExecOut"),
        VAR_SET,
        vec![
            ("VarRef", status.into()),
            ("Value", End::Node(value, "Output").into()),
        ],
    );
    let step = g.gate(
        grid,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", Lit::Int(0).into()),
            ("InputB", Lit::Int(1).into()),
            ("bSelectB", End::Node(lp.shell, "bAborted").into()),
        ],
    );
    let counted = g.exec(
        grid,
        End::Node(set, "ExecOut"),
        "BrickComponentType_Exec_Var_Increment",
        vec![
            ("VarRef", cut.into()),
            ("Value", End::Node(step, "Output").into()),
        ],
    );
    End::Node(counted, "ExecOut")
}

/// After a frame change, blank the status behind `status` if the change
/// replayed from a keyframe (`from_key`) and no loop was cut short (the
/// variable behind `cut` is still 0). Only a keyframe clears it: the deltas
/// between keyframes are small and always finish, so clearing on them would
/// show a cut-short keyframe's warning for a single frame. Raising the
/// server's limit therefore clears the warning at the next keyframe.
fn clear_status(g: &mut Graph, exec: End, from_key: End, status: End, cut: End) -> End {
    let cuts = g.exec(
        ROOT,
        exec,
        "BrickComponentType_Exec_Var_Get",
        vec![("VarRef", cut.into())],
    );
    let current = g.exec(
        ROOT,
        End::Node(cuts, "ExecOut"),
        "BrickComponentType_Exec_Var_Get",
        vec![("VarRef", status.clone().into())],
    );
    let none_cut = g.gate(
        ROOT,
        "BrickComponentType_Expr_CompareLessOrEqual",
        vec![
            ("InputA", End::Node(cuts, "Value").into()),
            ("InputB", Lit::Int(0).into()),
        ],
    );
    let clean = g.gate(
        ROOT,
        "BrickComponentType_Expr_LogicalAND",
        vec![
            ("bInputA", from_key.into()),
            ("bInputB", End::Node(none_cut, "bOutput").into()),
        ],
    );
    let value = g.gate(
        ROOT,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", End::Node(current, "Value").into()),
            ("InputB", Lit::Str(String::new()).into()),
            ("bSelectB", End::Node(clean, "bOutput").into()),
        ],
    );
    let set = g.exec(
        ROOT,
        End::Node(current, "ExecOut"),
        VAR_SET,
        vec![
            ("VarRef", status.into()),
            ("Value", End::Node(value, "Output").into()),
        ],
    );
    End::Node(set, "ExecOut")
}

/// Build the player graph for `packed`.
pub fn player_graph(packed: &PackedRecords, ports: PlayerPorts) -> Player {
    let mut g = Graph::new();
    let null_ref = || Lit::ComponentRef(graph::NULL_COMPONENT_REF);
    // The status text, and how many loops were cut short in the current
    // frame change.
    let status = ports
        .status
        .as_ref()
        .map(|_| g.gate(ROOT, VAR, vec![("Value", Lit::Str(String::new()).into())]));
    let cut = status.map(|_| g.gate(ROOT, VAR, vec![("Value", Lit::Int(0).into())]));

    // `apply_frame(f, pixels, a_banks, b_banks)`: look up which bank holds
    // the frame's entries, fetch that bank's two arrays through the bank
    // refs, and loop over the frame's entries in them.
    let func = g.function(
        ROOT,
        &[
            ("Frame", Lit::Int(0)),
            ("Pixels", null_ref()),
            ("A Banks", null_ref()),
            ("B Banks", null_ref()),
            ("Status", null_ref()),
            ("Cut", null_ref()),
        ],
    );
    let fp = func.plane;
    // Inside the function the status and the cut count are reached through
    // their `VarRef` parameters.
    let report = |g: &mut Graph, grid, lp: &graph::Loop, count: End, frame: End, what: &str| {
        if status.is_some() {
            report_abort(g, grid, lp, count, frame, func.input(4), func.input(5), what);
        }
    };
    let (f, pixels) = (func.input(0), func.input(1));
    let frame_bank = i64_array(&mut g, fp, &packed.frame_bank);
    let frame_first = i64_array(&mut g, fp, &packed.frame_first);
    let frame_entries = i64_array(&mut g, fp, &packed.frame_entries);
    let bank = get(&mut g, fp, func.body(), frame_bank, f.clone());
    let first = get(
        &mut g,
        fp,
        End::Node(bank, "ExecOut"),
        frame_first,
        f.clone(),
    );
    let count = get(&mut g, fp, End::Node(first, "ExecOut"), frame_entries, f.clone());
    let a = get(
        &mut g,
        fp,
        End::Node(count, "ExecOut"),
        func.input(2),
        End::Node(bank, "Value"),
    );
    let b = get(
        &mut g,
        fp,
        End::Node(a, "ExecOut"),
        func.input(3),
        End::Node(bank, "Value"),
    );
    let outer = g.count_for(
        fp,
        End::Node(b, "ExecOut"),
        End::Node(first, "Value"),
        End::Node(count, "Value"),
    );
    let outer_what = match packed.encoding {
        RecordEncoding::Hex => "record strings",
        RecordEncoding::ColorArray => "records",
    };
    report(&mut g, fp, &outer, End::Node(count, "Value"), f.clone(), outer_what);
    let lp = outer.plane;
    let (a_bank, b_bank) = (End::Node(a, "Value"), End::Node(b, "Value"));
    match packed.encoding {
        RecordEncoding::Hex if packed.dense => {
            // A string's record count is negative when it is dense: no index,
            // and record `j` paints pixel `(string - first) * per_chunk + j`.
            let gs = get(&mut g, lp, outer.body(), a_bank, outer.index());
            let gn = get(&mut g, lp, End::Node(gs, "ExecOut"), b_bank, outer.index());
            let n = End::Node(gn, "Value");
            let math = |g: &mut Graph, grid: graph::GridId, class: &'static str, a: In, b: In| {
                let node = g.gate(grid, class, vec![("InputA", a), ("InputB", b)]);
                End::Node(node, "Output")
            };
            let negated = math(
                &mut g,
                lp,
                "BrickComponentType_Expr_MathSubtract",
                Lit::Int(0).into(),
                n.clone().into(),
            );
            let records = math(
                &mut g,
                lp,
                "BrickComponentType_Expr_MathMax",
                n.clone().into(),
                negated.into(),
            );
            let dense = g.gate(
                lp,
                "BrickComponentType_Expr_CompareLess",
                vec![("InputA", n.into()), ("InputB", Lit::Int(0).into())],
            );
            let dense = End::Node(dense, "bOutput");
            let nth = math(
                &mut g,
                lp,
                "BrickComponentType_Expr_MathSubtract",
                outer.index().into(),
                End::Node(first, "Value").into(),
            );
            let base = math(
                &mut g,
                lp,
                "BrickComponentType_Expr_MathMultiply",
                nth.into(),
                Lit::Int(packed.dense_per_chunk as i64).into(),
            );
            let inner = g.count_for(lp, End::Node(gn, "ExecOut"), 0, records.clone());
            report(&mut g, lp, &inner, records, f.clone(), "records");
            let ip = inner.plane;
            let w = packed.index_width as i64;
            let select = |g: &mut Graph, a: In, b: In| {
                let node = g.gate(
                    ip,
                    "BrickComponentType_Expr_Select",
                    vec![
                        ("InputA", a),
                        ("InputB", b),
                        ("bSelectB", dense.clone().into()),
                    ],
                );
                End::Node(node, "Output")
            };
            let stride = select(
                &mut g,
                Lit::Int(hex_stride(packed.index_width) as i64).into(),
                Lit::Int(6).into(),
            );
            let index_chars = select(&mut g, Lit::Int(w).into(), Lit::Int(0).into());
            let at = math(
                &mut g,
                ip,
                "BrickComponentType_Expr_MathMultiply",
                inner.index().into(),
                stride.into(),
            );
            let text = End::Node(gs, "Value");
            let idx_text = g.gate(
                ip,
                "BrickComponentType_Expr_String_Substring",
                vec![
                    ("Input", text.clone().into()),
                    ("Start", at.clone().into()),
                    ("Length", Lit::Int(w).into()),
                ],
            );
            let parsed = g.gate(
                ip,
                "BrickComponentType_Expr_String_ParseInt",
                vec![("Input", End::Node(idx_text, "Output").into())],
            );
            let implied = math(
                &mut g,
                ip,
                "BrickComponentType_Expr_MathAdd",
                base.into(),
                inner.index().into(),
            );
            let idx = select(&mut g, End::Node(parsed, "Value").into(), implied.into());
            let hex_at = math(
                &mut g,
                ip,
                "BrickComponentType_Expr_MathAdd",
                at.into(),
                index_chars.into(),
            );
            let hex = g.gate(
                ip,
                "BrickComponentType_Expr_String_Substring",
                vec![
                    ("Input", text.into()),
                    ("Start", hex_at.into()),
                    ("Length", Lit::Int(6).into()),
                ],
            );
            let color = g.gate(
                ip,
                "BrickComponentType_Expr_MakeColorHex",
                vec![("Hex", End::Node(hex, "Output").into())],
            );
            paint(
                &mut g,
                ip,
                inner.body(),
                pixels,
                idx,
                End::Node(color, "Output"),
            );
        }
        RecordEncoding::Hex => {
            let gs = get(&mut g, lp, outer.body(), a_bank, outer.index());
            let gn = get(&mut g, lp, End::Node(gs, "ExecOut"), b_bank, outer.index());
            let inner = g.count_for(lp, End::Node(gn, "ExecOut"), 0, End::Node(gn, "Value"));
            report(&mut g, lp, &inner, End::Node(gn, "Value"), f.clone(), "records");
            let ip = inner.plane;
            let w = packed.index_width as i64;
            let at = g.gate(
                ip,
                "BrickComponentType_Expr_MathMultiply",
                vec![
                    ("InputA", inner.index().into()),
                    (
                        "InputB",
                        Lit::Int(hex_stride(packed.index_width) as i64).into(),
                    ),
                ],
            );
            let text = End::Node(gs, "Value");
            let idx_text = g.gate(
                ip,
                "BrickComponentType_Expr_String_Substring",
                vec![
                    ("Input", text.clone().into()),
                    ("Start", End::Node(at, "Output").into()),
                    ("Length", Lit::Int(w).into()),
                ],
            );
            let idx = g.gate(
                ip,
                "BrickComponentType_Expr_String_ParseInt",
                vec![("Input", End::Node(idx_text, "Output").into())],
            );
            let hex_at = g.gate(
                ip,
                "BrickComponentType_Expr_MathAdd",
                vec![
                    ("InputA", End::Node(at, "Output").into()),
                    ("InputB", Lit::Int(w).into()),
                ],
            );
            let hex = g.gate(
                ip,
                "BrickComponentType_Expr_String_Substring",
                vec![
                    ("Input", text.into()),
                    ("Start", End::Node(hex_at, "Output").into()),
                    ("Length", Lit::Int(6).into()),
                ],
            );
            let color = g.gate(
                ip,
                "BrickComponentType_Expr_MakeColorHex",
                vec![("Hex", End::Node(hex, "Output").into())],
            );
            paint(
                &mut g,
                ip,
                inner.body(),
                pixels,
                End::Node(idx, "Value"),
                End::Node(color, "Output"),
            );
        }
        RecordEncoding::ColorArray => {
            let gi = get(&mut g, lp, outer.body(), a_bank, outer.index());
            let gc = get(&mut g, lp, End::Node(gi, "ExecOut"), b_bank, outer.index());
            paint(
                &mut g,
                lp,
                End::Node(gc, "ExecOut"),
                pixels,
                End::Node(gi, "Value"),
                End::Node(gc, "Value"),
            );
        }
    }

    // The driver, in the screen chip: replay from the last frame shown, or
    // from the target's keyframe when that is later or playback went back.
    let target = ports.frame_index;
    let shown = g.gate(ROOT, VAR, vec![("Value", Lit::Int(-1).into())]);
    let seek = g.gate(
        ROOT,
        ARRAY_VAR,
        vec![(
            "Value",
            Lit::Array(WireArray::I64(packed.seek_from.clone())).into(),
        )],
    );
    let key = g.exec(
        ROOT,
        ports.on_change,
        ARRAY_GET,
        vec![
            ("ArrayVarRef", End::Node(seek, "ArrayVarRef").into()),
            ("Index", target.clone().into()),
        ],
    );
    let next = g.gate(
        ROOT,
        "BrickComponentType_Expr_MathAdd",
        vec![
            ("InputA", End::Node(shown, "Value").into()),
            ("InputB", 1.into()),
        ],
    );
    let ahead = g.gate(
        ROOT,
        "BrickComponentType_Expr_MathMax",
        vec![
            ("InputA", End::Node(next, "Output").into()),
            ("InputB", End::Node(key, "Value").into()),
        ],
    );
    let back = g.gate(
        ROOT,
        "BrickComponentType_Expr_CompareLess",
        vec![
            ("InputA", target.clone().into()),
            ("InputB", End::Node(shown, "Value").into()),
        ],
    );
    let start = g.gate(
        ROOT,
        "BrickComponentType_Expr_Select",
        vec![
            ("InputA", End::Node(ahead, "Output").into()),
            ("InputB", End::Node(key, "Value").into()),
            ("bSelectB", End::Node(back, "bOutput").into()),
        ],
    );
    let end = g.gate(
        ROOT,
        "BrickComponentType_Expr_MathAdd",
        vec![("InputA", target.clone().into()), ("InputB", 1.into())],
    );
    let count = g.gate(
        ROOT,
        "BrickComponentType_Expr_MathSubtract",
        vec![
            ("InputA", End::Node(end, "Output").into()),
            ("InputB", End::Node(start, "Output").into()),
        ],
    );
    // Each frame change counts its cut-short loops from 0.
    let replay = match cut {
        Some(cut) => {
            let reset = g.exec(
                ROOT,
                End::Node(key, "ExecOut"),
                VAR_SET,
                vec![
                    ("VarRef", End::Node(cut, "VarRef").into()),
                    ("Value", Lit::Int(0).into()),
                ],
            );
            End::Node(reset, "ExecOut")
        }
        None => End::Node(key, "ExecOut"),
    };
    let frames = g.count_for(
        ROOT,
        replay,
        End::Node(start, "Output"),
        End::Node(count, "Output"),
    );
    let mut args = vec![
        frames.index().into(),
        ports.pixels.into(),
        ports.a_banks.into(),
        ports.b_banks.into(),
    ];
    let mut after = frames.completed();
    if let (Some(status_var), Some(cut), Some(out)) = (status, cut, ports.status) {
        g.wire(End::Node(status_var, "Value"), out);
        let (status, cut) = (End::Node(status_var, "VarRef"), End::Node(cut, "VarRef"));
        args.push(status.clone().into());
        args.push(cut.clone().into());
        let reported = report_abort(
            &mut g,
            ROOT,
            &frames,
            End::Node(count, "Output"),
            target.clone(),
            status.clone(),
            cut.clone(),
            "frames",
        );
        // The replay began at a keyframe when it started at the target's
        // keyframe.
        let from_key = g.gate(
            ROOT,
            "BrickComponentType_Expr_CompareEqual",
            vec![
                ("InputA", End::Node(start, "Output").into()),
                ("InputB", End::Node(key, "Value").into()),
            ],
        );
        after = clear_status(&mut g, reported, End::Node(from_key, "bOutput"), status, cut);
    }
    g.call(frames.plane, &func, frames.body(), args);
    g.exec(
        ROOT,
        after,
        VAR_SET,
        vec![
            ("VarRef", End::Node(shown, "VarRef").into()),
            ("Value", target.into()),
        ],
    );
    Player {
        graph: g,
        shown,
        apply_frame: func,
        status,
    }
}

/// `pixels[index].Color = color`, fired by `exec`, in `grid`.
fn paint(g: &mut Graph, grid: graph::GridId, exec: End, pixels: End, index: End, color: End) {
    let get = g.exec(
        grid,
        exec,
        ARRAY_GET,
        vec![("ArrayVarRef", pixels.into()), ("Index", index.into())],
    );
    g.exec(
        grid,
        End::Node(get, "ExecOut"),
        SET_BRICK_PROPERTY,
        vec![
            ("Brick", End::Node(get, "Value").into()),
            ("Property", Lit::Int(BRICK_PROPERTY_COLOR).into()),
            ("Value", color.into()),
        ],
    );
}

/// Streams `source` into a ref-driven display-brick [`World`]. Same
/// cancellation contract as [`super::bricks::build_brick_world`].
pub fn build_ref_world(
    source: &dyn FrameSource,
    opts: &AnimOptions,
    encoding: RecordEncoding,
    progress: &mut dyn Progress,
) -> Result<World, String> {
    let info = source.info();
    let (w, h) = (info.width as i32, info.height as i32);

    FrameTotal::new(info.frame_count_hint, source.frame_count_estimate())
        .begin(progress, "packing frames");
    let mut packer = RecordPacker::new(info.width, info.height, opts.alpha_threshold, opts.records)
        .linearize(opts.srgb_to_linear);
    let mut first_frame: Option<RgbaImage> = None;
    let streamed: Result<(), String> = (|| {
        let mut stream = source.open()?;
        let mut seen = 0u64;
        while let Some(frame) = stream.next()? {
            packer.push_frame(&frame)?;
            progress.frame(frame.width(), frame.height(), frame.as_raw());
            if first_frame.is_none() {
                first_frame = Some(frame);
            }
            seen += 1;
            progress.tick(seen);
            if progress.is_cancelled() {
                break;
            }
        }
        Ok(())
    })();
    progress.finish();
    streamed?;
    if progress.is_cancelled() {
        return Ok(World::new());
    }
    let frame_count = packer.frame_count();
    if frame_count == 0 {
        return Err("clip has 0 frames: nothing to render (check --start/--duration against the source's length)".into());
    }
    let (frames, visible) = packer.finish();
    let packed = pack_records(
        &frames,
        &visible,
        encoding,
        MAX_COMPONENT_CHARS,
        opts.bank_size,
        opts.records.dense_keyframes,
    )?;
    let (fixed, per_record) = gate_runs(encoding, opts.records.dense_keyframes);
    let runs = |records: usize| fixed + per_record * records;
    log::info!(
        "refs: {} pixel(s), {} record(s) over {frame_count} frame(s); worst frame {} record(s) \
         (~{} gate runs), worst jump {} (~{} gate runs) (loop iterations in one tick); {} \
         keyframe(s)",
        packed.pixel_count,
        packed.total_records(),
        packed.worst_frame(),
        runs(packed.worst_frame()),
        packed.worst_catch_up(),
        runs(packed.worst_catch_up()),
        frames.iter().filter(|f| f.key).count(),
    );
    if runs(packed.worst_frame()) > super::cost::TICK_GATE_BUDGET {
        log::warn!(
            "the worst frame change runs ~{} gates in one tick, over the ~{} the game runs in \
             a tick before frames drop: turn Loop Gates off (--driver wires), use a smaller \
             picture, or use the color-array encoding",
            runs(packed.worst_frame()),
            super::cost::TICK_GATE_BUDGET
        );
    }
    if packed.longest_array() > opts.bank_size {
        return Err(format!(
            "the ref driver keeps each table in one array, and this render needs {} entries \
             (limit {}); shorten the clip or use --driver wires",
            packed.longest_array(),
            opts.bank_size
        ));
    }

    let mut world = World::new();
    world.meta.bundle.description = "Animation generated from image frames".to_string();

    // Display bricks start on frame 0's colours, so nothing flashes before
    // the first frame change.
    let (brick_of, geometry) = add_display_bricks(&mut world, opts, w, h, &visible);
    // `add_display_bricks` adds one brick per visible pixel, row-major, to a
    // fresh world: `world.bricks[k]` is the k-th visible pixel, the same
    // order `pack_records` numbers them in.
    let first = first_frame.expect("frame_count > 0");
    let visible_pixels: Vec<usize> = (0..visible.len()).filter(|&i| visible[i]).collect();
    debug_assert_eq!(brick_of.len(), visible_pixels.len());
    for (k, &pixel) in visible_pixels.iter().enumerate() {
        let brick = &mut world.bricks[k];
        debug_assert_eq!(brick.id, Some(brick_of[&pixel]));
        let p = &first.as_raw()[pixel * 4..][..4];
        brick.color = if p[3] < opts.alpha_threshold {
            (0u8, 0u8, 0u8).into()
        } else {
            (p[0], p[1], p[2]).into()
        };
        brick.components.clear();
    }
    let brick_index: Vec<usize> = (0..visible_pixels.len()).collect();

    let mut chip = new_screen_chip(&mut world, &geometry);
    let service = |col: i32, row: i32| {
        lattice_pos_staged(
            col,
            row,
            super::bricks::SERVICE_STAGE,
            h,
            GATE_HALF,
            STAGE_PITCH,
        )
    };

    // `pixels`, beside the chip shell on the main grid, with the display
    // bricks' refs in the absolute form.
    let shell_x = world
        .bricks
        .iter()
        .find(|b| b.id == Some(chip.brick_id))
        .map_or(0, |b| b.position.x);
    let array_at = Position {
        x: shell_x,
        y: -2 * super::layout::CELL,
        z: 2,
    };
    let refs = graph::main_grid_brick_refs(&world, &brick_index);
    let (array, array_id) = Brick {
        asset: BrickType::from("B_1x1_Gate_Variable_Array"),
        position: array_at,
        ..Default::default()
    }
    .with_component(LiteralComponent::new(ARRAY_VAR).with_data([(
        "Value",
        Box::new(WireArray::Bricks(refs)) as Box<dyn brdb::AsBrdbValue>,
    )]))
    .with_id_split();
    world.add_brick(array);
    // The record banks and the two arrays of refs to them, in a column below
    // `pixels`.
    let (a_arrays, b_arrays): (Vec<WireArray>, Vec<WireArray>) =
        packed.banks.iter().map(Bank::arrays).unzip();
    let column = |start: i32| {
        move |k: usize| Position {
            x: shell_x,
            y: -(start + k as i32) * super::layout::CELL,
            z: 2,
        }
    };
    // A long render's banks run to gigabytes, so they spread over chunks (a
    // chunk is one `.brz` blob, capped at 2 GiB).
    let a_banks =
        graph::ArrayOfArrays::add_spread(&mut world, a_arrays, column(3), opts.chunk_bytes);
    let b_banks = graph::ArrayOfArrays::add_spread(
        &mut world,
        b_arrays,
        column(4 + packed.banks.len() as i32),
        opts.chunk_bytes,
    );

    let (frame_index, control_pins) = if opts.external_clock {
        let pin = chip::add_input_pin(&mut chip, "Frame", service(0, -1));
        (chip::pin_source(pin, true), None)
    } else {
        let clock = clock::build_clock(
            &mut world,
            &mut chip,
            info.fps,
            frame_count,
            opts.loop_playback,
            service(0, -2),
        );
        let pins = (clock.pause_pin, clock.restart_pin, clock.resume_pin);
        (clock.frame_index, Some(pins))
    };
    let detector = clock::gate(
        &mut chip,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, -4),
        vec![],
    );
    world.add_wire_connection(
        frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );
    let on_change = WirePort::new(detector, CHANGE_DETECTOR, "OnChanged");

    // The status text rides on the chip shell, fed through an output pin:
    // blank until the server cuts a loop short, then a warning.
    let status_pin = chip::add_output_pin(&mut chip, "Status", service(1, -1));
    let label = crate::text::text_label_component(
        &mut world,
        String::new(),
        STATUS_LINE_HEIGHT,
        crate::text::FACE_Z_POSITIVE,
        // From the chip outward, line under line, rather than centred across
        // the record arrays beside it.
        crate::text::ANCHOR_TOP_LEFT,
        crate::text::OUTLINE_OUTLINED,
        crate::text::DEFAULT_OUTLINE_WIDTH,
        &crate::text::FontPreset::MonaspaceArgon.options(1.0),
    );
    world
        .bricks
        .iter_mut()
        .find(|b| b.id == Some(chip.brick_id))
        .expect("the chip shell is on the main grid")
        .add_component(label);
    world.add_wire_connection(
        chip::pin_source(status_pin, false),
        WirePort::new(chip.brick_id, super::text_refs::TEXT_DISPLAY, "Text"),
    );

    let player = player_graph(
        &packed,
        PlayerPorts {
            frame_index: End::ext(&frame_index),
            on_change: End::ext(&on_change),
            pixels: End::ext(&WirePort::new(array_id, ARRAY_VAR, "ArrayVarRef")),
            a_banks: a_banks.port(),
            b_banks: b_banks.port(),
            status: Some(End::ext(&chip::pin_target(status_pin, false))),
        },
    );
    let chip_origin = service(0, -6);
    player.graph.emit(
        &mut world,
        &mut chip,
        &EmitPlacement {
            root_origin: chip_origin,
            root_columns: 8,
        },
    )?;

    if let Some(subs) = &opts.subtitles {
        let per_frame = subs.per_frame(opts.source_start_s, info.fps as f64, frame_count)?;
        super::subtitle_display::add_subtitle_display(
            &mut world,
            &mut chip,
            super::subtitle_display::FrameIndex {
                index_of_bank: std::slice::from_ref(&frame_index),
                entry_of_bank: std::slice::from_ref(&on_change),
                ge: &[],
            },
            &per_frame,
            info.fps,
            opts,
            super::bricks::subtitle_extent(&geometry, w, h, opts.subtitle_lift)?,
        )?;
    }
    if let (true, Some((pause, restart, resume))) = (opts.control_buttons, control_pins) {
        let anchor = controls::control_anchor(&world);
        controls::add_control_buttons(&mut world, pause, restart, resume, anchor);
    }
    if progress.is_cancelled() {
        return Ok(World::new());
    }
    chip::finish(&mut world, chip)?;
    world.register_used_components();
    a_banks.fill(&mut world)?;
    b_banks.fill(&mut world)?;
    Ok(world)
}

/// The player's own `(gates, wires)`: fixed per encoding and keyframe form,
/// whatever the clip.
pub fn player_size(encoding: RecordEncoding, dense_keyframes: bool) -> (usize, usize) {
    let packed = pack_records(
        &[FrameRecords {
            key: true,
            records: vec![(0, [0, 0, 0, 255])],
        }],
        &[true],
        encoding,
        MAX_COMPONENT_CHARS,
        1,
        dense_keyframes,
    )
    .expect("one record fits a bank");
    let port = |p: &str| End::Ext(graph::ExtPort::new(0, "Ext", p));
    let player = player_graph(
        &packed,
        PlayerPorts {
            frame_index: port("Frame"),
            on_change: port("OnChanged"),
            pixels: port("Pixels"),
            a_banks: port("A Banks"),
            b_banks: port("B Banks"),
            status: Some(port("Status")),
        },
    );
    (player.graph.gate_count(), player.graph.wires.len())
}

/// Gate runs on one frame change of the player, as `(fixed, per_record)`: a
/// frame change that applies `n` records costs about `fixed + per_record * n`
/// gate runs, all in one tick.
///
/// Measured on the simulator rather than counted by hand, so it follows
/// whatever [`player_graph`] builds: one frame change applies a keyframe of
/// one record, another a keyframe of two, and the difference is one record.
/// Cached, because the GUI reads the estimate on every repaint.
pub fn gate_runs(encoding: RecordEncoding, dense_keyframes: bool) -> (usize, usize) {
    use std::sync::OnceLock;
    static MEASURED: [OnceLock<(usize, usize)>; 4] = [const { OnceLock::new() }; 4];
    let slot = match encoding {
        RecordEncoding::Hex => 0,
        RecordEncoding::ColorArray => 2,
    } + dense_keyframes as usize;
    *MEASURED[slot].get_or_init(|| {
        let one = frame_change_runs(encoding, dense_keyframes, 1);
        let two = frame_change_runs(encoding, dense_keyframes, 2);
        (2 * one - two, two - one)
    })
}

/// Simulated gate runs of the frame change to a keyframe of `n` records.
fn frame_change_runs(encoding: RecordEncoding, dense_keyframes: bool, n: usize) -> usize {
    use crate::graph::sim::{Sim, Val, array_vals};
    let port = |p: &str| graph::ExtPort::new(0, "Ext", p);
    let frame = FrameRecords {
        key: true,
        records: (0..n as u32).map(|i| (i, [1, 2, 3, 255])).collect(),
    };
    let packed = pack_records(
        &[frame.clone(), frame],
        &vec![true; n],
        encoding,
        MAX_COMPONENT_CHARS,
        2 * n,
        dense_keyframes,
    )
    .expect("two keyframes fit a bank");
    let player = player_graph(
        &packed,
        PlayerPorts {
            frame_index: End::Ext(port("Frame")),
            on_change: End::Ext(port("OnChanged")),
            pixels: End::Ext(port("Pixels")),
            a_banks: End::Ext(port("A Banks")),
            b_banks: End::Ext(port("B Banks")),
            status: Some(End::Ext(port("Status"))),
        },
    );
    let mut sim = Sim::new(&player.graph);
    let brick = |i: usize| brdb::schema::SavedBrickRef {
        grid_persistent_index: 0,
        chunk_index: brdb::ChunkIndex { x: 0, y: 0, z: 0 },
        brick_index_in_chunk: i as u32,
    };
    sim.bind_array(port("Pixels"), (0..n).map(|i| Val::Brick(brick(i))).collect());
    let (a, b): (Vec<Val>, Vec<Val>) = packed
        .banks
        .iter()
        .map(|bank| {
            let (a, b) = bank.arrays();
            (sim.new_array(array_vals(&a)), sim.new_array(array_vals(&b)))
        })
        .unzip();
    sim.bind_array(port("A Banks"), a);
    sim.bind_array(port("B Banks"), b);
    sim.bind(port("Frame"), Val::Int(1));
    sim.fire_ext(port("OnChanged"));
    assert_eq!(sim.brick_props.len(), n, "the frame change must paint every record");
    sim.gate_runs() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-record cost holds for a frame far bigger than the two it is
    /// measured on, in every encoding.
    #[test]
    fn gate_runs_grow_by_a_fixed_cost_per_record() {
        for encoding in [RecordEncoding::Hex, RecordEncoding::ColorArray] {
            for dense in [false, true] {
                let (fixed, per_record) = gate_runs(encoding, dense);
                assert!(per_record >= 2, "{encoding:?} dense {dense}: {per_record} per record");
                assert_eq!(
                    frame_change_runs(encoding, dense, 300),
                    fixed + 300 * per_record,
                    "{encoding:?} dense {dense}"
                );
            }
        }
    }
}
