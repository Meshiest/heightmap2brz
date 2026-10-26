//! MIDI playheads as functions (`--midi-functions`).
//!
//! The wired playhead ([`super::speakers::build_midi_event_world`]) builds
//! about 16 gates per speaker and advances at most one note per tick, so a
//! run of notes shorter than a tick falls behind. Here one shared function
//! definition holds the playhead logic and each speaker keeps only its note
//! array, its playhead `Var` and a Function Call:
//!
//! `voice_step(t, back, notes, playhead) -> (pitch, volume)`:
//! - `back` (time jumped backward: a restart or loop wrap) resets the
//!   playhead to 0;
//! - a `CountFor` with breaks advances past every note that has ended by `t`,
//!   however many that is (up to [`CATCH_UP`] per tick);
//! - the current note's pitch, and its volume while `start <= t <= end`
//!   (0 otherwise), come back as the call's outputs, wired to the emitter.
//!
//! Percussion lanes call `lane_step(t, back, hits, playhead)`, which advances
//! past every strike time reached; the playhead `Var`'s value stays wired to
//! the oneshot's `Play`, so a change fires one strike.

use crate::graph::{
    self, ARRAY_GET, ARRAY_VAR, CALLABLE_BREAK, End, FUNCTION_OUTPUT, Graph, GridId, Lit, ROOT,
    VAR, VAR_SET,
};
use crate::wire_array::WireArray;

/// A time no playback reaches. Every array ends in a sentinel at this time,
/// so a playhead never reads past the end: a function's gates are shared by
/// every call, and an out-of-bounds `Get` keeps whatever value the previous
/// call left in it, which would be another speaker's note.
pub const NEVER: f64 = 1.0e12;

/// Most notes one tick may skip past. Bounds the loop when something goes
/// wrong; a real tick passes a handful at most.
pub const CATCH_UP: i64 = 64;

const VAR_GET: &str = "BrickComponentType_Exec_Var_Get";
const VAR_INCREMENT: &str = "BrickComponentType_Exec_Var_Increment";
const BRANCH: &str = "BrickComponentType_Exec_Branch";
const UNION: &str = "BrickComponentType_Exec_Union";
const SPLIT_QUAT: &str = "BrickComponentType_Expr_SplitQuaternion";
const SELECT: &str = "BrickComponentType_Expr_Select";

/// One voice's notes, `(start, end, pitch, volume)`.
pub type Note = (f64, f64, f64, f64);

/// Where the playheads plug into the rest of the chip.
pub struct Ports {
    /// Playback time in seconds.
    pub time: End,
    /// True on the tick time went backward.
    pub back: End,
}

pub struct Voice {
    pub notes: Vec<Note>,
    /// Where the pitch and the (pre-master) volume go.
    pub pitch_to: End,
    pub volume_to: End,
}

pub struct Lane {
    pub hits: Vec<f64>,
    /// The oneshot's `Play` port.
    pub play_to: End,
}

/// The handles a test needs.
pub struct Playheads {
    pub graph: Graph,
    pub voice_calls: Vec<graph::NodeId>,
    pub voice_playheads: Vec<graph::NodeId>,
    pub lane_playheads: Vec<graph::NodeId>,
}

fn null_ref() -> Lit {
    Lit::ComponentRef(graph::NULL_COMPONENT_REF)
}

fn params() -> [(&'static str, Lit); 4] {
    [
        ("Time", Lit::Num(0.0)),
        ("Back", Lit::Bool(false)),
        ("Events", null_ref()),
        ("Playhead", null_ref()),
    ]
}

/// The shared prologue: reset on `back`, then a catch-up loop. Returns the
/// loop, whose body the caller fills, and the function.
fn prologue(g: &mut Graph) -> (graph::Func, graph::Loop) {
    let f = g.function(ROOT, &params());
    let fp = f.plane;
    let reset = g.exec(fp, f.body(), BRANCH, vec![("bCond", f.input(1).into())]);
    let zero = g.exec(
        fp,
        End::Node(reset, "ExecOutA"),
        VAR_SET,
        vec![("VarRef", f.input(3).into()), ("Value", Lit::Int(0).into())],
    );
    let join = g.gate(fp, UNION, vec![]);
    g.wire(End::Node(reset, "ExecOutB"), End::Node(join, "ExecA"));
    g.wire(End::Node(zero, "ExecOut"), End::Node(join, "ExecB"));
    let lp = g.count_for(fp, End::Node(join, "ExecOut"), 0, CATCH_UP);
    (f, lp)
}

/// In loop plane `lp` of `f`: read `events[playhead]`, break when out of
/// bounds or when `keep_going` (built from the read value) is false,
/// otherwise advance the playhead.
fn advance(
    g: &mut Graph,
    f: &graph::Func,
    lp: &graph::Loop,
    keep_going: impl FnOnce(&mut Graph, GridId, End) -> End,
) {
    let p = lp.plane;
    let at = g.exec(p, lp.body(), VAR_GET, vec![("VarRef", f.input(3).into())]);
    let get = g.exec(
        p,
        End::Node(at, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", f.input(2).into()),
            ("Index", End::Node(at, "Value").into()),
        ],
    );
    let oob = g.exec(
        p,
        End::Node(get, "ExecOut"),
        BRANCH,
        vec![("bCond", End::Node(get, "bOutOfBounds").into())],
    );
    g.exec(p, End::Node(oob, "ExecOutA"), CALLABLE_BREAK, vec![]);
    let cond = keep_going(g, p, End::Node(get, "Value"));
    let more = g.exec(
        p,
        End::Node(oob, "ExecOutB"),
        BRANCH,
        vec![("bCond", cond.into())],
    );
    g.exec(
        p,
        End::Node(more, "ExecOutA"),
        VAR_INCREMENT,
        vec![("VarRef", f.input(3).into()), ("Value", Lit::Int(1).into())],
    );
    g.exec(p, End::Node(more, "ExecOutB"), CALLABLE_BREAK, vec![]);
}

fn voice_function(g: &mut Graph) -> graph::Func {
    let (f, lp) = prologue(g);
    let t = f.input(0);
    // Keep going while the note at the playhead has ended: t > end.
    advance(g, &f, &lp, |g, p, value| {
        let split = g.gate(p, SPLIT_QUAT, vec![("Input", value.into())]);
        let past = g.gate(
            p,
            "BrickComponentType_Expr_CompareGreater",
            vec![
                ("InputA", t.clone().into()),
                ("InputB", End::Node(split, "Y").into()),
            ],
        );
        End::Node(past, "bOutput")
    });
    // Then read the current note for the outputs.
    let fp = f.plane;
    let at = g.exec(
        fp,
        lp.completed(),
        VAR_GET,
        vec![("VarRef", f.input(3).into())],
    );
    let get = g.exec(
        fp,
        End::Node(at, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", f.input(2).into()),
            ("Index", End::Node(at, "Value").into()),
        ],
    );
    let split = g.gate(
        fp,
        SPLIT_QUAT,
        vec![("Input", End::Node(get, "Value").into())],
    );
    let started = g.gate(
        fp,
        "BrickComponentType_Expr_CompareGreaterOrEqual",
        vec![
            ("InputA", f.input(0).into()),
            ("InputB", End::Node(split, "X").into()),
        ],
    );
    let not_ended = g.gate(
        fp,
        "BrickComponentType_Expr_CompareLessOrEqual",
        vec![
            ("InputA", f.input(0).into()),
            ("InputB", End::Node(split, "Y").into()),
        ],
    );
    let inner = g.gate(
        fp,
        SELECT,
        vec![
            ("InputA", Lit::Num(0.0).into()),
            ("InputB", End::Node(split, "W").into()),
            ("bSelectB", End::Node(not_ended, "bOutput").into()),
        ],
    );
    let volume = g.gate(
        fp,
        SELECT,
        vec![
            ("InputA", Lit::Num(0.0).into()),
            ("InputB", End::Node(inner, "Output").into()),
            ("bSelectB", End::Node(started, "bOutput").into()),
        ],
    );
    for (slot, (label, value)) in [
        ("Pitch", End::Node(split, "Z")),
        ("Volume", End::Node(volume, "Output")),
    ]
    .into_iter()
    .enumerate()
    {
        g.gate(
            fp,
            FUNCTION_OUTPUT,
            vec![
                ("PortLabel", Lit::Str(label.into()).into()),
                ("PortOrder", Lit::Int(slot as i64).into()),
                ("Value", value.into()),
            ],
        );
    }
    f
}

fn lane_function(g: &mut Graph) -> graph::Func {
    let (f, lp) = prologue(g);
    let t = f.input(0);
    // Keep going while the strike at the playhead has been reached.
    advance(g, &f, &lp, |g, p, value| {
        let reached = g.gate(
            p,
            "BrickComponentType_Expr_CompareGreaterOrEqual",
            vec![("InputA", t.into()), ("InputB", value.into())],
        );
        End::Node(reached, "bOutput")
    });
    f
}

const OUTPUTS: [&str; 8] = [
    "Output0", "Output1", "Output2", "Output3", "Output4", "Output5", "Output6", "Output7",
];

/// Build the playheads: a change detector on `ports.time`, then one call per
/// voice and per lane in a single exec chain.
pub fn playheads(voices: Vec<Voice>, lanes: Vec<Lane>, ports: Ports) -> Playheads {
    let mut g = Graph::new();
    let detector = g.gate(
        ROOT,
        "BrickComponentType_Expr_ChangeDetectorExec",
        vec![("Input", ports.time.clone().into())],
    );
    let mut exec = End::Node(detector, "OnChanged");
    let mut out = Playheads {
        graph: Graph::new(),
        voice_calls: Vec::new(),
        voice_playheads: Vec::new(),
        lane_playheads: Vec::new(),
    };

    if !voices.is_empty() {
        let f = voice_function(&mut g);
        for mut v in voices {
            // Silent, at the last pitch, so the speaker holds its last note's
            // pitch once the voice has finished.
            let last_pitch = v.notes.last().map_or(1.0, |n| n.2);
            v.notes.push((NEVER, NEVER, last_pitch, 0.0));
            let notes = g.gate(
                ROOT,
                ARRAY_VAR,
                vec![("Value", Lit::Array(WireArray::Quat(v.notes)).into())],
            );
            let head = g.gate(ROOT, VAR, vec![("Value", Lit::Int(0).into())]);
            let call = g.call(
                ROOT,
                &f,
                exec,
                vec![
                    ports.time.clone().into(),
                    ports.back.clone().into(),
                    End::Node(notes, "ArrayVarRef").into(),
                    End::Node(head, "VarRef").into(),
                ],
            );
            g.wire(
                End::Node(call, OUTPUTS[graph::call_pin_for_slot(0)]),
                v.pitch_to,
            );
            g.wire(
                End::Node(call, OUTPUTS[graph::call_pin_for_slot(1)]),
                v.volume_to,
            );
            exec = End::Node(call, "Completed");
            out.voice_calls.push(call);
            out.voice_playheads.push(head);
        }
    }
    if !lanes.is_empty() {
        let f = lane_function(&mut g);
        for mut l in lanes {
            l.hits.push(NEVER);
            let hits = g.gate(
                ROOT,
                ARRAY_VAR,
                vec![("Value", Lit::Array(WireArray::F64(l.hits)).into())],
            );
            let head = g.gate(ROOT, VAR, vec![("Value", Lit::Int(0).into())]);
            g.wire(End::Node(head, "Value"), l.play_to);
            let call = g.call(
                ROOT,
                &f,
                exec,
                vec![
                    ports.time.clone().into(),
                    ports.back.clone().into(),
                    End::Node(hits, "ArrayVarRef").into(),
                    End::Node(head, "VarRef").into(),
                ],
            );
            exec = End::Node(call, "Completed");
            out.lane_playheads.push(head);
        }
    }
    out.graph = g;
    out
}
