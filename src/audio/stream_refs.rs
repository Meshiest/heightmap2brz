//! Audio bank / voice modes driven by component refs (`--driver refs`).
//!
//! The wired modes give every stream (a speaker's volume, and in voice mode
//! its pitch) an `ArrayVar`, a `Get` and a bank cascade, plus a master-volume
//! multiply per speaker. Here each speaker's values are one member of an
//! array of arrays on the main grid, a second array holds component refs to
//! the `AudioEmitter`s, and one loop over the speakers runs
//! `SetComponentProperty` for each (measured to take effect live).
//!
//! The loop runs when the frame changes and also when the master volume
//! changes, so pausing (which drops the master volume to 0 while the frame
//! index stands still) still mutes.

use crate::graph::{ARRAY_GET, End, Graph, Lit, ROOT, SET_COMPONENT_PROPERTY};

pub const AUDIO_EMITTER: &str = "Component_AudioEmitter";

pub struct StreamPorts {
    pub frame_index: End,
    /// The master volume (0 while paused).
    pub master: End,
    /// `ArrayVarRef` of the array of per-speaker volume arrays.
    pub volumes: End,
    /// `ArrayVarRef` of the array of per-speaker pitch arrays (voice mode).
    pub pitches: Option<End>,
    /// `ArrayVarRef` of the array of refs to the speakers' `AudioEmitter`s.
    pub emitters: End,
}

fn set(
    g: &mut Graph,
    grid: crate::graph::GridId,
    exec: End,
    emitter: End,
    property: &str,
    value: End,
) -> usize {
    g.exec(
        grid,
        exec,
        SET_COMPONENT_PROPERTY,
        vec![
            ("Component", emitter.into()),
            ("ComponentType", Lit::ComponentClass(AUDIO_EMITTER).into()),
            ("PropertyName", Lit::Str(property.into()).into()),
            ("Value", value.into()),
        ],
    )
}

/// One loop over `speakers` speakers, setting each one's volume (frame value
/// times master) and, with `pitches`, its pitch.
pub fn stream_loop_graph(speakers: usize, ports: StreamPorts) -> Graph {
    let mut g = Graph::new();
    let on_frame = g.gate(
        ROOT,
        "BrickComponentType_Expr_ChangeDetectorExec",
        vec![("Input", ports.frame_index.clone().into())],
    );
    let on_master = g.gate(
        ROOT,
        "BrickComponentType_Expr_ChangeDetectorExec",
        vec![("Input", ports.master.clone().into())],
    );
    let either = g.gate(ROOT, "BrickComponentType_Exec_Union", vec![]);
    g.wire(End::Node(on_frame, "OnChanged"), End::Node(either, "ExecA"));
    g.wire(
        End::Node(on_master, "OnChanged"),
        End::Node(either, "ExecB"),
    );
    let lp = g.count_for(ROOT, End::Node(either, "ExecOut"), 0, speakers as i64);
    let p = lp.plane;
    let emitter = g.exec(
        p,
        lp.body(),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", ports.emitters.into()),
            ("Index", lp.index().into()),
        ],
    );
    let values = g.exec(
        p,
        End::Node(emitter, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", ports.volumes.into()),
            ("Index", lp.index().into()),
        ],
    );
    let value = g.exec(
        p,
        End::Node(values, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", End::Node(values, "Value").into()),
            ("Index", ports.frame_index.clone().into()),
        ],
    );
    let volume = g.gate(
        p,
        "BrickComponentType_Expr_MathMultiply",
        vec![
            ("InputA", End::Node(value, "Value").into()),
            ("InputB", ports.master.into()),
        ],
    );
    let done = set(
        &mut g,
        p,
        End::Node(value, "ExecOut"),
        End::Node(emitter, "Value"),
        "VolumeMultiplier",
        End::Node(volume, "Output"),
    );
    if let Some(pitches) = ports.pitches {
        let values = g.exec(
            p,
            End::Node(done, "ExecOut"),
            ARRAY_GET,
            vec![
                ("ArrayVarRef", pitches.into()),
                ("Index", lp.index().into()),
            ],
        );
        let pitch = g.exec(
            p,
            End::Node(values, "ExecOut"),
            ARRAY_GET,
            vec![
                ("ArrayVarRef", End::Node(values, "Value").into()),
                ("Index", ports.frame_index.into()),
            ],
        );
        set(
            &mut g,
            p,
            End::Node(pitch, "ExecOut"),
            End::Node(emitter, "Value"),
            "PitchMultiplier",
            End::Node(pitch, "Value"),
        );
    }
    g
}
