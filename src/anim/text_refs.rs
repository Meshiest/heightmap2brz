//! Text mode driven by component refs (`--driver refs --anim-mode text`).
//!
//! The wired text mode spends an `ArrayVar` and a `Get` per band and wires
//! each `Get` into its band's `TextDisplay`. Here every band's frame strings
//! are one member of an array of arrays on the main grid, a second array
//! holds component refs to the bands' `TextDisplay`s, and on a frame change
//! one `CountFor` over the bands runs
//! `SetComponentProperty(display[b], "Text", texts[b][frame])` (measured to
//! take effect live). The graph is the same size for any number of bands.

use crate::graph::{ARRAY_GET, End, Graph, Lit, ROOT, SET_COMPONENT_PROPERTY};

/// The `TextDisplay` component class, as `SetComponentProperty` names it.
pub const TEXT_DISPLAY: &str = "Component_TextDisplay";

pub struct TextLoopPorts {
    /// The frame index.
    pub frame_index: End,
    /// The exec pulse on a frame change.
    pub on_change: End,
    /// `ArrayVarRef` of the array of per-band string arrays.
    pub texts: End,
    /// `ArrayVarRef` of the array of refs to the bands' `TextDisplay`s.
    pub displays: End,
}

/// One loop over `bands` bands, setting each band's text for the frame.
pub fn text_loop_graph(bands: usize, ports: TextLoopPorts) -> Graph {
    let mut g = Graph::new();
    let lp = g.count_for(ROOT, ports.on_change, 0, bands as i64);
    let p = lp.plane;
    let band = g.exec(
        p,
        lp.body(),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", ports.texts.into()),
            ("Index", lp.index().into()),
        ],
    );
    let text = g.exec(
        p,
        End::Node(band, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", End::Node(band, "Value").into()),
            ("Index", ports.frame_index.into()),
        ],
    );
    let display = g.exec(
        p,
        End::Node(text, "ExecOut"),
        ARRAY_GET,
        vec![
            ("ArrayVarRef", ports.displays.into()),
            ("Index", lp.index().into()),
        ],
    );
    g.exec(
        p,
        End::Node(display, "ExecOut"),
        SET_COMPONENT_PROPERTY,
        vec![
            ("Component", End::Node(display, "Value").into()),
            ("ComponentType", Lit::ComponentClass(TEXT_DISPLAY).into()),
            ("PropertyName", Lit::Str("Text".into()).into()),
            ("Value", End::Node(text, "Value").into()),
        ],
    );
    g
}

/// Gate runs on one frame change of the text loop, as `(fixed, per_band)`:
/// a frame change costs about `fixed + per_band * bands` gate runs, all in
/// one tick. Measured on the simulator, as [`super::ref_player::gate_runs`]
/// is for the brick player, and cached the same way.
pub fn gate_runs() -> (usize, usize) {
    static MEASURED: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *MEASURED.get_or_init(measure_gate_runs)
}

fn measure_gate_runs() -> (usize, usize) {
    use crate::graph::ExtPort;
    use crate::graph::sim::{Sim, Val};
    use brdb::schema::{SavedBrickRef, SavedComponentRef};
    let port = |p: &str| ExtPort::new(0, "Ext", p);
    let runs = |bands: usize| {
        let g = text_loop_graph(
            bands,
            TextLoopPorts {
                frame_index: End::Ext(port("Frame")),
                on_change: End::Ext(port("OnChanged")),
                texts: End::Ext(port("Texts")),
                displays: End::Ext(port("Displays")),
            },
        );
        let mut sim = Sim::new(&g);
        let texts = (0..bands)
            .map(|_| sim.new_array(vec![Val::Str("a".into()), Val::Str("b".into())]))
            .collect();
        sim.bind_array(port("Texts"), texts);
        let displays = (0..bands)
            .map(|i| {
                Val::Component(SavedComponentRef {
                    brick: SavedBrickRef {
                        grid_persistent_index: 0,
                        chunk_index: brdb::ChunkIndex { x: 0, y: 0, z: 0 },
                        brick_index_in_chunk: i as u32,
                    },
                    component_type_index: 0,
                })
            })
            .collect();
        sim.bind_array(port("Displays"), displays);
        sim.bind(port("Frame"), Val::Int(1));
        sim.fire_ext(port("OnChanged"));
        sim.gate_runs() as usize
    };
    let (one, two) = (runs(1), runs(2));
    (2 * one - two, two - one)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each band is a Get of its strings, a Get of the frame's text, a Get
    /// of its display and the SetComponentProperty, plus what they read.
    #[test]
    fn a_band_costs_its_four_exec_gates_and_their_reads() {
        assert_eq!(gate_runs(), (1, 4));
    }
}
