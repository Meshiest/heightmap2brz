use heightmap::audio::midi_functions::{CATCH_UP, Lane, Note, Ports, Voice, playheads};
use heightmap::graph::sim::{Sim, Val};
use heightmap::graph::{End, ExtPort};

fn ext(name: &str) -> ExtPort {
    ExtPort::new(usize::MAX - name.len(), "Ext", name)
}

/// What a speaker should play at `t`, with its playhead caught up past every
/// note that has ended.
struct Reference {
    notes: Vec<Note>,
    head: usize,
}

impl Reference {
    fn step(&mut self, t: f64, back: bool) -> (f64, f64) {
        if back {
            self.head = 0;
        }
        let mut n = 0;
        while self.head < self.notes.len() && t > self.notes[self.head].1 && n < CATCH_UP {
            self.head += 1;
            n += 1;
        }
        match self.notes.get(self.head) {
            Some(&(start, end, pitch, vol)) => {
                (pitch, if start <= t && t <= end { vol } else { 0.0 })
            }
            None => (self.notes.last().unwrap().2, 0.0),
        }
    }
}

fn voices() -> Vec<Vec<Note>> {
    // Voice 0: ordinary notes with gaps. Voice 1: a burst of notes far
    // shorter than a tick, which the wired playhead could only take one per
    // tick.
    let v0 = vec![
        (0.05, 0.4, 1.0, 0.8),
        (0.5, 0.9, 1.5, 0.6),
        (1.2, 1.6, 0.75, 1.0),
    ];
    let mut v1: Vec<Note> = (0..20)
        .map(|i| {
            (
                0.3 + i as f64 * 0.002,
                0.301 + i as f64 * 0.002,
                1.0 + i as f64 * 0.05,
                0.5,
            )
        })
        .collect();
    v1.push((1.0, 1.3, 2.0, 0.9));
    vec![v0, v1]
}

#[test]
fn every_tick_plays_what_the_caught_up_playhead_says() {
    let notes = voices();
    let hits = vec![0.1, 0.1005, 0.5, 1.0, 1.0001, 1.5];
    let sink = |name: &str| End::Ext(ext(name));
    let built = playheads(
        notes
            .iter()
            .enumerate()
            .map(|(i, n)| Voice {
                notes: n.clone(),
                pitch_to: sink(&format!("pitch{i}")),
                volume_to: sink(&format!("vol{i}")),
            })
            .collect(),
        vec![Lane {
            hits: hits.clone(),
            play_to: sink("play"),
        }],
        Ports {
            time: sink("Time"),
            back: sink("Back"),
        },
    );
    let mut sim = Sim::new(&built.graph);
    let mut reference: Vec<Reference> = notes
        .iter()
        .map(|n| Reference {
            notes: n.clone(),
            head: 0,
        })
        .collect();

    // 60 Hz ticks through the piece, a wrap back to the start, and on again.
    let mut times: Vec<f64> = (1..=110).map(|i| i as f64 / 60.0).collect();
    times.extend((0..60).map(|i| 0.02 + i as f64 / 60.0));
    // The load tick: the detector records t = 0 without firing, as the
    // emitters' baked first pitch at volume 0 covers it.
    sim.bind(ext("Time"), Val::Num(0.0));
    sim.bind(ext("Back"), Val::Bool(false));
    sim.tick();
    let mut prev = 0.0;
    let mut lane_ref = 0usize;
    let mut checked = 0;
    for t in times {
        let back = t < prev;
        prev = t;
        sim.bind(ext("Time"), Val::Num(t));
        sim.bind(ext("Back"), Val::Bool(back));
        sim.tick();
        for (v, call) in built.voice_calls.iter().enumerate() {
            let (pitch, vol) = reference[v].step(t, back);
            assert_eq!(
                sim.eval(&End::Node(*call, "Output0")),
                Val::Num(pitch),
                "voice {v} pitch at t={t}"
            );
            assert_eq!(
                sim.eval(&End::Node(*call, "Output1")),
                Val::Num(vol),
                "voice {v} volume at t={t}"
            );
            checked += 1;
        }
        if back {
            lane_ref = 0;
        }
        while lane_ref < hits.len() && t >= hits[lane_ref] {
            lane_ref += 1;
        }
        assert_eq!(
            sim.var(built.lane_playheads[0]),
            &Val::Int(lane_ref as i64),
            "lane strikes at t={t}"
        );
    }
    assert_eq!(checked, 2 * 170);
}

#[test]
fn a_tick_passes_a_whole_burst_of_short_notes() {
    let built = playheads(
        vec![Voice {
            notes: voices()[1].clone(),
            pitch_to: End::Ext(ext("p")),
            volume_to: End::Ext(ext("v")),
        }],
        vec![],
        Ports {
            time: End::Ext(ext("Time")),
            back: End::Ext(ext("Back")),
        },
    );
    let mut sim = Sim::new(&built.graph);
    sim.bind(ext("Back"), Val::Bool(false));
    sim.bind(ext("Time"), Val::Num(0.0));
    sim.tick();
    // One tick from before the burst to after it: all 20 notes pass at once.
    sim.bind(ext("Time"), Val::Num(0.35));
    sim.tick();
    assert_eq!(sim.var(built.voice_playheads[0]), &Val::Int(20));
}

#[test]
fn a_speaker_is_four_gates_whatever_the_note_count() {
    let count = |speakers: usize| {
        playheads(
            (0..speakers)
                .map(|i| Voice {
                    notes: voices()[0].clone(),
                    pitch_to: End::Ext(ext(&format!("p{i}"))),
                    volume_to: End::Ext(ext(&format!("v{i}"))),
                })
                .collect(),
            vec![],
            Ports {
                time: End::Ext(ext("Time")),
                back: End::Ext(ext("Back")),
            },
        )
        .graph
        .gate_count()
    };
    // Notes array, playhead Var and Function Call per speaker; the shared
    // function and detector are paid once. (The master-volume multiply is
    // built outside this graph.)
    assert_eq!(count(9) - count(8), 3);
}
