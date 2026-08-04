//! The per-bank spillover cascade, shared by all three animation renderers.
//!
//! A clip longer than one `ArrayVar` can hold (`bank_size` frames) spills into
//! several banks. Given the frame index and `n_banks`, three parallel vectors
//! address those banks, and every renderer builds them identically -- so the
//! construction lives here once rather than copy-pasted per renderer:
//!
//! * `index_of_bank[k]`: the frame index rebased so bank `k`'s array is
//!   addressed from zero. Bank 0 is the raw frame index; bank `k >= 1` is a
//!   [`MathSubtract`](super::bricks::SUBTRACT) of `k * bank_size`.
//! * `ge[k - 1]`: a [`CompareGreaterOrEqual`](super::bricks::COMPARE_GE) that is
//!   true once the frame index reaches bank `k` -- exactly `Select`'s
//!   `bSelectB` sense (true picks `InputB`, the later bank).
//! * `entry_of_bank[k]`: the exec source that fires while bank `k` is live. A
//!   [`Branch`](super::bricks::BRANCH) cascade at the FRONT of the exec chain,
//!   so exactly one bank's chain runs and no exec input ever takes two sources
//!   (`ExecOutA` keeps descending, `ExecOutB` is "this bank").
//!
//! What hangs off these three is what differs per renderer, and stays in each
//! renderer: hex selects per CHUNK, colour-arrays per PIXEL, text per BAND.
//! Only the shared spine lives here.
//!
//! At `n_banks == 1` the cascade emits NO gate at all: `index_of_bank` is just
//! the frame index, `ge` is empty, and `entry_of_bank` is just the `exec_src`
//! passed in -- byte-structurally identical to the pre-spillover graph.
use super::bricks::{BRANCH, COMPARE_GE, SUBTRACT};
use super::chip::Chip;
use super::clock::gate;
use brdb::{AsBrdbValue, Position, WirePort, World, schema::WireVariant};

/// The three per-bank vectors every renderer feeds its own per-unit wiring
/// from. See the module documentation.
pub struct BankCascade {
    /// Per-bank frame index. One entry per bank.
    pub index_of_bank: Vec<WirePort>,
    /// Boundary comparators, `n_banks - 1` of them. `ge[k - 1]` is true from the
    /// first frame of bank `k` onward.
    pub ge: Vec<WirePort>,
    /// Per-bank exec entry. One entry per bank; the renderer's per-unit gets
    /// fan out from it (or, in hex mode, chain off it per chunk).
    pub entry_of_bank: Vec<WirePort>,
}

/// Build the per-bank index / comparator / branch spine.
///
/// `place(col, row)` positions a service gate at the given lattice column and
/// (negative) service row. The caller owns the stage and lattice height that
/// row maps into, so each renderer keeps its own service-row layout -- the
/// three gate classes go at rows `-6` ([`MathSubtract`](super::bricks::SUBTRACT)),
/// `-7` ([`CompareGreaterOrEqual`](super::bricks::COMPARE_GE)) and `-8`
/// ([`Branch`](super::bricks::BRANCH)), which is exactly where all three
/// renderers placed them inline.
///
/// `exec_src` is the exec output the branch cascade descends from -- the change
/// detector's `OnChanged` in every renderer. `frame_index` is the wrapped index
/// every bank's math and comparator reads.
///
/// `MathSubtract` is typed float and `Get.Index` takes an int, but both
/// operands are integral so the difference is exact -- the same coercion the
/// clock already relies on for `BitwiseOR |0`.
pub fn bank_cascade(
    world: &mut World,
    chip: &mut Chip,
    frame_index: &WirePort,
    exec_src: WirePort,
    n_banks: usize,
    bank_size: usize,
    place: impl Fn(i32, i32) -> Position,
) -> BankCascade {
    // Per-bank index. Bank 0 reads the frame index directly; bank k subtracts
    // k*bank_size so its own array is addressed from zero.
    let mut index_of_bank = Vec::with_capacity(n_banks);
    index_of_bank.push(frame_index.clone());
    for k in 1..n_banks {
        let sub = gate(chip, "B_1x1_Gate_Expr_MathSubtract", SUBTRACT,
            place(k as i32, -6), vec![(
                "InputB",
                Box::new(WireVariant::Number((k * bank_size) as f64)) as Box<dyn AsBrdbValue>,
            )]);
        world.add_wire_connection(frame_index.clone(), WirePort::new(sub, SUBTRACT, "InputA"));
        index_of_bank.push(WirePort::new(sub, SUBTRACT, "Output"));
    }

    // Boundary comparators. `ge[k-1]` is true once the frame index reaches
    // bank k, which is exactly `Select`'s `bSelectB` sense: true picks InputB,
    // the later bank.
    let mut ge = Vec::with_capacity(n_banks.saturating_sub(1));
    for k in 1..n_banks {
        let cmp = gate(chip, "B_1x1_Gate_Expr_CompareGreaterOrEqual", COMPARE_GE,
            place(k as i32, -7), vec![(
                "InputB",
                Box::new(WireVariant::Int((k * bank_size) as i64)) as Box<dyn AsBrdbValue>,
            )]);
        world.add_wire_connection(frame_index.clone(), WirePort::new(cmp, COMPARE_GE, "InputA"));
        ge.push(WirePort::new(cmp, COMPARE_GE, "bOutput"));
    }

    // Exec: branches cascade at the front, so exactly one bank's chain runs and
    // no exec input ever takes two sources. Branching per unit and rejoining
    // would require exec fan-in, which is untested here.
    //
    // With n_banks == 1 this emits no branch at all and `entry_of_bank[0]` is
    // simply the `exec_src` passed in.
    let mut entry_of_bank = Vec::with_capacity(n_banks);
    let mut exec_src = exec_src;
    for bi in 0..n_banks {
        if bi + 1 < n_banks {
            let br = gate(chip, "B_1x1_Gate_Exec_Branch", BRANCH,
                place(bi as i32, -8), vec![]);
            world.add_wire_connection(ge[bi].clone(), WirePort::new(br, BRANCH, "bCond"));
            world.add_wire_connection(exec_src, WirePort::new(br, BRANCH, "Exec"));
            // true -> keep descending; false -> this bank
            exec_src = WirePort::new(br, BRANCH, "ExecOutA");
            entry_of_bank.push(WirePort::new(br, BRANCH, "ExecOutB"));
        } else {
            entry_of_bank.push(exec_src.clone());
        }
    }

    BankCascade { index_of_bank, ge, entry_of_bank }
}
