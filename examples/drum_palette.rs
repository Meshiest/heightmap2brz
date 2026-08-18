//! Generate the drum audition palette save.
//!
//! ```sh
//! cargo run --example drum_palette -- <out.brz>
//! ```
//!
//! Writes a labelled row of candidate oneshot drum sounds -- one `B_Button`
//! wired to one `B_1x1_SoundEmitter` per sound -- so the sounds can be
//! auditioned and tuned in game and the tuned save handed back. Green buttons
//! are the user's approved sounds; amber buttons are placeholder guesses to
//! replace. See [`heightmap::audio::percussion`].
//!
//! It writes a PREFAB bundle, so the `.brz` drops into `Saved/Prefabs` and is
//! spawned from the prefab browser.

use heightmap::audio::percussion::{PALETTE_ROLES, build_drum_palette};
use heightmap::util::write_world;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "drum_palette.brz".to_string());

    let mut world = build_drum_palette(PALETTE_ROLES);
    world.make_prefab();
    match write_world(&world, &out) {
        Ok(()) => {
            let approved = PALETTE_ROLES.iter().filter(|r| r.approved).count();
            println!(
                "wrote {out}: {} sounds ({approved} approved, {} placeholders to replace)",
                PALETTE_ROLES.len(),
                PALETTE_ROLES.len() - approved,
            );
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
