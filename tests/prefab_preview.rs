//! Each generated save carries the source map as its in-game preview.
//!
//! The game shows a grid of pictures, and a generated save has no render of
//! itself. The colormap, or the heightmap without `-c`, gives a view of the
//! build from above.
//!
//! **This applies to all saves, not only prefabs.** The preview was applied
//! only with `--prefab` before. Each GUI save is a world bundle and thus had
//! no preview. These tests use both bundle types.
//!
//! The tests read a `.brz` file and not the `WorldMeta` fields, because the
//! path in the archive controls what the game finds.

use brdb::IntoReader;
use heightmap::util::save_screenshot;

fn source_image() -> image::RgbaImage {
    image::RgbaImage::from_pixel(800, 400, image::Rgba([40, 90, 140, 255]))
}

fn save_with_preview(name: &str, prefab: bool) -> brdb::World {
    let mut world = brdb::World::new();
    world.add_bricks(vec![brdb::Brick {
        asset: brdb::BrickType::Procedural {
            asset: brdb::assets::bricks::PB_DEFAULT_BRICK,
            size: brdb::BrickSize::new(5, 5, 6),
        },
        position: brdb::Position::new(0, 0, 6),
        ..Default::default()
    }]);
    world.meta.screenshot = Some(save_screenshot(&source_image()).expect("encode the preview"));
    world.meta.bundle.name = name.to_string();
    if prefab {
        world.make_prefab();
    }
    world
}

fn written(world: &brdb::World, tag: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("h2b_preview_{tag}_{}.brz", std::process::id()));
    std::fs::write(&path, world.to_brz_vec().expect("encode")).expect("write");
    path
}

#[test]
fn a_prefab_bundle_carries_the_source_map_as_its_preview() {
    let path = written(&save_with_preview("Test Prefab", true), "prefab");
    let db = brdb::Brz::open(&path).expect("reopen").into_reader();

    assert!(
        db.prefab_json().expect("read").is_some(),
        "the bundle must be a prefab, or this test repeats the world case"
    );
    assert_eq!(
        db.bundle_json().expect("read").name,
        "Test Prefab",
        "the browser tile shows only the bundle name"
    );

    let screenshot = db
        .screenshot()
        .expect("read")
        .expect("Meta/Screenshot.jpg must be in the archive, because it is the preview");
    assert_eq!(&screenshot[..2], &[0xFF, 0xD8], "Screenshot.jpg must be a JPEG");
    let shot = image::load_from_memory(&screenshot).expect("the game must decode it");
    assert_eq!((shot.width(), shot.height()), (1280, 640), "the 2:1 shape of the map");

    let _ = std::fs::remove_file(&path);
}

/// The world bundle. The GUI writes this type only, and the CLI writes it
/// without `--prefab`.
#[test]
fn a_world_bundle_carries_one_too() {
    let path = written(&save_with_preview("Test World", false), "world");
    let db = brdb::Brz::open(&path).expect("reopen").into_reader();

    assert!(
        db.prefab_json().expect("read").is_none(),
        "this test uses the WORLD bundle"
    );
    assert!(
        db.screenshot().expect("read").is_some(),
        "a world save must also carry a preview, because the GUI writes this type"
    );
    assert_eq!(db.bundle_json().expect("read").name, "Test World");

    let _ = std::fs::remove_file(&path);
}
