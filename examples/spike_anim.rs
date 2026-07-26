//! Phase-1 de-risking spike: a 2x2-pixel, 3-frame animation, built by hand.
//! Exists to answer four questions that source cannot: does the chip I/O
//! convention work, does 3D gate layering survive a paste, does remote-wire
//! fan-out across the chip boundary hold, and is the array limit real.
use brdb::{
    AsBrdbValue, Brick, IntVector, Position, Vector3f, WirePort, World,
    assets::LiteralComponent,
    schema::WireArrayVariant,
};
use heightmap::anim::{
    chip,
    clock::{build_clock, gate},
    layout::{GATE_HALF, lattice_pos},
};

const SUBSTRING: &str = "BrickComponentType_WireGraph_Expr_String_Substring";
const MAKE_COLOR_HEX: &str = "BrickComponentType_WireGraph_Expr_MakeColorHex";
const ARRAY_VAR: &str = "BrickComponentType_WireGraphPseudo_ArrayVar";
const ARRAY_GET: &str = "BrickComponentType_WireGraph_Exec_ArrayVar_Get";
const CHANGE_DETECTOR: &str = "BrickComponentType_WireGraph_Expr_ChangeDetectorExec";
const PROP_CHANGER: &str = "Component_BrickPropertyChanger";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Pixel order is row-major: (0,0) (1,0) (0,1) (1,1).
    // Frame 2 gives every pixel a distinct color, so a transposed or mirrored
    // axis mapping is obvious on sight.
    let frames = vec![
        "FF0000FF0000FF0000FF0000".to_string(), // all red
        "00FF0000FF0000FF0000FF00".to_string(), // all green
        "FF000000FF000000FFFFFFFF".to_string(), // red, green, blue, white
    ];
    let (w, h) = (2i32, 2i32);

    let mut world = World::new();
    world.meta.bundle.description = "Animation spike".to_string();

    // Display bricks, spaced so each is individually visible.
    let mut brick_ids = Vec::new();
    for row in 0..h {
        for col in 0..w {
            let (brick, id) = Brick {
                position: Position { x: col * 10, y: row * 10, z: 2 },
                ..Default::default()
            }
            .with_component(LiteralComponent::new(PROP_CHANGER))
            .with_id_split();
            world.add_brick(brick);
            brick_ids.push(id);
        }
    }

    // The chip shell sits BESIDE the screen, not on it. A default display
    // brick is half-extent 5,5,6 (z span [-4,8] at z=2) while the shell is
    // half-extent 5,5,2 — putting the shell at z=6 over the origin pixel
    // overlapped both, and the game silently dropped one, leaving a 3-brick
    // L with unconnectable wires. One clear cell of -X keeps them disjoint.
    // `chip::finish` now fully replaces this with the true extent it
    // recomputes from every placed brick, so a generous value here buys
    // nothing -- only a legal, non-degenerate placeholder is needed. See
    // `chip::new_chip`'s docs.
    let mut c = chip::new_chip(
        &mut world,
        Position { x: -20, y: 0, z: 2 },
        Vector3f { x: 0.0, y: 0.0, z: 40.0 },
        IntVector { x: 5, y: 5, z: 5 },
    );
    let service = |col: i32, row: i32| lattice_pos(col, row, 2, h, GATE_HALF);

    let clock = build_clock(&mut world, &mut c, 2.0, frames.len(), service(0, -2));
    let detector = gate(
        &mut c,
        "B_1x1_Gate_Expr_ChangeDetectorExec",
        CHANGE_DETECTOR,
        service(0, 0),
        vec![],
    );
    world.add_wire_connection(
        clock.frame_index.clone(),
        WirePort::new(detector, CHANGE_DETECTOR, "Input"),
    );

    let array = gate(
        &mut c,
        "B_1x1_Gate_Variable_Array",
        ARRAY_VAR,
        service(0, 1),
        vec![(
            "Value",
            Box::new(WireArrayVariant::StringArray(frames)) as Box<dyn AsBrdbValue>,
        )],
    );
    let get = gate(&mut c, "B_1x1_Gate_Exec_ArrayVar_Get", ARRAY_GET, service(1, 1), vec![]);
    world.add_wire_connection(
        WirePort::new(array, ARRAY_VAR, "ArrayVarRef"),
        WirePort::new(get, ARRAY_GET, "ArrayVarRef"),
    );
    world.add_wire_connection(clock.frame_index.clone(), WirePort::new(get, ARRAY_GET, "Index"));
    world.add_wire_connection(
        WirePort::new(detector, CHANGE_DETECTOR, "OnChanged"),
        WirePort::new(get, ARRAY_GET, "Exec"),
    );

    for (i, &brick_id) in brick_ids.iter().enumerate() {
        let (col, row) = (i as i32 % w, i as i32 / w);
        let sub = gate(
            &mut c,
            "B_1x1_Gate_Expr_String_Substring",
            SUBSTRING,
            lattice_pos(col, row, 1, h, GATE_HALF),
            vec![
                ("Start", Box::new(i as i64 * 6) as Box<dyn AsBrdbValue>),
                ("Length", Box::new(6i64) as Box<dyn AsBrdbValue>),
            ],
        );
        let mk = gate(
            &mut c,
            "B_1x1_Gate_Expr_MakeColorHex",
            MAKE_COLOR_HEX,
            lattice_pos(col, row, 0, h, GATE_HALF),
            vec![],
        );
        world.add_wire_connection(
            WirePort::new(get, ARRAY_GET, "Value"),
            WirePort::new(sub, SUBSTRING, "Input"),
        );
        world.add_wire_connection(
            WirePort::new(sub, SUBSTRING, "Output"),
            WirePort::new(mk, MAKE_COLOR_HEX, "Hex"),
        );
        world.add_wire_connection(
            WirePort::new(mk, MAKE_COLOR_HEX, "Output"),
            WirePort::new(brick_id, PROP_CHANGER, "Color"),
        );
    }

    chip::finish(&mut world, c)?;
    world.register_used_components();
    std::fs::write("./spike_anim.brz", world.to_brz_vec()?)?;
    println!("wrote ./spike_anim.brz");
    Ok(())
}
