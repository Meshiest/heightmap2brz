//! A gate graph with loop and function planes.
//!
//! The renderers in [`crate::anim`] place gates and wires straight into a
//! brdb [`World`]. That stops working once a graph has `for` loops and
//! functions: each loop or function is a microchip *plane*, a grid of its
//! own whose shell brick sits in the parent grid, and values cross a loop
//! wall only through `MicrochipInput` pins. [`Graph`] records gates, planes
//! and wires first, then [`Graph::emit`] lays them out and writes them. The
//! same graph runs offline in [`sim`], the only check of loop and function
//! wiring short of loading the save in game.
//!
//! The rules this module encodes, all from bearilog's emitter and in-game
//! notes:
//! - a loop or function shell needs its own brick asset AND component class;
//!   the game rejects an entry gate on a plain microchip;
//! - a value read inside a loop plane comes in through a `MicrochipInput`
//!   pin in that plane ([`Graph::bring`] adds them);
//! - nothing crosses a function plane's wall but the call's `Definition`
//!   wire: a body sees its arguments through `FunctionInput` gates;
//! - an input port takes at most one wire (fan-in fails the load).

pub mod sim;

use crate::anim::chip::{self, Chip};
use crate::anim::layout::{CELL, GATE_HALF};
use crate::wire_array::WireArray;
use brdb::{
    AsBrdbValue, BString, BrdbComponent, BrdbSchemaError, Brick, BrickSize, BrickType, ChunkIndex,
    IntVector, Position, WirePort, World,
    assets::components::{component, procedural_gate_for_component},
    schema::{
        BrdbInterned, BrdbSchema, BrdbSchemaStructProperty, BrdbValue, SavedBrickRef,
        SavedComponentRef, WireValue, as_brdb::BrdbArrayIter,
    },
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

pub type NodeId = usize;
pub type GridId = usize;

/// The host grid: an existing microchip's inner grid, passed to
/// [`Graph::emit`].
pub const ROOT: GridId = 0;

pub const COUNT_FOR_SHELL: &str = "Component_Internal_Microchip_CountFor";
pub const COUNT_FOR_ENTRY: &str = "BrickComponentType_CallableEntry_CountFor";
pub const FUNCTION_SHELL: &str = "Component_Internal_Microchip_FunctionDefinition";
pub const FUNCTION_ENTRY: &str = "BrickComponentType_CallableEntry_Function";
pub const FUNCTION_INPUT: &str = "BrickComponentType_FunctionInput";
pub const FUNCTION_OUTPUT: &str = "BrickComponentType_FunctionOutput";
pub const FUNCTION_CALL: &str = "BrickComponentType_FunctionCall";
pub const CALLABLE_BREAK: &str = "BrickComponentType_Exec_CallableBreak";
pub const VAR: &str = "BrickComponentType_Var";
pub const VAR_SET: &str = "BrickComponentType_Exec_Var_Set";
pub const ARRAY_VAR: &str = "BrickComponentType_ArrayVar";
pub const ARRAY_GET: &str = "BrickComponentType_Exec_ArrayVar_Get";
pub const SET_BRICK_PROPERTY: &str = "BrickComponentType_Exec_SetBrickProperty";
pub const SET_COMPONENT_PROPERTY: &str = "BrickComponentType_Exec_SetComponentProperty";

/// `EBRBrickProperty::Color`.
pub const BRICK_PROPERTY_COLOR: i64 = 1;

/// Which `Input<n>` pin of a Function Call feeds the `FunctionInput` whose
/// `PortOrder` is `slot`. bearilog found the order positional but left open
/// whether `PortOrder` or brick position decides it; the in-game probe
/// settles it, and this is the one place to change.
pub fn call_pin_for_slot(slot: usize) -> usize {
    slot
}

/// A constant value for a gate's data field. [`Graph::emit`] writes it in the
/// form the field's schema type needs (a `WireValue` port gets the matching
/// alternative, a native `i64` field a bare integer, and so on).
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    /// Linear `(R, G, B, A)`.
    Color(f32, f32, f32, f32),
    Array(WireArray),
    BrickRef(SavedBrickRef),
    ComponentRef(SavedComponentRef),
    /// A component class, for `ComponentType` asset fields
    /// (`BrickComponentTypeBase/<class>`).
    ComponentClass(&'static str),
}

/// The game's unset component ref.
pub const NULL_COMPONENT_REF: SavedComponentRef = SavedComponentRef {
    brick: SavedBrickRef {
        grid_persistent_index: 0,
        chunk_index: ChunkIndex { x: 0, y: 0, z: 0 },
        brick_index_in_chunk: u32::MAX,
    },
    component_type_index: 0,
};

/// One end of a wire: a port of a node in this graph, or a port of a brick
/// built outside it (the clock, a display brick). An external end counts
/// as living in [`ROOT`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum End {
    Node(NodeId, &'static str),
    Ext(ExtPort),
}

/// A port on a brick outside the graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExtPort {
    pub brick_id: usize,
    pub component: String,
    pub port: String,
}

impl ExtPort {
    pub fn new(brick_id: usize, component: &str, port: &str) -> Self {
        ExtPort {
            brick_id,
            component: component.to_owned(),
            port: port.to_owned(),
        }
    }
    fn wire_port(&self) -> WirePort {
        WirePort::new(self.brick_id, self.component.clone(), self.port.clone())
    }
}

impl From<&WirePort> for ExtPort {
    fn from(p: &WirePort) -> Self {
        ExtPort::new(
            p.brick_id,
            &p.component_type.to_string(),
            &p.port_name.to_string(),
        )
    }
}

impl End {
    /// An end on a brick outside the graph.
    pub fn ext(p: &WirePort) -> End {
        End::Ext(p.into())
    }
}

/// A gate input: a constant or a wire.
#[derive(Clone, Debug)]
pub enum In {
    Lit(Lit),
    Wire(End),
}

impl From<Lit> for In {
    fn from(l: Lit) -> Self {
        In::Lit(l)
    }
}
impl From<End> for In {
    fn from(e: End) -> Self {
        In::Wire(e)
    }
}
impl From<i64> for In {
    fn from(v: i64) -> Self {
        In::Lit(Lit::Int(v))
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub grid: GridId,
    pub class: &'static str,
    pub data: Vec<(&'static str, Lit)>,
    /// The plane this node is the shell of.
    pub plane: Option<GridId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridKind {
    Root,
    Loop,
    Function,
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub kind: GridKind,
    pub parent: Option<GridId>,
    pub shell: Option<NodeId>,
}

/// A `for` loop: its shell (driven by `Exec`, `Count`, `FirstIndex`, `Step`;
/// continues on `Completed`), its plane, and the entry gate whose `Body`
/// starts each iteration and whose `Index` is the loop variable.
#[derive(Clone, Copy, Debug)]
pub struct Loop {
    pub shell: NodeId,
    pub plane: GridId,
    pub entry: NodeId,
}

impl Loop {
    pub fn body(&self) -> End {
        End::Node(self.entry, "Body")
    }
    pub fn index(&self) -> End {
        End::Node(self.entry, "Index")
    }
    pub fn completed(&self) -> End {
        End::Node(self.shell, "Completed")
    }
}

/// A function definition: its shell, plane, entry gate and one
/// `FunctionInput` per parameter (by slot).
#[derive(Clone, Debug)]
pub struct Func {
    pub shell: NodeId,
    pub plane: GridId,
    pub entry: NodeId,
    pub inputs: Vec<NodeId>,
}

impl Func {
    pub fn body(&self) -> End {
        End::Node(self.entry, "Body")
    }
    pub fn input(&self, slot: usize) -> End {
        End::Node(self.inputs[slot], "Value")
    }
}

#[derive(Default)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub grids: Vec<Grid>,
    pub wires: Vec<(End, End)>,
    targets: HashSet<End>,
    pins: HashMap<(GridId, End), End>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            grids: vec![Grid {
                kind: GridKind::Root,
                parent: None,
                shell: None,
            }],
            ..Default::default()
        }
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    fn push(
        &mut self,
        grid: GridId,
        class: &'static str,
        data: Vec<(&'static str, Lit)>,
    ) -> NodeId {
        self.nodes.push(Node {
            grid,
            class,
            data,
            plane: None,
        });
        self.nodes.len() - 1
    }

    /// A gate in `grid`. Each input is a constant (written as data) or a
    /// wire, which [`Self::bring`] routes in through any loop walls.
    pub fn gate(
        &mut self,
        grid: GridId,
        class: &'static str,
        inputs: Vec<(&'static str, In)>,
    ) -> NodeId {
        let id = self.push(grid, class, Vec::new());
        for (port, input) in inputs {
            match input {
                In::Lit(l) => self.nodes[id].data.push((port, l)),
                In::Wire(src) => {
                    let src = self.bring(src, grid);
                    self.wire(src, End::Node(id, port));
                }
            }
        }
        id
    }

    /// An exec gate in `grid`, fired by `exec`.
    pub fn exec(
        &mut self,
        grid: GridId,
        exec: End,
        class: &'static str,
        inputs: Vec<(&'static str, In)>,
    ) -> NodeId {
        let id = self.gate(grid, class, inputs);
        self.wire(exec, End::Node(id, "Exec"));
        id
    }

    pub fn wire(&mut self, src: End, dst: End) {
        assert!(
            self.targets.insert(dst.clone()),
            "fan-in: {dst:?} already has a wire (the game rejects the load)"
        );
        self.wires.push((src, dst));
    }

    pub fn grid_of(&self, end: &End) -> GridId {
        match end {
            End::Node(n, _) => self.nodes[*n].grid,
            End::Ext(_) => ROOT,
        }
    }

    /// `src` as seen from `grid`: unchanged in its own grid, otherwise
    /// passed down through one `MicrochipInput` pin per loop plane between
    /// them. Panics on a function wall, which only arguments may cross.
    pub fn bring(&mut self, src: End, grid: GridId) -> End {
        let from = self.grid_of(&src);
        if from == grid {
            return src;
        }
        let mut path = Vec::new();
        let mut g = grid;
        while g != from {
            path.push(g);
            g = self.grids[g].parent.unwrap_or_else(|| {
                panic!("grid {grid} is not inside grid {from}, which {src:?} lives in")
            });
        }
        let mut value = src;
        for &plane in path.iter().rev() {
            assert_eq!(
                self.grids[plane].kind,
                GridKind::Loop,
                "only arguments cross a function wall; pass {value:?} as a FunctionInput"
            );
            value = self.pin_in(plane, value);
        }
        value
    }

    fn pin_in(&mut self, plane: GridId, src: End) -> End {
        if let Some(end) = self.pins.get(&(plane, src.clone())) {
            return end.clone();
        }
        let label = format!("in{}", self.pins.len());
        let pin = self.push(
            plane,
            chip::MICROCHIP_INPUT,
            vec![("PortLabel", Lit::Str(label))],
        );
        self.wire(src.clone(), End::Node(pin, "RER_Input"));
        let out = End::Node(pin, "RER_Output");
        self.pins.insert((plane, src), out.clone());
        out
    }

    fn plane(
        &mut self,
        grid: GridId,
        kind: GridKind,
        shell_class: &'static str,
        data: Vec<(&'static str, Lit)>,
    ) -> (NodeId, GridId) {
        let shell = self.push(grid, shell_class, data);
        self.grids.push(Grid {
            kind,
            parent: Some(grid),
            shell: Some(shell),
        });
        let plane = self.grids.len() - 1;
        self.nodes[shell].plane = Some(plane);
        (shell, plane)
    }

    /// A `CountFor` loop in `grid`, started by `exec`: `count` iterations
    /// with `Index = first + i`. Runs every iteration within one tick.
    pub fn count_for(
        &mut self,
        grid: GridId,
        exec: End,
        first: impl Into<In>,
        count: impl Into<In>,
    ) -> Loop {
        let (shell, plane) = self.plane(
            grid,
            GridKind::Loop,
            COUNT_FOR_SHELL,
            vec![("Step", Lit::Int(1))],
        );
        for (port, input) in [("FirstIndex", first.into()), ("Count", count.into())] {
            match input {
                In::Lit(l) => self.nodes[shell].data.push((port, l)),
                In::Wire(src) => {
                    let src = self.bring(src, grid);
                    self.wire(src, End::Node(shell, port));
                }
            }
        }
        self.wire(exec, End::Node(shell, "Exec"));
        let entry = self.push(plane, COUNT_FOR_ENTRY, Vec::new());
        Loop {
            shell,
            plane,
            entry,
        }
    }

    /// A function definition in `grid` with one `FunctionInput` per entry of
    /// `params`: `(label, default)`. The default fixes the parameter's type.
    pub fn function(&mut self, grid: GridId, params: &[(&str, Lit)]) -> Func {
        let (shell, plane) = self.plane(grid, GridKind::Function, FUNCTION_SHELL, Vec::new());
        let entry = self.push(plane, FUNCTION_ENTRY, Vec::new());
        let inputs = params
            .iter()
            .enumerate()
            .map(|(slot, (label, default))| {
                self.push(
                    plane,
                    FUNCTION_INPUT,
                    vec![
                        ("PortLabel", Lit::Str(label.to_string())),
                        ("PortOrder", Lit::Int(slot as i64)),
                        ("DefaultValue", default.clone()),
                        ("Value", default.clone()),
                    ],
                )
            })
            .collect();
        Func {
            shell,
            plane,
            entry,
            inputs,
        }
    }

    /// A call of `f` in `grid`, fired by `exec`, with one argument per
    /// parameter. Continue on `Completed`.
    pub fn call(&mut self, grid: GridId, f: &Func, exec: End, args: Vec<In>) -> NodeId {
        assert_eq!(args.len(), f.inputs.len(), "one argument per parameter");
        let call = self.push(grid, FUNCTION_CALL, Vec::new());
        // The definition wire is the one wire allowed across a plane wall.
        self.wire(
            End::Node(f.shell, "Definition"),
            End::Node(call, "Definition"),
        );
        for (slot, arg) in args.into_iter().enumerate() {
            let port = INPUT_PINS[call_pin_for_slot(slot)];
            match arg {
                In::Lit(l) => self.nodes[call].data.push((port, l)),
                In::Wire(src) => {
                    let src = self.bring(src, grid);
                    self.wire(src, End::Node(call, port));
                }
            }
        }
        self.wire(exec, End::Node(call, "Exec"));
        call
    }

    /// The nodes of `grid`, in creation order.
    pub fn nodes_in(&self, grid: GridId) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).filter(move |&n| self.nodes[n].grid == grid)
    }

    /// Every gate brick the graph emits, shells and pins included.
    pub fn gate_count(&self) -> usize {
        self.nodes.len()
    }
}

const INPUT_PINS: [&str; 8] = [
    "Input0", "Input1", "Input2", "Input3", "Input4", "Input5", "Input6", "Input7",
];

// --- emission ---------------------------------------------------------------

/// The schema type name of `class`'s data field `field`, `"[]"` for an array
/// field, or `None` when the field does not exist.
fn field_type(class: &str, field: &str) -> Option<&'static str> {
    static TABLE: OnceLock<HashMap<&'static str, HashMap<&'static str, &'static str>>> =
        OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let schema: &'static BrdbSchema = brdb::schemas::bricks_components_schema_max();
        let mut m = HashMap::new();
        for (comp, strct) in brdb::component_db::COMPONENT_TYPE_STRUCT_PAIRS {
            let Some(s) = schema.get_struct(strct) else {
                continue;
            };
            let fields = s
                .iter()
                .filter_map(|(k, prop)| {
                    let name = schema.intern.lookup_ref(*k)?;
                    let ty = match prop {
                        BrdbSchemaStructProperty::Type(t) => schema.intern.lookup_ref(*t)?,
                        _ => "[]",
                    };
                    Some((name, ty))
                })
                .collect();
            m.insert(*comp, fields);
        }
        m
    });
    table.get(class)?.get(field).copied()
}

fn is_enum(ty: &str) -> bool {
    brdb::schemas::bricks_components_schema_max()
        .get_enum(ty)
        .is_some()
}

/// `lit` in the form a field of schema type `ty` is written as.
fn boxed(
    world: &mut World,
    class: &str,
    field: &str,
    ty: &str,
    lit: &Lit,
) -> Result<Box<dyn AsBrdbValue>, String> {
    let bad = || format!("{class}.{field}: a {ty} field cannot hold {lit:?}");
    Ok(match (ty, lit) {
        ("WireValue", Lit::Bool(b)) => Box::new(WireValue::Bool(*b)),
        ("WireValue", Lit::Int(i)) => Box::new(WireValue::Int(*i)),
        ("WireValue", Lit::Num(n)) => Box::new(WireValue::Number(*n)),
        ("WireValue", Lit::Str(s)) => Box::new(WireValue::Str(s.clone())),
        ("WireValue", Lit::Color(r, g, b, a)) => Box::new(WireValue::LinearColor {
            r: *r,
            g: *g,
            b: *b,
            a: *a,
        }),
        ("WireValue", Lit::BrickRef(r)) => Box::new(WireValue::BrickRef(*r)),
        ("WireValue", Lit::ComponentRef(r)) => Box::new(WireValue::ComponentRef(*r)),
        ("bool", Lit::Bool(b)) => Box::new(*b),
        ("i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64", Lit::Int(i)) => Box::new(*i),
        ("f32" | "f64", Lit::Num(n)) => Box::new(*n),
        ("f32" | "f64", Lit::Int(i)) => Box::new(*i as f64),
        ("str", Lit::Str(s)) => Box::new(s.clone()),
        ("dynamic_array", Lit::Array(a)) => Box::new(a.clone()),
        ("BRSavedBrickRef", Lit::BrickRef(r)) => Box::new(*r),
        ("BRSavedComponentRef", Lit::ComponentRef(r)) => Box::new(*r),
        ("LinearColor", Lit::Color(r, g, b, a)) => Box::new((*r, *g, *b, *a)),
        ("object" | "class", Lit::ComponentClass(name)) => {
            let (index, _) = world
                .global_data
                .external_asset_references
                .insert_full(("BrickComponentTypeBase".into(), (*name).into()));
            Box::new(BrdbValue::Asset(Some(index)))
        }
        (ty, Lit::Int(i)) if is_enum(ty) => Box::new(*i as i32),
        _ => return Err(bad()),
    })
}

/// A component with scalar fields and `u8` array fields (`LiteralComponent`
/// cannot write array fields, and a Function Call's `InputSources` is one).
#[derive(Clone)]
struct GateComponent {
    class: BString,
    fields: Arc<HashMap<String, Box<dyn AsBrdbValue>>>,
    arrays: Arc<HashMap<String, Vec<u8>>>,
}

impl AsBrdbValue for GateComponent {
    fn has_brdb_struct_props(&self) -> bool {
        !self.fields.is_empty() || !self.arrays.is_empty()
    }
    fn has_brdb_struct_prop(
        &self,
        schema: &BrdbSchema,
        _s: BrdbInterned,
        prop: BrdbInterned,
    ) -> bool {
        prop.get(schema)
            .is_some_and(|p| self.fields.contains_key(p) || self.arrays.contains_key(p))
    }
    fn as_brdb_struct_prop_opt(
        &self,
        schema: &BrdbSchema,
        _s: BrdbInterned,
        prop: BrdbInterned,
    ) -> Result<Option<&dyn AsBrdbValue>, BrdbSchemaError> {
        Ok(prop
            .get(schema)
            .and_then(|p| self.fields.get(p))
            .map(|v| v.as_ref()))
    }
    fn as_brdb_struct_prop_value(
        &self,
        schema: &BrdbSchema,
        _s: BrdbInterned,
        prop: BrdbInterned,
    ) -> Result<&dyn AsBrdbValue, BrdbSchemaError> {
        let p = prop.get(schema).unwrap_or("");
        self.fields.get(p).map(|v| v.as_ref()).ok_or_else(|| {
            BrdbSchemaError::MissingStructField(self.class.to_string(), p.to_owned())
        })
    }
    fn as_brdb_struct_prop_array(
        &self,
        schema: &BrdbSchema,
        _s: BrdbInterned,
        prop: BrdbInterned,
    ) -> Result<BrdbArrayIter<'_>, BrdbSchemaError> {
        let p = prop.get(schema).unwrap_or("");
        match self.arrays.get(p) {
            Some(v) => Ok(Box::new(v.iter().map(|x| x as &dyn AsBrdbValue))),
            None => Err(BrdbSchemaError::MissingStructField(
                self.class.to_string(),
                p.to_owned(),
            )),
        }
    }
}

impl BrdbComponent for GateComponent {
    fn component_type(&self) -> Option<BString> {
        Some(self.class.clone())
    }
    fn component_type_ref(&self) -> Option<&str> {
        Some(self.class.as_ref())
    }
}

/// The brick a node is placed as, and its half-size.
fn brick_for(class: &'static str) -> Result<(BrickType, IntVector), String> {
    if let Some(gate) = procedural_gate_for_component(class) {
        let v = gate
            .variants
            .first()
            .ok_or_else(|| format!("{class}: no procedural sizes"))?;
        let [x, y, z] = v.half_size;
        let size = BrickSize {
            x: x as u16,
            y: y as u16,
            z: z as u16,
        };
        return Ok((
            BrickType::Procedural {
                asset: gate.asset.into(),
                size,
            },
            IntVector { x, y, z },
        ));
    }
    let info = component(class).ok_or_else(|| format!("{class} is not in the brdb catalog"))?;
    let asset = info
        .bricks
        .first()
        .ok_or_else(|| format!("{class} has no host brick"))?;
    // Not every gate is 1x1: Format String is 18 units long. A wrong size
    // here packs gates into each other, and the game drops the overlap.
    let half = brdb::assets::brick_sizes::brick_half_extent(asset)
        .map(|[x, y, z]| IntVector { x, y, z })
        .unwrap_or(GATE_HALF);
    Ok((BrickType::from(*asset), half))
}

/// Where emission puts things that are not in the graph.
pub struct EmitPlacement {
    /// Centre of the first root gate in the host chip; the rest follow in
    /// rows along +y, `root_columns` per row, stepping +x per row.
    pub root_origin: Position,
    pub root_columns: i32,
}

/// Lay `halves` out in rows of at most `columns` cells, returning centres
/// relative to a lattice whose first cell's min corner is the origin. A row
/// is as deep in x as its deepest brick, so a gate longer than one cell
/// never reaches into the next row.
fn pack_rows(halves: &[IntVector], columns: i32) -> Vec<Position> {
    let cells = |half: i32| (2 * half + CELL - 1) / CELL;
    let mut out = Vec::with_capacity(halves.len());
    let (mut x, mut depth, mut y) = (0, 1, 0);
    for h in halves {
        let wide = cells(h.y);
        if y > 0 && y + wide > columns {
            y = 0;
            x += depth;
            depth = 1;
        }
        depth = depth.max(cells(h.x));
        out.push(Position {
            x: x * CELL + h.x,
            y: y * CELL + h.y,
            z: h.z,
        });
        y += wide;
    }
    out
}

impl Graph {
    /// Write the graph: root nodes into `host`, each plane as a grid of its
    /// own linked to its shell. The caller still finishes `host`.
    pub fn emit(
        &self,
        world: &mut World,
        host: &mut Chip,
        place: &EmitPlacement,
    ) -> Result<Emitted, String> {
        let mut brick_ids = vec![usize::MAX; self.nodes.len()];
        let mut components: Vec<Option<GateComponent>> = Vec::with_capacity(self.nodes.len());
        for id in 0..self.nodes.len() {
            components.push(Some(self.component_for(world, id)?));
        }

        let mut planes: Vec<Option<Chip>> = (0..self.grids.len()).map(|_| None).collect();
        for grid in 0..self.grids.len() {
            let ids: Vec<NodeId> = self.nodes_in(grid).collect();
            let shapes: Vec<(BrickType, IntVector)> = ids
                .iter()
                .map(|&n| brick_for(self.nodes[n].class))
                .collect::<Result<_, _>>()?;
            let halves: Vec<IntVector> = shapes.iter().map(|(_, h)| *h).collect();
            let (origin, columns) = if grid == ROOT {
                (place.root_origin, place.root_columns)
            } else {
                (
                    Position {
                        x: CELL / 2,
                        y: CELL / 2,
                        // On the plane, as the game places gates: each
                        // gate's underside at the plane's own z of 0.
                        z: 0,
                    },
                    8,
                )
            };
            let positions = pack_rows(&halves, columns);
            let mut target = if grid == ROOT {
                None
            } else {
                let shell = self.grids[grid].shell.expect("a plane has a shell");
                let shell_brick = brick_ids[shell];
                assert_ne!(
                    shell_brick,
                    usize::MAX,
                    "a plane's shell is emitted before the plane"
                );
                Some(chip::new_plane(world, shell_brick))
            };
            for (k, &n) in ids.iter().enumerate() {
                let (asset, half) = shapes[k].clone();
                let rel = positions[k];
                let position = if grid == ROOT {
                    Position {
                        x: origin.x - GATE_HALF.x + rel.x,
                        y: origin.y - GATE_HALF.y + rel.y,
                        z: origin.z - GATE_HALF.z + rel.z,
                    }
                } else {
                    Position {
                        x: rel.x,
                        y: rel.y,
                        z: origin.z + rel.z,
                    }
                };
                let comp = components[n].take().expect("each node emits once");
                let brick = Brick {
                    asset,
                    position,
                    ..Default::default()
                }
                .with_component(comp);
                brick_ids[n] = match target.as_mut() {
                    Some(c) => c.add_brick(brick, half),
                    None => host.add_brick(brick, half),
                };
            }
            planes[grid] = target;
        }
        for plane in planes.into_iter().flatten() {
            chip::finish(world, plane)?;
        }

        let port = |end: &End| -> WirePort {
            match end {
                End::Node(n, p) => WirePort::new(brick_ids[*n], self.nodes[*n].class, *p),
                End::Ext(e) => e.wire_port(),
            }
        };
        for (src, dst) in &self.wires {
            world.add_wire_connection(port(src), port(dst));
        }
        Ok(Emitted { brick_ids })
    }

    fn component_for(&self, world: &mut World, id: NodeId) -> Result<GateComponent, String> {
        let node = &self.nodes[id];
        let mut fields: HashMap<String, Box<dyn AsBrdbValue>> = HashMap::new();
        for (field, lit) in &node.data {
            let ty = field_type(node.class, field)
                .ok_or_else(|| format!("{} has no data field {field}", node.class))?;
            fields.insert(field.to_string(), boxed(world, node.class, field, ty, lit)?);
        }
        let mut arrays = HashMap::new();
        if node.class == FUNCTION_CALL {
            // Every pin this call wires or sets overrides the definition's
            // default (`ECallableInputSource::Override`).
            let given: HashSet<&str> = node
                .data
                .iter()
                .map(|(f, _)| *f)
                .chain(self.wires.iter().filter_map(|(_, dst)| match dst {
                    End::Node(n, p) if *n == id => Some(*p),
                    _ => None,
                }))
                .collect();
            arrays.insert(
                "InputSources".to_string(),
                INPUT_PINS.iter().map(|p| given.contains(p) as u8).collect(),
            );
        }
        Ok(GateComponent {
            class: node.class.into(),
            fields: Arc::new(fields),
            arrays: Arc::new(arrays),
        })
    }
}

/// What [`Graph::emit`] produced: the brick id of every node.
pub struct Emitted {
    pub brick_ids: Vec<usize>,
}

/// The persistent index of the main grid in a save.
pub const MAIN_GRID: u32 = 1;

/// Brick refs to `world.bricks[i]` for each `i` in `indices` (main-grid
/// bricks), in the absolute form: the main grid's persistent index, the
/// brick's own chunk, and its index among the main-grid bricks before it in
/// that chunk.
///
/// Measured in game with a pasted prefab: this form resolves from an array
/// in any chunk. The game itself writes grid 0 with chunk (0,0,0) for a ref
/// within the array's own chunk, but neither sign of a chunk offset under
/// grid 0 resolved across chunks, and neither did grid 0 with the absolute
/// chunk.
///
/// Valid only while every brick up to the last of `indices` keeps its place:
/// the save numbers a chunk's bricks in `world.bricks` order.
pub fn main_grid_brick_refs(world: &World, indices: &[usize]) -> Vec<SavedBrickRef> {
    let last = indices.iter().copied().max().map_or(0, |m| m + 1);
    let mut seen: HashMap<ChunkIndex, u32> = HashMap::new();
    let mut slot = Vec::with_capacity(last);
    for brick in &world.bricks[..last] {
        let n = seen.entry(brick.position.to_relative().0).or_insert(0);
        slot.push(*n);
        *n += 1;
    }
    indices
        .iter()
        .map(|&i| SavedBrickRef {
            grid_persistent_index: MAIN_GRID,
            chunk_index: world.bricks[i].position.to_relative().0,
            brick_index_in_chunk: slot[i],
        })
        .collect()
}

/// Component refs to `(world.bricks[i], component type)` pairs, in the same
/// absolute form as [`main_grid_brick_refs`].
///
/// A ref's `component_type_index` is the type's position in the save's
/// component type table, which [`World::register_used_components`] rebuilds
/// in first-seen order. So call this after it, put the refs into a brick's
/// data without changing any brick's component *types*, and do not register
/// again.
pub fn main_grid_component_refs(
    world: &World,
    items: &[(usize, &str)],
) -> Result<Vec<SavedComponentRef>, String> {
    let indices: Vec<usize> = items.iter().map(|(i, _)| *i).collect();
    let bricks = main_grid_brick_refs(world, &indices);
    items
        .iter()
        .zip(bricks)
        .map(|((_, ty), brick)| {
            let component_type_index =
                world
                    .global_data
                    .get_component_type_index(ty)
                    .ok_or_else(|| {
                        format!("{ty} is not registered; call register_used_components first")
                    })?;
            Ok(SavedComponentRef {
                brick,
                component_type_index,
            })
        })
        .collect()
}

/// A `component_handle` `ArrayVar` on the main grid holding refs to
/// components on main-grid bricks: `(index in world.bricks, component type)`
/// per element.
///
/// The refs need the save's component type table, so they are written by
/// [`RefArray::fill`] after [`World::register_used_components`]; until then
/// the array is empty.
pub struct RefArray {
    at: usize,
    id: usize,
    targets: Vec<(usize, &'static str)>,
}

impl RefArray {
    pub fn add(world: &mut World, position: Position, targets: Vec<(usize, &'static str)>) -> Self {
        let (brick, id) = array_brick(position, WireArray::Components(Vec::new())).with_id_split();
        world.add_brick(brick);
        RefArray {
            at: world.bricks.len() - 1,
            id,
            targets,
        }
    }

    /// The array's `ArrayVarRef`.
    pub fn port(&self) -> End {
        End::Ext(ExtPort::new(self.id, ARRAY_VAR, "ArrayVarRef"))
    }

    /// Write the refs. Call after [`World::register_used_components`], and
    /// do not register again.
    pub fn fill(&self, world: &mut World) -> Result<(), String> {
        let refs = main_grid_component_refs(world, &self.targets)?;
        world.bricks[self.at].components = vec![Box::new(
            brdb::assets::LiteralComponent::new(ARRAY_VAR).with_data([(
                "Value",
                Box::new(WireArray::Components(refs)) as Box<dyn AsBrdbValue>,
            )]),
        )];
        Ok(())
    }
}

/// An array of arrays on the main grid: one `ArrayVar` per member array and
/// a [`RefArray`] of refs to them in order. Reading element `i` of the outer
/// array gives an `ArrayVarRef` that drives `Get` on member `i`.
pub struct ArrayOfArrays(RefArray);

/// Most array data, in [`WireArray::approx_bytes`], that
/// [`ArrayOfArrays::add`] puts in one main-grid chunk. A chunk's components
/// are one `.brz` blob, whose size is an `i32`, so a chunk must stay well
/// under 2 GiB; the refs player puts two arrays of arrays in the same chunks,
/// so each gets half of a safe 1 GiB.
pub const CHUNK_DATA_BYTES: usize = 512 << 20;

impl ArrayOfArrays {
    /// Add the member arrays and the outer array to the main grid, member
    /// `k` at `position(k)` and the outer one at `position(arrays.len())`,
    /// spread as [`Self::add_spread`] does at [`CHUNK_DATA_BYTES`].
    pub fn add(
        world: &mut World,
        arrays: Vec<WireArray>,
        position: impl Fn(usize) -> Position,
    ) -> Self {
        Self::add_spread(world, arrays, position, CHUNK_DATA_BYTES)
    }

    /// As [`Self::add`], but once the members placed so far in a chunk hold
    /// `limit` bytes, the next ones move along +x by a whole chunk. Members
    /// under the limit keep `position(k)` exactly. The refs to them are in
    /// the absolute form, which resolves across chunks.
    pub fn add_spread(
        world: &mut World,
        arrays: Vec<WireArray>,
        position: impl Fn(usize) -> Position,
        limit: usize,
    ) -> Self {
        let n = arrays.len();
        let (mut shift, mut held) = (0, 0usize);
        let members = arrays
            .into_iter()
            .enumerate()
            .map(|(k, a)| {
                let bytes = a.approx_bytes();
                if held > 0 && held + bytes > limit {
                    shift += 1;
                    held = 0;
                }
                held += bytes;
                let mut at = position(k);
                at.x += shift * brdb::CHUNK_SIZE;
                world.add_brick(array_brick(at, a));
                (world.bricks.len() - 1, ARRAY_VAR)
            })
            .collect();
        ArrayOfArrays(RefArray::add(world, position(n), members))
    }

    pub fn port(&self) -> End {
        self.0.port()
    }

    pub fn fill(&self, world: &mut World) -> Result<(), String> {
        self.0.fill(world)
    }
}

/// An `ArrayVar` brick holding `value`.
pub fn array_brick(position: Position, value: WireArray) -> Brick {
    Brick {
        asset: BrickType::from("B_1x1_Gate_Variable_Array"),
        position,
        ..Default::default()
    }
    .with_component(
        brdb::assets::LiteralComponent::new(ARRAY_VAR)
            .with_data([("Value", Box::new(value) as Box<dyn AsBrdbValue>)]),
    )
}
