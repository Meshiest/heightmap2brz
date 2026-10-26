//! Runs a [`Graph`] offline, as far as the gates this crate emits go.
//!
//! This checks wiring, not the game: the semantics below are the ones the
//! design relies on, and the in-game probes are what confirm them.
//! - Pure gates are evaluated when something reads them, so a loop body sees
//!   the current `Index` every iteration.
//! - An exec gate's outputs hold the value from its last run.
//! - A `CountFor` runs all its iterations inside the exec that starts it;
//!   `CallableBreak` stops the iterations after the current one.
//! - [`Sim::gate_limit`] models the server's per-tick gate budget: a loop
//!   about to pass it stops, aborted, and fires `Completed` as usual.
//! - A Function Call binds each `FunctionInput` by `PortOrder` through
//!   [`super::call_pin_for_slot`], the same mapping emission uses.
//! - A `ChangeDetectorExec` fires when its input differs from the value it
//!   saw on the previous tick. Whether the game also fires on the first
//!   tick is a [`Sim::fire_on_first_tick`] switch, so a design can be checked
//!   under both.
//! - A `SetBrickProperty` whose `Brick` is not a brick ref stops its exec
//!   chain silently, as the game does with a non-ref value.

use super::*;

#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Nil,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    /// `(R, G, B, A)`. From `MakeColorHex` these are the hex bytes over 255,
    /// with no colour-space conversion.
    Color([f32; 4]),
    Quat([f64; 4]),
    Arr(ArrKey),
    Var(NodeId),
    Brick(SavedBrickRef),
    Component(SavedComponentRef),
    Func(NodeId),
}

impl Val {
    pub fn int(&self) -> i64 {
        match self {
            Val::Int(i) => *i,
            Val::Num(n) => *n as i64,
            Val::Bool(b) => *b as i64,
            _ => 0,
        }
    }
    fn num(&self) -> f64 {
        match self {
            Val::Int(i) => *i as f64,
            Val::Num(n) => *n,
            Val::Bool(b) => *b as i64 as f64,
            _ => 0.0,
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Val::Bool(b) => *b,
            Val::Int(i) => *i != 0,
            Val::Num(n) => *n != 0.0,
            _ => false,
        }
    }
    fn str(&self) -> String {
        match self {
            Val::Str(s) => s.clone(),
            Val::Int(i) => i.to_string(),
            Val::Num(n) => n.to_string(),
            _ => String::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArrKey {
    Node(NodeId),
    Ext(usize),
}

fn lit_val(l: &Lit) -> Val {
    match l {
        Lit::Bool(b) => Val::Bool(*b),
        Lit::Int(i) => Val::Int(*i),
        Lit::Num(n) => Val::Num(*n),
        Lit::Str(s) => Val::Str(s.clone()),
        Lit::Color(r, g, b, a) => Val::Color([*r, *g, *b, *a]),
        Lit::BrickRef(r) => Val::Brick(*r),
        Lit::ComponentRef(r) => Val::Component(*r),
        Lit::Array(_) | Lit::ComponentClass(_) => Val::Nil,
    }
}

pub fn array_vals(a: &WireArray) -> Vec<Val> {
    match a {
        WireArray::Str(v) => v.iter().map(|s| Val::Str(s.clone())).collect(),
        WireArray::F64(v) => v.iter().map(|n| Val::Num(*n)).collect(),
        WireArray::I64(v) => v.iter().map(|i| Val::Int(*i)).collect(),
        WireArray::LinearColor(v) => v.iter().map(|c| Val::Color([c.0, c.1, c.2, c.3])).collect(),
        WireArray::Quat(v) => v.iter().map(|q| Val::Quat([q.0, q.1, q.2, q.3])).collect(),
        WireArray::Bricks(v) => v.iter().map(|r| Val::Brick(*r)).collect(),
        WireArray::Components(v) => v.iter().map(|r| Val::Component(*r)).collect(),
    }
}

fn arith(a: &Val, b: &Val, int: impl Fn(i64, i64) -> i64, float: impl Fn(f64, f64) -> f64) -> Val {
    match (a, b) {
        (Val::Num(_), _) | (_, Val::Num(_)) => Val::Num(float(a.num(), b.num())),
        _ => Val::Int(int(a.int(), b.int())),
    }
}

pub struct Sim<'g> {
    g: &'g Graph,
    source: HashMap<End, End>,
    fanout: HashMap<End, Vec<End>>,
    ext: HashMap<ExtPort, Val>,
    arrays: HashMap<ArrKey, Vec<Val>>,
    vars: HashMap<NodeId, Val>,
    latched: HashMap<(NodeId, &'static str), Val>,
    loop_index: HashMap<NodeId, i64>,
    call_args: Vec<HashMap<NodeId, Val>>,
    breaks: Vec<bool>,
    detectors: HashMap<NodeId, Val>,
    ticks: u64,
    /// Whether a `ChangeDetectorExec` fires on the first tick it sees a value.
    pub fire_on_first_tick: bool,
    /// Every `SetBrickProperty`: `(brick, property) -> value`.
    pub brick_props: HashMap<(SavedBrickRef, i64), Val>,
    /// Every `SetComponentProperty`: `(component, property name) -> value`.
    pub component_props: HashMap<(SavedComponentRef, String), Val>,
    /// Exec gate runs and loop iterations, cumulative.
    pub exec_runs: u64,
    pub iterations: u64,
    /// Expression gate evaluations, cumulative. One read twice counts
    /// twice, so with `exec_runs` this is an upper bound on gate work.
    pub evals: u64,
    /// The server's per-tick gate budget: a `CountFor` stops early, aborted,
    /// before an iteration once a tick has run this many gates. Every
    /// enclosing loop then stops too. `None` runs every loop to the end.
    pub gate_limit: Option<u64>,
    tick_start: u64,
}

const CHANGE_DETECTOR: &str = "BrickComponentType_Expr_ChangeDetectorExec";
const READ_BRICK_GRID: &str = "BrickComponentType_Internal_ReadBrickGrid";

impl<'g> Sim<'g> {
    pub fn new(g: &'g Graph) -> Self {
        let mut source = HashMap::new();
        let mut fanout: HashMap<End, Vec<End>> = HashMap::new();
        for (s, d) in &g.wires {
            source.insert(d.clone(), s.clone());
            fanout.entry(s.clone()).or_default().push(d.clone());
        }
        let mut arrays = HashMap::new();
        let mut vars = HashMap::new();
        for (n, node) in g.nodes.iter().enumerate() {
            let value = node
                .data
                .iter()
                .find(|(f, _)| *f == "Value")
                .map(|(_, l)| l);
            match (node.class, value) {
                (ARRAY_VAR, Some(Lit::Array(a))) => {
                    arrays.insert(ArrKey::Node(n), array_vals(a));
                }
                (ARRAY_VAR, _) => {
                    arrays.insert(ArrKey::Node(n), Vec::new());
                }
                (VAR, v) => {
                    vars.insert(n, v.map(lit_val).unwrap_or(Val::Nil));
                }
                _ => {}
            }
        }
        Sim {
            g,
            source,
            fanout,
            ext: HashMap::new(),
            arrays,
            vars,
            latched: HashMap::new(),
            loop_index: HashMap::new(),
            call_args: Vec::new(),
            breaks: Vec::new(),
            detectors: HashMap::new(),
            ticks: 0,
            fire_on_first_tick: false,
            brick_props: HashMap::new(),
            component_props: HashMap::new(),
            exec_runs: 0,
            iterations: 0,
            evals: 0,
            gate_limit: None,
            tick_start: 0,
        }
    }

    /// Give an external port a value (re-bind each tick to drive it).
    pub fn bind(&mut self, port: ExtPort, value: Val) {
        self.ext.insert(port, value);
    }

    /// Make an external port an array ref holding `items` (an `ArrayVar`
    /// built outside the graph).
    pub fn bind_array(&mut self, port: ExtPort, items: Vec<Val>) {
        let key = ArrKey::Ext(self.arrays.len());
        self.arrays.insert(key, items);
        self.ext.insert(port, Val::Arr(key));
    }

    /// Fire an exec source outside the graph (a change detector built
    /// alongside it, say).
    pub fn fire_ext(&mut self, port: ExtPort) {
        self.tick_start = self.gate_runs();
        self.fire(&End::Ext(port));
    }

    /// A new array not bound to any port, as the value an array ref holds:
    /// for building an array of arrays.
    pub fn new_array(&mut self, items: Vec<Val>) -> Val {
        let key = ArrKey::Ext(self.arrays.len());
        self.arrays.insert(key, items);
        Val::Arr(key)
    }

    pub fn array(&self, key: ArrKey) -> &[Val] {
        &self.arrays[&key]
    }

    pub fn var(&self, node: NodeId) -> &Val {
        &self.vars[&node]
    }

    /// Exec runs plus pure evaluations so far: the gate work to compare
    /// against a per-tick budget.
    pub fn gate_runs(&self) -> u64 {
        self.exec_runs + self.evals
    }

    /// One wire-graph tick: load events on the first, then every change
    /// detector in node order.
    pub fn tick(&mut self) {
        let first = self.ticks == 0;
        self.ticks += 1;
        self.tick_start = self.gate_runs();
        if first {
            for n in 0..self.g.nodes.len() {
                if self.g.nodes[n].class == READ_BRICK_GRID {
                    self.fire(&End::Node(n, "BrickGrid"));
                }
            }
        }
        for n in 0..self.g.nodes.len() {
            if self.g.nodes[n].class != CHANGE_DETECTOR {
                continue;
            }
            let v = self.input(n, "Input");
            let prev = self.detectors.insert(n, v.clone());
            let fire = match prev {
                Some(p) => p != v,
                None => self.fire_on_first_tick,
            };
            if fire {
                self.fire(&End::Node(n, "OnChanged"));
            }
        }
    }

    fn data(&self, n: NodeId, field: &str) -> Option<&Lit> {
        self.g.nodes[n]
            .data
            .iter()
            .find(|(f, _)| *f == field)
            .map(|(_, l)| l)
    }

    /// The value on input `port` of `n`: its wire, else its data, else nil.
    pub fn input(&mut self, n: NodeId, port: &'static str) -> Val {
        if let Some(src) = self.source.get(&End::Node(n, port)).cloned() {
            return self.eval(&src);
        }
        self.data(n, port).map(lit_val).unwrap_or(Val::Nil)
    }

    pub fn eval(&mut self, end: &End) -> Val {
        let (n, port) = match end {
            End::Ext(e) => return self.ext.get(e).cloned().unwrap_or(Val::Nil),
            End::Node(n, p) => (*n, *p),
        };
        let class = self.g.nodes[n].class;
        // Only expression gates compute when read. Reading a variable, a loop
        // index or an exec gate's held output runs nothing.
        if class.contains("Type_Expr_") {
            self.evals += 1;
        }
        let a = |s: &mut Self| s.input(n, "InputA");
        let b = |s: &mut Self| s.input(n, "InputB");
        match (class, port) {
            (ARRAY_VAR, "ArrayVarRef") => Val::Arr(ArrKey::Node(n)),
            (VAR, "Value") => self.vars[&n].clone(),
            (VAR, "VarRef") => Val::Var(n),
            (chip::MICROCHIP_INPUT, "RER_Output") => self.input(n, "RER_Input"),
            (FUNCTION_INPUT, "Value") => match self.call_args.last().and_then(|f| f.get(&n)) {
                Some(v) => v.clone(),
                None => self.input(n, "Value"),
            },
            (COUNT_FOR_ENTRY, "Index") => Val::Int(*self.loop_index.get(&n).unwrap_or(&0)),
            (FUNCTION_SHELL, "Definition") => Val::Func(n),
            ("BrickComponentType_Expr_MathAdd", _) => {
                let (x, y) = (a(self), b(self));
                arith(&x, &y, |p, q| p + q, |p, q| p + q)
            }
            ("BrickComponentType_Expr_MathSubtract", _) => {
                let (x, y) = (a(self), b(self));
                arith(&x, &y, |p, q| p - q, |p, q| p - q)
            }
            ("BrickComponentType_Expr_MathMultiply", _) => {
                let (x, y) = (a(self), b(self));
                arith(&x, &y, |p, q| p * q, |p, q| p * q)
            }
            ("BrickComponentType_Expr_MathModuloFloored", _) => {
                let (x, y) = (a(self), b(self));
                arith(
                    &x,
                    &y,
                    |p, q| p.rem_euclid(q),
                    |p, q| p - q * (p / q).floor(),
                )
            }
            ("BrickComponentType_Expr_MathMax", _) => {
                let (x, y) = (a(self), b(self));
                arith(&x, &y, i64::max, f64::max)
            }
            ("BrickComponentType_Expr_MathMin", _) => {
                let (x, y) = (a(self), b(self));
                arith(&x, &y, i64::min, f64::min)
            }
            ("BrickComponentType_Expr_BitwiseOR", _) => Val::Int(a(self).int() | b(self).int()),
            ("BrickComponentType_Expr_CompareLess", _) => Val::Bool(a(self).num() < b(self).num()),
            ("BrickComponentType_Expr_CompareLessOrEqual", _) => {
                Val::Bool(a(self).num() <= b(self).num())
            }
            ("BrickComponentType_Expr_CompareGreater", _) => {
                Val::Bool(a(self).num() > b(self).num())
            }
            ("BrickComponentType_Expr_CompareGreaterOrEqual", _) => {
                Val::Bool(a(self).num() >= b(self).num())
            }
            ("BrickComponentType_Expr_CompareNotEqual", _) => Val::Bool(a(self) != b(self)),
            ("BrickComponentType_Expr_CompareEqual", _) => Val::Bool(a(self) == b(self)),
            ("BrickComponentType_Expr_LogicalAND", _) => Val::Bool(
                self.input(n, "bInputA").truthy() && self.input(n, "bInputB").truthy(),
            ),
            ("BrickComponentType_Expr_String_FormatText", _) => {
                let mut out = self.input(n, "FormatString").str();
                for (k, port) in ["InputA", "InputB", "InputC", "InputD", "InputE", "InputF", "InputG"]
                    .into_iter()
                    .enumerate()
                {
                    let v = self.input(n, port).str();
                    out = out.replace(&format!("{{{k}}}"), &v);
                }
                Val::Str(out)
            }
            ("BrickComponentType_Expr_Select", _) => {
                if self.input(n, "bSelectB").truthy() {
                    b(self)
                } else {
                    a(self)
                }
            }
            ("BrickComponentType_Expr_String_Substring", _) => {
                let s = self.input(n, "Input").str();
                let start = self.input(n, "Start").int().max(0) as usize;
                let len = self.input(n, "Length").int().max(0) as usize;
                Val::Str(s.chars().skip(start).take(len).collect())
            }
            ("BrickComponentType_Expr_String_Length", _) => {
                Val::Int(self.input(n, "Input").str().chars().count() as i64)
            }
            ("BrickComponentType_Expr_String_ParseInt", p) => {
                let parsed = self.input(n, "Input").str().trim().parse::<i64>();
                match p {
                    "bSuccess" => Val::Bool(parsed.is_ok()),
                    _ => Val::Int(parsed.unwrap_or(0)),
                }
            }
            ("BrickComponentType_Expr_MakeColorHex", _) => {
                let hex = self.input(n, "Hex").str();
                let byte = |i: usize| {
                    u8::from_str_radix(hex.get(i..i + 2).unwrap_or("00"), 16).unwrap_or(0) as f32
                        / 255.0
                };
                Val::Color([byte(0), byte(2), byte(4), 1.0])
            }
            ("BrickComponentType_Expr_Convert", _) => self.input(n, "Input"),
            ("BrickComponentType_Expr_SplitQuaternion", p) => {
                let q = match self.input(n, "Input") {
                    Val::Quat(q) => q,
                    _ => [0.0; 4],
                };
                let i = match p {
                    "X" => 0,
                    "Y" => 1,
                    "Z" => 2,
                    _ => 3,
                };
                Val::Num(q[i])
            }
            _ => self.latched.get(&(n, port)).cloned().unwrap_or(Val::Nil),
        }
    }

    pub fn fire(&mut self, src: &End) {
        let Some(targets) = self.fanout.get(src).cloned() else {
            return;
        };
        for t in targets {
            if let End::Node(n, port) = t {
                self.run(n, port);
            }
        }
    }

    fn latch(&mut self, n: NodeId, port: &'static str, v: Val) {
        self.latched.insert((n, port), v);
    }

    fn array_of(&mut self, n: NodeId) -> Option<ArrKey> {
        match self.input(n, "ArrayVarRef") {
            Val::Arr(k) => Some(k),
            _ => None,
        }
    }

    fn run(&mut self, n: NodeId, port: &'static str) {
        let class = self.g.nodes[n].class;
        let out = |n| End::Node(n, "ExecOut");
        if class == chip::MICROCHIP_INPUT && port == "RER_Input" {
            return self.fire(&End::Node(n, "RER_Output"));
        }
        self.exec_runs += 1;
        match class {
            ARRAY_GET => {
                let Some(key) = self.array_of(n) else { return };
                let i = self.input(n, "Index").int();
                let v = usize::try_from(i)
                    .ok()
                    .and_then(|i| self.arrays[&key].get(i).cloned());
                self.latch(n, "bOutOfBounds", Val::Bool(v.is_none()));
                if let Some(v) = v {
                    self.latch(n, "Value", v);
                }
                self.fire(&out(n));
            }
            "BrickComponentType_Exec_ArrayVar_GetLength" => {
                let Some(key) = self.array_of(n) else { return };
                let len = self.arrays[&key].len() as i64;
                self.latch(n, "Length", Val::Int(len));
                self.fire(&out(n));
            }
            "BrickComponentType_Exec_ArrayVar_Push" => {
                let Some(key) = self.array_of(n) else { return };
                let v = self.input(n, "Value");
                self.arrays.get_mut(&key).expect("array").push(v);
                self.fire(&out(n));
            }
            "BrickComponentType_Exec_ArrayVar_SetAtIndex" => {
                let Some(key) = self.array_of(n) else { return };
                let (i, v) = (self.input(n, "Index").int(), self.input(n, "Value"));
                if let Some(slot) = usize::try_from(i)
                    .ok()
                    .and_then(|i| self.arrays.get_mut(&key)?.get_mut(i))
                {
                    *slot = v;
                }
                self.fire(&out(n));
            }
            VAR_SET | "BrickComponentType_Exec_Var_Increment" => {
                let Val::Var(target) = self.input(n, "VarRef") else {
                    return;
                };
                let v = self.input(n, "Value");
                let v = if class == VAR_SET {
                    v
                } else {
                    let cur = self.vars[&target].clone();
                    arith(&cur, &v, |p, q| p + q, |p, q| p + q)
                };
                self.vars.insert(target, v);
                self.fire(&out(n));
            }
            "BrickComponentType_Exec_Var_Get" => {
                let Val::Var(target) = self.input(n, "VarRef") else {
                    return;
                };
                let v = self.vars[&target].clone();
                self.latch(n, "Value", v);
                self.fire(&out(n));
            }
            "BrickComponentType_Exec_Branch" => {
                let next = if self.input(n, "bCond").truthy() {
                    "ExecOutA"
                } else {
                    "ExecOutB"
                };
                self.fire(&End::Node(n, next));
            }
            "BrickComponentType_Exec_Union" => self.fire(&out(n)),
            SET_BRICK_PROPERTY => {
                let Val::Brick(brick) = self.input(n, "Brick") else {
                    return;
                };
                let prop = self
                    .data(n, "Property")
                    .map_or(BRICK_PROPERTY_COLOR, |l| match l {
                        Lit::Int(i) => *i,
                        _ => BRICK_PROPERTY_COLOR,
                    });
                let v = self.input(n, "Value");
                self.brick_props.insert((brick, prop), v);
                self.fire(&out(n));
            }
            SET_COMPONENT_PROPERTY => {
                let Val::Component(c) = self.input(n, "Component") else {
                    return;
                };
                let name = match self.data(n, "PropertyName") {
                    Some(Lit::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                let v = self.input(n, "Value");
                self.component_props.insert((c, name), v);
                self.fire(&out(n));
            }
            COUNT_FOR_SHELL => {
                let count = self.input(n, "Count").int();
                let first = self.input(n, "FirstIndex").int();
                let step = match self.input(n, "Step") {
                    Val::Nil => 1,
                    v => v.int(),
                };
                let plane = self.g.nodes[n].plane.expect("a loop shell has a plane");
                let entry = self
                    .g
                    .nodes_in(plane)
                    .find(|&m| self.g.nodes[m].class == COUNT_FOR_ENTRY)
                    .expect("a loop plane has an entry gate");
                self.breaks.push(false);
                let mut done = 0;
                let mut cut = false;
                for i in 0..count.max(0) {
                    if self
                        .gate_limit
                        .is_some_and(|limit| self.gate_runs() - self.tick_start >= limit)
                    {
                        cut = true;
                        break;
                    }
                    self.loop_index.insert(entry, first + i * step);
                    self.iterations += 1;
                    done += 1;
                    self.fire(&End::Node(entry, "Body"));
                    if *self.breaks.last().expect("pushed") {
                        break;
                    }
                }
                let aborted = self.breaks.pop().expect("pushed") || cut;
                self.latch(n, "IterationsExecuted", Val::Int(done));
                self.latch(n, "bAborted", Val::Bool(aborted));
                self.fire(&End::Node(n, "Completed"));
            }
            CALLABLE_BREAK => {
                if let Some(b) = self.breaks.last_mut() {
                    *b = true;
                }
                self.fire(&out(n));
            }
            FUNCTION_CALL => {
                let Val::Func(shell) = self.input(n, "Definition") else {
                    return;
                };
                let plane = self.g.nodes[shell]
                    .plane
                    .expect("a function shell has a plane");
                let members: Vec<NodeId> = self.g.nodes_in(plane).collect();
                let mut args = HashMap::new();
                for &m in &members {
                    if self.g.nodes[m].class != FUNCTION_INPUT {
                        continue;
                    }
                    let slot = match self.data(m, "PortOrder") {
                        Some(Lit::Int(i)) => *i as usize,
                        _ => 0,
                    };
                    let v = self.input(n, INPUT_PINS[call_pin_for_slot(slot)]);
                    args.insert(m, v);
                }
                let entry = members
                    .iter()
                    .copied()
                    .find(|&m| self.g.nodes[m].class == FUNCTION_ENTRY)
                    .expect("a function plane has an entry gate");
                self.call_args.push(args);
                self.breaks.push(false);
                self.fire(&End::Node(entry, "Body"));
                self.breaks.pop();
                let mut outputs = Vec::new();
                for &m in &members {
                    if self.g.nodes[m].class == FUNCTION_OUTPUT {
                        let slot = match self.data(m, "PortOrder") {
                            Some(Lit::Int(i)) => *i as usize,
                            _ => 0,
                        };
                        let v = self.input(m, "Value");
                        outputs.push((slot, v));
                    }
                }
                self.call_args.pop();
                for (slot, v) in outputs {
                    const OUTS: [&str; 8] = [
                        "Output0", "Output1", "Output2", "Output3", "Output4", "Output5",
                        "Output6", "Output7",
                    ];
                    self.latch(n, OUTS[call_pin_for_slot(slot)], v);
                }
                self.fire(&End::Node(n, "Completed"));
            }
            other => panic!("the simulator does not run {other} (port {port})"),
        }
    }
}
