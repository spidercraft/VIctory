//! A compute shader of ours, not a translation: it bounds the arguments of
//! indirect execution on the GPU, just before they are read. Built with the
//! module builder the translations use, so the host needs no HLSL compiler.
//!
//! What the host binds ([`argument_clamp`]):
//!
//! * root constants `b0, space 0`: the record kind ([`DRAW`]..[`MESH`]), the
//!   bytes from one record to the next, the number of records;
//! * the arguments as a root UAV, `u<ARGUMENTS>` in [`ROOT_CBV_SPACE`], at
//!   the address of the first record;
//! * the list's budget as a root UAV, `u<BUDGET>` there: one dword, the
//!   work the list's records have asked for so far.
//!
//! One thread per record, 64 to a group: `ceil(records / 64)` groups.
//!
//! [`ROOT_CBV_SPACE`]: super::ROOT_CBV_SPACE

use super::bitcode::{Attr, Bin, Term, Ty, V, pred};
use super::shader::Shader;

/// Record kinds: the first dword is a vertex count, an index count, or
/// the first of three group counts.
pub const DRAW: u32 = 0;
pub const DRAW_INDEXED: u32 = 1;
pub const DISPATCH: u32 = 2;
pub const MESH: u32 = 3;

/// Root parameter indices of the two buffers (the constants are 0).
pub const ARGUMENTS: u8 = 1;
pub const BUDGET: u8 = 2;

/// Threads in a group of the clamp itself.
pub const GROUP: u32 = 64;

/// What a record may ask for. A count above its limit is lowered to it; a
/// record over the list's budget runs empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Vertices (or indices) of a draw.
    pub vertices: u32,
    /// Vertices times instances of a draw.
    pub vertex_instances: u32,
    /// Thread groups of a dispatch in one dimension.
    pub groups_per_dimension: u32,
    /// Thread groups of a dispatch; one asking for more runs empty.
    pub groups: u32,
    /// What a compute group and a mesh group cost, in the budget's unit
    /// (a vertex instance or a thread).
    pub group_cost: u32,
    pub mesh_group_cost: u32,
    /// The work of one list.
    pub budget: u32,
}

struct Clamp {
    s: Shader,
    ret: Ty,
    args: V,
    budget: V,
}

impl Clamp {
    /// Dwords `at..` of the arguments.
    fn load(&mut self, block: usize, at: V, n: u32) -> Vec<V> {
        let (undef, mask, align) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8((1u8 << n) - 1),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            block,
            "rawBufferLoad.i32",
            Attr::ReadOnly,
            self.ret,
            139,
            &[self.args, at, undef, mask, align],
        );
        (0..n).map(|k| self.s.m.extract(block, r, k)).collect()
    }

    /// Writes `values` at dword `at` of the arguments.
    fn store(&mut self, block: usize, at: V, values: &[V]) {
        let undef = self.s.m.undef(self.s.t.i32);
        let mut args = vec![self.args, at, undef];
        args.extend(values);
        args.resize(3 + 4, undef);
        args.push(self.s.m.i8((1u8 << values.len()) - 1));
        args.push(self.s.m.i32(4));
        let void = self.s.t.void;
        self.s
            .op(block, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
    }

    fn min(&mut self, block: usize, a: V, limit: u32) -> V {
        let limit = self.s.m.i32(limit);
        let less = self.s.m.cmp(block, pred::IULT, a, limit);
        self.s.m.select(block, less, a, limit)
    }

    fn mul(&mut self, block: usize, a: V, b: V) -> V {
        self.s.m.bin(block, Bin::Mul, a, b)
    }
}

/// The clamp for `limits`, as a DXIL container without a root signature.
pub fn argument_clamp(limits: &Limits) -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([GROUP, 1, 1]);
    s.user_data = 3;
    let i32t = s.t.i32;
    let ret = s
        .m
        .named_struct("dx.types.ResRet.i32", vec![i32t, i32t, i32t, i32t, i32t]);
    let entry = s.entry;
    let constants = s.user_data_handle();
    let (args, budget) = (s.root_uav_handle(ARGUMENTS), s.root_uav_handle(BUDGET));
    let mut c = Clamp {
        s,
        ret,
        args,
        budget,
    };
    let [first, draws, groups, dispatch, mesh, spend, empty, done] =
        std::array::from_fn(|_| c.s.m.new_block());

    // One thread per record; the last group's spare threads do nothing.
    let zero = c.s.m.i32(0);
    let row = c.s.op(
        entry,
        "cbufferLoadLegacy.i32",
        Attr::ReadOnly,
        c.s.t.cbuf_ret,
        59,
        &[constants, zero],
    );
    let [kind, stride, records] = [0, 1, 2].map(|k| c.s.m.extract(entry, row, k));
    let id = c
        .s
        .op(entry, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);
    let at = c.mul(entry, id, stride);
    let one = c.s.m.i32(1);
    let last_draw = c.s.m.i32(DRAW_INDEXED);
    let is_draw = c.s.m.cmp(entry, pred::IULE, kind, last_draw);
    let live = c.s.m.cmp(entry, pred::IULT, id, records);
    c.s.m.terminate(entry, Term::CondBr(live, first, done));
    c.s.m.terminate(first, Term::CondBr(is_draw, draws, groups));

    // Draws: the vertex or index count, then the instances.
    let v = c.load(draws, at, 2);
    let count = c.min(draws, v[0], limits.vertices);
    let some = c.s.m.cmp(draws, pred::IUGT, count, zero);
    let divisor = c.s.m.select(draws, some, count, one);
    let total = c.s.m.i32(limits.vertex_instances);
    let room = c.s.m.bin(draws, Bin::UDiv, total, divisor);
    let fits = c.s.m.cmp(draws, pred::IULT, v[1], room);
    let instances = c.s.m.select(draws, fits, v[1], room);
    let draw_cost = c.mul(draws, count, instances);
    c.store(draws, at, &[count, instances]);
    c.s.m.terminate(draws, Term::Br(spend));

    // Groups: three counts. The console's mesh dispatches are
    // one-dimensional.
    let g = c.load(groups, at, 3);
    let is_dispatch = {
        let k = c.s.m.i32(DISPATCH);
        c.s.m.cmp(groups, pred::EQ, kind, k)
    };
    c.s.m.terminate(groups, Term::CondBr(is_dispatch, dispatch, mesh));

    let d: Vec<V> = g
        .iter()
        .map(|&x| c.min(dispatch, x, limits.groups_per_dimension))
        .collect();
    // x * y fits 32 bits when both are limited to 16; the whole count may
    // not, so it is compared by division.
    let xy = c.mul(dispatch, d[0], d[1]);
    let deep = c.s.m.cmp(dispatch, pred::IUGT, d[2], zero);
    let divisor = c.s.m.select(dispatch, deep, d[2], one);
    let total = c.s.m.i32(limits.groups);
    let room = c.s.m.bin(dispatch, Bin::UDiv, total, divisor);
    let over = c.s.m.cmp(dispatch, pred::IUGT, xy, room);
    let over = c.s.m.bin(dispatch, Bin::And, over, deep);
    let d: Vec<V> = d
        .iter()
        .map(|&x| c.s.m.select(dispatch, over, zero, x))
        .collect();
    let dispatch_cost = {
        let xy = c.mul(dispatch, d[0], d[1]);
        let xyz = c.mul(dispatch, xy, d[2]);
        let each = c.s.m.i32(limits.group_cost);
        c.mul(dispatch, xyz, each)
    };
    c.store(dispatch, at, &d);
    c.s.m.terminate(dispatch, Term::Br(spend));

    let m = [
        c.min(mesh, g[0], limits.groups_per_dimension),
        c.min(mesh, g[1], 1),
        c.min(mesh, g[2], 1),
    ];
    let mesh_cost = {
        let xy = c.mul(mesh, m[0], m[1]);
        let xyz = c.mul(mesh, xy, m[2]);
        let each = c.s.m.i32(limits.mesh_group_cost);
        c.mul(mesh, xyz, each)
    };
    c.store(mesh, at, &m);
    c.s.m.terminate(mesh, Term::Br(spend));

    // The list's budget: a record that starts past it, or ends past it,
    // runs empty. Draws keep their layout with no vertices; groups become
    // zero in x.
    let cost = c.s.m.phi(spend, i32t);
    c.s.m.set_phi(
        cost,
        vec![
            (draw_cost, draws),
            (dispatch_cost, dispatch),
            (mesh_cost, mesh),
        ],
    );
    let undef = c.s.m.undef(i32t);
    let before = c.s.op(
        spend,
        "atomicBinOp.i32",
        Attr::NoUnwind,
        i32t,
        78,
        &[c.budget, zero, zero, undef, undef, cost],
    );
    let limit = c.s.m.i32(limits.budget);
    let spent = c.s.m.cmp(spend, pred::IUGE, before, limit);
    let after = c.s.m.bin(spend, Bin::Add, before, cost);
    let past = c.s.m.cmp(spend, pred::IUGT, after, limit);
    let out = c.s.m.bin(spend, Bin::Or, spent, past);
    c.s.m.terminate(spend, Term::CondBr(out, empty, done));
    c.store(empty, at, &[zero]);
    c.s.m.terminate(empty, Term::Br(done));
    c.s.m.terminate(done, Term::Ret);
    c.s.finish()
}

/// What the clamp makes of one record (`dwords` are its first three) with
/// `spent` of the budget used: the record, and what it adds to the budget.
/// The shader's arithmetic, for tests and for readers.
pub fn clamped(limits: &Limits, kind: u32, dwords: [u32; 3], spent: u32) -> ([u32; 3], u32) {
    let [x, y, z] = dwords;
    let (mut out, cost) = if kind <= DRAW_INDEXED {
        let count = x.min(limits.vertices);
        let instances = y.min(limits.vertex_instances / count.max(1));
        ([count, instances, z], count.wrapping_mul(instances))
    } else if kind == DISPATCH {
        let d = dwords.map(|v| v.min(limits.groups_per_dimension));
        let over = d[2] != 0 && d[0].wrapping_mul(d[1]) > limits.groups / d[2];
        let d = if over { [0; 3] } else { d };
        let groups = d[0].wrapping_mul(d[1]).wrapping_mul(d[2]);
        (d, groups.wrapping_mul(limits.group_cost))
    } else {
        let m = [x.min(limits.groups_per_dimension), y.min(1), z.min(1)];
        let groups = m[0] * m[1] * m[2];
        (m, groups.wrapping_mul(limits.mesh_group_cost))
    };
    if spent >= limits.budget || spent.wrapping_add(cost) > limits.budget {
        out[0] = 0;
    }
    (out, cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits = Limits {
        vertices: 1 << 24,
        vertex_instances: 1 << 26,
        groups_per_dimension: 65535,
        groups: 1 << 20,
        group_cost: 64,
        mesh_group_cost: 128,
        budget: 1 << 27,
    };

    #[test]
    fn the_container_is_a_compute_shader_with_its_bindings() {
        let dxil = argument_clamp(&LIMITS);
        assert_eq!(dxil, argument_clamp(&LIMITS), "not deterministic");
        let c = crate::dxbc::Container::parse(&dxil).unwrap();
        let has = |name: &[u8; 4]| c.part(crate::dxbc::FourCC(*name)).is_some();
        assert!(has(b"DXIL") && has(b"PSV0") && !has(b"RTS0"));
        // The validator, where one is installed, takes it too.
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil.clone()).unwrap();
        }
    }

    #[test]
    fn the_model_bounds_every_kind() {
        let l = &LIMITS;
        let big = u32::MAX;
        // Draws: the count, then instances by what the count leaves.
        assert_eq!(clamped(l, DRAW, [3, 2, 9], 0), ([3, 2, 9], 6));
        assert_eq!(clamped(l, DRAW, [big, big, 9], 0), ([1 << 24, 4, 9], 1 << 26));
        assert_eq!(clamped(l, DRAW_INDEXED, [0, big, 9], 0), ([0, 1 << 26, 9], 0));
        // Dispatches: each dimension, then the whole count or nothing.
        assert_eq!(clamped(l, DISPATCH, [8, 8, 1], 0), ([8, 8, 1], 64 * 64));
        assert_eq!(clamped(l, DISPATCH, [big, 1, 1], 0), ([65535, 1, 1], 65535 * 64));
        assert_eq!(clamped(l, DISPATCH, [1024, 1024, 1], 0), ([1024, 1024, 1], 1 << 26));
        assert_eq!(clamped(l, DISPATCH, [1024, 1025, 1], 0), ([0; 3], 0));
        assert_eq!(clamped(l, DISPATCH, [5, 0, 5], 0), ([5, 0, 5], 0));
        assert_eq!(clamped(l, DISPATCH, [1, 1, 0], 0), ([1, 1, 0], 0));
        // Mesh dispatches are one-dimensional.
        assert_eq!(clamped(l, MESH, [big, big, big], 0), ([65535, 1, 1], 65535 * 128));
        // The budget: a record that starts or ends past it runs empty.
        assert_eq!(clamped(l, DRAW, [3, 2, 9], l.budget - 6).0, [3, 2, 9]);
        assert_eq!(clamped(l, DRAW, [3, 2, 9], l.budget - 5).0, [0, 2, 9]);
        assert_eq!(clamped(l, DISPATCH, [8, 8, 1], l.budget).0, [0, 8, 1]);
    }
}
