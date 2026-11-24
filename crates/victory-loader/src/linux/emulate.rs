//! Emulation of Zen 2 / AMD-only instructions that Intel CPUs lack.
//!
//! Xbox Series X|S titles are compiled for AMD Zen 2 and execute `clzero`,
//! `monitorx`/`mwaitx`, `clwb` (absent on pre-Ice Lake Intel), and SSE4a
//! (`extrq`/`insertq`). On Linux we handle their `#UD` (`SIGILL`) directly
//! inside the signal handler on the kernel's `ucontext_t`, preserving full
//! XSTATE/YMM/x87 state and avoiding guest-stack SEH dispatch overhead.

use super::context::M128A;

const GPR_ORDER: [i32; 16] = [
    libc::REG_RAX,
    libc::REG_RCX,
    libc::REG_RDX,
    libc::REG_RBX,
    libc::REG_RSP,
    libc::REG_RBP,
    libc::REG_RSI,
    libc::REG_RDI,
    libc::REG_R8,
    libc::REG_R9,
    libc::REG_R10,
    libc::REG_R11,
    libc::REG_R12,
    libc::REG_R13,
    libc::REG_R14,
    libc::REG_R15,
];

/// Attempts to emulate an AMD-only / Zen 2 instruction at `uc`'s `RIP`.
/// Returns `true` if the instruction was emulated and `uc` was updated so
/// execution can resume directly from `rt_sigreturn`.
///
/// # Safety
/// `uc` must be a valid `ucontext_t` passed to a `SIGILL` handler, with
/// `RIP` pointing at readable code bytes (the CPU just fetched them).
pub(crate) unsafe fn emulate_amd_only(uc: &mut libc::ucontext_t) -> bool {
    let rip = uc.uc_mcontext.gregs[libc::REG_RIP as usize] as u64;
    let code = rip as *const u8;
    // Only read within the faulting instruction's page for lookahead; the
    // next page may not be mapped.
    let room = 0x1000 - (rip as usize & 0xfff);
    let mut bytes = [0u8; 32];
    let n = room.min(bytes.len());
    // SAFETY: `rip` is mapped readable (the CPU just fetched from it), and
    // `n` does not cross the page boundary.
    unsafe { std::ptr::copy_nonoverlapping(code, bytes.as_mut_ptr(), n) };

    // SAFETY: the 3-byte opcode is already fetched by the CPU even if it
    // straddles a page boundary.
    let first3 = unsafe { code.cast::<[u8; 3]>().read_unaligned() };
    match first3 {
        [0x0f, 0x01, 0xfc] => {
            // `clzero`: zeroes the 64-byte cache line containing `rax`.
            // Collapse tight `clzero` loops into one write_bytes pass.
            let rax = uc.uc_mcontext.gregs[libc::REG_RAX as usize] as u64;
            if let Some(pass) = clzero_loop(&bytes[..n])
                && rax & 63 == 0
            {
                let limit =
                    uc.uc_mcontext.gregs[GPR_ORDER[(pass.limit & 15) as usize] as usize] as u64;
                let lines = if pass.inclusive && rax <= limit {
                    Some((limit - rax) / 64 + 1)
                } else if !pass.inclusive && rax < limit {
                    Some((limit - rax).div_ceil(64))
                } else {
                    None
                };
                if let Some(lines) = lines
                    && let Some(total) = lines.checked_mul(64)
                    && total <= 64 * 1024 * 1024
                    && let Some(end) = rax.checked_add(total)
                {
                    // SAFETY: the title's `clzero` loop zeroes `[rax, end)`.
                    unsafe { (rax as *mut u8).write_bytes(0, total as usize) };
                    uc.uc_mcontext.gregs[libc::REG_RAX as usize] = end as i64;
                    let efl = uc.uc_mcontext.gregs[libc::REG_EFL as usize] as u64;
                    let flags = u64::from(cmp_flags(end, limit));
                    uc.uc_mcontext.gregs[libc::REG_EFL as usize] =
                        ((efl & !u64::from(CMP_FLAGS)) | flags) as i64;
                    uc.uc_mcontext.gregs[libc::REG_RIP as usize] =
                        rip.wrapping_add(pass.len as u64) as i64;
                    return true;
                }
            }
            // SAFETY: `clzero` zeroes the 64-byte line holding `rax`.
            unsafe { ((rax & !63) as *mut u8).write_bytes(0, 64) };
            uc.uc_mcontext.gregs[libc::REG_RIP as usize] = rip.wrapping_add(3) as i64;
            true
        }
        [0x0f, 0x01, 0xfa] => {
            // `monitorx`: wait hint, no-op.
            uc.uc_mcontext.gregs[libc::REG_RIP as usize] = rip.wrapping_add(3) as i64;
            true
        }
        [0x0f, 0x01, 0xfb] => {
            // `mwaitx`: wait hint; yield (async-signal-safe).
            // SAFETY: `sched_yield` has no preconditions and is async-signal-safe.
            unsafe { libc::sched_yield() };
            uc.uc_mcontext.gregs[libc::REG_RIP as usize] = rip.wrapping_add(3) as i64;
            true
        }
        _ => {
            if let Some(pass) = clwb_loop(&bytes[..n]) {
                let limit_idx = GPR_ORDER[(pass.limit & 15) as usize] as usize;
                let base_idx = GPR_ORDER[(pass.base & 15) as usize] as usize;
                let limit = uc.uc_mcontext.gregs[limit_idx] as u64;
                let base = uc.uc_mcontext.gregs[base_idx] as u64;
                let next = if base <= limit {
                    base.wrapping_add(((limit - base) / 64 + 1) * 64)
                } else {
                    base.wrapping_add(64)
                };
                uc.uc_mcontext.gregs[base_idx] = next as i64;
                let efl = uc.uc_mcontext.gregs[libc::REG_EFL as usize] as u64;
                let flags = u64::from(cmp_flags(next, limit));
                uc.uc_mcontext.gregs[libc::REG_EFL as usize] =
                    ((efl & !u64::from(CMP_FLAGS)) | flags) as i64;
                uc.uc_mcontext.gregs[libc::REG_RIP as usize] =
                    rip.wrapping_add(pass.len as u64) as i64;
                return true;
            }
            if n >= 15
                && let Some(len) = clwb_len(bytes[..15].try_into().unwrap_or([0; 15]))
            {
                uc.uc_mcontext.gregs[libc::REG_RIP as usize] = rip.wrapping_add(len as u64) as i64;
                return true;
            }
            // SAFETY: `code` is mapped readable for the faulting instruction.
            if let Some(op) = unsafe { decode_sse4a(code) } {
                // SAFETY: `uc` is a valid signal ucontext.
                let Some(xmm) = (unsafe { xmm_from_ucontext(uc) }) else {
                    return false;
                };
                apply_sse4a(op, xmm);
                uc.uc_mcontext.gregs[libc::REG_RIP as usize] =
                    rip.wrapping_add(op.len as u64) as i64;
                return true;
            }
            false
        }
    }
}

/// Returns mutable access to XMM0..15 in the kernel's saved FP/XSTATE frame,
/// ensuring the XSAVE SSE bit is set so `rt_sigreturn` reloads the changes.
///
/// # Safety
/// `uc` must be a valid `ucontext_t` from a Linux x86-64 signal frame.
unsafe fn xmm_from_ucontext(uc: &mut libc::ucontext_t) -> Option<&mut [M128A; 16]> {
    let fp = uc.uc_mcontext.fpregs as *mut u8;
    if fp.is_null() {
        return None;
    }
    // SAFETY: Linux x86-64 fpregs starts with a 512-byte FXSAVE area (XMM0..15 at +160).
    // If FP_XSTATE_MAGIC1 is present at +464, xstate_bv is at +512; bit 1 is SSE.
    unsafe {
        let magic = (fp.add(464) as *const u32).read_unaligned();
        if magic == 0x4650_5853 {
            let bv_ptr = fp.add(512) as *mut u64;
            let bv = bv_ptr.read_unaligned();
            if bv & 2 == 0 {
                std::ptr::write_bytes(fp.add(160), 0, 256);
                bv_ptr.write_unaligned(bv | 2);
            }
        }
        Some(&mut *(fp.add(160) as *mut [M128A; 16]))
    }
}

/// A `clzero` cache-line zeroing loop:
/// ```text
/// top: clzero
///      add  rax, 0x40
///      cmp  rax, limit
///      jb|jbe top
/// ```
struct ClzeroLoop {
    limit: u8,
    inclusive: bool,
    len: usize,
}

fn clzero_loop(code: &[u8]) -> Option<ClzeroLoop> {
    // clzero: 0f 01 fc
    if code.get(0..3)? != [0x0f, 0x01, 0xfc] {
        return None;
    }
    // add rax, 0x40: 48 83 c0 40
    if code.get(3..7)? != [0x48, 0x83, 0xc0, 0x40] {
        return None;
    }
    // cmp rax, limit: REX.W(+R) 39 /r with mod=3, rm=0 (rax), REX.B=0.
    let &[cmp_rex, 0x39, cmp_modrm] = code.get(7..10)? else {
        return None;
    };
    if !matches!(cmp_rex, 0x48 | 0x4c) || (cmp_modrm & 0xc7) != 0xc0 {
        return None;
    }
    let limit = ((cmp_modrm >> 3) & 7) | ((cmp_rex & 4) << 1);
    if limit == 0 {
        return None;
    }
    // jb (0x72) or jbe (0x76) back to top (-12).
    let &[jcc, rel] = code.get(10..12)? else {
        return None;
    };
    let inclusive = match jcc {
        0x72 => false,
        0x76 => true,
        _ => return None,
    };
    if i64::from(rel as i8) != -12 {
        return None;
    }
    Some(ClzeroLoop {
        limit,
        inclusive,
        len: 12,
    })
}

/// `clwb m8` (`66 [rex] 0f ae /6`, a memory operand), which Intel CPUs before
/// Ice Lake lack: its length, or `None` for any other instruction.
fn clwb_len(code: [u8; 15]) -> Option<usize> {
    let at = if code[1] & 0xf0 == 0x40 { 2 } else { 1 };
    if code[0] != 0x66 || code[at] != 0x0f || code[at + 1] != 0xae {
        return None;
    }
    let modrm = code[at + 2];
    let (mode, rm) = (modrm >> 6, modrm & 7);
    if mode == 3 || (modrm >> 3) & 7 != 6 {
        return None;
    }
    let mut len = at + 3;
    let mut base = rm;
    if rm == 4 {
        base = code[len] & 7;
        len += 1;
    }
    len += match mode {
        0 if rm == 5 || (rm == 4 && base == 5) => 4,
        1 => 1,
        2 => 4,
        _ => 0,
    };
    Some(len)
}

/// A `clwb` write-back loop as the title's compilers build it:
/// ```text
/// top: clwb [base]
///      add  base, 0x40
///      cmp  base, limit
///      jbe  top
/// ```
struct ClwbLoop {
    base: u8,
    limit: u8,
    len: usize,
}

fn clwb_loop(code: &[u8]) -> Option<ClwbLoop> {
    let (rex, at) = match code.get(1)? {
        0x40 | 0x41 => (code[1], 2),
        _ => (0, 1),
    };
    if *code.first()? != 0x66 || code.get(at..at + 2)? != [0x0f, 0xae] {
        return None;
    }
    let modrm = *code.get(at + 2)?;
    let rm = modrm & 7;
    if modrm >> 6 != 0 || (modrm >> 3) & 7 != 6 || rm == 4 || rm == 5 {
        return None;
    }
    let base = rm | (rex & 1) << 3;
    let add = at + 3;
    if *code.get(add..add + 4)? != [0x48 | base >> 3, 0x83, 0xc0 | (base & 7), 0x40] {
        return None;
    }
    let cmp = add + 4;
    let &[cmp_rex @ 0x48..=0x4f, 0x39, cmp_modrm] = code.get(cmp..cmp + 3)? else {
        return None;
    };
    if cmp_modrm >> 6 != 3 || (cmp_modrm & 7 | (cmp_rex & 1) << 3) != base {
        return None;
    }
    let limit = (cmp_modrm >> 3) & 7 | (cmp_rex & 4) << 1;
    let jump = cmp + 3;
    let len = jump + 2;
    if *code.get(jump)? != 0x76 || i64::from(*code.get(jump + 1)? as i8) != -(len as i64) {
        return None;
    }
    (limit != base).then_some(ClwbLoop { base, limit, len })
}

/// The status flags `cmp a, b` leaves (CF, PF, AF, ZF, SF, OF) as `EFLAGS` bits.
fn cmp_flags(a: u64, b: u64) -> u32 {
    let r = a.wrapping_sub(b);
    u32::from(a < b)
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | (((a ^ b ^ r) >> 4) as u32 & 1) << 4
        | u32::from(r == 0) << 6
        | ((r >> 63) as u32) << 7
        | ((((a ^ b) & (a ^ r)) >> 63) as u32) << 11
}
const CMP_FLAGS: u32 = 0x8d5;

#[derive(Clone, Copy)]
struct Sse4aOp {
    extrq: bool,
    dst: usize,
    src: usize,
    imm: Option<(u8, u8)>,
    len: usize,
}

/// Decodes SSE4a `extrq`/`insertq` (register forms) without mutating state.
unsafe fn decode_sse4a(code: *const u8) -> Option<Sse4aOp> {
    // SAFETY: the caller's code bytes are readable up to the instruction's end.
    let b = |i: usize| unsafe { code.add(i).read() };
    let prefix = b(0);
    let (rex, at) = if b(1) & 0xf0 == 0x40 {
        (b(1), 2)
    } else {
        (0, 1)
    };
    if !matches!(prefix, 0x66 | 0xf2) || b(at) != 0x0f || !matches!(b(at + 1), 0x78 | 0x79) {
        return None;
    }
    let modrm = b(at + 2);
    if modrm >> 6 != 3 {
        return None;
    }
    let extrq = prefix == 0x66;
    let immediate = b(at + 1) == 0x78;
    if extrq && immediate && (modrm >> 3) & 7 != 0 {
        return None;
    }
    let dst = usize::from((modrm >> 3) & 7 | (rex & 4) << 1);
    let src = usize::from(modrm & 7 | (rex & 1) << 3);
    let (imm, len) = if immediate {
        (Some((b(at + 3), b(at + 4))), at + 5)
    } else {
        (None, at + 3)
    };
    Some(Sse4aOp {
        extrq,
        dst,
        src,
        imm,
        len,
    })
}

fn apply_sse4a(op: Sse4aOp, xmm: &mut [M128A; 16]) {
    let fields = match op.imm {
        Some(f) => f,
        None => {
            let f = if op.extrq {
                xmm[op.src].low
            } else {
                xmm[op.src].high
            };
            (f as u8, (f >> 8) as u8)
        }
    };
    let (n, at_bit) = (u32::from(fields.0 & 63), u32::from(fields.1 & 63));
    let bits = if n == 0 { 64 } else { n };
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    if op.extrq {
        let target = if op.imm.is_some() { op.src } else { op.dst };
        xmm[target].low = (xmm[target].low >> at_bit) & mask;
    } else {
        let value = (xmm[op.src].low & mask) << at_bit;
        xmm[op.dst].low = xmm[op.dst].low & !(mask << at_bit) | value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clzero_loop_matches_jb_and_jbe() {
        // clzero; add rax, 0x40; cmp rax, r10; jb top
        let jb = clzero_loop(&[
            0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4,
        ])
        .expect("jb loop");
        assert_eq!((jb.limit, jb.inclusive, jb.len), (10, false, 12));
        // clzero; add rax, 0x40; cmp rax, rdx; jbe top
        let jbe = clzero_loop(&[
            0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x76, 0xf4,
        ])
        .expect("jbe loop");
        assert_eq!((jbe.limit, jbe.inclusive, jbe.len), (2, true, 12));
        // wrong jump target
        assert!(
            clzero_loop(&[
                0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf0
            ])
            .is_none()
        );
    }

    #[test]
    fn clwb_loops_and_lengths() {
        let a = clwb_loop(&[
            0x66, 0x0f, 0xae, 0x31, 0x48, 0x83, 0xc1, 0x40, 0x48, 0x39, 0xc1, 0x76, 0xf3,
        ])
        .expect("rcx loop");
        assert_eq!((a.base, a.limit, a.len), (1, 0, 13));
        let pad = |b: &[u8]| {
            let mut c = [0u8; 15];
            c[..b.len()].copy_from_slice(b);
            c
        };
        assert_eq!(clwb_len(pad(&[0x66, 0x41, 0x0f, 0xae, 0x31])), Some(5));
        assert_eq!(clwb_len(pad(&[0x66, 0x0f, 0xae, 0x38])), None);
    }

    #[test]
    fn amd_only_instructions_execute_on_any_cpu() {
        super::super::signals::install();
        #[repr(align(64))]
        struct Lines([u8; 256]);
        let mut buf = Lines([0xff; 256]);
        let at = buf.0.as_mut_ptr().wrapping_add(5);
        // Single clzero on unaligned pointer inside line 0.
        // SAFETY: zeroes only the 64-byte line containing `at`, inside `buf`.
        unsafe { std::arch::asm!(".byte 0x0f, 0x01, 0xfc", in("rax") at) };
        assert!(buf.0[..64].iter().all(|&b| b == 0));
        assert!(buf.0[64..].iter().all(|&b| b == 0xff));

        // Clzero loop over lines 1..3 (128 bytes), leaving line 3 untouched.
        let start = buf.0.as_mut_ptr().wrapping_add(64);
        let end = buf.0.as_mut_ptr().wrapping_add(192);
        let final_rax: usize;
        // SAFETY: zeroes `[start, end)` inside `buf`.
        unsafe {
            std::arch::asm!(
                "2:",
                ".byte 0x0f, 0x01, 0xfc",
                "add rax, 0x40",
                "cmp rax, r10",
                "jb 2b",
                inout("rax") start => final_rax,
                in("r10") end,
            );
        }
        assert_eq!(final_rax, end as usize);
        assert!(buf.0[..192].iter().all(|&b| b == 0));
        assert!(buf.0[192..].iter().all(|&b| b == 0xff));

        // SSE4a insertq + extrq.
        let out: u64;
        // SAFETY: touches only the named registers.
        unsafe {
            std::arch::asm!(
                "movq xmm0, {a}",
                "movq xmm1, {b}",
                ".byte 0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10",
                "movq {a}, xmm0",
                ".byte 0x66, 0x0f, 0x78, 0xc0, 0x08, 0x14",
                "movq {b}, xmm0",
                a = inout(reg) 0x1111_2222_3333_4444u64 => _,
                b = inout(reg) 0xabcdu64 => out,
                out("xmm0") _,
                out("xmm1") _,
            );
        }
        assert_eq!(out, 0xbc);

        // monitorx / mwaitx.
        // SAFETY: wait hints that write no memory.
        unsafe {
            std::arch::asm!(
                ".byte 0x0f, 0x01, 0xfa",
                ".byte 0x0f, 0x01, 0xfb",
                in("rax") at,
                in("ecx") 0,
                in("edx") 0,
            );
        }
    }

    #[repr(C, align(64))]
    struct FakeXstate {
        fxsave: [u8; 512],
        xstate_bv: u64,
        xcomp_bv: u64,
        reserved: [u8; 48],
    }

    impl FakeXstate {
        fn new(xmm: &[M128A; 16], xstate_bv: u64) -> Self {
            let mut f = FakeXstate {
                fxsave: [0; 512],
                xstate_bv,
                xcomp_bv: 0,
                reserved: [0; 48],
            };
            // FP_XSTATE_MAGIC1 at FXSAVE offset 464.
            f.fxsave[464..468].copy_from_slice(&0x4650_5853u32.to_le_bytes());
            // SAFETY: XMM0..15 live at FXSAVE offset 160 (256 bytes, 16-byte aligned).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    xmm.as_ptr() as *const u8,
                    f.fxsave.as_mut_ptr().add(160),
                    256,
                );
            }
            f
        }

        fn xmms(&self) -> [M128A; 16] {
            let mut out = [M128A::default(); 16];
            // SAFETY: reads the 16 XMM registers at FXSAVE offset 160.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.fxsave.as_ptr().add(160),
                    out.as_mut_ptr() as *mut u8,
                    256,
                );
            }
            out
        }
    }

    fn reg(low: u64, high: u64) -> M128A {
        M128A { low, high }
    }

    /// Runs `emulate_amd_only` directly against a synthetic `ucontext_t`
    /// without executing `code`, so AMD CPUs and CI exercise every emulation
    /// branch.
    fn emulate(
        code: &[u8],
        setup_gprs: impl FnOnce(&mut libc::ucontext_t),
        xmm: &mut [M128A; 16],
        xstate_bv: &mut u64,
    ) -> Option<(u64, libc::ucontext_t)> {
        // Fill trailing bytes with `int3` so an over-eager decoder never sees
        // accidental matches.
        let mut bytes = [0xcc_u8; 32];
        bytes[..code.len()].copy_from_slice(code);
        let start = bytes.as_ptr() as u64;
        let mut xstate = FakeXstate::new(xmm, *xstate_bv);
        // SAFETY: zeroed `ucontext_t` is valid; `fpregs` points at our aligned
        // `FakeXstate` for the duration of the call.
        unsafe {
            let mut uc: libc::ucontext_t = std::mem::zeroed();
            uc.uc_mcontext.gregs[libc::REG_RIP as usize] = start as i64;
            uc.uc_mcontext.fpregs = (&raw mut xstate).cast();
            setup_gprs(&mut uc);
            let emulated = emulate_amd_only(&mut uc);
            *xmm = xstate.xmms();
            *xstate_bv = xstate.xstate_bv;
            let advanced = (uc.uc_mcontext.gregs[libc::REG_RIP as usize] as u64).wrapping_sub(start);
            emulated.then_some((advanced, uc))
        }
    }

    #[test]
    fn emulated_clzero_zeroes_its_line() {
        #[repr(align(64))]
        struct Line([u8; 128]);
        let mut line = Line([0xff; 128]);
        let at = line.0.as_mut_ptr().wrapping_add(5) as u64;
        let mut xmm = [M128A::default(); 16];
        let mut bv = 2;
        let (len, _) = emulate(
            &[0x0f, 0x01, 0xfc],
            |uc| uc.uc_mcontext.gregs[libc::REG_RAX as usize] = at as i64,
            &mut xmm,
            &mut bv,
        )
        .expect("clzero emulated");
        assert_eq!(len, 3);
        assert!(line.0[..64].iter().all(|&b| b == 0));
        assert!(line.0[64..].iter().all(|&b| b == 0xff));
    }

    #[test]
    fn emulated_clzero_loops_jb_and_jbe() {
        #[repr(align(64))]
        struct Buf([u8; 256]);
        let mut buf = Buf([0xaa; 256]);
        let start = buf.0.as_mut_ptr() as u64;
        let mut xmm = [M128A::default(); 16];
        let mut bv = 2;

        // jb: zeroes [0..128), leaves [128..256) untouched, final rax == start + 128.
        let jb_code = [
            0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4,
        ];
        let (len, uc) = emulate(
            &jb_code,
            |uc| {
                uc.uc_mcontext.gregs[libc::REG_RAX as usize] = start as i64;
                uc.uc_mcontext.gregs[libc::REG_R10 as usize] = (start + 128) as i64;
            },
            &mut xmm,
            &mut bv,
        )
        .expect("clzero jb loop emulated");
        assert_eq!(len, 12);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_RAX as usize] as u64, start + 128);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_EFL as usize] as u32 & 0x41, 0x40); // ZF=1, CF=0
        assert!(buf.0[..128].iter().all(|&b| b == 0));
        assert!(buf.0[128..].iter().all(|&b| b == 0xaa));

        // jbe: zeroes [128..192) (limit == 128 -> one inclusive pass), leaves [192..256) untouched.
        let jbe_code = [
            0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x76, 0xf4,
        ];
        let (len, uc) = emulate(
            &jbe_code,
            |uc| {
                uc.uc_mcontext.gregs[libc::REG_RAX as usize] = (start + 128) as i64;
                uc.uc_mcontext.gregs[libc::REG_RDX as usize] = (start + 128) as i64;
            },
            &mut xmm,
            &mut bv,
        )
        .expect("clzero jbe loop emulated");
        assert_eq!(len, 12);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_RAX as usize] as u64, start + 192);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_EFL as usize] as u32 & 0x41, 0); // ZF=0, CF=0
        assert!(buf.0[..192].iter().all(|&b| b == 0));
        assert!(buf.0[192..].iter().all(|&b| b == 0xaa));
    }

    #[test]
    fn emulated_monitorx_mwaitx_and_clwb() {
        let mut xmm = [reg(1, 2); 16];
        let mut bv = 2;
        assert_eq!(emulate(&[0x0f, 0x01, 0xfa], |_| {}, &mut xmm, &mut bv).map(|r| r.0), Some(3));
        assert_eq!(emulate(&[0x0f, 0x01, 0xfb], |_| {}, &mut xmm, &mut bv).map(|r| r.0), Some(3));
        assert!(xmm.iter().all(|r| r.low == 1 && r.high == 2));

        // Single clwb [r9] (5 bytes).
        assert_eq!(
            emulate(&[0x66, 0x41, 0x0f, 0xae, 0x31], |_| {}, &mut xmm, &mut bv).map(|r| r.0),
            Some(5)
        );
        // clwb [rcx]; add rcx, 0x40; cmp rcx, rax; jbe top (13 bytes).
        let clwb_loop_code = [
            0x66, 0x0f, 0xae, 0x31, 0x48, 0x83, 0xc1, 0x40, 0x48, 0x39, 0xc1, 0x76, 0xf3,
        ];
        let (len, uc) = emulate(
            &clwb_loop_code,
            |uc| {
                uc.uc_mcontext.gregs[libc::REG_RCX as usize] = 0x1000;
                uc.uc_mcontext.gregs[libc::REG_RAX as usize] = 0x1080;
            },
            &mut xmm,
            &mut bv,
        )
        .expect("clwb loop emulated");
        assert_eq!(len, 13);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_RCX as usize] as u64, 0x10c0);
        assert_eq!(uc.uc_mcontext.gregs[libc::REG_EFL as usize] as u32 & 0x41, 0);
    }

    #[test]
    fn emulated_sse4a_variants_and_xstate_init() {
        let mut xmm = [M128A::default(); 16];
        let mut bv = 2;

        // Immediate insertq + extrq.
        xmm[0] = reg(0x1111_2222_3333_4444, 0x5555);
        xmm[1] = reg(0xabcd, 0x6666);
        assert_eq!(
            emulate(&[0xf2, 0x0f, 0x78, 0xc1, 0x10, 0x10], |_| {}, &mut xmm, &mut bv).map(|r| r.0),
            Some(6)
        );
        assert_eq!(xmm[0].low, 0x1111_2222_abcd_4444);
        assert_eq!(
            emulate(&[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x14], |_| {}, &mut xmm, &mut bv).map(|r| r.0),
            Some(6)
        );
        assert_eq!(xmm[0].low, 0xbc);

        // Register-form insertq + extrq.
        xmm[0] = reg(0x1111_2222_3333_4444, 0);
        xmm[1] = reg(0xabcd, 0x10_10);
        assert_eq!(
            emulate(&[0xf2, 0x0f, 0x79, 0xc1], |_| {}, &mut xmm, &mut bv).map(|r| r.0),
            Some(4)
        );
        assert_eq!(xmm[0].low, 0x1111_2222_abcd_4444);
        xmm[1] = reg(0x14_08, 0);
        assert_eq!(
            emulate(&[0x66, 0x0f, 0x79, 0xc1], |_| {}, &mut xmm, &mut bv).map(|r| r.0),
            Some(4)
        );
        assert_eq!(xmm[0].low, 0xbc);

        // High registers xmm8/xmm9 (REX.R / REX.B) + length 0 == 64 bits.
        xmm[8] = reg(0x1111_2222_3333_4444, 0);
        xmm[9] = reg(0xabcd, 0);
        assert_eq!(
            emulate(&[0xf2, 0x45, 0x0f, 0x78, 0xc1, 0x10, 0x10], |_| {}, &mut xmm, &mut bv)
                .map(|r| r.0),
            Some(7)
        );
        assert_eq!(xmm[8].low, 0x1111_2222_abcd_4444);
        assert_eq!(
            emulate(&[0x66, 0x41, 0x0f, 0x78, 0xc1, 0x00, 0x00], |_| {}, &mut xmm, &mut bv)
                .map(|r| r.0),
            Some(7)
        );
        assert_eq!(xmm[9].low, 0xabcd);

        // XSAVE SSE init-state (xstate_bv bit 1 == 0): stale FXSAVE bytes are
        // cleared and bit 1 is set so rt_sigreturn reloads the result.
        let mut dirty = [reg(0xdead_beef, 0xcafe_babe); 16];
        let mut init_bv = 0u64;
        assert_eq!(
            emulate(&[0x66, 0x0f, 0x78, 0xc0, 0x08, 0x00], |_| {}, &mut dirty, &mut init_bv)
                .map(|r| r.0),
            Some(6)
        );
        assert_eq!(init_bv & 2, 2);
        assert!(dirty.iter().all(|r| r.low == 0 && r.high == 0));
    }

    #[test]
    fn other_instructions_are_not_emulated() {
        let mut xmm = [M128A::default(); 16];
        let mut bv = 0u64;
        assert!(emulate(&[0x0f, 0x0b], |_| {}, &mut xmm, &mut bv).is_none()); // ud2
        assert!(emulate(&[0x66, 0x0f, 0x79, 0x00], |_| {}, &mut xmm, &mut bv).is_none()); // mem extrq
        assert!(emulate(&[0x66, 0x0f, 0x78, 0xc8, 0x08, 0x14], |_| {}, &mut xmm, &mut bv).is_none());
        // Failed emulation must not touch xstate_bv.
        assert_eq!(bv, 0);
    }
}
