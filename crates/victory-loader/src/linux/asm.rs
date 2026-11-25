//! The machine-level pieces of exception handling: capturing and loading a
//! [`Context`], a memory read that cannot fault, and the frame every call
//! from the dispatcher into title code goes through.
//!
//! Our Rust frames have DWARF unwind data, not `.pdata`, so a walk by the
//! title's unwinder cannot cross them. [`frame_call`] has Windows unwind
//! data (registered by [`super::unwind`]): saved nonvolatile registers and
//! a machine frame that make the caller of its frame the `resume` context,
//! which is where the exception was raised (dispatch), the caller of
//! `RtlUnwindEx` (unwind) or the frame a catch block belongs to
//! (`STATUS_UNWIND_CONSOLIDATE`). A walk that starts in a handler goes on
//! in title frames where `ntdll`'s own frames would lead it on Windows. Its
//! language handler answers nested exceptions and collided unwinds as
//! `RtlpExecuteHandlerForException` / `ForUnwind` do (`seh.rs`).
//!
//! A pc in our code (no `.pdata`) gets [`boundary`]: unwinding it sets Rip
//! to 0, the bottom of a Windows thread's stack, so walks end there.

use super::context::{Context, RuntimeFunction};

// frame_call(resume rdi, kind rsi, fn rdx, a1 rcx, a2 r8, a3 r9, a4 [rsp+8]): calls the win64
// fn(a1, a2, a3, a4). Its frame at the call, from rsp (the establisher frame):
//   0x00  home space of fn's arguments       0x30  resume's rbx rbp rsi rdi r12 r13 r14 r15
//   0x20  kind                               0x70  resume's xmm6..xmm15
//   0x28  a4                                 0x118 machine frame: resume's rip, cs, eflags, rsp, ss
// victory_seh_frame_info describes exactly this; the two change together. Every code is at
// prologue offset 0, so it applies anywhere in the routine; the only pc a walk sees is the
// return address of the call, a `nop` (not an epilogue).
//
// restore(context rdi): loads every register, the ymm halves as vector_control says (YMM_SAVED,
// YMM_CLEAN), and continues at its rip through iretq (rip, rsp and rflags at once). Does not
// touch the target stack.
//
// capture_context(context rcx): RtlCaptureContext, callable from title code as is. Rip is the
// return address, Rsp the caller's after the return; flags and every register are kept;
// vector_control is 0 (no ymm halves).
//
// probe_read(address rdi, out rsi) -> eax: 1 and *out = [address], or 0 when the load faulted
// (the signal handler moves a fault at victory_seh_probe_load to victory_seh_probe_fail).
//
// probe_write(address rdi) -> eax: 1 when the 8 bytes at address can be written, or 0 when the
// store faulted (at victory_seh_probe_store, recovered the same way). An atomic or of zero: the
// bytes are left as they are, whatever other threads write there meanwhile.
//
// The two function entries (frame call, boundary) and their UNWIND_INFO are RVAs from
// victory_seh_asm_base, so they are all in this section.
core::arch::global_asm!(
    r#"
    .text
    .p2align 4
    .globl victory_seh_asm_base
    .hidden victory_seh_asm_base
victory_seh_asm_base:
    .globl victory_seh_frame_call
    .hidden victory_seh_frame_call
victory_seh_frame_call:
    pushq %rbp
    movq %rsp, %rbp
    andq $-16, %rsp
    subq $0x140, %rsp
    movq 0xf8(%rdi), %rax
    movq %rax, 0x118(%rsp)
    movq $0x33, 0x120(%rsp)
    movl 0x44(%rdi), %eax
    movq %rax, 0x128(%rsp)
    movq 0x98(%rdi), %rax
    movq %rax, 0x130(%rsp)
    movq $0x2b, 0x138(%rsp)
    movq %rsi, 0x20(%rsp)
    movq 16(%rbp), %rax
    movq %rax, 0x28(%rsp)
    movq 0x90(%rdi), %rax
    movq %rax, 0x30(%rsp)
    movq 0xa0(%rdi), %rax
    movq %rax, 0x38(%rsp)
    movq 0xa8(%rdi), %rax
    movq %rax, 0x40(%rsp)
    movq 0xb0(%rdi), %rax
    movq %rax, 0x48(%rsp)
    movq 0xd8(%rdi), %rax
    movq %rax, 0x50(%rsp)
    movq 0xe0(%rdi), %rax
    movq %rax, 0x58(%rsp)
    movq 0xe8(%rdi), %rax
    movq %rax, 0x60(%rsp)
    movq 0xf0(%rdi), %rax
    movq %rax, 0x68(%rsp)
    movups 0x200(%rdi), %xmm0
    movups %xmm0, 0x70(%rsp)
    movups 0x210(%rdi), %xmm0
    movups %xmm0, 0x80(%rsp)
    movups 0x220(%rdi), %xmm0
    movups %xmm0, 0x90(%rsp)
    movups 0x230(%rdi), %xmm0
    movups %xmm0, 0xa0(%rsp)
    movups 0x240(%rdi), %xmm0
    movups %xmm0, 0xb0(%rsp)
    movups 0x250(%rdi), %xmm0
    movups %xmm0, 0xc0(%rsp)
    movups 0x260(%rdi), %xmm0
    movups %xmm0, 0xd0(%rsp)
    movups 0x270(%rdi), %xmm0
    movups %xmm0, 0xe0(%rsp)
    movups 0x280(%rdi), %xmm0
    movups %xmm0, 0xf0(%rsp)
    movups 0x290(%rdi), %xmm0
    movups %xmm0, 0x100(%rsp)
    movq %rdx, %rax
    movq %r8, %rdx
    movq %r9, %r8
    movq 0x28(%rsp), %r9
    call *%rax
    nop
    movq %rbp, %rsp
    popq %rbp
    ret
victory_seh_frame_call_end:

victory_seh_frame_handler_thunk:
    jmp {frame_handler}

    .globl victory_seh_restore
    .hidden victory_seh_restore
victory_seh_restore:
    movabsq $0x56494354594d4d30, %rax
    cmpq %rax, 0x4a0(%rdi)
    jne 2f
    vzeroupper
2:  testb $8, 0x30(%rdi)
    jz 1f
    movups 0x1a0(%rdi), %xmm0
    movups 0x1b0(%rdi), %xmm1
    movups 0x1c0(%rdi), %xmm2
    movups 0x1d0(%rdi), %xmm3
    movups 0x1e0(%rdi), %xmm4
    movups 0x1f0(%rdi), %xmm5
    movups 0x200(%rdi), %xmm6
    movups 0x210(%rdi), %xmm7
    movups 0x220(%rdi), %xmm8
    movups 0x230(%rdi), %xmm9
    movups 0x240(%rdi), %xmm10
    movups 0x250(%rdi), %xmm11
    movups 0x260(%rdi), %xmm12
    movups 0x270(%rdi), %xmm13
    movups 0x280(%rdi), %xmm14
    movups 0x290(%rdi), %xmm15
1:  movabsq $0x56494354594d4d31, %rax
    cmpq %rax, 0x4a0(%rdi)
    jne 3f
    vinsertf128 $1, 0x300(%rdi), %ymm0, %ymm0
    vinsertf128 $1, 0x310(%rdi), %ymm1, %ymm1
    vinsertf128 $1, 0x320(%rdi), %ymm2, %ymm2
    vinsertf128 $1, 0x330(%rdi), %ymm3, %ymm3
    vinsertf128 $1, 0x340(%rdi), %ymm4, %ymm4
    vinsertf128 $1, 0x350(%rdi), %ymm5, %ymm5
    vinsertf128 $1, 0x360(%rdi), %ymm6, %ymm6
    vinsertf128 $1, 0x370(%rdi), %ymm7, %ymm7
    vinsertf128 $1, 0x380(%rdi), %ymm8, %ymm8
    vinsertf128 $1, 0x390(%rdi), %ymm9, %ymm9
    vinsertf128 $1, 0x3a0(%rdi), %ymm10, %ymm10
    vinsertf128 $1, 0x3b0(%rdi), %ymm11, %ymm11
    vinsertf128 $1, 0x3c0(%rdi), %ymm12, %ymm12
    vinsertf128 $1, 0x3d0(%rdi), %ymm13, %ymm13
    vinsertf128 $1, 0x3e0(%rdi), %ymm14, %ymm14
    vinsertf128 $1, 0x3f0(%rdi), %ymm15, %ymm15
3:  movq %ss, %rax
    pushq %rax
    pushq 0x98(%rdi)
    movl 0x44(%rdi), %eax
    andl $0x240dd5, %eax
    orl $0x202, %eax
    pushq %rax
    movq %cs, %rax
    pushq %rax
    pushq 0xf8(%rdi)
    movq 0x78(%rdi), %rax
    movq 0x80(%rdi), %rcx
    movq 0x88(%rdi), %rdx
    movq 0x90(%rdi), %rbx
    movq 0xa0(%rdi), %rbp
    movq 0xa8(%rdi), %rsi
    movq 0xb8(%rdi), %r8
    movq 0xc0(%rdi), %r9
    movq 0xc8(%rdi), %r10
    movq 0xd0(%rdi), %r11
    movq 0xd8(%rdi), %r12
    movq 0xe0(%rdi), %r13
    movq 0xe8(%rdi), %r14
    movq 0xf0(%rdi), %r15
    movq 0xb0(%rdi), %rdi
    iretq

    .globl victory_seh_capture_context
    .hidden victory_seh_capture_context
victory_seh_capture_context:
    pushfq
    movq %rax, 0x78(%rcx)
    movq %rcx, 0x80(%rcx)
    movq %rdx, 0x88(%rcx)
    movq %rbx, 0x90(%rcx)
    leaq 0x10(%rsp), %rax
    movq %rax, 0x98(%rcx)
    movq %rbp, 0xa0(%rcx)
    movq %rsi, 0xa8(%rcx)
    movq %rdi, 0xb0(%rcx)
    movq %r8, 0xb8(%rcx)
    movq %r9, 0xc0(%rcx)
    movq %r10, 0xc8(%rcx)
    movq %r11, 0xd0(%rcx)
    movq %r12, 0xd8(%rcx)
    movq %r13, 0xe0(%rcx)
    movq %r14, 0xe8(%rcx)
    movq %r15, 0xf0(%rcx)
    movq 8(%rsp), %rax
    movq %rax, 0xf8(%rcx)
    movq (%rsp), %rax
    movl %eax, 0x44(%rcx)
    movw %cs, 0x38(%rcx)
    movw %ds, 0x3a(%rcx)
    movw %es, 0x3c(%rcx)
    movw %fs, 0x3e(%rcx)
    movw %gs, 0x40(%rcx)
    movw %ss, 0x42(%rcx)
    fxsave64 0x100(%rcx)
    stmxcsr 0x34(%rcx)
    movl $0x10000f, 0x30(%rcx)
    movq $0, 0x4a0(%rcx)
    movq 0x78(%rcx), %rax
    popfq
    ret

    .globl victory_seh_probe_read
    .hidden victory_seh_probe_read
victory_seh_probe_read:
    .globl victory_seh_probe_load
    .hidden victory_seh_probe_load
victory_seh_probe_load:
    movq (%rdi), %rax
    movq %rax, (%rsi)
    movl $1, %eax
    ret
    .globl victory_seh_probe_fail
    .hidden victory_seh_probe_fail
victory_seh_probe_fail:
    xorl %eax, %eax
    ret

    .globl victory_seh_probe_write
    .hidden victory_seh_probe_write
victory_seh_probe_write:
    .globl victory_seh_probe_store
    .hidden victory_seh_probe_store
victory_seh_probe_store:
    lock orq $0, (%rdi)
    movl $1, %eax
    ret

    .globl victory_seh_copy_bytes
    .hidden victory_seh_copy_bytes
victory_seh_copy_bytes:
    movq %rdx, %rcx
    .globl victory_seh_copy_move
    .hidden victory_seh_copy_move
victory_seh_copy_move:
    rep movsb
    movl $1, %eax
    ret

    .p2align 2
victory_seh_frame_info:
    .byte 0x19, 0, 39, 0
    .byte 0, 0x34
    .short 6
    .byte 0, 0x54
    .short 7
    .byte 0, 0x64
    .short 8
    .byte 0, 0x74
    .short 9
    .byte 0, 0xc4
    .short 10
    .byte 0, 0xd4
    .short 11
    .byte 0, 0xe4
    .short 12
    .byte 0, 0xf4
    .short 13
    .byte 0, 0x68
    .short 7
    .byte 0, 0x78
    .short 8
    .byte 0, 0x88
    .short 9
    .byte 0, 0x98
    .short 10
    .byte 0, 0xa8
    .short 11
    .byte 0, 0xb8
    .short 12
    .byte 0, 0xc8
    .short 13
    .byte 0, 0xd8
    .short 14
    .byte 0, 0xe8
    .short 15
    .byte 0, 0xf8
    .short 16
    .byte 0, 0x01
    .short 0x118 / 8
    .byte 0, 0x0a
    .short 0
    .long victory_seh_frame_handler_thunk - victory_seh_asm_base

    .globl victory_seh_frame_entry
    .hidden victory_seh_frame_entry
victory_seh_frame_entry:
    .long victory_seh_frame_call - victory_seh_asm_base
    .long victory_seh_frame_call_end - victory_seh_asm_base
    .long victory_seh_frame_info - victory_seh_asm_base

victory_seh_boundary_info:
    .byte 1, 0, 0, 0
    .globl victory_seh_boundary
    .hidden victory_seh_boundary
victory_seh_boundary:
    .long 0
    .long 0xffffffff
    .long victory_seh_boundary_info - victory_seh_asm_base
    "#,
    frame_handler = sym super::seh::frame_call_handler,
    options(att_syntax)
);

unsafe extern "C" {
    static victory_seh_asm_base: u8;
    static victory_seh_frame_entry: RuntimeFunction;
    static victory_seh_boundary: RuntimeFunction;
    fn victory_seh_frame_call(
        resume: *const Context,
        kind: u64,
        function: usize,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
    ) -> u64;
    fn victory_seh_restore(context: *const Context) -> !;
    fn victory_seh_probe_read(address: u64, out: *mut u64) -> u32;
    static victory_seh_probe_load: u8;
    static victory_seh_probe_fail: u8;
    fn victory_seh_probe_write(address: u64) -> u32;
    static victory_seh_probe_store: u8;
    pub(super) fn victory_seh_copy_bytes(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_seh_copy_move: u8;
}

unsafe extern "win64" {
    fn victory_seh_capture_context(context: *mut Context);
}

/// The base the entries below are RVAs from.
pub(crate) fn base() -> usize {
    // A label in .text; only its address is taken.
    &raw const victory_seh_asm_base as usize
}

/// The function entry of [`frame_call`], RVAs from [`base`].
pub(crate) fn frame_entry() -> *const RuntimeFunction {
    &raw const victory_seh_frame_entry
}

/// The entry a pc in our own code gets (begin 0, end `u32::MAX`, from
/// [`base`]): unwinding it ends the walk.
pub(crate) fn boundary() -> *const RuntimeFunction {
    &raw const victory_seh_boundary
}

/// Calls the win64 `function(a1, a2, a3, a4)` from a frame whose unwind data
/// leads to `resume`; `kind` (at frame + 0x20, `a4` at + 0x28) tells its
/// language handler what to answer.
///
/// # Safety
/// `function` is code with that signature; `resume` a context of this thread.
pub(crate) unsafe fn frame_call(
    resume: &Context,
    kind: u64,
    function: usize,
    a: [u64; 4],
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { victory_seh_frame_call(resume, kind, function, a[0], a[1], a[2], a[3]) }
}

/// Loads `context` and continues there.
///
/// # Safety
/// `context` is a state this thread may continue in; its stack is live.
pub(crate) unsafe fn restore(context: &Context) -> ! {
    // SAFETY: forwarded from the caller.
    unsafe { victory_seh_restore(context) }
}

/// `RtlCaptureContext`: the address title code can call.
pub(crate) fn capture_context_address() -> usize {
    victory_seh_capture_context as *const () as usize
}

/// The state of the caller of this function as it returns: a context for
/// tests.
#[cfg(test)]
#[inline(always)]
pub(crate) fn capture() -> Context {
    let mut c = Context::zeroed();
    // SAFETY: writes the context it is given.
    unsafe { victory_seh_capture_context(&mut c) };
    c
}

/// Reads the 8 bytes at `address`, or `None` where that would fault (with
/// the signal handler installed; without it the fault is a fault).
pub(crate) fn read64(address: u64) -> Option<u64> {
    let mut value = 0;
    // SAFETY: the probe either loads or reports the fault (see the asm).
    (unsafe { victory_seh_probe_read(address, &mut value) } != 0).then_some(value)
}

/// Whether the 8 bytes at `address` (8-aligned) can be written now; they
/// are left as they are. False where a write would fault (with the signal
/// handler installed).
pub(crate) fn writable64(address: u64) -> bool {
    // SAFETY: the probe either stores nothing new or reports the fault (see the asm).
    unsafe { victory_seh_probe_write(address) != 0 }
}

/// Where a fault at `rip` continues if it is a probe's load or store.
pub(crate) fn probe_recovery(rip: u64) -> Option<u64> {
    let (load, store, fail) = (
        &raw const victory_seh_probe_load as u64,
        &raw const victory_seh_probe_store as u64,
        &raw const victory_seh_probe_fail as u64,
    );
    (rip == load || rip == store).then_some(fail)
}

/// The copy's faulting instruction uses the probes' zero-return epilogue.
pub(super) fn copy_recovery(rip: u64) -> Option<u64> {
    (rip == &raw const victory_seh_copy_move as u64)
        .then_some(&raw const victory_seh_probe_fail as u64)
}
