//! Faults to exceptions: one handler for SIGSEGV, SIGBUS, SIGILL, SIGFPE
//! and SIGTRAP, on an alternate signal stack. A fault in title code becomes
//! an `EXCEPTION_RECORD` and a `CONTEXT` and is dispatched on the faulting
//! thread's own stack, below the faulting rsp, as Windows does. A fault in
//! our own code is not a title exception: it gets one line on stderr and
//! the handler that was there before ours (Rust's stack-overflow report, or
//! the default action).

use std::sync::Once;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use super::asm;
use super::context::*;
use super::{seh, unwind};

const SIGNALS: [i32; 5] = [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE, libc::SIGTRAP];
static INSTALL: Once = Once::new();
/// The actions before ours, by index in [`SIGNALS`].
static mut PREVIOUS: [std::mem::MaybeUninit<libc::sigaction>; 5] =
    [const { std::mem::MaybeUninit::uninit() }; 5];
/// The offset of the ymm upper halves in a standard-format XSAVE area.
static YMM_OFFSET: AtomicU32 = AtomicU32::new(0);
/// `fn(address) -> bool` of the region table: whether `address` was in a
/// `PAGE_GUARD` page (which it then opens).
static GUARD_HOOK: AtomicUsize = AtomicUsize::new(0);

/// Below StackLimit: the thread's guard page(s). A fault there is the stack
/// running out.
const GUARD_SPAN: u64 = 0x10000;
/// What a dispatch needs below the faulting rsp before it can call a handler.
const DISPATCH_ROOM: u64 = 0x6000;
const ALT_STACK_BYTES: usize = 64 * 1024;

/// Installs the handler, once per process; also gives the calling thread
/// an alternate signal stack. [`seh::register_image`] calls it.
pub fn install() {
    INSTALL.call_once(|| {
        // CPUID leaf 0xD sub-leaf 2: the AVX state component (zero where absent).
        let ymm = core::arch::x86_64::__cpuid_count(0xD, 2);
        YMM_OFFSET.store(ymm.ebx, Ordering::Relaxed);
        for (i, &signal) in SIGNALS.iter().enumerate() {
            // SAFETY: the previous action is saved before any fault can read it (the handler is
            // installed after).
            unsafe {
                let previous = &raw mut PREVIOUS;
                libc::sigaction(signal, &ours(), (*previous)[i].as_mut_ptr());
            }
        }
    });
    set_guard_page_hook(super::vm::take_guard);
    attach_thread();
}

/// Copies into host memory, returning false on an inaccessible source page.
/// A failed copy can have written a prefix of the destination.
///
/// # Safety
/// The signal handler must be installed. The destination must be writable
/// for `len` bytes and must not overlap the source.
pub unsafe fn copy_readable(dst: *mut u8, src: *const u8, len: usize) -> bool {
    // SAFETY: the destination and non-overlap are the caller's guarantees.
    unsafe { asm::victory_seh_copy_bytes(dst, src, len) != 0 }
}

fn ours() -> libc::sigaction {
    // SAFETY: a zeroed sigaction is valid.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = on_signal as *const () as usize;
    // SA_NODEFER: a probe inside the handler may fault and must reach it again.
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
    // SAFETY: our own set.
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    action
}

/// Puts our handler back where a library replaced it: DXC's LLVM installs
/// its own (`SA_RESETHAND`) at its first compile, after which the next
/// fault in title code, or write to a block the GPU side guards, ended the
/// process. What was there before ours stays the fallback.
pub fn reclaim() {
    if !INSTALL.is_completed() {
        return;
    }
    for &signal in &SIGNALS {
        // SAFETY: our action, as `install` set it.
        unsafe { libc::sigaction(signal, &ours(), std::ptr::null_mut()) };
    }
}

/// Gives the calling thread a 64 KiB alternate signal stack unless it has
/// one that large (std gives its threads a smaller one), freed when the
/// thread ends. Guest threads
/// call it as they start, so a stack overflow can still be reported.
pub fn attach_thread() {
    struct AltStack(*mut libc::c_void);
    impl Drop for AltStack {
        fn drop(&mut self) {
            let disable = libc::stack_t { ss_sp: std::ptr::null_mut(), ss_flags: libc::SS_DISABLE, ss_size: 0 };
            // SAFETY: switches this thread's alternate stack off, then frees ours.
            unsafe {
                libc::sigaltstack(&disable, std::ptr::null_mut());
                libc::munmap(self.0, ALT_STACK_BYTES);
            }
        }
    }
    thread_local! {
        static ALT: std::cell::RefCell<Option<AltStack>> = const { std::cell::RefCell::new(None) };
    }
    // SAFETY: queries the current alternate stack, and sets one we own if there is none.
    unsafe {
        let mut current: libc::stack_t = std::mem::zeroed();
        libc::sigaltstack(std::ptr::null(), &mut current);
        if current.ss_flags & libc::SS_DISABLE == 0 && current.ss_size >= ALT_STACK_BYTES {
            return;
        }
        let memory = libc::mmap(
            std::ptr::null_mut(),
            ALT_STACK_BYTES,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if memory == libc::MAP_FAILED {
            return; // the handler then runs on the thread's stack; only an overflow is lost
        }
        let stack = libc::stack_t { ss_sp: memory, ss_flags: 0, ss_size: ALT_STACK_BYTES };
        libc::sigaltstack(&stack, std::ptr::null_mut());
        ALT.with(|a| *a.borrow_mut() = Some(AltStack(memory)));
    }
}

/// Lets the region table say which faults are `PAGE_GUARD` pages: `hook`
/// gets the address of a write or read fault in title code and returns
/// true if it was in a guard page (whose guard it has removed); the fault
/// is then raised as `STATUS_GUARD_PAGE_VIOLATION`. It runs in the signal
/// handler: async-signal-safe work only, and no lock title code could hold.
pub fn set_guard_page_hook(hook: fn(usize) -> bool) {
    GUARD_HOOK.store(hook as usize, Ordering::Release);
}

/// The registers and the FXSAVE image the kernel saved, as a CONTEXT.
fn context_from(uc: &libc::ucontext_t, c: &mut Context) {
    let g = &uc.uc_mcontext.gregs;
    // SAFETY: all-zero is a valid Context; written in place (no stack copy).
    unsafe { std::ptr::write_bytes(c as *mut Context, 0, 1) };
    c.context_flags = CONTEXT_FULL | CONTEXT_SEGMENTS;
    // What a Windows x64 thread has: user code and data, the TEB segment in fs for 32-bit code.
    c.seg_cs = 0x33;
    (c.seg_ds, c.seg_es, c.seg_gs, c.seg_ss) = (0x2B, 0x2B, 0x2B, 0x2B);
    c.seg_fs = 0x53;
    c.eflags = g[libc::REG_EFL as usize] as u32;
    const ORDER: [i32; 16] = [
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
    for (r, &at) in c.gpr.iter_mut().zip(&ORDER) {
        *r = g[at as usize] as u64;
    }
    c.rip = g[libc::REG_RIP as usize] as u64;
    let fp = uc.uc_mcontext.fpregs;
    if fp.is_null() {
        c.context_flags &= !0x8; // no floating-point state to restore
        return;
    }
    let frame = fp as *const u8;
    // SAFETY: the kernel's FXSAVE image (512 bytes), then, when its software bytes say so, the
    // XSAVE header and components in the standard format.
    unsafe {
        std::ptr::copy_nonoverlapping(frame, &mut c.flt_save as *mut XmmSaveArea32 as *mut u8, 512);
        c.mx_csr = c.flt_save.mx_csr;
        let magic = (frame.add(464) as *const u32).read_unaligned();
        let sw_bv = (frame.add(472) as *const u64).read_unaligned();
        let size = (frame.add(480) as *const u32).read_unaligned();
        let bv = (frame.add(512) as *const u64).read_unaligned();
        let comp = (frame.add(520) as *const u64).read_unaligned();
        let offset = YMM_OFFSET.load(Ordering::Relaxed);
        if magic == 0x4650_5853 && sw_bv & 4 != 0 && comp >> 63 == 0 && offset != 0 && size >= offset + 256 {
            // Component 2: the upper halves of ymm0..15; its bit clear means all zero.
            if bv & 4 != 0 {
                std::ptr::copy_nonoverlapping(
                    frame.add(offset as usize),
                    c.vector_register.as_mut_ptr() as *mut u8,
                    256,
                );
            }
            c.vector_control = if bv & 4 != 0 { YMM_SAVED } else { YMM_CLEAN };
        }
    }
}

/// SIGFPE's `si_code`s (FPE_INTDIV ..), as Linux numbers them.
fn float_code(si_code: i32) -> u32 {
    match si_code {
        1 => STATUS_INTEGER_DIVIDE_BY_ZERO,
        2 => STATUS_INTEGER_OVERFLOW,
        3 => STATUS_FLOAT_DIVIDE_BY_ZERO,
        4 => STATUS_FLOAT_OVERFLOW,
        5 => STATUS_FLOAT_UNDERFLOW,
        6 => STATUS_FLOAT_INEXACT_RESULT,
        _ => STATUS_FLOAT_INVALID_OPERATION,
    }
}

/// What the one handler knows about a fault.
struct Fault {
    signal: i32,
    si_code: i32,
    trap: u64,
    error: u64,
    /// The address accessed, for a page fault.
    address: u64,
}

/// The exception Windows reports for this trap; `c` is adjusted where
/// Windows reports another context than Linux saved (the breakpoint's Rip,
/// a single step's trap flag).
fn record_from(f: &Fault, c: &mut Context, r: &mut ExceptionRecord) {
    // SAFETY: all-zero is a valid record (a null chain pointer).
    unsafe { std::ptr::write_bytes(r as *mut ExceptionRecord, 0, 1) };
    let mut access_violation = |kind: usize, address: usize| {
        r.code = STATUS_ACCESS_VIOLATION;
        r.number_parameters = 2;
        r.information[0] = kind;
        r.information[1] = address;
    };
    let write = f.error & 2 != 0;
    let execute = f.error & 0x10 != 0;
    let access = if execute { 8 } else { usize::from(write) };
    match f.trap {
        // ponytail: Windows tells a zero divisor from a quotient that does not fit (INT_MIN / -1 is
        // STATUS_INTEGER_OVERFLOW) by decoding the instruction; both are divide by zero here.
        0 => r.code = STATUS_INTEGER_DIVIDE_BY_ZERO,
        1 => {
            r.code = STATUS_SINGLE_STEP;
            c.eflags &= !0x100; // reported with the trap flag already clear
        }
        3 => {
            // Linux saves the address after the int3, Windows reports the int3 itself.
            r.code = STATUS_BREAKPOINT;
            c.rip -= 1;
            r.number_parameters = 1;
        }
        6 => r.code = STATUS_ILLEGAL_INSTRUCTION,
        12 | 13 if f.trap == 13 && f.error == (0x29 << 3) | 2 => {
            // int 0x29 through the IDT: __fastfail(rcx).
            r.code = STATUS_STACK_BUFFER_OVERRUN;
            r.flags = EXCEPTION_NONCONTINUABLE;
            r.number_parameters = 1;
            r.information[0] = c.gpr[RCX] as usize;
        }
        12 | 13 => access_violation(0, usize::MAX), // a non-canonical address, as Windows reports it
        14 if f.signal == libc::SIGBUS => {
            // A mapped file without the page behind it.
            r.code = STATUS_IN_PAGE_ERROR;
            r.number_parameters = 3;
            r.information[..3].copy_from_slice(&[access, f.address as usize, 0xC000_0001]);
        }
        14 => access_violation(access, f.address as usize),
        16 | 19 => r.code = float_code(f.si_code),
        17 => r.code = STATUS_DATATYPE_MISALIGNMENT,
        _ => match f.signal {
            libc::SIGILL => r.code = STATUS_ILLEGAL_INSTRUCTION,
            libc::SIGFPE => r.code = float_code(f.si_code),
            libc::SIGTRAP => r.code = STATUS_BREAKPOINT,
            _ => access_violation(0, f.address as usize),
        },
    }
    r.address = c.rip as usize;
}

/// Where a title fault continues once the handler returns: on its own
/// stack, outside signal context, as if called from the faulting code.
extern "C" fn fault_entry(record: *mut ExceptionRecord, context: *mut Context) -> ! {
    // SAFETY: the handler put both on this stack just above us.
    let (record, context) = unsafe { (&mut *record, &mut *context) };
    if record.code == STATUS_STACK_BUFFER_OVERRUN {
        seh::terminate(record, context); // a fast fail is offered to no handler
    }
    seh::dispatch(record, context)
}

/// Writes to stderr without allocating.
struct Line {
    bytes: [u8; 256],
    len: usize,
}

impl std::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let n = s.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// A fault that is not a title exception: say so, then pass it to the
/// handler before ours, or the default action.
///
/// # Safety
/// The arguments of the signal handler.
unsafe fn host_fault(signal: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void, rip: u64, address: u64) {
    use std::fmt::Write;
    let mut line = Line { bytes: [0; 256], len: 0 };
    let _ = writeln!(
        line,
        "VIctory: signal {signal} at {rip:#x} (address {address:#x}): a fault in our own code, not a title exception"
    );
    // SAFETY: write(2) of our buffer.
    unsafe { libc::write(2, line.bytes.as_ptr().cast(), line.len) };
    let index = SIGNALS.iter().position(|&s| s == signal).unwrap_or(0);
    // SAFETY: PREVIOUS was written before our handler was installed.
    let previous = unsafe { PREVIOUS[index].assume_init() };
    let handler = previous.sa_sigaction;
    if handler == libc::SIG_IGN {
        return;
    }
    if handler != libc::SIG_DFL {
        // SAFETY: the handler that was installed for this signal, called as it expects.
        unsafe {
            if previous.sa_flags & libc::SA_SIGINFO != 0 {
                let f: extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) = std::mem::transmute(handler);
                f(signal, info, raw);
            } else {
                let f: extern "C" fn(i32) = std::mem::transmute(handler);
                f(signal);
            }
        }
        return;
    }
    // SAFETY: back to the default action; a fault repeats when the instruction restarts, a
    // trap or a signal someone sent does not.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = libc::SIG_DFL;
        libc::sigaction(signal, &action, std::ptr::null_mut());
        if signal == libc::SIGTRAP || (*info).si_code <= 0 {
            libc::raise(signal);
        }
    }
}

extern "C" fn on_signal(signal: i32, info: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid siginfo and ucontext.
    let (si, uc) = unsafe { (&*info, &mut *(raw as *mut libc::ucontext_t)) };
    let g = uc.uc_mcontext.gregs;
    let rip = g[libc::REG_RIP as usize] as u64;
    if let Some(to) = asm::probe_recovery(rip) {
        uc.uc_mcontext.gregs[libc::REG_RIP as usize] = to as i64;
        return;
    }
    let trap = g[libc::REG_TRAPNO as usize] as u64;
    // A write to a block the GPU side guards, by title code or ours: taken
    // before anyone sees it, and the write is retried.
    if signal == libc::SIGSEGV
        && trap == 14
        && g[libc::REG_ERR as usize] & 2 != 0
        && super::watch::take_write(g[libc::REG_CR2 as usize] as usize)
    {
        return;
    }
    if (signal == libc::SIGSEGV || signal == libc::SIGBUS)
        && si.si_code > 0
        && let Some(to) = asm::copy_recovery(rip)
    {
        uc.uc_mcontext.gregs[libc::REG_RIP as usize] = to as i64;
        return;
    }
    let fault = Fault {
        signal,
        si_code: si.si_code,
        trap,
        error: g[libc::REG_ERR as usize] as u64,
        address: if trap == 14 { g[libc::REG_CR2 as usize] as u64 } else { 0 },
    };
    if si.si_code <= 0 || unwind::is_host_code(rip as usize) {
        // SAFETY: our arguments.
        unsafe { host_fault(signal, info, raw, rip, fault.address) };
        return;
    }

    let rsp = g[libc::REG_RSP as usize] as u64;
    // The stack the title thread runs on, from its TEB: an overflow ends the process from here,
    // as far as one gets, when there is no room left to dispatch on.
    let mut overflow = false;
    if let Some((limit, base)) = seh::stack_bounds() {
        let on_stack = rsp >= limit - GUARD_SPAN && rsp < base;
        overflow = on_stack
            && (rsp < limit || trap == 14 && fault.address >= limit - GUARD_SPAN && fault.address < limit);
        if on_stack && rsp < limit + DISPATCH_ROOM {
            fatal(STATUS_STACK_OVERFLOW, rip);
        }
    }

    // The record and context go on the thread's own stack below the faulting rsp (title code
    // has no red zone), or below this handler if it runs on that stack too, and are built in
    // place: the alternate stack std gives a thread is small. The thread goes on in fault_entry
    // as if called from there.
    let marker = 0u8;
    let here = &marker as *const u8 as u64;
    let top = if here < rsp && rsp - here < 1 << 20 { here - 0x2000 } else { rsp };
    let context_at = (top - size_of::<Context>() as u64) & !15;
    let record_at = (context_at - size_of::<ExceptionRecord>() as u64) & !15;
    let sp = record_at - 8;
    if asm::read64(sp & !0xFFF).is_none() || asm::read64((context_at + 0x4C8) & !0xFFF).is_none() {
        fatal(STATUS_STACK_OVERFLOW, rip); // the title destroyed rsp: nothing to dispatch on
    }
    // SAFETY: below the faulting thread's live stack, and mapped (checked above).
    let (context, record) = unsafe { (&mut *(context_at as *mut Context), &mut *(record_at as *mut ExceptionRecord)) };
    context_from(uc, context);
    record_from(&fault, context, record);
    if overflow {
        record.code = STATUS_STACK_OVERFLOW; // a probe below a stack that still has room
    } else if record.code == STATUS_ACCESS_VIOLATION && signal == libc::SIGSEGV {
        let hook = GUARD_HOOK.load(Ordering::Acquire);
        // SAFETY: a hook the region table set, with the signature it was set with.
        if hook != 0 && unsafe { std::mem::transmute::<usize, fn(usize) -> bool>(hook) }(fault.address as usize) {
            record.code = STATUS_GUARD_PAGE_VIOLATION;
        }
    }
    // SAFETY: as above. fault_entry has no caller: a zero return address ends a walk there.
    unsafe { (sp as *mut u64).write(0) };
    let g = &mut uc.uc_mcontext.gregs;
    g[libc::REG_RIP as usize] = fault_entry as *const () as usize as i64;
    g[libc::REG_RDI as usize] = record_at as i64;
    g[libc::REG_RSI as usize] = context_at as i64;
    g[libc::REG_RSP as usize] = sp as i64; // as after a call: 16-aligned plus the return slot
    g[libc::REG_EFL as usize] &= !0x500; // our code runs with the direction and trap flags clear
}

/// Ends the process from the signal handler, where no report can be built
/// (no stack left): one line on stderr, the exception code as exit status.
fn fatal(code: u32, rip: u64) -> ! {
    use std::fmt::Write;
    let mut line = Line { bytes: [0; 256], len: 0 };
    let _ = writeln!(line, "VIctory: unhandled exception {code:#010x} at {rip:#x} (no stack left to dispatch on)");
    // SAFETY: write(2) of our buffer, then the end of the process.
    unsafe {
        libc::write(2, line.bytes.as_ptr().cast(), line.len);
        libc::_exit(code as i32)
    }
}
