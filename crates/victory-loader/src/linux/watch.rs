//! Write watching for the GPU side's copies of title memory (Linux has no
//! `GetWriteWatch`). A 64 KiB block a copy took is made read-only and
//! noted guarded; the first write to it, from title code or ours, is taken
//! in the fault handler before anything else sees it ([`take_write`]): the
//! block is writable again and no longer guarded, which is what "written"
//! means. [`super::vm::guard`] asks and guards again, and keeps the guard
//! out of what `VirtualQuery` and `VirtualProtect` see.
//!
//! The kernel does not fault into the handler: a system call writing a
//! guarded block fails (`read(2)` returns `EFAULT`), so whoever lets the
//! kernel write title memory opens it first ([`open`]).
//!
//! What the handler touches is lock-free and allocates nothing: one bit per
//! block of the address space, in a reservation whose pages cost memory
//! once a block near them is guarded.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use super::vm::HIGHEST;

pub const BLOCK: usize = 0x1_0000;
const WORDS: usize = (HIGHEST + 1) / BLOCK / 64;

static BITS: AtomicPtr<AtomicU64> = AtomicPtr::new(std::ptr::null_mut());
/// Writes taken by the handler, for the diagnostics line.
static FAULTS: AtomicU64 = AtomicU64::new(0);

fn bits() -> Option<&'static [AtomicU64]> {
    let p = BITS.load(Ordering::Acquire);
    // SAFETY: once set, a mapping of WORDS words that is never unmapped.
    (!p.is_null()).then(|| unsafe { std::slice::from_raw_parts(p, WORDS) })
}

/// The bits, reserved on first use (never from the handler).
fn made() -> &'static [AtomicU64] {
    if let Some(b) = bits() {
        return b;
    }
    // SAFETY: a fresh anonymous mapping; zero pages are read until written.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            WORDS * 8,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    assert!(p != libc::MAP_FAILED, "no room for the write-watch bits");
    if BITS
        .compare_exchange(std::ptr::null_mut(), p.cast(), Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        // SAFETY: ours, never published.
        unsafe { libc::munmap(p, WORDS * 8) };
    }
    bits().expect("set")
}

/// Whether block number `block` is guarded.
pub(crate) fn guarded(block: usize) -> bool {
    bits().is_some_and(|b| block / 64 < WORDS && b[block / 64].load(Ordering::Acquire) >> (block % 64) & 1 != 0)
}

/// Appends the unguarded blocks of a private run, skipping clean bitmap words.
pub(crate) fn extend_unguarded(first: usize, last: usize, out: &mut Vec<usize>) {
    let Some(bits) = bits() else {
        out.extend(first..last);
        return;
    };
    let mut block = first;
    while block < last {
        let word = block / 64;
        let base = word * 64;
        let end = last.min(base + 64);
        let mut dirty = !bits.get(word).map_or(0, |bits| bits.load(Ordering::Acquire));
        if dirty != 0 {
            dirty &= (u64::MAX << (block - base)) & (u64::MAX >> (64 - (end - base)));
            while dirty != 0 {
                out.push(base + dirty.trailing_zeros() as usize);
                dirty &= dirty - 1;
            }
        }
        block = end;
    }
}

/// Notes `block` guarded; the caller then makes it read-only.
pub(crate) fn mark(block: usize) {
    made()[block / 64].fetch_or(1 << (block % 64), Ordering::AcqRel);
}

/// Notes `block` not guarded, when making it read-only failed.
pub(crate) fn forget(block: usize) {
    clear(made(), block);
}

fn clear(bits: &[AtomicU64], block: usize) -> bool {
    bits[block / 64].fetch_and(!(1 << (block % 64)), Ordering::AcqRel) >> (block % 64) & 1 != 0
}

fn writable(block: usize) -> bool {
    // SAFETY: a guarded block is committed read-write memory of the region
    // table; only its protection changes back.
    unsafe {
        libc::mprotect((block * BLOCK) as *mut _, BLOCK, libc::PROT_READ | libc::PROT_WRITE) == 0
    }
}

/// Opens guarded `block`: writable, and no longer guarded (written). The
/// protection goes back before the bit, so a thread that faulted on the
/// block meanwhile and finds the bit clear finds the block writable too.
fn open_block(bits: &[AtomicU64], block: usize) -> bool {
    if bits[block / 64].load(Ordering::Acquire) >> (block % 64) & 1 == 0 {
        return false;
    }
    writable(block);
    clear(bits, block);
    true
}

/// Before the kernel (or anything that does not fault into our handler)
/// writes `[at, at + len)`: opens the guarded blocks there.
pub fn open(at: usize, len: usize) {
    let Some(bits) = bits() else {
        return;
    };
    let end = (at.saturating_add(len)).div_ceil(BLOCK).min(WORDS * 64);
    let mut block = at / BLOCK;
    while block < end {
        if block.is_multiple_of(64) && bits[block / 64].load(Ordering::Relaxed) == 0 {
            block += 64; // nothing guarded in this word (releases of big reservations)
            continue;
        }
        open_block(bits, block);
        block += 1;
    }
}

/// For the fault handler, a write fault at `address`: true when it was a
/// guarded block (now open) or one another thread opened since (writable
/// now), so the write is retried. Async-signal-safe.
pub(crate) fn take_write(address: usize) -> bool {
    let Some(bits) = bits() else {
        return false;
    };
    let block = address / BLOCK;
    if block / 64 >= WORDS {
        return false;
    }
    if open_block(bits, block) {
        FAULTS.fetch_add(1, Ordering::Relaxed);
        return true;
    }
    super::asm::writable64(address as u64 & !7)
}

/// Writes the handler took since the last call.
pub fn take_faults() -> u64 {
    FAULTS.swap(0, Ordering::Relaxed)
}
