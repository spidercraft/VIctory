//! kernel32: interlocked singly linked lists (`SLIST_HEADER`), lock-free
//! and in the x64 layout, since title code may read the header inline: 16
//! bytes, 16-aligned; depth in the low 16 bits, a sequence number in the
//! next 48, then the first entry. Every push and pop changes the
//! sequence, so a pop is not fooled by an entry popped and pushed back in
//! between (ABA).
//!
//! `InterlockedPopEntrySList` reads the first entry's `Next` before it
//! knows the entry is still listed. Windows survives another thread having
//! popped and unmapped it meanwhile by catching the fault; here that read
//! faults for real. Freed but mapped memory (the usual case) is harmless:
//! the exchange then fails.

#![allow(non_snake_case)]

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::SeqCst};

use crate::{HleModule, hle_export};

/// `SLIST_HEADER`.
#[repr(C, align(16))]
pub struct SListHeader {
    depth_sequence: u64,
    first: usize,
}

/// `SLIST_ENTRY`.
#[repr(C)]
pub struct SListEntry {
    next: *mut SListEntry,
}

fn pack(depth_sequence: u64, first: usize) -> u128 {
    u128::from(depth_sequence) | (first as u128) << 64
}

/// The header word after a change of `delta` to the depth: sequence + 1.
fn stepped(depth_sequence: u64, delta: i32) -> u64 {
    ((depth_sequence + 0x1_0000) & !0xFFFF) | u64::from((depth_sequence as i32 + delta) as u16)
}

/// The header as two halves read one after the other; a torn pair fails
/// the exchange, which compares both.
fn read(header: *mut SListHeader) -> (u64, usize) {
    // SAFETY: the title's header, 16-aligned and live for the call.
    unsafe {
        (
            AtomicU64::from_ptr(&raw mut (*header).depth_sequence).load(SeqCst),
            AtomicUsize::from_ptr(&raw mut (*header).first).load(SeqCst),
        )
    }
}

/// Replaces `old` with `new` if the header still holds `old`.
fn exchange(header: *mut SListHeader, old: (u64, usize), new: (u64, usize)) -> Result<(), (u64, usize)> {
    let (old, new) = (pack(old.0, old.1), pack(new.0, new.1));
    // SAFETY: cmpxchg16b exists on every processor Windows 8.1+ x64 runs
    // on; the header is 16-aligned.
    let seen = unsafe { cas16(header.cast(), old, new) };
    if seen == old { Ok(()) } else { Err((seen as u64, (seen >> 64) as usize)) }
}

/// `lock cmpxchg16b`: what `target` held (`old` when it was replaced).
/// Inline assembly, since the intrinsic becomes a libatomic call in builds
/// that do not inline it.
unsafe fn cas16(target: *mut u128, old: u128, new: u128) -> u128 {
    let (mut low, mut high) = (old as u64, (old >> 64) as u64);
    // SAFETY: by the caller; rbx (reserved by LLVM) is swapped in and back.
    unsafe {
        std::arch::asm!(
            "xchg rsi, rbx",
            "lock cmpxchg16b [rdi]",
            "mov rbx, rsi",
            // Fixed registers: with `reg`, LLVM may hand out rbx for an
            // operand, which the swap above then overwrites.
            in("rdi") target,
            inout("rsi") new as u64 => _,
            in("rcx") (new >> 64) as u64,
            inout("rax") low,
            inout("rdx") high,
            options(nostack),
        )
    };
    u128::from(low) | u128::from(high) << 64
}

pub extern "win64" fn InitializeSListHead(header: *mut SListHeader) {
    // SAFETY: the title's header.
    unsafe { header.write(SListHeader { depth_sequence: 0, first: 0 }) };
}

pub extern "win64" fn QueryDepthSList(header: *mut SListHeader) -> u16 {
    read(header).0 as u16
}

/// Returns the entry that was first before.
pub extern "win64" fn InterlockedPushEntrySList(header: *mut SListHeader, entry: *mut SListEntry) -> *mut SListEntry {
    let mut old = read(header);
    loop {
        // SAFETY: the title's entry, not listed yet.
        unsafe { (*entry).next = old.1 as *mut SListEntry };
        match exchange(header, old, (stepped(old.0, 1), entry as usize)) {
            Ok(()) => return old.1 as *mut SListEntry,
            Err(now) => old = now,
        }
    }
}

pub extern "win64" fn InterlockedPopEntrySList(header: *mut SListHeader) -> *mut SListEntry {
    let mut old = read(header);
    loop {
        if old.1 == 0 {
            return std::ptr::null_mut();
        }
        // SAFETY: an entry that was listed; see the module docs.
        let next = unsafe { (*(old.1 as *const SListEntry)).next } as usize;
        match exchange(header, old, (stepped(old.0, -1), next)) {
            Ok(()) => return old.1 as *mut SListEntry,
            Err(now) => old = now,
        }
    }
}

/// Takes the whole list.
pub extern "win64" fn InterlockedFlushSList(header: *mut SListHeader) -> *mut SListEntry {
    let mut old = read(header);
    loop {
        if old.1 == 0 {
            return std::ptr::null_mut();
        }
        match exchange(header, old, (stepped(old.0, 0) & !0xFFFF, 0)) {
            Ok(()) => return old.1 as *mut SListEntry,
            Err(now) => old = now,
        }
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernel32-interlocked",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-interlocked-*"],
        exports: vec![
            hle_export!(InitializeSListHead),
            hle_export!(QueryDepthSList),
            hle_export!(InterlockedPushEntrySList),
            hle_export!(InterlockedPopEntrySList),
            hle_export!(InterlockedFlushSList),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of};

    #[test]
    fn guest_layouts_match_windows() {
        assert_eq!((size_of::<SListHeader>(), align_of::<SListHeader>()), (16, 16));
        assert_eq!(offset_of!(SListHeader, first), 8);
        assert_eq!(size_of::<SListEntry>(), 8);
    }

    #[test]
    fn push_pop_flush_and_depth() {
        let header = Box::leak(Box::new(SListHeader { depth_sequence: 7, first: 0 }));
        InitializeSListHead(header);
        let entries: Vec<_> = (0..3).map(|_| Box::leak(Box::new(SListEntry { next: std::ptr::null_mut() })) as *mut _).collect();
        assert!(InterlockedPushEntrySList(header, entries[0]).is_null());
        assert_eq!(InterlockedPushEntrySList(header, entries[1]), entries[0]);
        assert_eq!(QueryDepthSList(header), 2);
        assert_eq!(InterlockedPopEntrySList(header), entries[1]);
        InterlockedPushEntrySList(header, entries[2]);
        assert_eq!(InterlockedFlushSList(header), entries[2]);
        assert_eq!(QueryDepthSList(header), 0);
        assert!(InterlockedPopEntrySList(header).is_null());
        // The sequence moved on with every change.
        assert_eq!(header.depth_sequence >> 16, 5);
    }

    #[test]
    fn concurrent_pushes_and_pops_keep_every_entry() {
        let header = Box::leak(Box::new(SListHeader { depth_sequence: 0, first: 0 })) as *mut SListHeader as usize;
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(move || {
                    let header = header as *mut SListHeader;
                    let mine: Vec<usize> = (0..64)
                        .map(|_| Box::leak(Box::new(SListEntry { next: std::ptr::null_mut() })) as *mut _ as usize)
                        .collect();
                    for _ in 0..200 {
                        for &e in &mine {
                            InterlockedPushEntrySList(header, e as *mut _);
                        }
                        for _ in 0..mine.len() {
                            while InterlockedPopEntrySList(header).is_null() {}
                        }
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(QueryDepthSList(header as *mut _), 0);
    }
}
