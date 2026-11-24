//! The guest heap: `HeapAlloc` and friends, `LocalAlloc`/`GlobalAlloc`,
//! and the allocator the UCRT's `malloc` family uses ([`alloc`],
//! [`resize`], [`size`], [`free`]).
//!
//! Blocks are plain host `malloc` blocks: the pointer is glibc's own and
//! [`free`] is `free`. So memory the host allocates and the guest frees
//! (`free`, `HeapFree`, `LocalFree`, in any pairing) is freed right, and
//! the other way round. Blocks are 16-byte aligned, as on Windows x64.
//!
//! What this adds is the exact size: `HeapSize` and `_msize` report what
//! was asked for, as on Windows (`malloc_usable_size` alone gives the size
//! class). One byte more is requested and the slack from the asked size to
//! the end of the usable block is kept in its last byte (in the 8 bytes
//! before that when it does not fit). A block not made here has no such
//! byte: its size is then something from 0 to its usable size.
//!
//! Every heap handle is the process heap: `HeapCreate` is not imported.

#![allow(non_snake_case)] // named like the exports

use std::ffi::c_void;

use crate::HleModule;
use crate::hle_export;

const HEAP_ZERO_MEMORY: u32 = 0x8;
const HEAP_REALLOC_IN_PLACE_ONLY: u32 = 0x10;
const LMEM_ZEROINIT: u32 = 0x40;

/// The block's bytes, hidden from the optimiser: it knows how many bytes
/// `malloc` was asked for and the size record lies beyond them.
fn bytes(block: *const c_void) -> *mut u8 {
    std::hint::black_box(block as *mut u8)
}

/// Records that `block`, asked for with `size + 1` bytes, holds `size`.
fn stamp(block: *mut c_void, size: usize) -> *mut c_void {
    if block.is_null() {
        return block;
    }
    // SAFETY: a live malloc block.
    let usable = unsafe { libc::malloc_usable_size(block) };
    let slack = usable - size; // at least 1
    let p = bytes(block);
    // SAFETY: inside the usable block; the 8 bytes before the last are
    // beyond `size` when the slack is 255 or more.
    unsafe {
        p.add(usable - 1).write(slack.min(255) as u8);
        if slack >= 255 {
            p.add(usable - 9).cast::<usize>().write_unaligned(slack);
        }
    }
    block
}

/// A block of `size` bytes aligned to `align` (0: 16); null when out of
/// memory or `align` is not a power of two.
pub(crate) fn alloc(size: usize, align: usize, zero: bool) -> *mut c_void {
    if align & align.wrapping_sub(1) != 0 || size >= usize::MAX - align - 64 {
        return std::ptr::null_mut();
    }
    if align <= 16 {
        // SAFETY: plain allocation.
        let block = unsafe {
            if zero {
                libc::calloc(1, size + 1)
            } else {
                libc::malloc(size + 1)
            }
        };
        return stamp(block, size);
    }
    // SAFETY: a power of two; aligned_alloc rounds nothing for glibc.
    let block = unsafe { libc::aligned_alloc(align, size + 1) };
    if zero && !block.is_null() {
        // SAFETY: the block holds `size` bytes.
        unsafe { block.cast::<u8>().write_bytes(0, size) };
    }
    stamp(block, size)
}

/// `realloc`: null `block` allocates; contents kept up to the smaller
/// size; null on failure with `block` untouched.
pub(crate) fn resize(block: *mut c_void, size: usize, align: usize) -> *mut c_void {
    if align <= 16 {
        if size == usize::MAX {
            return std::ptr::null_mut();
        }
        // SAFETY: a malloc block or null.
        return stamp(unsafe { libc::realloc(block, size + 1) }, size);
    }
    // glibc has no aligned realloc.
    let fresh = alloc(size, align, false);
    if !fresh.is_null() && !block.is_null() {
        let keep = self::size(block).min(size);
        // SAFETY: both blocks hold `keep` bytes and are distinct.
        unsafe {
            std::ptr::copy_nonoverlapping(block.cast::<u8>(), fresh.cast::<u8>(), keep);
            libc::free(block);
        }
    }
    fresh
}

/// The size `block` was asked for.
pub(crate) fn size(block: *const c_void) -> usize {
    if block.is_null() {
        return 0;
    }
    // SAFETY: a live malloc block.
    let usable = unsafe { libc::malloc_usable_size(block as *mut c_void) };
    if usable == 0 {
        return 0;
    }
    let p = bytes(block);
    // SAFETY: inside the usable block.
    let mut slack = usize::from(unsafe { p.add(usable - 1).read() });
    if slack == 255 && usable >= 9 {
        // SAFETY: as above.
        slack = unsafe { p.add(usable - 9).cast::<usize>().read_unaligned() };
    }
    usable.saturating_sub(slack)
}

/// `HEAP_REALLOC_IN_PLACE_ONLY`: whether `block` can hold `size` bytes
/// where it is; that is then its size.
fn resize_in_place(block: *mut c_void, size: usize) -> bool {
    // SAFETY: a live malloc block (null is refused first).
    if block.is_null() || size >= unsafe { libc::malloc_usable_size(block) } {
        return false; // one byte is the size record's
    }
    stamp(block, size);
    true
}

pub(crate) fn free(block: *mut c_void) {
    // SAFETY: a malloc block or null.
    unsafe { libc::free(block) };
}

static PROCESS_HEAP: u8 = 0;

/// A stable non-null handle; nothing reads through it.
pub extern "win64" fn GetProcessHeap() -> *mut c_void {
    (&raw const PROCESS_HEAP).cast_mut().cast()
}

pub extern "win64" fn HeapAlloc(_heap: *mut c_void, flags: u32, size: usize) -> *mut c_void {
    alloc(size, 0, flags & HEAP_ZERO_MEMORY != 0)
}

pub extern "win64" fn HeapFree(_heap: *mut c_void, _flags: u32, block: *mut c_void) -> i32 {
    free(block);
    1
}

pub extern "win64" fn HeapReAlloc(
    _heap: *mut c_void,
    flags: u32,
    block: *mut c_void,
    size: usize,
) -> *mut c_void {
    if block.is_null() {
        return block;
    }
    let old = self::size(block);
    // The caller keeps using `block`: one that would have to move fails.
    let fresh = if flags & HEAP_REALLOC_IN_PLACE_ONLY != 0 {
        if !resize_in_place(block, size) {
            return std::ptr::null_mut();
        }
        block
    } else {
        resize(block, size, 0)
    };
    if !fresh.is_null() && flags & HEAP_ZERO_MEMORY != 0 && size > old {
        // SAFETY: the block holds `size` bytes.
        unsafe { fresh.cast::<u8>().add(old).write_bytes(0, size - old) };
    }
    fresh
}

pub extern "win64" fn HeapSize(_heap: *mut c_void, _flags: u32, block: *const c_void) -> usize {
    if block.is_null() {
        return usize::MAX;
    }
    size(block)
}

pub extern "win64" fn HeapValidate(_heap: *mut c_void, _flags: u32, _block: *const c_void) -> i32 {
    1
}

/// `LMEM_MOVEABLE` is a fixed block too: the handle is the pointer.
pub extern "win64" fn LocalAlloc(flags: u32, size: usize) -> *mut c_void {
    alloc(size, 0, flags & LMEM_ZEROINIT != 0)
}

/// Null is success.
pub extern "win64" fn LocalFree(block: *mut c_void) -> *mut c_void {
    free(block);
    std::ptr::null_mut()
}

pub extern "win64" fn GlobalAlloc(flags: u32, size: usize) -> *mut c_void {
    LocalAlloc(flags, size)
}

pub extern "win64" fn GlobalFree(block: *mut c_void) -> *mut c_void {
    LocalFree(block)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "kernel32-heap",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-heap-*"],
        exports: vec![
            hle_export!(GetProcessHeap),
            hle_export!(HeapAlloc),
            hle_export!(HeapFree),
            hle_export!(HeapReAlloc),
            hle_export!(HeapSize),
            hle_export!(HeapValidate),
            hle_export!(LocalAlloc),
            hle_export!(LocalFree),
            hle_export!(GlobalAlloc),
            hle_export!(GlobalFree),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_exact_and_blocks_aligned() {
        let heap = GetProcessHeap();
        for n in [0usize, 1, 15, 16, 17, 254, 255, 256, 300, 4096, 1 << 20] {
            let p = HeapAlloc(heap, HEAP_ZERO_MEMORY, n);
            assert!(!p.is_null());
            assert_eq!(p as usize % 16, 0);
            assert_eq!(HeapSize(heap, 0, p), n);
            // SAFETY: n zeroed bytes.
            let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) };
            assert!(bytes.iter().all(|&b| b == 0));
            assert_eq!(HeapFree(heap, 0, p), 1);
        }
        let a = alloc(100, 256, true);
        assert_eq!((a as usize % 256, size(a)), (0, 100));
        let b = resize(a, 5000, 256);
        assert_eq!((b as usize % 256, size(b)), (0, 5000));
        free(b);
    }

    #[test]
    fn realloc_keeps_contents_and_zeroes_growth() {
        let heap = GetProcessHeap();
        let p = HeapAlloc(heap, 0, 10).cast::<u8>();
        // SAFETY: 10 bytes.
        unsafe { p.write_bytes(0xAB, 10) };
        let q = HeapReAlloc(heap, HEAP_ZERO_MEMORY, p.cast(), 1000).cast::<u8>();
        assert_eq!(HeapSize(heap, 0, q.cast()), 1000);
        // SAFETY: 1000 bytes.
        let bytes = unsafe { std::slice::from_raw_parts(q, 1000) };
        assert!(bytes[..10].iter().all(|&b| b == 0xAB) && bytes[10..].iter().all(|&b| b == 0));
        // Shrinking in place works; growing past the block does not.
        assert_eq!(
            HeapReAlloc(heap, HEAP_REALLOC_IN_PLACE_ONLY, q.cast(), 3),
            q.cast()
        );
        assert_eq!(HeapSize(heap, 0, q.cast()), 3);
        assert!(HeapReAlloc(heap, HEAP_REALLOC_IN_PLACE_ONLY, q.cast(), 1 << 20).is_null());
        assert!(HeapReAlloc(heap, 0, std::ptr::null_mut(), 8).is_null());
        HeapFree(heap, 0, q.cast());
    }

    #[test]
    fn host_and_guest_blocks_mix() {
        // SAFETY: a host block freed by the guest's LocalFree.
        // calloc: size() reads the last usable byte (garbage in a recycled
        // block), and a reused chunk can be wider than 48, so only a floor holds.
        let host = unsafe { libc::calloc(1, 40) };
        assert!(size(host) >= 40);
        assert!(LocalFree(host).is_null());
        let g = GlobalAlloc(LMEM_ZEROINIT, 32);
        assert_eq!(size(g), 32);
        // SAFETY: the guest's block, freed by the host.
        unsafe { libc::free(g) };
    }
}
