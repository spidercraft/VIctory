//! The region table behind `VirtualAlloc`, `VirtualFree`, `VirtualProtect`
//! and `VirtualQuery`: Windows virtual-memory semantics over `mmap`,
//! `mprotect` and `madvise`, one table for the whole process.
//!
//! - A new allocation's base is a multiple of 64 KiB ([`GRANULARITY`]);
//!   sizes and sub-ranges are whole 4 KiB pages ([`PAGE`]).
//! - `MEM_RESERVE` alone is address space only (`PROT_NONE`,
//!   `MAP_NORESERVE`). `MEM_COMMIT` gives zero pages on first touch;
//!   `MEM_DECOMMIT` drops the pages at once, a later commit reads zero.
//! - Every page knows its allocation base, the allocation's protection, its
//!   state and its own protection, so [`query`] answers one
//!   `MEMORY_BASIC_INFORMATION` per run of like pages in one allocation.
//! - `MEM_TOP_DOWN` changes nothing: the kernel already places a mapping at
//!   the highest free range. `MEM_RESET` is `MADV_FREE`, `MEM_RESET_UNDO`
//!   succeeds, `MEM_WRITE_WATCH` is `ERROR_NOT_SUPPORTED`.
//! - `MEM_LARGE_PAGES` (reserve and commit together, a multiple of 2 MiB):
//!   `MAP_HUGETLB` when the host has huge pages reserved, else normal pages
//!   with `MADV_HUGEPAGE`.
//! - `PAGE_GUARD` pages are `PROT_NONE`; the fault handler asks
//!   [`take_guard`], which lifts the guard once, as Windows does.
//! - Memory the table did not hand out (the host heap, libraries, thread
//!   stacks) is answered from the kernel's own table: committed with the
//!   host's protection, or reserved for `PROT_NONE`. A walk over the whole
//!   address space sees every mapping once and ends at [`HIGHEST`].
//!
//! Placeholders (`MEM_RESERVE_PLACEHOLDER`, `MEM_REPLACE_PLACEHOLDER`,
//! `MEM_PRESERVE_PLACEHOLDER`, `MEM_COALESCE_PLACEHOLDERS`) are bookkeeping:
//! replacing part of one splits it implicitly.
//!
//! Shared memory, for `CreateFileMapping`/`MapViewOfFile3` and the xmem
//! physical pages: a [`Section`] is a `memfd`; [`map_view`] maps it (into a
//! placeholder or free space), [`unmap_view`] takes it back. The physical
//! pool ([`phys_alloc`], [`phys_map`]) is one growing section of 64 KiB
//! pages mapped into a reservation; [`alias`] shows whatever section backs
//! one range at another as well, and `MEM_DECOMMIT` unmaps it again. On
//! Linux xmem needs no placeholders: reserve, `phys_map`, decommit. A
//! placeholder the title reserves itself takes physical pages too: the
//! mapped range becomes a reservation of its own, as a view would.
//!
//! Write watching for the GPU side ([`guard`], `watch`): blocks it copied
//! are read-only underneath the table, which goes on answering the
//! protection the title set. Every change of protection or mapping opens
//! the guarded blocks it touches first, and the table knows which blocks
//! show the same section pages, so a write through any of them is seen.
//!
//! Errors are Windows codes; the kernel32 layer sets the last error.
//! Thread-safe: one leaf lock, never held across a read of `/proc`.

use std::collections::BTreeMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// `SYSTEM_INFO.dwPageSize`.
pub const PAGE: usize = 0x1000;
/// `SYSTEM_INFO.dwAllocationGranularity`.
pub const GRANULARITY: usize = 0x1_0000;
/// `GetLargePageMinimum`.
pub const LARGE_PAGE: usize = 0x20_0000;
/// One page of the physical pool.
pub const PHYS_PAGE: usize = 0x1_0000;
/// `lpMinimumApplicationAddress`.
pub const LOWEST: usize = 0x1_0000;
/// `lpMaximumApplicationAddress`.
pub const HIGHEST: usize = 0x7FFF_FFFE_FFFF;

pub const PAGE_NOACCESS: u32 = 0x01;
pub const PAGE_READONLY: u32 = 0x02;
pub const PAGE_READWRITE: u32 = 0x04;
pub const PAGE_WRITECOPY: u32 = 0x08;
pub const PAGE_EXECUTE: u32 = 0x10;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;
pub const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;
pub const PAGE_GUARD: u32 = 0x100;
pub const PAGE_NOCACHE: u32 = 0x200;
pub const PAGE_WRITECOMBINE: u32 = 0x400;

pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;
pub const MEM_DECOMMIT: u32 = 0x4000;
pub const MEM_RELEASE: u32 = 0x8000;
pub const MEM_FREE: u32 = 0x1_0000;
pub const MEM_PRIVATE: u32 = 0x2_0000;
pub const MEM_MAPPED: u32 = 0x4_0000;
pub const MEM_RESET: u32 = 0x8_0000;
pub const MEM_TOP_DOWN: u32 = 0x10_0000;
pub const MEM_WRITE_WATCH: u32 = 0x20_0000;
pub const MEM_RESET_UNDO: u32 = 0x100_0000;
pub const MEM_IMAGE: u32 = 0x100_0000;
pub const MEM_LARGE_PAGES: u32 = 0x2000_0000;
/// `VirtualAlloc2`.
pub const MEM_REPLACE_PLACEHOLDER: u32 = 0x4000;
/// `VirtualAlloc2`.
pub const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
/// `VirtualFree` with `MEM_RELEASE`.
pub const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;
/// `VirtualFree` with `MEM_RELEASE`.
pub const MEM_PRESERVE_PLACEHOLDER: u32 = 0x2;

pub const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
pub const ERROR_NOT_SUPPORTED: u32 = 50;
pub const ERROR_INVALID_PARAMETER: u32 = 87;
pub const ERROR_INVALID_ADDRESS: u32 = 487;

/// `MEMORY_BASIC_INFORMATION` as title code reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryBasicInformation {
    pub base_address: usize,
    pub allocation_base: usize,
    pub allocation_protect: u32,
    pub partition_id: u16,
    pub region_size: usize,
    /// `MEM_COMMIT`, `MEM_RESERVE` or `MEM_FREE`.
    pub state: u32,
    /// `PAGE_*`; 0 while reserved, `PAGE_NOACCESS` when free.
    pub protect: u32,
    /// `MEM_PRIVATE`, `MEM_MAPPED` or `MEM_IMAGE`; 0 when free.
    pub kind: u32,
}

/// A `memfd` that views map: a file mapping object, or the physical pool.
/// Its memory goes back to the host once it and every view are gone.
#[derive(Debug)]
pub struct Section {
    fd: OwnedFd,
    size: AtomicU64,
}

impl Section {
    /// A section of `size` zero bytes; untouched pages cost nothing.
    pub fn new(size: u64) -> Result<Arc<Section>, u32> {
        // SAFETY: a NUL-terminated name and plain flags.
        let fd = unsafe { libc::memfd_create(c"victory-section".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(ERROR_NOT_ENOUGH_MEMORY);
        }
        // SAFETY: a descriptor we just opened and nobody else owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let section = Section {
            fd,
            size: AtomicU64::new(0),
        };
        section.resize(size)?;
        Ok(Arc::new(section))
    }

    /// The `memfd`, for a layer that needs the same pages elsewhere.
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    pub fn size(&self) -> u64 {
        self.size.load(Ordering::Relaxed)
    }

    fn resize(&self, size: u64) -> Result<(), u32> {
        let len = libc::off_t::try_from(size).map_err(|_| ERROR_NOT_ENOUGH_MEMORY)?;
        // SAFETY: our descriptor.
        if unsafe { libc::ftruncate(self.fd(), len) } != 0 {
            return Err(ERROR_NOT_ENOUGH_MEMORY);
        }
        self.size.store(size, Ordering::Relaxed);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `VirtualAlloc`'s.
    Private,
    /// A mapping somebody else made and owns ([`adopt`]).
    Image,
    /// Address space nothing can be committed in.
    Placeholder,
    /// A view of a section ([`map_view`]).
    Mapped,
}

/// Pages `[start, end)` with the same attributes inside one allocation;
/// the start is the run's key. An allocation is the adjacent runs that
/// share `base`.
#[derive(Debug, Clone)]
struct Run {
    end: usize,
    base: usize,
    alloc_protect: u32,
    /// Of committed pages, modifiers included; 0 while reserved.
    protect: u32,
    /// The granularity of sizes in this allocation.
    page: usize,
    committed: bool,
    kind: Kind,
    /// The section shown at the run's start and its offset there.
    view: Option<(Arc<Section>, u64)>,
}

type Runs = BTreeMap<usize, Run>;

#[derive(Default)]
struct State {
    runs: Runs,
    /// The physical pool, its size in pages and the numbers handed back.
    pool: Option<Arc<Section>>,
    pool_pages: u64,
    pool_free: Vec<u64>,
    /// Section blocks (section, offset / 64 KiB) → the address blocks that
    /// show them, for write watching.
    shown: Shown,
}

type Shown = BTreeMap<(usize, u64), Vec<usize>>;

static STATE: Mutex<State> = Mutex::new(State {
    runs: BTreeMap::new(),
    pool: None,
    pool_pages: 0,
    pool_free: Vec::new(),
    shown: BTreeMap::new(),
});

fn state() -> MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn down(a: usize, step: usize) -> usize {
    a & !(step - 1)
}

fn up(a: usize, step: usize) -> usize {
    (a + step - 1) & !(step - 1)
}

const RESERVE: libc::c_int = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;

/// The start of the run holding `a`.
fn find(runs: &Runs, a: usize) -> Option<usize> {
    runs.range(..=a)
        .next_back()
        .filter(|(_, r)| a < r.end)
        .map(|(&s, _)| s)
}

/// One past the last page of the allocation the run at `start` is in.
fn extent(runs: &Runs, start: usize) -> usize {
    let base = runs[&start].base;
    let mut end = start;
    for (&s, r) in runs.range(start..) {
        if s != end || r.base != base {
            break;
        }
        end = r.end;
    }
    end
}

/// The run holding `a` when all of `[a, b)` lies in its allocation.
fn whole(runs: &Runs, a: usize, b: usize) -> Option<usize> {
    find(runs, a).filter(|&s| b <= extent(runs, s))
}

fn split(runs: &mut Runs, at: usize) {
    let Some(start) = find(runs, at).filter(|&s| s != at) else {
        return;
    };
    let head = runs.get_mut(&start).expect("found");
    let mut tail = head.clone();
    head.end = at;
    if let Some((_, offset)) = &mut tail.view {
        *offset += (at - start) as u64;
    }
    runs.insert(at, tail);
}

/// Gives `[base, end)` of the placeholder whose run starts at `start` an
/// allocation of its own, `fill` setting each run there; what is left
/// below and above stays placeholders of their own.
fn carve(runs: &mut Runs, start: usize, base: usize, end: usize, fill: impl FnMut(usize, &mut Run)) {
    let (first, limit) = (runs[&start].base, extent(runs, start));
    each(runs, first, base, |_, r| r.base = first);
    each(runs, base, end, fill);
    each(runs, end, limit, |_, r| r.base = end);
}

fn joinable(a_start: usize, a: &Run, b_start: usize, b: &Run) -> bool {
    a.end == b_start
        && a.base == b.base
        && a.alloc_protect == b.alloc_protect
        && a.protect == b.protect
        && a.page == b.page
        && a.committed == b.committed
        && a.kind == b.kind
        && match (&a.view, &b.view) {
            (None, None) => true,
            (Some((sa, oa)), Some((sb, ob))) => {
                Arc::ptr_eq(sa, sb) && *ob == oa + (b_start - a_start) as u64
            }
            _ => false,
        }
}

/// Changes every run in `[a, b)`, which the caller has checked is covered,
/// then joins like neighbours again. `change` gets each run's start.
fn each(runs: &mut Runs, a: usize, b: usize, mut change: impl FnMut(usize, &mut Run)) {
    if a >= b {
        return;
    }
    split(runs, a);
    split(runs, b);
    for (&s, r) in runs.range_mut(a..b) {
        change(s, r);
    }
    let from = runs.range(..a).next_back().map_or(a, |(&s, _)| s);
    let keys: Vec<usize> = runs.range(from..=b).map(|(&s, _)| s).collect();
    let mut current: Option<usize> = None;
    for k in keys {
        if let Some(c) = current
            && joinable(c, &runs[&c], k, &runs[&k])
        {
            let end = runs.remove(&k).expect("listed").end;
            runs.get_mut(&c).expect("listed").end = end;
            continue;
        }
        current = Some(k);
    }
}

/// `PAGE_*` as `PROT_*`: `PAGE_GUARD` and `PAGE_NOACCESS` are `PROT_NONE`;
/// `None` for a value that is not one protection.
pub fn native_protect(protect: u32) -> Option<libc::c_int> {
    if protect & !(0xFF | PAGE_GUARD | PAGE_NOCACHE | PAGE_WRITECOMBINE) != 0 {
        return None;
    }
    let prot = match protect & 0xFF {
        PAGE_NOACCESS => libc::PROT_NONE,
        PAGE_READONLY => libc::PROT_READ,
        PAGE_READWRITE | PAGE_WRITECOPY => libc::PROT_READ | libc::PROT_WRITE,
        // x86 has no execute-only pages; Windows makes them readable too.
        PAGE_EXECUTE | PAGE_EXECUTE_READ => libc::PROT_READ | libc::PROT_EXEC,
        PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => {
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC
        }
        _ => return None,
    };
    Some(if protect & PAGE_GUARD != 0 {
        libc::PROT_NONE
    } else {
        prot
    })
}

/// `PROT_*` as `PAGE_*`.
pub fn win_protect(prot: libc::c_int) -> u32 {
    let (r, w, x) = (
        prot & libc::PROT_READ != 0,
        prot & libc::PROT_WRITE != 0,
        prot & libc::PROT_EXEC != 0,
    );
    match (x, w, r) {
        (true, true, _) => PAGE_EXECUTE_READWRITE,
        (true, false, true) => PAGE_EXECUTE_READ,
        (true, false, false) => PAGE_EXECUTE,
        (false, true, _) => PAGE_READWRITE,
        (false, false, true) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

/// Whether pages with this protection can be written.
pub fn writable(protect: u32) -> bool {
    native_protect(protect).is_some_and(|p| p & libc::PROT_WRITE != 0)
}

/// What Windows calls the commit limit: memory and swap together. The
/// mappings are `MAP_NORESERVE`, so without this a title that sizes a pool
/// by what it can commit would be told it can have 128 TiB.
// ponytail: the machine's numbers, not a cgroup limit nor what is committed already.
fn commit_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        // SAFETY: an all-zero sysinfo is valid; the call fills it.
        let mut info: libc::sysinfo = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is writable.
        if unsafe { libc::sysinfo(&mut info) } != 0 {
            return usize::MAX;
        }
        (info.totalram as usize + info.totalswap as usize).saturating_mul(info.mem_unit as usize)
    })
}

/// Address space of `len` bytes on a multiple of `align`, wherever the
/// kernel likes: that much more is mapped, then the ends are cut off.
fn reserve_anywhere(len: usize, align: usize) -> Result<usize, u32> {
    // SAFETY: a fresh anonymous mapping; nothing is replaced.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len + align,
            libc::PROT_NONE,
            RESERVE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    let raw = p as usize;
    let base = up(raw, align);
    // SAFETY: both pieces are parts of the mapping just made.
    let cut = unsafe {
        (base == raw || libc::munmap(p, base - raw) == 0)
            && libc::munmap((base + len) as *mut _, raw + align - base) == 0
    };
    if !cut {
        // At vm.max_map_count the kernel cannot cut a mapping in two.
        // SAFETY: as above.
        unsafe { libc::munmap(p, len + align) };
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    Ok(base)
}

/// Address space at exactly `[base, base + len)`, which must be free.
fn reserve_at(base: usize, len: usize, prot: libc::c_int, flags: libc::c_int) -> Result<(), u32> {
    // SAFETY: MAP_FIXED_NOREPLACE never replaces a mapping.
    let p = unsafe {
        libc::mmap(
            base as *mut _,
            len,
            prot,
            flags | libc::MAP_FIXED_NOREPLACE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error().raw_os_error();
        return Err(if errno == Some(libc::ENOMEM) {
            ERROR_NOT_ENOUGH_MEMORY
        } else {
            ERROR_INVALID_ADDRESS
        });
    }
    if p as usize != base {
        // A kernel older than 4.17 takes the address as a hint.
        // SAFETY: the mapping just made.
        unsafe { libc::munmap(p, len) };
        return Err(ERROR_INVALID_ADDRESS);
    }
    Ok(())
}

/// A fresh reservation over `[lo, hi)`: the pages (private or a view) are
/// gone at once and no other mapping can take the addresses.
fn back_to_reserved(lo: usize, hi: usize) -> Result<(), u32> {
    // SAFETY: a range the table owns.
    let p = unsafe {
        libc::mmap(
            lo as *mut _,
            hi - lo,
            libc::PROT_NONE,
            RESERVE | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    Ok(())
}

fn mprotect(lo: usize, hi: usize, prot: libc::c_int) -> Result<(), u32> {
    // SAFETY: a range the table owns.
    if unsafe { libc::mprotect(lo as *mut _, hi - lo, prot) } != 0 {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    Ok(())
}

/// `VirtualAlloc` (and `VirtualAlloc2`'s placeholder forms). `page` is the
/// size granularity of a new allocation: [`PAGE`], 64 KiB or
/// [`LARGE_PAGE`] (also chosen by `MEM_LARGE_PAGES`). A range that is not
/// free (reserve) or not in one reservation (commit) is
/// `ERROR_INVALID_ADDRESS`; nothing is replaced.
pub fn alloc(
    address: usize,
    size: usize,
    kind: u32,
    protect: u32,
    page: usize,
) -> Result<usize, u32> {
    const KNOWN: u32 = MEM_COMMIT
        | MEM_RESERVE
        | MEM_RESET
        | MEM_RESET_UNDO
        | MEM_TOP_DOWN
        | MEM_WRITE_WATCH
        | MEM_LARGE_PAGES
        | MEM_REPLACE_PLACEHOLDER
        | MEM_RESERVE_PLACEHOLDER;
    let page = if kind & MEM_LARGE_PAGES != 0 {
        LARGE_PAGE
    } else {
        page
    };
    let a = address;
    let commit = kind & MEM_COMMIT != 0;
    // A commit nowhere in particular reserves too.
    let reserve = kind & MEM_RESERVE != 0 || (commit && a == 0);
    let reset = kind & (MEM_RESET | MEM_RESET_UNDO) != 0;
    let placeholder = kind & (MEM_REPLACE_PLACEHOLDER | MEM_RESERVE_PLACEHOLDER);
    let prot = native_protect(protect);
    let invalid = size == 0
        || kind & !KNOWN != 0
        || !(reserve || commit || reset)
        || (prot.is_none() && !reset)
        // "Not supported by VirtualAlloc."
        || protect & (PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY) != 0
        || ![PAGE, GRANULARITY, LARGE_PAGE].contains(&page)
        || size > HIGHEST
        || a > HIGHEST - size
        || (placeholder != 0 && kind & MEM_RESERVE == 0)
        || (kind & MEM_REPLACE_PLACEHOLDER != 0 && a == 0)
        || (kind & MEM_RESERVE_PLACEHOLDER != 0 && commit)
        // Large pages: reserved and committed together, whole pages.
        || (reserve && page == LARGE_PAGE && (!commit || !size.is_multiple_of(LARGE_PAGE)))
        // "Cannot be used with any other value."
        || (reset && kind != MEM_RESET && kind != MEM_RESET_UNDO);
    if invalid {
        return Err(ERROR_INVALID_PARAMETER);
    }
    if kind & MEM_WRITE_WATCH != 0 {
        return Err(ERROR_NOT_SUPPORTED);
    }
    if commit && size > commit_limit() {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    let prot = prot.unwrap_or(libc::PROT_NONE);

    let mut state = state();
    let runs = &mut state.runs;

    if reset {
        // "No longer of interest": only whole pages inside the range.
        whole(runs, down(a, PAGE), up(a + size, PAGE)).ok_or(ERROR_INVALID_ADDRESS)?;
        let (lo, hi) = (up(a, PAGE), down(a + size, PAGE));
        if kind == MEM_RESET && lo < hi {
            // SAFETY: committed or reserved pages the table owns.
            unsafe { libc::madvise(lo as *mut _, hi - lo, libc::MADV_FREE) };
        }
        return Ok(down(a, PAGE));
    }

    let (mut base, mut end) = (0, 0);
    // This call mapped [base, end): a failed commit gives it back.
    let mut fresh = false;
    // Committed by the reservation itself (huge pages).
    let mut done = false;
    if reserve {
        let align = page.max(GRANULARITY);
        let new = Run {
            end: 0,
            base: 0,
            alloc_protect: protect,
            protect: 0,
            page,
            committed: false,
            kind: if kind & MEM_RESERVE_PLACEHOLDER != 0 {
                Kind::Placeholder
            } else {
                Kind::Private
            },
            view: None,
        };
        if a != 0 {
            base = down(a, align);
            end = up(a + size, page);
            if base < LOWEST {
                return Err(ERROR_INVALID_PARAMETER);
            }
        }
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            let start = whole(runs, base, end).ok_or(ERROR_INVALID_ADDRESS)?;
            if runs[&start].kind != Kind::Placeholder {
                return Err(ERROR_INVALID_ADDRESS);
            }
            carve(runs, start, base, end, |_, r| {
                *r = Run {
                    end: r.end,
                    base,
                    ..new.clone()
                }
            });
        } else {
            let len = if a != 0 { end - base } else { size };
            if page == LARGE_PAGE
                && let Some(at) = huge(a, base, len, prot)
            {
                (base, end, done) = (at, at + len, true);
            } else if a != 0 {
                reserve_at(base, end - base, libc::PROT_NONE, RESERVE)?;
            } else {
                end = up(size, page);
                base = reserve_anywhere(end, align)?;
                end += base;
            }
            fresh = true;
            runs.insert(
                base,
                Run {
                    end,
                    base,
                    ..new.clone()
                },
            );
        }
    }
    if commit {
        let mut start = find(runs, if reserve { base } else { a });
        if !reserve && let Some(s) = start {
            let page = runs[&s].page;
            base = down(a, page);
            end = up(a + size, page);
            start = whole(runs, base, end);
        }
        let ok = start.is_some_and(|s| runs[&s].kind != Kind::Placeholder);
        if ok {
            super::watch::open(base, end - base);
        }
        let result = if !ok {
            Err(ERROR_INVALID_ADDRESS)
        } else if done {
            Ok(())
        } else {
            mprotect(base, end, prot)
        };
        if let Err(e) = result {
            if fresh {
                // SAFETY: the mapping this call made.
                unsafe { libc::munmap(base as *mut _, end - base) };
                runs.remove(&base);
            }
            return Err(e);
        }
        if page == LARGE_PAGE && !done {
            // SAFETY: committed pages the table owns.
            unsafe { libc::madvise(base as *mut _, end - base, libc::MADV_HUGEPAGE) };
        }
        // Pages committed already keep their contents and get the new protection.
        each(runs, base, end, |_, r| {
            r.committed = true;
            r.protect = protect;
        });
    }
    Ok(base)
}

/// Huge pages of the host's reserved pool, committed; `None` when it has
/// none to give (the caller then uses normal pages).
fn huge(a: usize, base: usize, len: usize, prot: libc::c_int) -> Option<usize> {
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB;
    if a != 0 {
        return reserve_at(base, len, prot, flags).ok().map(|()| base);
    }
    // SAFETY: a fresh mapping; the kernel aligns huge pages to their size.
    let p = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, flags, -1, 0) };
    (p != libc::MAP_FAILED).then_some(p as usize)
}

/// `VirtualFree`: `MEM_DECOMMIT` (size 0: the whole allocation `address`
/// is the base of), `MEM_RELEASE` (size 0 and the base), or the
/// placeholder forms (`MEM_RELEASE` with `MEM_PRESERVE_PLACEHOLDER` splits
/// a placeholder or turns a replaced one back; with
/// `MEM_COALESCE_PLACEHOLDERS` joins the placeholders filling the range).
pub fn free(address: usize, size: usize, kind: u32) -> Result<(), u32> {
    let a = address;
    let op = kind & (MEM_DECOMMIT | MEM_RELEASE);
    let extra = kind & !op;
    if (op != MEM_DECOMMIT && op != MEM_RELEASE)
        || (extra != 0 && op != MEM_RELEASE)
        || (extra != 0 && extra != MEM_COALESCE_PLACEHOLDERS && extra != MEM_PRESERVE_PLACEHOLDER)
        || (extra != 0 && (size == 0 || !a.is_multiple_of(PAGE) || !size.is_multiple_of(PAGE)))
        || size > HIGHEST
        || a > HIGHEST - size
    {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut state = state();
    let State { runs, shown, .. } = &mut *state;
    let start = find(runs, a).ok_or(ERROR_INVALID_ADDRESS)?;
    let first = runs[&start].clone();
    let (base, limit) = (first.base, extent(runs, start));
    if matches!(first.kind, Kind::Image | Kind::Mapped) {
        return Err(ERROR_INVALID_PARAMETER); // not VirtualAlloc's to free
    }
    let reserved = |r: &mut Run| {
        r.committed = false;
        r.protect = 0;
        r.view = None;
    };

    if op == MEM_DECOMMIT {
        let (lo, hi) = if size == 0 {
            (base, limit)
        } else {
            (down(a, first.page), up(a + size, first.page))
        };
        if (size == 0 && a != base) || hi > limit || first.kind == Kind::Placeholder {
            return Err(ERROR_INVALID_ADDRESS);
        }
        unwatch(runs, shown, lo, hi);
        back_to_reserved(lo, hi)?;
        each(runs, lo, hi, |_, r| reserved(r));
        return Ok(());
    }
    if extra == MEM_COALESCE_PLACEHOLDERS {
        let mut pos = a;
        while pos < a + size {
            let p = find(runs, pos)
                .filter(|p| runs[p].kind == Kind::Placeholder && runs[p].base == pos)
                .ok_or(ERROR_INVALID_ADDRESS)?;
            pos = extent(runs, p);
        }
        if pos != a + size {
            return Err(ERROR_INVALID_ADDRESS);
        }
        each(runs, a, pos, |_, r| r.base = a);
        return Ok(());
    }
    if extra == MEM_PRESERVE_PLACEHOLDER {
        let hi = a + size;
        if hi > limit {
            return Err(ERROR_INVALID_ADDRESS);
        }
        if first.kind == Kind::Placeholder {
            // Split: [base, a) [a, hi) [hi, limit).
            each(runs, a, hi, |_, r| r.base = a);
            each(runs, hi, limit, |_, r| r.base = hi);
            return Ok(());
        }
        if a != base || hi != limit {
            return Err(ERROR_INVALID_ADDRESS);
        }
        unwatch(runs, shown, a, hi);
        back_to_reserved(a, hi)?;
        each(runs, a, hi, |_, r| {
            reserved(r);
            r.alloc_protect = PAGE_NOACCESS;
            r.kind = Kind::Placeholder;
        });
        return Ok(());
    }
    if size != 0 {
        return Err(ERROR_INVALID_PARAMETER); // MEM_RELEASE takes 0
    }
    if a != base {
        return Err(ERROR_INVALID_ADDRESS);
    }
    unwatch(runs, shown, base, limit);
    // At vm.max_map_count this can fail; then nothing was unmapped.
    // SAFETY: the whole allocation, which the table owns.
    if unsafe { libc::munmap(base as *mut _, limit - base) } != 0 {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    let keys: Vec<usize> = runs.range(base..limit).map(|(&s, _)| s).collect();
    for k in keys {
        runs.remove(&k);
    }
    Ok(())
}

/// `VirtualProtect`: every page of the range committed and in one
/// allocation (else `ERROR_INVALID_ADDRESS`). Returns the first page's old
/// protection.
pub fn protect(address: usize, size: usize, new: u32) -> Result<u32, u32> {
    let prot = native_protect(new).ok_or(ERROR_INVALID_PARAMETER)?;
    if size == 0 || size > HIGHEST || address > HIGHEST - size {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let (lo, hi) = (down(address, PAGE), up(address + size, PAGE));
    let mut state = state();
    let runs = &mut state.runs;
    let first = whole(runs, lo, hi).ok_or(ERROR_INVALID_ADDRESS)?;
    let start = find(runs, lo).expect("whole");
    if runs.range(start..hi).any(|(_, r)| !r.committed) {
        return Err(ERROR_INVALID_ADDRESS);
    }
    super::watch::open(lo, hi - lo);
    mprotect(lo, hi, prot)?;
    let old = runs[&first].protect;
    each(runs, lo, hi, |_, r| r.protect = new);
    Ok(old)
}

/// `VirtualQuery`. `ERROR_INVALID_PARAMETER` only above [`HIGHEST`], which
/// is how a walk ends.
pub fn query(address: usize) -> Result<MemoryBasicInformation, u32> {
    let a = down(address, PAGE);
    if a > HIGHEST {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut info = MemoryBasicInformation {
        base_address: a,
        ..Default::default()
    };
    let (prev, next) = {
        let state = state();
        let runs = &state.runs;
        if let Some(start) = find(runs, a) {
            let r = &runs[&start];
            info.allocation_base = r.base;
            info.allocation_protect = r.alloc_protect;
            info.region_size = r.end - a;
            info.state = if r.committed { MEM_COMMIT } else { MEM_RESERVE };
            info.protect = r.protect;
            info.kind = match r.kind {
                Kind::Image => MEM_IMAGE,
                Kind::Mapped => MEM_MAPPED,
                Kind::Private | Kind::Placeholder => MEM_PRIVATE,
            };
            return Ok(info);
        }
        // Not the table's: free or the host's. Either way the answer ends
        // where the table's next allocation begins and starts no lower than
        // where the one below ends.
        let next = runs.range(a..).next().map_or(HIGHEST + 1, |(&s, _)| s);
        let prev = runs.range(..a).next_back().map_or(0, |(_, r)| r.end);
        (prev, next)
    };
    match host_vma(a) {
        Some(v) if v.start <= a => {
            let (lo, hi) = (v.start.max(prev), v.end.min(next));
            info.allocation_base = lo;
            info.region_size = hi - a;
            info.state = if v.prot != 0 { MEM_COMMIT } else { MEM_RESERVE };
            info.protect = if v.prot != 0 { win_protect(v.prot) } else { 0 };
            info.allocation_protect = win_protect(v.prot);
            info.kind = MEM_PRIVATE;
        }
        v => {
            let host_next = v.map_or(HIGHEST + 1, |v| v.start.min(HIGHEST + 1));
            info.region_size = host_next.min(next) - a;
            info.state = MEM_FREE;
            info.protect = PAGE_NOACCESS;
        }
    }
    Ok(info)
}

#[derive(Debug, Clone, Copy)]
struct Vma {
    start: usize,
    end: usize,
    prot: libc::c_int,
}

/// `struct procmap_query` of `<linux/fs.h>` (Linux 6.11).
#[repr(C)]
#[derive(Default)]
struct ProcmapQuery {
    size: u64,
    query_flags: u64,
    query_addr: u64,
    vma_start: u64,
    vma_end: u64,
    vma_flags: u64,
    vma_page_size: u64,
    vma_offset: u64,
    inode: u64,
    dev_major: u32,
    dev_minor: u32,
    vma_name_size: u32,
    build_id_size: u32,
    vma_name_addr: u64,
    build_id_addr: u64,
}

const PROCMAP_QUERY: libc::c_ulong = 0xC068_6611;
const PROCMAP_QUERY_COVERING_OR_NEXT_VMA: u64 = 0x10;

/// The host mapping that holds `a`, or the next one above it. One ioctl on
/// Linux 6.11 and later, else a scan of `/proc/self/maps`.
fn host_vma(a: usize) -> Option<Vma> {
    use std::io::BufRead;
    static MAPS: std::sync::OnceLock<Option<std::fs::File>> = std::sync::OnceLock::new();
    let maps = MAPS.get_or_init(|| std::fs::File::open("/proc/self/maps").ok());
    if let Some(maps) = maps {
        let mut q = ProcmapQuery {
            size: size_of::<ProcmapQuery>() as u64,
            query_flags: PROCMAP_QUERY_COVERING_OR_NEXT_VMA,
            query_addr: a as u64,
            ..Default::default()
        };
        // SAFETY: a procmap_query the kernel fills; no name or build id buffers.
        if unsafe { libc::ioctl(maps.as_raw_fd(), PROCMAP_QUERY, &mut q) } == 0 {
            let f = q.vma_flags;
            let prot = (if f & 1 != 0 { libc::PROT_READ } else { 0 })
                | (if f & 2 != 0 { libc::PROT_WRITE } else { 0 })
                | (if f & 4 != 0 { libc::PROT_EXEC } else { 0 });
            return Some(Vma {
                start: q.vma_start as usize,
                end: q.vma_end as usize,
                prot,
            });
        }
        if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
            return None;
        }
    }
    let file = std::fs::File::open("/proc/self/maps").ok()?;
    for line in std::io::BufReader::new(file).lines() {
        let line = line.ok()?;
        let (range, rest) = line.split_once(' ')?;
        let (start, end) = range.split_once('-')?;
        let (start, end) = (
            usize::from_str_radix(start, 16).ok()?,
            usize::from_str_radix(end, 16).ok()?,
        );
        if end > a {
            let p = rest.as_bytes();
            let prot = (if p.first() == Some(&b'r') {
                libc::PROT_READ
            } else {
                0
            }) | (if p.get(1) == Some(&b'w') {
                libc::PROT_WRITE
            } else {
                0
            }) | (if p.get(2) == Some(&b'x') {
                libc::PROT_EXEC
            } else {
                0
            });
            return Some(Vma { start, end, prot });
        }
    }
    None
}

/// Records a mapping made elsewhere (an image, thunk memory) so that
/// `VirtualQuery` and `VirtualProtect` see it (`MEM_IMAGE`, committed).
/// `protect` is a Windows `PAGE_*` value. A range inside an image adopted
/// earlier only changes the protection recorded there, so adopt the whole
/// image first, then each section. `VirtualFree` refuses it; the owner
/// calls [`forget`] before it unmaps.
pub fn adopt(base: usize, size: usize, protect: u32) {
    let end = up(base + size, PAGE);
    let mut state = state();
    let runs = &mut state.runs;
    if whole(runs, base, end).is_some_and(|s| runs[&s].kind == Kind::Image) {
        each(runs, base, end, |_, r| r.protect = protect);
        return;
    }
    split(runs, base);
    split(runs, end);
    let keys: Vec<usize> = runs.range(base..end).map(|(&s, _)| s).collect();
    for k in keys {
        runs.remove(&k);
    }
    runs.insert(
        base,
        Run {
            end,
            base,
            alloc_protect: protect,
            protect,
            page: PAGE,
            committed: true,
            kind: Kind::Image,
            view: None,
        },
    );
}

/// Forgets the adopted allocation at `base`.
pub fn forget(base: usize) {
    let mut state = state();
    let runs = &mut state.runs;
    if !runs
        .get(&base)
        .is_some_and(|r| r.base == base && r.kind == Kind::Image)
    {
        return;
    }
    let limit = extent(runs, base);
    let keys: Vec<usize> = runs.range(base..limit).map(|(&s, _)| s).collect();
    for k in keys {
        runs.remove(&k);
    }
}

/// For the fault handler: true when `address` was in a guard page, whose
/// guard is now gone (raise `STATUS_GUARD_PAGE_VIOLATION`); false for an
/// ordinary access violation.
pub fn take_guard(address: usize) -> bool {
    let a = down(address, PAGE);
    let mut state = state();
    let runs = &mut state.runs;
    let Some(start) = find(runs, a) else {
        return false;
    };
    let r = &runs[&start];
    if !r.committed || r.protect & PAGE_GUARD == 0 {
        return false;
    }
    let without = r.protect & !PAGE_GUARD;
    // At vm.max_map_count this fails and the page stays a guard page.
    if mprotect(
        a,
        a + PAGE,
        native_protect(without).unwrap_or(libc::PROT_NONE),
    )
    .is_err()
    {
        return false;
    }
    each(runs, a, a + PAGE, |_, r| r.protect = without);
    true
}

/// Maps `[offset, offset + len)` of `section` (`len` 0: to its end) at
/// `address`, which must be free, or with `replace_placeholder` lie in one
/// placeholder; 0 is anywhere. `MapViewOfFile3`. The view is committed.
pub fn map_view(
    section: &Arc<Section>,
    offset: u64,
    len: usize,
    address: usize,
    protect: u32,
    replace_placeholder: bool,
) -> Result<usize, u32> {
    let prot = native_protect(protect).ok_or(ERROR_INVALID_PARAMETER)?;
    let size = section.size();
    let len = if len == 0 {
        size.saturating_sub(offset) as usize
    } else {
        len
    };
    if !offset.is_multiple_of(GRANULARITY as u64)
        || len == 0
        || offset
            .checked_add(len as u64)
            .is_none_or(|e| e > up(size as usize, PAGE) as u64)
        || !address.is_multiple_of(PAGE)
        || address > HIGHEST - len.min(HIGHEST)
        || (replace_placeholder && address == 0)
    {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let end_len = up(len, PAGE);
    let mut state = state();
    let State { runs, shown, .. } = &mut *state;
    let base = if replace_placeholder {
        let start = whole(runs, address, address + end_len).ok_or(ERROR_INVALID_ADDRESS)?;
        if runs[&start].kind != Kind::Placeholder {
            return Err(ERROR_INVALID_ADDRESS);
        }
        address
    } else if address != 0 {
        if !address.is_multiple_of(GRANULARITY) {
            return Err(ERROR_INVALID_ADDRESS);
        }
        reserve_at(address, end_len, libc::PROT_NONE, RESERVE)?;
        address
    } else {
        reserve_anywhere(end_len, GRANULARITY)?
    };
    let off = libc::off_t::try_from(offset).map_err(|_| ERROR_INVALID_PARAMETER)?;
    // SAFETY: over address space the table owns (a placeholder or a
    // reservation just made).
    let p = unsafe {
        libc::mmap(
            base as *mut _,
            end_len,
            prot,
            libc::MAP_SHARED | libc::MAP_FIXED,
            section.fd(),
            off,
        )
    };
    if p == libc::MAP_FAILED {
        if !replace_placeholder {
            // SAFETY: the reservation just made.
            unsafe { libc::munmap(base as *mut _, end_len) };
        }
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    let new = Run {
        end: base + end_len,
        base,
        alloc_protect: protect,
        protect,
        page: PAGE,
        committed: true,
        kind: Kind::Mapped,
        view: Some((section.clone(), offset)),
    };
    if replace_placeholder {
        let start = find(runs, base).expect("checked");
        carve(runs, start, base, base + end_len, |s, r| {
            *r = Run {
                end: r.end,
                view: Some((section.clone(), offset + (s - base) as u64)),
                ..new.clone()
            }
        });
    } else {
        runs.insert(base, new);
    }
    show(runs, shown, base, base + end_len);
    Ok(base)
}

/// `UnmapViewOfFile2`: the view at `base` goes; with
/// `preserve_placeholder` its range becomes a placeholder again.
pub fn unmap_view(base: usize, preserve_placeholder: bool) -> Result<(), u32> {
    let mut state = state();
    let State { runs, shown, .. } = &mut *state;
    let start = find(runs, base).ok_or(ERROR_INVALID_ADDRESS)?;
    let r = &runs[&start];
    if r.kind != Kind::Mapped || r.base != base {
        return Err(ERROR_INVALID_ADDRESS);
    }
    let limit = extent(runs, start);
    unwatch(runs, shown, base, limit);
    if preserve_placeholder {
        back_to_reserved(base, limit)?;
        each(runs, base, limit, |_, r| {
            r.committed = false;
            r.protect = 0;
            r.view = None;
            r.alloc_protect = PAGE_NOACCESS;
            r.kind = Kind::Placeholder;
        });
        return Ok(());
    }
    // SAFETY: the view, which the table owns.
    if unsafe { libc::munmap(base as *mut _, limit - base) } != 0 {
        return Err(ERROR_NOT_ENOUGH_MEMORY);
    }
    let keys: Vec<usize> = runs.range(base..limit).map(|(&s, _)| s).collect();
    for k in keys {
        runs.remove(&k);
    }
    Ok(())
}

/// Shows the section pages mapped at `[from, from + len)` at `to` as well,
/// with the same protection, in place of what `to` shows: the aliases
/// share their bytes. `to` must lie in one reservation or view; every page
/// of the source must be a view or physical pages.
pub fn alias(to: usize, from: usize, len: usize) -> Result<(), u32> {
    if len == 0
        || !to.is_multiple_of(PAGE)
        || !from.is_multiple_of(PAGE)
        || !len.is_multiple_of(PAGE)
    {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut state = state();
    let State { runs, shown, .. } = &mut *state;
    let target = whole(runs, to, to + len).ok_or(ERROR_INVALID_ADDRESS)?;
    if !matches!(runs[&target].kind, Kind::Private | Kind::Mapped) {
        return Err(ERROR_INVALID_ADDRESS);
    }
    // The source's pieces: where, the section and offset there, protection.
    let mut pieces = Vec::new();
    let mut at = from;
    while at < from + len {
        let s = find(runs, at).ok_or(ERROR_INVALID_ADDRESS)?;
        let r = &runs[&s];
        let Some((section, offset)) = &r.view else {
            return Err(ERROR_INVALID_ADDRESS);
        };
        let end = r.end.min(from + len);
        pieces.push((
            at,
            end,
            section.clone(),
            offset + (at - s) as u64,
            r.protect,
        ));
        at = end;
    }
    unwatch(runs, shown, to, to + len);
    let mut result = Ok(());
    for (lo, hi, section, offset, protect) in pieces {
        let dst = to + (lo - from);
        let prot = native_protect(protect).unwrap_or(libc::PROT_NONE);
        // SAFETY: inside an allocation the table owns.
        let p = unsafe {
            libc::mmap(
                dst as *mut _,
                hi - lo,
                prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                section.fd(),
                offset as libc::off_t,
            )
        };
        if p == libc::MAP_FAILED {
            result = Err(ERROR_NOT_ENOUGH_MEMORY);
            break;
        }
        each(runs, dst, dst + (hi - lo), |s, r| {
            r.committed = true;
            r.protect = protect;
            r.view = Some((section.clone(), offset + (s - dst) as u64));
        });
    }
    show(runs, shown, to, to + len);
    result
}

/// Hands out `pages.len()` physical page numbers (64 KiB each): recycled
/// numbers first, then new ones from the end of the pool, ascending.
pub fn phys_alloc(pages: &mut [u64]) -> Result<(), u32> {
    let mut state = state();
    if state.pool.is_none() {
        state.pool = Some(Section::new(0)?);
    }
    let recycled = pages.len().min(state.pool_free.len());
    let grown = state.pool_pages + (pages.len() - recycled) as u64;
    if grown != state.pool_pages {
        let size = grown
            .checked_mul(PHYS_PAGE as u64)
            .ok_or(ERROR_NOT_ENOUGH_MEMORY)?;
        state.pool.as_ref().expect("made").resize(size)?;
    }
    for (i, page) in pages.iter_mut().enumerate() {
        *page = if i < recycled {
            state.pool_free.pop().expect("counted")
        } else {
            state.pool_pages += 1;
            state.pool_pages - 1
        };
    }
    Ok(())
}

/// Returns physical pages; their memory goes back to the host (a hole is
/// punched), so views still showing them read zero.
pub fn phys_free(pages: &[u64]) {
    let mut state = state();
    let Some(pool) = state.pool.clone() else {
        return;
    };
    for &page in pages {
        if page >= state.pool_pages || state.pool_free.contains(&page) {
            continue;
        }
        // SAFETY: our memfd.
        unsafe {
            libc::fallocate(
                pool.fd(),
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                (page * PHYS_PAGE as u64) as libc::off_t,
                PHYS_PAGE as libc::off_t,
            )
        };
        // Views still showing it now read zero: written.
        open_shown(&state.shown, &pool, page);
        state.pool_free.push(page);
    }
}

/// Maps `pages[i]` at `va + i * 64 KiB`, replacing what is there; the
/// range must lie in one reservation or placeholder of this table. A page can be mapped
/// at several addresses at once. `MEM_DECOMMIT` unmaps.
pub fn phys_map(va: usize, pages: &[u64], protect: u32) -> Result<(), u32> {
    let prot = native_protect(protect).ok_or(ERROR_INVALID_PARAMETER)?;
    let n = pages.len();
    if n == 0
        || !va.is_multiple_of(PHYS_PAGE)
        || n > HIGHEST / PHYS_PAGE
        || va > HIGHEST - n * PHYS_PAGE
    {
        return Err(ERROR_INVALID_PARAMETER);
    }
    let mut state = state();
    let pool_pages = state.pool_pages;
    let pool = state.pool.clone().ok_or(ERROR_INVALID_PARAMETER)?;
    let State { runs, shown, .. } = &mut *state;
    let end = va + n * PHYS_PAGE;
    let start = whole(runs, va, end).ok_or(ERROR_INVALID_ADDRESS)?;
    if !matches!(runs[&start].kind, Kind::Private | Kind::Placeholder) {
        return Err(ERROR_INVALID_ADDRESS);
    }
    if pages.iter().any(|&p| p >= pool_pages) {
        return Err(ERROR_INVALID_PARAMETER);
    }
    if runs[&start].kind == Kind::Placeholder {
        // As `MapViewOfFile3` into a placeholder: the range becomes an
        // allocation of its own, here a reservation the pages go into.
        carve(runs, start, va, end, |_, r| {
            *r = Run {
                end: r.end,
                base: va,
                alloc_protect: protect,
                protect: 0,
                page: PHYS_PAGE,
                committed: false,
                kind: Kind::Private,
                view: None,
            }
        });
    }
    unwatch(runs, shown, va, end);
    let mut done = 0;
    while done < n {
        // One mapping per run of consecutive page numbers.
        let mut k = 1;
        while done + k < n && pages[done + k] == pages[done] + k as u64 {
            k += 1;
        }
        let (at, offset) = (va + done * PHYS_PAGE, pages[done] * PHYS_PAGE as u64);
        // SAFETY: inside a reservation the table owns.
        let p = unsafe {
            libc::mmap(
                at as *mut _,
                k * PHYS_PAGE,
                prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                pool.fd(),
                offset as libc::off_t,
            )
        };
        if p == libc::MAP_FAILED {
            break;
        }
        each(runs, at, at + k * PHYS_PAGE, |s, r| {
            r.committed = true;
            r.protect = protect;
            r.view = Some((pool.clone(), offset + (s - at) as u64));
        });
        done += k;
    }
    show(runs, shown, va, end);
    if done == n {
        Ok(())
    } else {
        Err(ERROR_NOT_ENOUGH_MEMORY)
    }
}

/// Physical pages were written other than through the table's views (into
/// the pool's file, or through a view of our own): the blocks showing them
/// are open again, which counts them written.
pub fn pool_written(pages: &[u64]) {
    let state = state();
    if let Some(pool) = &state.pool {
        for &page in pages {
            open_shown(&state.shown, pool, page);
        }
    }
}

fn open_shown(shown: &Shown, section: &Arc<Section>, block: u64) {
    for &b in shown.get(&(Arc::as_ptr(section) as usize, block)).into_iter().flatten() {
        super::watch::open(b * BLOCK, BLOCK);
    }
}

const BLOCK: usize = super::watch::BLOCK;

/// The section blocks block `block` shows, each once.
fn showing(runs: &Runs, block: usize) -> Vec<(usize, u64)> {
    let (lo, hi) = (block * BLOCK, (block + 1) * BLOCK);
    let mut keys = Vec::new();
    let from = find(runs, lo).unwrap_or(lo);
    for (&s, r) in runs.range(from..hi) {
        let Some((section, offset)) = &r.view else {
            continue;
        };
        let (a, b) = (s.max(lo), r.end.min(hi));
        let at = |x: usize| (offset + (x - s) as u64) / BLOCK as u64;
        let id = Arc::as_ptr(section) as usize;
        keys.extend((at(a)..=at(b - 1)).map(|k| (id, k)));
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// The blocks of `[lo, hi)` where a run shows a section.
fn view_blocks(runs: &Runs, lo: usize, hi: usize) -> Vec<usize> {
    let from = find(runs, lo).unwrap_or(lo);
    let mut blocks: Vec<usize> = runs
        .range(from..hi)
        .filter(|(_, r)| r.view.is_some())
        .flat_map(|(&s, r)| s.max(lo) / BLOCK..r.end.min(hi).div_ceil(BLOCK))
        .collect();
    blocks.dedup();
    blocks
}

/// After the mappings of `[lo, hi)` changed: notes which blocks there show
/// which section pages.
fn show(runs: &Runs, shown: &mut Shown, lo: usize, hi: usize) {
    for block in view_blocks(runs, down(lo, BLOCK), up(hi, BLOCK)) {
        for key in showing(runs, block) {
            let blocks = shown.entry(key).or_default();
            if !blocks.contains(&block) {
                blocks.push(block);
            }
        }
    }
}

/// Before the protection or mappings of `[lo, hi)` change: its guarded
/// blocks are open, and what its blocks show is forgotten ([`show`] notes
/// it again).
fn unwatch(runs: &Runs, shown: &mut Shown, lo: usize, hi: usize) {
    super::watch::open(lo, hi - lo);
    for block in view_blocks(runs, down(lo, BLOCK), up(hi, BLOCK)) {
        for key in showing(runs, block) {
            if let Some(blocks) = shown.get_mut(&key) {
                blocks.retain(|&b| b != block);
                if blocks.is_empty() {
                    shown.remove(&key);
                }
            }
        }
    }
}

/// For the GPU side's copies (`victory_gpu`'s watch, see `watch`): the
/// 64 KiB blocks of `[lo, hi)` written since this last guarded them, which
/// are guarded again. A block is watched when one run of committed
/// read-write memory holds it, and every block showing the same section
/// pages is watched with it (guarded too: a write through any of them
/// counts); a block that cannot be counts as written every time. `None`
/// when the table has nothing in the range, or (`views` false) when part
/// of it is a view.
pub fn guard(lo: usize, hi: usize, views: bool) -> Option<Vec<usize>> {
    use super::watch::{extend_unguarded, guarded, mark};
    let state = state();
    let (runs, shown) = (&state.runs, &state.shown);
    // The run holding all of `block` when it is watchable, as (start, run).
    let watchable = |block: usize| {
        find(runs, block * BLOCK)
            .map(|s| (s, &runs[&s]))
            .filter(|(_, r)| r.committed && r.protect == PAGE_READWRITE && r.end >= (block + 1) * BLOCK)
    };
    // The section block a run shows all of at `block`, if it lines up.
    let shows = |s: usize, r: &Run, block: usize| {
        let (section, offset) = r.view.as_ref()?;
        let at = offset + (block * BLOCK - s) as u64;
        at.is_multiple_of(BLOCK as u64)
            .then(|| (Arc::as_ptr(section) as usize, at / BLOCK as u64))
    };
    let (first, last) = (lo / BLOCK, hi.div_ceil(BLOCK));
    let (mut written, mut protect, mut ours) = (Vec::new(), Vec::new(), false);
    let mut next = first; // blocks below are decided
    let from = find(runs, first * BLOCK).unwrap_or(first * BLOCK);
    for (&s, r) in runs.range(from..last * BLOCK) {
        ours = true;
        if r.view.is_some() && !views {
            return None;
        }
        let (a, b) = (s.div_ceil(BLOCK).clamp(first, last), (r.end / BLOCK).clamp(first, last));
        // Below this run: outside the table, or in more than one run.
        written.extend(next..a.max(next));
        next = next.max(a);
        if !(r.committed && r.protect == PAGE_READWRITE) {
            continue;
        }
        if r.view.is_none() {
            let before = protect.len();
            extend_unguarded(next, b, &mut protect);
            written.extend_from_slice(&protect[before..]);
        } else {
            for block in next..b {
                let group = shows(s, r, block).and_then(|key| {
                    shown.get(&key).filter(|group| {
                        group.iter().all(|&peer| {
                            watchable(peer).is_some_and(|(start, run)| shows(start, run, peer) == Some(key))
                        })
                    })
                });
                let mut clean = group.is_some();
                for &peer in group.into_iter().flatten() {
                    if !guarded(peer) {
                        clean = false;
                        protect.push(peer);
                    }
                }
                if !clean {
                    written.push(block);
                }
            }
        }
        next = next.max(b);
    }
    if !ours {
        return None;
    }
    written.extend(next..last);
    protect.sort_unstable();
    protect.dedup();
    // Marked first: a write before the protection lands, and is copied.
    let mut i = 0;
    while i < protect.len() {
        let mut j = i + 1;
        while j < protect.len() && protect[j] == protect[j - 1] + 1 {
            j += 1;
        }
        let blocks = &protect[i..j];
        blocks.iter().for_each(|&b| mark(b));
        let (from, to) = (blocks[0] * BLOCK, (blocks[j - i - 1] + 1) * BLOCK);
        if mprotect(from, to, libc::PROT_READ).is_err() {
            // At vm.max_map_count: block by block, the ones that fail unguarded.
            for &b in blocks {
                if mprotect(b * BLOCK, (b + 1) * BLOCK, libc::PROT_READ).is_err() {
                    super::watch::forget(b);
                }
            }
        }
        i = j;
    }
    Some(written)
}

/// The physical pool's `memfd`, once a page was allocated.
pub fn phys_fd() -> Option<RawFd> {
    state().pool.as_ref().map(|p| p.fd())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The physical pool is one per process: tests that check which pages
    /// they get back take turns.
    static POOL_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn q(a: usize) -> MemoryBasicInformation {
        query(a).unwrap()
    }

    #[test]
    fn memory_basic_information_has_the_windows_layout() {
        assert_eq!(size_of::<MemoryBasicInformation>(), 48);
        assert_eq!(
            std::mem::offset_of!(MemoryBasicInformation, allocation_protect),
            16
        );
        assert_eq!(
            std::mem::offset_of!(MemoryBasicInformation, partition_id),
            20
        );
        assert_eq!(
            std::mem::offset_of!(MemoryBasicInformation, region_size),
            24
        );
        assert_eq!(std::mem::offset_of!(MemoryBasicInformation, state), 32);
        assert_eq!(std::mem::offset_of!(MemoryBasicInformation, protect), 36);
        assert_eq!(std::mem::offset_of!(MemoryBasicInformation, kind), 40);
        assert_eq!(size_of::<ProcmapQuery>(), 104);
    }

    #[test]
    fn reserve_commit_decommit_release() {
        let size = 4 * GRANULARITY;
        let base = alloc(0, size, MEM_RESERVE, PAGE_READWRITE, PAGE).unwrap();
        assert_eq!(base % GRANULARITY, 0);
        let i = q(base + 5);
        assert_eq!(
            (i.base_address, i.allocation_base, i.region_size),
            (base, base, size)
        );
        assert_eq!(
            (i.state, i.protect, i.allocation_protect, i.kind),
            (MEM_RESERVE, 0, PAGE_READWRITE, MEM_PRIVATE)
        );

        // Commit two pages in the middle: three regions.
        let mid = base + 3 * PAGE + 1;
        assert_eq!(
            alloc(mid, PAGE + 1, MEM_COMMIT, PAGE_READWRITE, PAGE),
            Ok(base + 3 * PAGE)
        );
        // SAFETY: committed read-write just now.
        unsafe { ((base + 4 * PAGE) as *mut u8).write(7) };
        assert_eq!(q(base).region_size, 3 * PAGE);
        let c = q(base + 3 * PAGE);
        assert_eq!(
            (c.state, c.protect, c.region_size),
            (MEM_COMMIT, PAGE_READWRITE, 2 * PAGE)
        );
        assert_eq!(q(base + 5 * PAGE).region_size, size - 5 * PAGE);

        // Committing again keeps the contents.
        alloc(base + 4 * PAGE, PAGE, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        // SAFETY: still committed.
        assert_eq!(unsafe { ((base + 4 * PAGE) as *const u8).read() }, 7);

        // Decommit one: it reads zero after a new commit.
        free(base + 4 * PAGE, PAGE, MEM_DECOMMIT).unwrap();
        assert_eq!(q(base + 4 * PAGE).state, MEM_RESERVE);
        alloc(base + 4 * PAGE, PAGE, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        // SAFETY: committed again.
        assert_eq!(unsafe { ((base + 4 * PAGE) as *const u8).read() }, 0);

        // A commit outside the reservation fails.
        assert_eq!(
            alloc(base + size, PAGE, MEM_COMMIT, PAGE_READWRITE, PAGE),
            Err(ERROR_INVALID_ADDRESS)
        );
        // Release takes the base and 0.
        assert_eq!(free(base, size, MEM_RELEASE), Err(ERROR_INVALID_PARAMETER));
        assert_eq!(
            free(base + PAGE, 0, MEM_RELEASE),
            Err(ERROR_INVALID_ADDRESS)
        );
        free(base, 0, MEM_DECOMMIT).unwrap();
        assert_eq!((q(base).state, q(base).region_size), (MEM_RESERVE, size));
        // Another test may take the space at once: nothing more to ask of it.
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn reserving_taken_space_fails_and_a_fixed_reserve_rounds_down() {
        let base = alloc(0, GRANULARITY * 2, MEM_RESERVE, PAGE_NOACCESS, PAGE).unwrap();
        assert_eq!(
            alloc(base, PAGE, MEM_RESERVE, PAGE_READWRITE, PAGE),
            Err(ERROR_INVALID_ADDRESS)
        );
        free(base, 0, MEM_RELEASE).unwrap();
        // Far from where the kernel places mappings, so no other test takes it.
        let base = 0x3000_0000_0000;
        let at = alloc(
            base + 0x1234,
            PAGE,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
            PAGE,
        )
        .unwrap();
        assert_eq!(at, base);
        assert_eq!(q(base).region_size, 3 * PAGE);
        free(base, 0, MEM_RELEASE).unwrap();
        assert_eq!(q(base).state, MEM_FREE);
        assert_eq!(free(base, 0, MEM_RELEASE), Err(ERROR_INVALID_ADDRESS));
    }

    #[test]
    fn protect_splits_and_merges() {
        let base = alloc(0, 8 * PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, PAGE).unwrap();
        assert_eq!(
            protect(base + 2 * PAGE, 2 * PAGE, PAGE_READONLY),
            Ok(PAGE_READWRITE)
        );
        assert_eq!(q(base).region_size, 2 * PAGE);
        let r = q(base + 2 * PAGE);
        assert_eq!(
            (r.protect, r.region_size, r.allocation_base),
            (PAGE_READONLY, 2 * PAGE, base)
        );
        assert_eq!(
            protect(base + 2 * PAGE, 2 * PAGE, PAGE_READWRITE),
            Ok(PAGE_READONLY)
        );
        assert_eq!(q(base).region_size, 8 * PAGE);
        // Not committed: refused.
        free(base + 6 * PAGE, PAGE, MEM_DECOMMIT).unwrap();
        assert_eq!(
            protect(base, 8 * PAGE, PAGE_READONLY),
            Err(ERROR_INVALID_ADDRESS)
        );
        assert_eq!(protect(base, PAGE, 0x3), Err(ERROR_INVALID_PARAMETER));
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn guard_pages_fire_once() {
        let base = alloc(0, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE | PAGE_GUARD, PAGE).unwrap();
        assert!(take_guard(base + PAGE + 8));
        assert!(!take_guard(base + PAGE + 8));
        assert_eq!(q(base + PAGE).protect, PAGE_READWRITE);
        assert_eq!(q(base).protect, PAGE_READWRITE | PAGE_GUARD);
        // SAFETY: the guard is gone from the second page.
        unsafe { ((base + PAGE) as *mut u8).write(1) };
        assert!(!take_guard(0x1000));
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn reset_and_bad_combinations() {
        let base = alloc(0, GRANULARITY, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        assert_eq!(
            alloc(base, GRANULARITY, MEM_RESET, PAGE_NOACCESS, PAGE),
            Ok(base)
        );
        assert_eq!(alloc(base, PAGE, MEM_RESET_UNDO, 0, PAGE), Ok(base));
        assert_eq!(
            alloc(base, PAGE, MEM_RESET | MEM_COMMIT, PAGE_READWRITE, PAGE),
            Err(ERROR_INVALID_PARAMETER)
        );
        assert_eq!(
            alloc(0, PAGE, MEM_COMMIT, PAGE_WRITECOPY, PAGE),
            Err(ERROR_INVALID_PARAMETER)
        );
        assert_eq!(
            alloc(0, 0, MEM_COMMIT, PAGE_READWRITE, PAGE),
            Err(ERROR_INVALID_PARAMETER)
        );
        assert_eq!(
            alloc(0, PAGE, MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE, PAGE),
            Err(ERROR_NOT_SUPPORTED)
        );
        assert_eq!(
            alloc(0, 1 << 46, MEM_COMMIT, PAGE_READWRITE, PAGE),
            Err(ERROR_NOT_ENOUGH_MEMORY)
        );
        free(base, 0, MEM_RELEASE).unwrap();
        // A reservation larger than memory is only address space.
        let big = alloc(0, 1 << 33, MEM_RESERVE | MEM_TOP_DOWN, PAGE_READWRITE, PAGE).unwrap();
        free(big, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn large_pages_are_aligned_and_committed() {
        assert_eq!(
            alloc(
                0,
                LARGE_PAGE,
                MEM_RESERVE | MEM_LARGE_PAGES,
                PAGE_READWRITE,
                PAGE
            ),
            Err(ERROR_INVALID_PARAMETER)
        );
        let base = alloc(
            0,
            LARGE_PAGE,
            MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
            PAGE_READWRITE,
            PAGE,
        )
        .unwrap();
        assert_eq!(base % LARGE_PAGE, 0);
        let i = q(base);
        assert_eq!((i.state, i.region_size), (MEM_COMMIT, LARGE_PAGE));
        // SAFETY: committed read-write.
        unsafe { ((base + LARGE_PAGE - 1) as *mut u8).write(1) };
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn placeholders_split_replace_and_coalesce() {
        let g = GRANULARITY;
        let base = alloc(
            0,
            4 * g,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            PAGE,
        )
        .unwrap();
        // Nothing commits in a placeholder.
        assert_eq!(
            alloc(base, PAGE, MEM_COMMIT, PAGE_READWRITE, PAGE),
            Err(ERROR_INVALID_ADDRESS)
        );
        // Replace the second granule: three allocations.
        let mid = alloc(
            base + g,
            g,
            MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            PAGE,
        )
        .unwrap();
        assert_eq!(mid, base + g);
        // SAFETY: committed read-write.
        unsafe { (mid as *mut u8).write(3) };
        assert_eq!((q(base).allocation_base, q(base).region_size), (base, g));
        assert_eq!((q(mid).allocation_base, q(mid).state), (mid, MEM_COMMIT));
        assert_eq!(
            (q(base + 2 * g).allocation_base, q(base + 2 * g).region_size),
            (base + 2 * g, 2 * g)
        );
        // Back to a placeholder, then join all of it.
        free(mid, g, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER).unwrap();
        assert_eq!(q(mid).state, MEM_RESERVE);
        free(base, 4 * g, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS).unwrap();
        assert_eq!(
            (q(base).allocation_base, q(base).region_size),
            (base, 4 * g)
        );
        // Split off the last granule, release both.
        free(base + 3 * g, g, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER).unwrap();
        assert_eq!(q(base).region_size, 3 * g);
        free(base + 3 * g, 0, MEM_RELEASE).unwrap();
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn views_alias_the_same_bytes() {
        let g = GRANULARITY;
        let section = Section::new(2 * g as u64).unwrap();
        let a = map_view(&section, 0, 0, 0, PAGE_READWRITE, false).unwrap();
        let b = map_view(&section, g as u64, g, 0, PAGE_READWRITE, false).unwrap();
        // SAFETY: both views are mapped read-write.
        unsafe {
            ((a + g + 5) as *mut u8).write(9);
            assert_eq!((b as *const u8).add(5).read(), 9);
        }
        assert_eq!(
            (q(a).kind, q(a).region_size, q(a).state),
            (MEM_MAPPED, 2 * g, MEM_COMMIT)
        );
        assert_eq!(free(a, 0, MEM_RELEASE), Err(ERROR_INVALID_PARAMETER));
        assert_eq!(
            map_view(&section, 1, g, 0, PAGE_READWRITE, false),
            Err(ERROR_INVALID_PARAMETER)
        );

        // Into a placeholder, and back.
        let p = alloc(
            0,
            3 * g,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            PAGE,
        )
        .unwrap();
        assert_eq!(
            map_view(&section, 0, g, p + g, PAGE_READONLY, true),
            Ok(p + g)
        );
        // SAFETY: mapped read-only.
        assert_eq!(unsafe { ((p + g + 5) as *const u8).read() }, 0);
        assert_eq!(
            (q(p + g).allocation_base, q(p + g).kind),
            (p + g, MEM_MAPPED)
        );
        assert_eq!(q(p + 2 * g).allocation_base, p + 2 * g);
        unmap_view(p + g, true).unwrap();
        assert_eq!((q(p + g).state, q(p + g).kind), (MEM_RESERVE, MEM_PRIVATE));
        free(p, 3 * g, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS).unwrap();
        free(p, 0, MEM_RELEASE).unwrap();

        // An alias of a view inside a reservation.
        let r = alloc(0, g, MEM_RESERVE, PAGE_READWRITE, PAGE).unwrap();
        alias(r, a + g, g).unwrap();
        // SAFETY: the alias is mapped read-write.
        assert_eq!(unsafe { ((r + 5) as *const u8).read() }, 9);
        assert_eq!(q(r).state, MEM_COMMIT);
        free(r, 0, MEM_DECOMMIT).unwrap();
        assert_eq!(alias(r, r, g), Err(ERROR_INVALID_ADDRESS));
        free(r, 0, MEM_RELEASE).unwrap();
        unmap_view(a, false).unwrap();
        unmap_view(b, false).unwrap();
    }

    #[test]
    fn physical_pages_map_alias_and_free() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let mut pages = [0u64; 3];
        phys_alloc(&mut pages).unwrap();
        assert!(phys_fd().is_some());
        let va = alloc(0, 4 * PHYS_PAGE, MEM_RESERVE, PAGE_READWRITE, PAGE).unwrap();
        // Page 2 at 0 and 3, pages 0 and 1 at 1 and 2.
        phys_map(
            va,
            &[pages[2], pages[0], pages[1], pages[2]],
            PAGE_READWRITE,
        )
        .unwrap();
        // SAFETY: mapped read-write.
        unsafe {
            (va as *mut u8).write(42);
            assert_eq!(((va + 3 * PHYS_PAGE) as *const u8).read(), 42);
        }
        assert_eq!(q(va).region_size, PHYS_PAGE);
        // Pages 0, 1 and 2 are one run of the pool there.
        assert_eq!(q(va + PHYS_PAGE).region_size, 3 * PHYS_PAGE);
        free(va + 3 * PHYS_PAGE, PHYS_PAGE, MEM_DECOMMIT).unwrap();
        assert_eq!(q(va + 3 * PHYS_PAGE).state, MEM_RESERVE);
        phys_free(&pages);
        // SAFETY: still mapped; the hole reads zero.
        assert_eq!(unsafe { (va as *const u8).read() }, 0);
        let mut again = [0u64; 1];
        phys_alloc(&mut again).unwrap();
        assert!(pages.contains(&again[0]));
        phys_free(&again);
        free(va, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn physical_pages_map_into_a_placeholder() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let mut pages = [0u64; 2];
        phys_alloc(&mut pages).unwrap();
        let g = PHYS_PAGE;
        let va = alloc(0, 4 * g, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE).unwrap();
        phys_map(va + g, &pages, PAGE_READWRITE).unwrap();
        // SAFETY: mapped read-write.
        unsafe { ((va + g) as *mut u8).write(7) };
        // The mapped range is an allocation of its own; the rest stay placeholders.
        assert_eq!((q(va + g).allocation_base, q(va + g).state), (va + g, MEM_COMMIT));
        assert_eq!(q(va).allocation_base, va);
        assert_eq!(q(va + 3 * g).allocation_base, va + 3 * g);
        // Mapped again in place, unmapped, then back to a placeholder.
        phys_map(va + g, &[pages[1], pages[0]], PAGE_READWRITE).unwrap();
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { ((va + 2 * g) as *const u8).read() }, 7);
        free(va + g, 2 * g, MEM_DECOMMIT).unwrap();
        free(va + g, 2 * g, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER).unwrap();
        free(va, 4 * g, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS).unwrap();
        free(va, 0, MEM_RELEASE).unwrap();
        phys_free(&pages);
    }

    #[test]
    fn host_memory_and_images_answer_too() {
        let boxed = Box::new([1u8; 64]);
        let i = q(boxed.as_ptr() as usize);
        assert_eq!(
            (i.state, i.protect, i.kind),
            (MEM_COMMIT, PAGE_READWRITE, MEM_PRIVATE)
        );
        assert!(i.region_size > 0);
        let local = 0u64;
        assert_eq!(q(&local as *const u64 as usize).state, MEM_COMMIT);
        assert_eq!(query(HIGHEST + 1), Err(ERROR_INVALID_PARAMETER));

        // A walk over everything ends.
        let mut at = LOWEST;
        let mut steps = 0;
        while let Ok(i) = query(at) {
            assert!(i.region_size > 0);
            at = i.base_address + i.region_size;
            steps += 1;
        }
        assert!(steps > 3);

        // An image the loader mapped: adopted, protected per section, not freeable.
        let image = alloc(0, 3 * PAGE, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        adopt(image, 3 * PAGE, PAGE_READONLY);
        adopt(image + PAGE, PAGE, PAGE_EXECUTE_READ);
        let i = q(image + PAGE);
        assert_eq!(
            (i.allocation_base, i.protect, i.kind),
            (image, PAGE_EXECUTE_READ, MEM_IMAGE)
        );
        assert_eq!(free(image, 0, MEM_RELEASE), Err(ERROR_INVALID_PARAMETER));
        assert_eq!(protect(image, PAGE, PAGE_READWRITE), Ok(PAGE_READONLY));
        forget(image);
        assert_ne!(q(image).kind, MEM_IMAGE);
        // SAFETY: the mapping made above, no longer in the table.
        unsafe { libc::munmap(image as *mut _, 3 * PAGE) };
    }

    /// Write watching: a block counts written until guarded, then is clean
    /// until a write from our code, from title code or (opened first) from
    /// the kernel; the title's own protection calls never see the guard.
    #[test]
    fn guarded_blocks_see_every_writer() {
        use std::os::unix::fs::FileExt;
        crate::linux::signals::install();
        let base = alloc(0, 4 * BLOCK, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        let (first, end) = (base / BLOCK, base + 4 * BLOCK);
        assert_eq!(guard(base, end, true), Some((first..first + 4).collect()));
        assert_eq!(guard(base, end, true), Some(vec![]));
        assert_eq!((q(base).protect, q(base).region_size), (PAGE_READWRITE, 4 * BLOCK));
        // SAFETY: committed read-write (to the table); the handler takes the write.
        unsafe { ((base + BLOCK + 8) as *mut u8).write(1) };
        // Title-like code (not ours, where a write fault is a title exception):
        // mov [rdi], sil; ret.
        let code = alloc(0, PAGE, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE).unwrap();
        // SAFETY: four bytes of fresh code memory, then a call of them as the function they are.
        let store = unsafe {
            std::ptr::copy_nonoverlapping([0x40u8, 0x88, 0x37, 0xC3].as_ptr(), code as *mut u8, 4);
            std::mem::transmute::<usize, extern "sysv64" fn(*mut u8, u8)>(code)
        };
        store((base + 3 * BLOCK + 1) as *mut u8, 5);
        assert_eq!(guard(base, end, true), Some(vec![first + 1, first + 3]));
        // SAFETY: committed.
        assert_eq!(unsafe { ((base + 3 * BLOCK + 1) as *const u8).read() }, 5);
        // The kernel fails on a guarded block, unless it was opened first.
        let file = tempfile::tempfile().unwrap();
        file.write_all_at(b"disk", 0).unwrap();
        // SAFETY: four committed bytes nothing else refers to.
        let into = unsafe { std::slice::from_raw_parts_mut((base + 2 * BLOCK) as *mut u8, 4) };
        assert_eq!(file.read_at(into, 0).map_err(|e| e.raw_os_error()), Err(Some(libc::EFAULT)));
        super::super::watch::open(base + 2 * BLOCK, 4);
        assert_eq!(file.read_at(into, 0).unwrap(), 4);
        assert_eq!(into, b"disk");
        assert_eq!(guard(base, end, true), Some(vec![first + 2]));
        // VirtualProtect: read-only is not watched; read-write again counts written once.
        assert_eq!(protect(base, PAGE, PAGE_READONLY), Ok(PAGE_READWRITE));
        assert_eq!(guard(base, end, true), Some(vec![first]));
        assert_eq!(protect(base, PAGE, PAGE_READWRITE), Ok(PAGE_READONLY));
        assert_eq!(guard(base, end, true), Some(vec![first]));
        assert_eq!(guard(base, end, true), Some(vec![]));
        // A decommit and commit: new pages, written.
        free(base + BLOCK, BLOCK, MEM_DECOMMIT).unwrap();
        alloc(base + BLOCK, BLOCK, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        // SAFETY: committed again.
        unsafe { ((base + BLOCK) as *mut u8).write(2) };
        assert_eq!(guard(base, end, true), Some(vec![first + 1]));
        // Not the table's memory.
        let host = vec![0u8; 2 * BLOCK];
        assert_eq!(guard(host.as_ptr() as usize, host.as_ptr() as usize + BLOCK, true), None);
        free(base, 0, MEM_RELEASE).unwrap();
        free(code, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    fn guard_limits_partial_bitmap_words_to_the_requested_range() {
        let base = alloc(0, 72 * BLOCK, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        let edge = (base + 2 * BLOCK).next_multiple_of(64 * BLOCK);
        let start = edge - BLOCK;
        let end = edge + BLOCK;
        let left = start - BLOCK;
        let right = end + BLOCK;
        let block = edge / BLOCK;
        assert_eq!(guard(start, end, true), Some(vec![block - 1, block]));
        assert_eq!(guard(start, end, true), Some(vec![]));

        super::super::watch::open(start, 2 * BLOCK);
        assert_eq!(guard(start, end, true), Some(vec![block - 1, block]));
        assert_eq!(guard(left, right, true), Some(vec![block - 2, block + 1]));
        free(base, 0, MEM_RELEASE).unwrap();
    }

    #[test]
    #[ignore = "measures large guarded ranges"]
    fn large_guarded_range_scan() {
        const BLOCKS: usize = 174 * 16;
        const PASSES: usize = 4096;
        let base = alloc(0, BLOCKS * BLOCK, MEM_COMMIT, PAGE_READWRITE, PAGE).unwrap();
        let end = base + BLOCKS * BLOCK;
        assert_eq!(guard(base, end, true).unwrap().len(), BLOCKS);

        let mut clean = Vec::new();
        let mut sparse = Vec::new();
        let blocks = [0, 63, 64, BLOCKS - 1];
        let expected = blocks.map(|block| base / BLOCK + block);
        for _ in 0..9 {
            let start = std::time::Instant::now();
            for _ in 0..PASSES {
                assert!(guard(base, end, true).unwrap().is_empty());
            }
            clean.push(start.elapsed().as_nanos() / PASSES as u128);

            let start = std::time::Instant::now();
            for _ in 0..PASSES / 8 {
                for block in blocks {
                    super::super::watch::open(base + block * BLOCK, BLOCK);
                }
                assert_eq!(guard(base, end, true).unwrap(), expected);
            }
            sparse.push(start.elapsed().as_nanos() / (PASSES as u128 / 8));
        }
        clean.sort_unstable();
        sparse.sort_unstable();
        println!("{BLOCKS} blocks, median ns: clean {}, four dirty {}", clean[4], sparse[4]);
        free(base, 0, MEM_RELEASE).unwrap();
    }

    /// A physical page shown at several addresses is guarded at all of
    /// them: a write through any counts for each, and so do writes into the
    /// pool other than through a view.
    #[test]
    fn writes_through_any_alias_count() {
        crate::linux::signals::install();
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let g = PHYS_PAGE;
        let mut pages = [0u64; 2];
        phys_alloc(&mut pages).unwrap();
        let va = alloc(0, 4 * g, MEM_RESERVE, PAGE_READWRITE, PAGE).unwrap();
        let (b, end) = (va / g, va + 2 * g);
        // Page 0 at 0 and 2, page 1 at 1.
        phys_map(va, &[pages[0], pages[1], pages[0]], PAGE_READWRITE).unwrap();
        assert_eq!(guard(va, end, true), Some(vec![b, b + 1]));
        assert_eq!(guard(va, end, true), Some(vec![]));
        // SAFETY: mapped read-write (to the table).
        unsafe { ((va + 2 * g + 5) as *mut u8).write(9) };
        assert_eq!(guard(va, end, true), Some(vec![b]));
        // SAFETY: mapped.
        assert_eq!(unsafe { ((va + 5) as *const u8).read() }, 9);
        // A new alias of page 1 (the GPU's page-mapping copies) counts once,
        // then writes through it count.
        alias(va + 3 * g, va + g, g).unwrap();
        assert_eq!(guard(va, end, true), Some(vec![b + 1]));
        assert_eq!(guard(va, end, true), Some(vec![]));
        // SAFETY: mapped read-write.
        unsafe { ((va + 3 * g) as *mut u8).write(1) };
        assert_eq!(guard(va, end, true), Some(vec![b + 1]));
        // Written into the pool outside the views (DirectStorage).
        pool_written(&[pages[0]]);
        assert_eq!(guard(va, end, true), Some(vec![b]));
        assert_eq!(guard(va, end, false), None, "views are left to the caller");
        free(va, 0, MEM_RELEASE).unwrap();
        let state = state();
        let pool = Arc::as_ptr(state.pool.as_ref().unwrap()) as usize;
        assert!(!state.shown.keys().any(|&(s, p)| s == pool && pages.contains(&p)));
        drop(state);
        phys_free(&pages);
    }
}
