//! The platform under title code on Linux: what the Windows loader and
//! kernel give a title on Windows. See `docs/linux.md`.
//!
//! - [`image`]: mapping PE images, binding imports, entry points.
//! - [`teb`]: each thread's TEB at its GS base, the PEB, last error.
//! - [`thread`]: guest threads.
//! - [`tls`]: static TLS.
//! - [`seh`]: exception dispatch and unwinding ([`context`] the structures,
//!   [`signals`] faults to exceptions).
//! - [`vm`]: the region table behind `VirtualAlloc` and friends.
//! - [`watch`]: write watching for the GPU side's copies of title memory.
//! - `stacks`: `VICTORY_STACKS` (diagnostics builds).

mod asm;
pub mod context;
mod emulate;
pub mod image;
pub mod seh;
pub mod signals;
#[cfg(feature = "diagnostics")]
pub mod stacks;
pub mod teb;
pub mod thread;
pub mod tls;
mod unwind;
pub mod vm;
pub mod watch;
