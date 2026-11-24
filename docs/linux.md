# Native Linux runtime

Goal: `victory run <exe> --execute` on Linux as an ELF program, with no
Wine and no Windows DLL. The title's x86-64 code still runs unchanged in
our process; on Linux we also supply what the Windows loader, kernel and
system DLLs supply on Windows. The Windows build keeps working unchanged.

## What changes, and where

| On Windows | On Linux | Where |
|------------|----------|-------|
| `LoadLibraryExW` maps the prepared image | our PE mapper: `mmap`, relocations, protections | `victory-loader/src/linux/image.rs` |
| the TEB, static TLS (`gs:[0x58]`), last error | a TEB at each title thread's GS base (`wrgsbase`, else `arch_prctl(ARCH_SET_GS)`) | `victory-loader/src/linux/teb.rs` |
| `CreateThread` | guest threads: TEB, stack with guard page, TLS callbacks | `victory-loader/src/linux/thread.rs` |
| SEH: the kernel and `ntdll` dispatch faults through `.pdata` | signals to `EXCEPTION_RECORD` + `CONTEXT`, our own dispatcher and unwinder | `victory-loader/src/linux/seh.rs`, `signals.rs` |
| `VirtualAlloc` and friends | the region table over `mmap` (64 KiB granularity, `memfd` for physical pages) | `victory-loader/src/linux/vm.rs` |
| host pass-through (`HostPolicy`) | our implementation of each allowed DLL | `victory-hle/src/linux/<family>_<slice>.rs` |
| host D3D12 and DXGI | host D3D12 on Vulkan (see GPU below) | `victory-gpu` |
| Win32 window, XInput | a Linux window and gamepads | `victory-gpu` (window), `victory-hle` (input) |

`victory-loader::linux` is the platform (memory, threads, exceptions);
`victory-hle::linux` is the Win32 API surface built on it. `victory-hle`
depends on `victory-loader`, never the other way round.

## Rules

- **Calling convention.** Every function title code calls, and every
  title function we call, is `extern "win64"`. `extern "system"` is only
  for callbacks handed to a Windows host API; on Linux it is SysV.
- **Import resolution is unchanged.** `HostPolicy` still decides which
  DLLs pass through; on Linux, `host::imp::resolve` looks the import up in
  `victory-hle::linux` (`lookup`, `lookup_ordinal`). The modules in
  `victory-hle::modules` still win over it, and DLLs the title ships
  (`vcruntime140`, `msvcp140`, ...) are still mapped and run, not
  replaced.
- **Adding a function.** In `victory-hle/src/linux/<family>_<slice>.rs`
  (family: the real DLL name, lower case, without `.dll`): an
  `extern "win64"` function named like the export, listed with
  `hle_export!` in the file's `module()`, and that `module()` added to
  `linux::modules()`. A module's `dlls` lists the DLL and the API sets
  that forward to it (`kernel32.dll`, `kernelbase.dll`,
  `api-ms-win-core-synch-*`). An export by ordinal is named `#<n>`.
- **Guest types.** `DWORD`/`LONG`/`BOOL` are 32 bits, wide text is `u16`,
  `HANDLE` is pointer-sized. Never `c_long` or `wchar_t` (64 and 32 bits
  on Linux). Structures the title reads keep their Windows layout, with
  their size and field offsets checked by tests.
- **Errors.** Report through `victory_loader::linux::teb::set_last_error`
  with Windows error codes; NTSTATUS where the API returns one.
- **Paths.** The `victory-hle` file module rewrites console paths to host
  paths before the Linux layer sees them. The Linux layer takes a host
  path as Windows code spells it: `\` and `/` both separate, no drive
  letter means the current directory.
- **Shared state lives in one place.** One handle table (files, events,
  threads, all waitable through the same `WaitForMultipleObjects`), one
  region table, one TEB layout.
- **No Wine, no Windows DLL, no system DLL forwarding.** Where a Linux
  library already does the job (Vulkan, a D3D12-on-Vulkan library, an
  audio API), call it through its C API.
- **Unsafe code** keeps the repository's `// SAFETY:` comments.
- **Tests run without game data** (`cargo test --workspace` on Linux).
  Guest code under test is a PE we build: `tests/fixtures/hle-probe`
  (`x86_64-pc-windows-msvc`, linked with `lld-link` against an xwin SDK).
  Tests that need the game take `VICTORY_GAME_DIR` and skip when it is
  unset; they assert properties, never snapshots
  ([asset policy](asset-policy.md)).

## GPU

`victory-gpu` translates D3D12.X to host D3D12 and console shaders to
DXIL. On Linux the host D3D12 is vkd3d-proton built as native Linux
libraries and loaded at run time, so the D3D12 code and the shader
translator stay shared.

- **Calling convention.** vkd3d-proton's native build defines
  `STDMETHODCALLTYPE` as nothing (`include/vkd3d_windows.h`): its COM is
  SysV, which is what `extern "system"` means on Linux, so the
  `windows` crate's vtables match it. Objects the title receives from
  the host (root signature blobs, deserializers) are copied into `win64`
  objects of ours (`victory-gpu/src/root.rs`).
- **Bindings.** The `windows` crate does not build on Linux (its
  `windows-future` uses parts of `windows-core` that only exist on
  Windows). `victory-windows` generates the same `Direct3D`, `D3D12` and
  `DXGI` bindings with `windows-bindgen` at build time (nothing generated
  is committed) and supplies the few core items (`PCSTR`, `PCWSTR`,
  `HSTRING`, `Free`); `victory-gpu` and `victory-shader` name it
  `windows`. vkd3d-proton's `WCHAR` is 16-bit, so wide names work as is.
- **Loading.** `victory_shader::hostlib` opens
  `libvkd3d-proton-d3d12core.so`, then `libvkd3d-proton-d3d12.so`, from
  `VICTORY_VKD3D` (their directory, or the d3d12 library itself), else
  the loader's search path. Missing, the runtime warns once and nothing
  executes, as without a GPU; device tests skip.
- **Presentation** has no DXGI: vkd3d-proton's command queues answer
  `IDXGIVkSwapChainFactory` (what its DXGI uses), which takes our
  `IDXGIVkSurfaceFactory` for the window (`victory-gpu/src/linux.rs`).
  vkd3d-proton's native instance enables the Wayland, Xlib and XCB
  surface extensions.
- **Window and input: SDL3** (`libSDL3.so.0`, run time, `VICTORY_SDL3`):
  window, Vulkan surface, keyboard, and gamepads with rumble from one
  library already on most desktops, no crate. winit would need a
  gamepad library and a surface helper beside it. One thread owns SDL;
  `victory_gpu::linux::{pad, keys, rumble}` feed `gameinput`.
- **What it supports.** vkd3d-proton 3.x: feature level 12_x, SM 6.6
  (DXIL through dxil-spirv), mesh shaders, directly indexed heaps,
  pipeline libraries. Fences need no event: `SetEventOnCompletion` with
  none waits.
- **Write watching** (only what the title wrote since a copy is copied
  again; Windows: `GetWriteWatch` and a guard on views): the region table
  guards every 64 KiB block a copy took with `mprotect`
  (`victory-loader/src/linux/vm.rs` `guard`, `watch.rs`), and the
  signal handler takes the first write to one, from title code or ours,
  before anything else sees it. `VirtualQuery` and `VirtualProtect` never
  see the guard; a physical page shown at several addresses is guarded at
  all of them. The kernel does not fault into the handler (`read(2)` into
  a guarded block is `EFAULT`): code that lets it write title memory calls
  `victory_gpu::kernel_will_write` first (the file layer and UCRT reads,
  DirectStorage), and writes into the physical pool outside the views call
  `vm::pool_written`.
- **Not yet:** the DXIL validator by default (below), fence events until the handle table registers its `SetEvent`
  (`victory_gpu::fence::on_set_event`), the debug layer and WARP.

Building vkd3d-proton (meson, ninja, glslang; Vulkan headers come with
it):

```sh
git clone --recurse-submodules https://github.com/HansKristian-Work/vkd3d-proton
cd vkd3d-proton
meson setup build --buildtype=release --prefix="$HOME/.local/vkd3d-proton"
ninja -C build install   # if it fails on a generated header, run it again
export VICTORY_VKD3D="$HOME/.local/vkd3d-proton/lib"
```

CachyOS packages `vkd3d-proton-git` (unchecked whether as native
libraries); Arch's `vkd3d` is upstream vkd3d, a different library.
vkd3d-proton writes `vkd3d-proton.cache` into the working directory
(`VKD3D_SHADER_CACHE_PATH` moves it).

DXC's validator is optional and opt-in on Linux: `VICTORY_DXIL` names
`libdxil.so` for `victory translate --validate`. Arch's
`directx-shader-compiler` 1.10.2605 validator traps on some of our
shaders, so it is not loaded unasked. Microsoft's own Linux release works
(tested: v1.9.2609). Nothing needs an HLSL compiler: the indirect argument
clamp is built by our own emitter.

```sh
gh release download v1.9.2609 -R microsoft/DirectXShaderCompiler -p 'linux_dxc_*.tar.gz'
mkdir -p ~/.local/dxc && tar xzf linux_dxc_*.tar.gz -C ~/.local/dxc
export VICTORY_DXIL=~/.local/dxc/lib/libdxil.so
```

Tests with a device: `VICTORY_VKD3D=... cargo test -p victory-shader
--features host -p victory-gpu`; `VICTORY_TEST_WINDOW=1` also presents
to a window.

## Status

| Part | State |
|------|-------|
| Workspace builds and tests on Linux | done |
| Title-facing ABI is `win64` | done |
| PE mapper, TEB, guest threads, TLS, entry | done |
| SEH, signals, unwinding | done: dispatcher, unwinder, signals, `Rtl*`/`RaiseException`, crash report (`linux/seh.rs`) |
| kernel32: handles, waits, sync objects and locks, threads, TLS/FLS, thread pool, time, system, process | done (`victory-hle/src/linux/handles.rs`, `kernel32_*.rs`; SuspendThread uses `SIGRTMIN+6`) |
| Region table (`vm.rs`), kernel32 memory, heap, files at host level, text, locale | done: `kernel32_{memory,heap,file,io,path,text,locale}.rs`; overlapped I/O completes inside the call |
| UCRT | every import the title and its DLLs take (`ucrt_*.rs`); C locale only |
| GPU, window, input | device, pipelines, window, pads and keyboard on vkd3d-proton and SDL3; every vertex, pixel and mesh program of the title makes a pipeline; write watching |
| Audio, network (offline), crypto, COM and the other host DLLs | done: every import of these DLLs resolves. Sound through PulseAudio/PipeWire (`libpulse-simple`, loaded at run time): the game's Spatial Audio stream (`mmdevapi.rs`) and Bink's XAudio2 (`xaudio2.rs`); loopback-only Winsock, WinHTTP that cannot connect, Media Foundation without sources, an empty registry |
| xmem, virtual drives (`files`), persistent local storage | done: xmem on the region table (reservations inside the console's address limits, the physical pool, `alias` for the GPU's page mappings, `MEM_DECOMMIT` to unmap; no placeholders of ours, while the title's own take pages); `G:`/`T:`/storage drives rewritten to Linux folders, names found whatever their case; the title's own `msvcp140` reaches them through its imports |
| `victory-hle` modules lifecycle, xsystem, apu, dstorage | done: suspend on SIGINT/SIGTERM/SIGHUP (the window's owner raises SIGTERM when the user closes it), relaunch waits on a pidfd, the device type over the empty registry; dstorage reads through the Linux file layer and writes into physical pages |
| `net`, `media` | Windows only: on Linux `ws2_32`, `winhttp` and `mfplat` keep the same offline policy themselves |
