//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside. On
//! Linux DXC compiles it, as 6.0, if `VICTORY_DXCOMPILER` names one).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
#[cfg(windows)]
use windows::core::s;

use crate::host::{HostList, host, transition};

/// A counted execution puts its count before the clamped records.
pub(crate) const COUNT_BYTES: u64 = size_of::<u32>() as u64;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u1), UAV(u0), SRV(t1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint count_bytes; };
ByteAddressBuffer args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint active_records = records;
    if (count_bytes != 0) {
        active_records = min(active_records, count_buf.Load(0));
        if (id.x == 0) dest.Store(0, active_records);
    }
    if (id.x >= active_records) return;
    uint at = id.x * stride;
    uint dst = at + count_bytes;
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(dst, uint2(count, instances));
        if (kind == 1) {
            dest.Store3(dst + 8, args.Load3(at + 8));
        } else {
            dest.Store2(dst + 8, args.Load2(at + 8));
        }
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(dst, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(dst, g);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        dest.Store(dst, 0);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

/// The clamp's DXIL, with its root signature, compiled once. DXC's LLVM
/// replaces our fault handlers at its first compile (it registers its own
/// once per process), so they are taken back right after; the device's
/// creation compiles it, before any copy guards title memory.
#[cfg(not(windows))]
pub(crate) fn compiled() -> windows::core::Result<Vec<u8>> {
    static DXIL: std::sync::OnceLock<Option<Vec<u8>>> = std::sync::OnceLock::new();
    DXIL.get_or_init(|| {
        let dxil = victory_shader::hostlib::compile_hlsl(SOURCE, "main", "cs_6_0");
        victory_loader::linux::signals::reclaim();
        dxil.inspect_err(|e| tracing::warn!("argument clamp: {e}")).ok()
    })
    .clone()
    .ok_or_else(|| windows::core::Error::from_hresult(windows::Win32::Foundation::E_FAIL))
}

/// The clamp's DXBC, with its root signature.
#[cfg(windows)]
fn compiled() -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    Ok(unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    }
    .to_vec())
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let bytes = &compiled()?[..];
    // SAFETY: the root signature is inside the shader.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart. The list's private buffer
/// holds the snapped count (when present) and results until [`done`].
/// Replaces the list's compute root signature and pipeline.
pub(crate) fn bound(
    list: &HostList,
    args: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    if !args.1.is_multiple_of(4) || !stride.is_multiple_of(4)
        || count.is_some_and(|(_, offset)| !offset.is_multiple_of(4))
    {
        return None;
    }
    let c = clamp()?;
    let budget = list.budget()?;
    let count_bytes = if count.is_some() { COUNT_BYTES } else { 0 };
    let bytes = (u64::from(stride) * u64::from(records.max(1))).checked_add(count_bytes)?;
    let bytes = u32::try_from(bytes).ok()?;
    let out = list.arguments(u64::from(bytes))?;
    let sources = [Some(args.0), count.map(|(buffer, _)| buffer).filter(|&buffer| buffer != args.0)];
    let written = sources.map(|source| source.filter(|buffer| list.gpu_written(buffer)));
    // SAFETY: each buffer lives through this recording and its execution.
    let gpu = unsafe { args.0.GetGPUVirtualAddress() }.checked_add(args.1)?;
    let count_gpu = match count {
        Some((buffer, offset)) => unsafe { buffer.GetGPUVirtualAddress() }.checked_add(offset)?,
        None => gpu,
    };
    let l = &list.list;
    let constants = [kind, stride, records, count_bytes as u32];
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // SAFETY: an open list; the caller holds the input alive and the list
    // keeps the output and budget allocated.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, uav),
        ]);
        for buffer in written.into_iter().flatten() {
            l.ResourceBarrier(&[transition(buffer, uav, read)]);
        }
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        l.SetComputeRootShaderResourceView(4, count_gpu);
        l.Dispatch(records.div_ceil(64).max(u32::from(count.is_some())), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &out,
                uav,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
        // Execution reads the snapshot, so both title buffers can return
        // to their previous state before the draw binds its resources.
        for buffer in written.into_iter().flatten() {
            l.ResourceBarrier(&[transition(buffer, read, uav)]);
        }
    }
    Some(out)
}

/// Restore the private buffer to COMMON for the next clamp.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return; // no HLSL compiler named
        }
        #[cfg(not(windows))]
        let handler = || {
            victory_loader::linux::signals::install();
            // SAFETY: a query into a zeroed action of ours.
            unsafe {
                let mut action: libc::sigaction = std::mem::zeroed();
                libc::sigaction(libc::SIGSEGV, std::ptr::null(), &mut action);
                action.sa_sigaction
            }
        };
        #[cfg(not(windows))]
        let ours = handler();
        assert!(super::clamp().is_some());
        // DXC's LLVM took the fault handler at its first compile; ours is back.
        #[cfg(not(windows))]
        assert_eq!(handler(), ours);
    }

    #[test]
    fn argument_clamp_preserves_title_records() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        // Two indexed records with a nonzero base offset and padded stride.
        let title = [0xfeed, u32::MAX, 9, 31, u32::MAX - 16, 7, 0xbeef, 3, 2, 12, 0, 6, 0];
        let (input, data) = host.shared_buffer(std::mem::size_of_val(&title) as u64).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: the mapping covers the title words and the buffer is idle.
        unsafe {
            std::ptr::copy_nonoverlapping(title.as_ptr(), data.cast::<u32>(), title.len());
        };
        let output = bound(&list, (&input, 4), None, (1, 24, 2)).expect("clamp recorded");
        assert_ne!(output, input);
        assert_eq!(output, list.arguments(48).unwrap());
        done(&list, &output);
        list.close();
        host.execute(&[&list]);

        let mut mapped = std::ptr::null_mut();
        // SAFETY: the GPU finished. Both mapped buffers cover the inspected words.
        unsafe {
            output.Map(0, None, Some(&mut mapped)).unwrap();
            let clamped = std::slice::from_raw_parts(mapped.cast::<u32>(), 11);
            assert_eq!(&clamped[..5], &[1 << 24, 4, 31, u32::MAX - 16, 7]);
            assert_eq!(&clamped[6..], &[3, 2, 12, 0, 6]);
            assert_eq!(std::slice::from_raw_parts(data.cast::<u32>(), title.len()), title);
        }
    }

    #[test]
    fn argument_clamp_counted_dispatches_only_active_records() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        let c = clamp().expect("clamp pipeline");
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a live description of one dispatch argument with no root changes.
        unsafe {
            host.device.CreateCommandSignature(
                &D3D12_COMMAND_SIGNATURE_DESC {
                    ByteStride: 12,
                    NumArgumentDescs: 1,
                    pArgumentDescs: &argument,
                    NodeMask: 0,
                },
                None::<&ID3D12RootSignature>,
                &mut signature,
            ).unwrap();
        }
        let signature = signature.unwrap();
        for count in [0u32, 1, 7] {
            let title = [2u32, 1, 1, 3, 1, 1, count];
            let sentinel = [2u32, 1, 1, 1, 1, 1, 1];
            let (input, data) = host.shared_buffer(size_of_val(&title) as u64).unwrap();
            let (witness, witnessed) = host.shared_buffer(512).unwrap();
            let list = host.new_list().unwrap();
            let output = list.arguments(size_of_val(&sentinel) as u64).unwrap();
            let mut mapped = std::ptr::null_mut();
            // SAFETY: the idle mappings cover all initialized words.
            unsafe {
                std::ptr::copy_nonoverlapping(title.as_ptr(), data.cast::<u32>(), title.len());
                witnessed.cast::<u32>().write(0);
                output.Map(0, None, Some(&mut mapped)).unwrap();
                std::ptr::copy_nonoverlapping(sentinel.as_ptr(), mapped.cast::<u32>(), sentinel.len());
            }
            let clamped = bound(&list, (&input, 0), Some((&input, 24)), (2, 12, 2)).unwrap();
            assert_eq!(clamped, output);

            // Reuse the clamp as a test compute pipeline. Each indirect
            // dispatch adds 128 to its separate budget at `witnessed`.
            let constants = [2u32, 12, 1, 0];
            // SAFETY: all buffers live through execution. The private buffer
            // is in INDIRECT_ARGUMENT and has distinct count and argument offsets.
            unsafe {
                let command = &list.list;
                let gpu = input.GetGPUVirtualAddress();
                let witness = witness.GetGPUVirtualAddress();
                command.SetComputeRootSignature(&c.root);
                command.SetPipelineState(&c.pso);
                command.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                command.SetComputeRootShaderResourceView(1, gpu);
                command.SetComputeRootUnorderedAccessView(2, witness);
                command.SetComputeRootUnorderedAccessView(3, witness + 256);
                command.SetComputeRootShaderResourceView(4, gpu);
                command.ExecuteIndirect(&signature, 2, &clamped, COUNT_BYTES, Some(&clamped), 0);
            }
            done(&list, &clamped);
            list.close();
            host.execute(&[&list]);

            let active = count.min(2) as usize;
            // SAFETY: the GPU finished and each mapping covers the inspected words.
            unsafe {
                let words = std::slice::from_raw_parts(mapped.cast::<u32>(), sentinel.len());
                assert_eq!(words[0], active as u32);
                let tail = active * 3;
                assert_eq!(&words[1..1 + tail], &title[..tail]);
                assert_eq!(&words[1 + tail..], &sentinel[1 + tail..]);
                assert_eq!(witnessed.cast::<u32>().read(), (active * 128) as u32);
                assert_eq!(std::slice::from_raw_parts(data.cast::<u32>(), title.len()), title);
                output.Unmap(0, None);
            }
        }
    }
}
