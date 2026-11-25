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
//! counted as 64 threads, mesh groups as 128): records past it run empty.
//! Groups reserve work from a saturating counter so rejected costs cannot
//! wrap it and admit more records.
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
uint clamp_record(uint at, uint dst) {
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
    return cost;
}

// Reserve one group's work. A single active group is the only writer, and
// the UAV barriers between clamps already order its load and store.
uint reserve_budget(uint cost, bool only_group) {
    uint before = 0;
    if (only_group) {
        before = min(budget.Load(0), BUDGET);
        if (before < BUDGET) budget.Store(0, before + min(cost, BUDGET - before));
    } else {
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint after = before + min(cost, BUDGET - before);
            uint observed = before;
            budget.InterlockedCompareExchange(0, before, after, observed);
            if (observed == before) break;
            before = observed;
        }
    }
    return BUDGET - min(before, BUDGET);
}

groupshared uint costs[64];
groupshared uint totals[2];
groupshared uint allowance;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint active_records = records;
    if (count_bytes != 0) {
        active_records = min(active_records, count_buf.Load(0));
        if (group.x == 0 && lane == 0) dest.Store(0, active_records);
    }
    uint first = group.x * 64u;
    if (first >= active_records) return;
    uint active_lanes = min(64u, active_records - first);
    uint at = (first + lane) * stride;
    uint dst = at + count_bytes;
    uint cost = 0;
    [branch]
    if (lane < active_lanes) cost = clamp_record(at, dst);
    if (active_lanes == 1) {
        if (lane == 0 && cost != 0) {
            uint left = reserve_budget(cost, active_records == 1);
            if (cost > left) dest.Store(dst, 0);
        }
        return;
    }
    costs[lane] = cost;
    if (lane < 2) totals[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    // A half group fits in uint even if every record costs 2^26.
    if (cost != 0) {
        uint unused;
        InterlockedAdd(totals[lane / 32], cost, unused);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        uint total = min(totals[0], BUDGET + 1u) + min(totals[1], BUDGET + 1u);
        allowance = 0;
        if (total != 0) allowance = reserve_budget(total, active_records <= 64);
        if (allowance >= total) {
            allowance = BUDGET;
        } else if (allowance != 0) {
            // Just the group that straddles the budget needs a prefix.
            uint prefix = 0;
            for (uint i = 0; i < active_lanes; i++) {
                prefix = min(prefix + costs[i], BUDGET + 1u);
                costs[i] = prefix;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (cost != 0 && costs[lane] > allowance) {
        // Clear only the count; the other argument fields keep their layout.
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
    compile_fxc(SOURCE)
}

#[cfg(windows)]
fn compile_fxc(source: &str) -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
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
    make_pipeline(&compiled()?)
}

fn make_pipeline(bytes: &[u8]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
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
/// holds the snapped count (when present) and stays readable until the
/// next clamp writes it.
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
    let sources = [Some(args.0), count.map(|(buffer, _)| buffer).filter(|&buffer| buffer != args.0)];
    let written = sources.map(|source| source.filter(|buffer| list.gpu_written(buffer)));
    // SAFETY: each buffer lives through this recording and its execution.
    let gpu = unsafe { args.0.GetGPUVirtualAddress() }.checked_add(args.1)?;
    let count_gpu = match count {
        Some((buffer, offset)) => unsafe { buffer.GetGPUVirtualAddress() }.checked_add(offset)?,
        None => gpu,
    };
    let (out, before) = list.arguments(u64::from(bytes))?;
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
            transition(&out, before, uav),
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

    fn read_arguments(
        host: &crate::host::Host,
        list: &HostList,
        args: &ID3D12Resource,
        bytes: u64,
    ) -> (ID3D12Resource, *mut u8) {
        let (readback, data) = host.shared_buffer(bytes).unwrap();
        let indirect = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let common = D3D12_RESOURCE_STATE_COMMON;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        // SAFETY: an open list, live buffers of at least `bytes`, and a
        // readable argument buffer left by this list's clamp.
        unsafe {
            list.list.ResourceBarrier(&[
                transition(args, indirect, source),
                transition(&readback, common, dest),
            ]);
            list.list.CopyBufferRegion(&readback, 0, args, 0, bytes);
            list.list.ResourceBarrier(&[
                transition(args, source, indirect),
                transition(&readback, dest, common),
            ]);
        }
        (readback, data)
    }

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
        let (_readback, mapped) = read_arguments(host, &list, &output, 11 * 4);
        list.close();
        host.execute(&[&list]);

        // SAFETY: the GPU finished. Both mapped buffers cover the inspected words.
        unsafe {
            let clamped = std::slice::from_raw_parts(mapped.cast::<u32>(), 11);
            assert_eq!(&clamped[..5], &[1 << 24, 4, 31, u32::MAX - 16, 7]);
            assert_eq!(&clamped[6..], &[3, 2, 12, 0, 6]);
            assert_eq!(std::slice::from_raw_parts(data.cast::<u32>(), title.len()), title);
        }
    }

    #[test]
    fn argument_clamp_budget_never_wraps() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        let clamp_draws = |draws: &[[u32; 4]]| {
            let bytes = size_of_val(draws) as u64;
            let (input, data) = host.shared_buffer(bytes).unwrap();
            let list = host.new_list().unwrap();
            // SAFETY: the idle mapping holds all supplied draw records.
            unsafe {
                std::ptr::copy_nonoverlapping(draws.as_ptr(), data.cast(), draws.len());
            }
            let output = bound(&list, (&input, 0), None, (0, 16, draws.len() as u32))
                .expect("clamp recorded");
            let (_readback, mapped) = read_arguments(host, &list, &output, bytes);
            list.close();
            host.execute(&[&list]);
            // SAFETY: the clamp completed and each mapping covers the inspected records.
            unsafe {
                let records = std::slice::from_raw_parts(mapped.cast::<[u32; 4]>(), draws.len());
                let counts = records.iter().map(|r| r[0]).collect::<Vec<_>>();
                assert!(records.iter().zip(draws).all(|(clamped, src)| clamped[2..] == src[2..]));
                assert_eq!(std::slice::from_raw_parts(data.cast::<[u32; 4]>(), draws.len()), draws);
                counts
            }
        };

        // Each clamped draw costs 2^26, so only two fit. The old add
        // wrapped at 64 draws and let a further two through.
        let large = [u32::MAX, 4, 11, 17];
        let counts = clamp_draws(&[large; 128]);
        assert_eq!(counts.iter().filter(|&&vertices| vertices != 0).count(), 2);

        // Lane order within a group is fixed. After two draws, the third
        // consumes the remainder and the small fourth draw is rejected.
        let almost = [1 << 24, 3, 11, 17];
        assert_eq!(
            clamp_draws(&[almost, almost, almost, [1, 1, 11, 17]]),
            [1 << 24, 1 << 24, 0, 0]
        );

        let small = vec![[1, 1, 11, 17]; 1 << 16];
        assert!(clamp_draws(&small).iter().all(|&vertices| vertices == 1));
    }

    /// An indirect dispatch increments the witness once, regardless of its
    /// grid size or whether the driver overlaps it with another dispatch.
    fn dispatch_witness(host: &crate::host::Host) -> (Clamp, ID3D12CommandSignature) {
        const WITNESS: &str = r#"
RWByteAddressBuffer output : register(u0);
[RootSignature("UAV(u0)")]
[numthreads(1, 1, 1)]
void main(uint3 group : SV_GroupID) {
    if (all(group == 0)) {
        uint unused;
        output.InterlockedAdd(0, 128, unused);
    }
}
"#;
        #[cfg(windows)]
        let witness = compile_fxc(WITNESS).expect("witness shader");
        #[cfg(not(windows))]
        let witness =
            victory_shader::hostlib::compile_hlsl(WITNESS, "main", "cs_6_0").expect("witness shader");
        let c = make_pipeline(&witness).expect("witness pipeline");
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
        (c, signature.unwrap())
    }

    fn record_witness(
        list: &HostList,
        (pipeline, signature): &(Clamp, ID3D12CommandSignature),
        args: &ID3D12Resource,
        (witness, offset): (&ID3D12Resource, u64),
        count: u32,
    ) {
        // SAFETY: an open list; the arguments and count are in
        // INDIRECT_ARGUMENT, and all resources live through execution.
        unsafe {
            let command = &list.list;
            command.SetComputeRootSignature(&pipeline.root);
            command.SetPipelineState(&pipeline.pso);
            command.SetComputeRootUnorderedAccessView(0, witness.GetGPUVirtualAddress() + offset);
            command.ExecuteIndirect(signature, count, args, COUNT_BYTES, Some(args), 0);
        }
    }

    #[test]
    fn argument_clamp_counted_dispatches_only_active_records() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        let pipeline = dispatch_witness(host);
        for count in [0u32, 1, 7] {
            let title = [2u32, 1, 1, 3, 1, 1, count];
            let sentinel = [2u32, 1, 1, 1, 1, 1, 1];
            let seed = [1u32, 1, 1, 1, 1, 1, 2];
            let (input, data) = host.shared_buffer(size_of_val(&title) as u64).unwrap();
            let (seed_input, seed_data) = host.shared_buffer(size_of_val(&seed) as u64).unwrap();
            let (witness, witnessed) = host.shared_buffer(512).unwrap();
            let list = host.new_list().unwrap();
            // SAFETY: the idle mappings cover all initialized words.
            unsafe {
                std::ptr::copy_nonoverlapping(title.as_ptr(), data.cast::<u32>(), title.len());
                std::ptr::copy_nonoverlapping(seed.as_ptr(), seed_data.cast::<u32>(), seed.len());
                witnessed.cast::<u32>().write(0);
            }
            // Populate both slots on the GPU, then check that the next
            // clamp changes only the slots inside its active count.
            let seeded = bound(&list, (&seed_input, 0), Some((&seed_input, 24)), (2, 12, 2)).unwrap();
            let output = bound(&list, (&input, 0), Some((&input, 24)), (2, 12, 2)).unwrap();
            assert_eq!(seeded, output);

            record_witness(&list, &pipeline, &output, (&witness, 0), 2);
            let (_readback, mapped) = read_arguments(host, &list, &output, size_of_val(&sentinel) as u64);
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
            }
        }
    }

    #[test]
    fn argument_clamp_reuses_grows_and_replays_scratch() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        const LARGE: u32 = 8192;
        let count = u64::from(LARGE) * 12;
        let (input, data) = host.shared_buffer(count + 8).unwrap();
        let (witness, witnessed) = host.shared_buffer(768).unwrap();
        // SAFETY: both mappings are idle and cover the initialized words.
        unsafe {
            std::ptr::copy_nonoverlapping([1u32; 3].as_ptr(), data.cast::<u32>(), 3);
            data.add(count as usize).cast::<u32>().write(1);
            data.add(count as usize + 4).cast::<u32>().write(0);
            witnessed.write_bytes(0, 768);
        }
        let pipeline = dispatch_witness(host);
        let list = host.new_list().unwrap();
        let record = |max, counter, offset| {
            let clamped = bound(&list, (&input, 0), Some((&input, counter)), (2, 12, max))
                .expect("clamp recorded");
            record_witness(&list, &pipeline, &clamped, (&witness, offset), max);
            clamped
        };
        let first = record(1, count, 0);
        let second = record(1, count + 4, 256);
        assert_eq!(first, second);
        let grown = record(LARGE, count, 512);
        assert_ne!(second, grown);
        drop((first, second));
        list.close();
        let check = |expected| {
            host.execute(&[&list]);
            // SAFETY: the list finished and the witness mapping is live.
            let observed = [0, 256, 512].map(|i| unsafe { witnessed.add(i).cast::<u32>().read() });
            assert_eq!(observed, expected);
        };
        check([128, 0, 128]);
        check([256, 0, 256]);

        list.reset();
        assert_eq!(record(1, count, 0), grown);
        list.close();
        check([384, 0, 256]);
    }

    #[test]
    fn argument_clamp_budget_resets_on_replay() {
        let Some(host) = host() else { return };
        if cfg!(not(windows)) && std::env::var_os("VICTORY_DXCOMPILER").is_none() {
            return;
        }
        const VERTICES: u32 = 1 << 24;
        let draws = [[VERTICES, 4, 11, 17], [VERTICES, 4, 11, 17], [1, 1, 11, 17]];
        let (input, data) = host.shared_buffer(size_of_val(&draws) as u64).unwrap();
        // SAFETY: the idle mapping covers all three initialized draws.
        unsafe { std::ptr::copy_nonoverlapping(draws.as_ptr(), data.cast(), draws.len()) };
        let list = host.new_list().unwrap();
        for _ in 0..2 {
            // Clamp only: the two large draws are never executed.
            let full = bound(&list, (&input, 0), None, (0, 16, 2)).unwrap();
            let (_full_readback, full_data) = read_arguments(host, &list, &full, 32);
            let spent = bound(&list, (&input, 32), None, (0, 16, 1)).unwrap();
            let (_spent_readback, spent_data) = read_arguments(host, &list, &spent, 16);
            list.close();
            for _ in 0..2 {
                host.execute(&[&list]);
                // SAFETY: execution completed and the two snapshots are live.
                let counts = unsafe {
                    let full = full_data.cast::<u32>();
                    [full.read(), full.add(4).read(), spent_data.cast::<u32>().read()]
                };
                assert_eq!(counts, [VERTICES, VERTICES, 0]);
            }
            list.reset();
        }
        list.close();
    }
}
