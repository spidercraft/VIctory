//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours
//! (`victory_shader::emit::clamp`, built as the translations are: no HLSL
//! compiler on any host).
//!
//! Each list also has a budget ([`LIMITS`]: vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): records past it run
//! empty, so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use victory_shader::emit::clamp::{self, Limits};
use victory_shader::emit::{ROOT_CBV_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host, transition};

const LIMITS: Limits = Limits {
    vertices: 1 << 24,
    vertex_instances: 1 << 26,
    groups_per_dimension: 65535,
    groups: 1 << 20,
    group_cost: 64,
    mesh_group_cost: 128,
    budget: 1 << 27,
};

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn pipeline() -> Option<&'static Clamp> {
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

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let uav = |slot: u8| D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: u32::from(slot),
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    // In the order `bound` sets them: the constants, then the buffers at
    // the root parameter indices the shader names.
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        uav(clamp::ARGUMENTS),
        uav(clamp::BUDGET),
    ];
    let (root, _) = crate::pipeline::host::root_signature(device, &params, &[], true)?;
    let bytes = clamp::argument_clamp(&LIMITS);
    // SAFETY: the description points at live data.
    let pso = unsafe {
        device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?
    };
    Ok(Clamp { root, pso })
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (pipeline(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(clamp::ARGUMENTS.into(), gpu);
        l.SetComputeRootUnorderedAccessView(clamp::BUDGET.into(), budget);
        l.Dispatch(records.div_ceil(clamp::GROUP), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
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
    use victory_shader::emit::clamp::{DISPATCH, DRAW, DRAW_INDEXED, MESH, clamped};

    use super::*;

    /// Runs the clamp over `records` (three dwords each, `stride` bytes
    /// apart) in a list of its own and returns them; None without a host.
    fn run(kind: u32, stride: u32, records: &[[u32; 3]]) -> Option<Vec<[u32; 3]>> {
        let host = host()?;
        let step = (stride / 4) as usize;
        let (buffer, data) = host.shared_buffer(u64::from(stride) * records.len() as u64)?;
        // SAFETY: the buffer's own mapped dwords, `step` to a record.
        let dwords = unsafe { std::slice::from_raw_parts_mut(data.cast::<u32>(), step * records.len()) };
        // Dwords past the third are the record's other arguments: untouched.
        dwords.fill(0xdead_beef);
        for (to, from) in dwords.chunks_mut(step).zip(records) {
            to[..3].copy_from_slice(from);
        }
        let list = host.new_list().unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        assert!(bound(&list, &buffer, gpu, (kind, stride, records.len() as u32)));
        done(&list, &buffer);
        list.close();
        host.execute(&[&list]);
        assert!(dwords.chunks(step).all(|r| r[3..].iter().all(|&d| d == 0xdead_beef)));
        Some(dwords.chunks(step).map(|r| [r[0], r[1], r[2]]).collect())
    }

    /// The host takes the shader we build and runs it as the arithmetic in
    /// `clamped` says, for every kind of record.
    #[test]
    fn the_host_clamps_as_the_model_does() {
        const BIG: u32 = u32::MAX;
        let cases: [(u32, u32, &[[u32; 3]]); 4] = [
            (DRAW, 16, &[[3, 1, 7], [BIG, 1, 7], [6, 1000, 7], [0, BIG, 7], [1 << 24, 5, 7]]),
            (DRAW_INDEXED, 20, &[[36, 2, 9], [BIG, BIG, 9], [0, 0, 9]]),
            (
                DISPATCH,
                12,
                &[[8, 8, 1], [BIG, 1, 1], [1024, 1024, 1], [1024, 1025, 1], [2, 2, BIG], [5, 0, 5], [1, 1, 0]],
            ),
            (MESH, 12, &[[40, 1, 1], [BIG, BIG, BIG], [9, 0, 1], [0, 1, 1]]),
        ];
        for (kind, stride, records) in cases {
            // Under the budget together, so the order the threads ran in
            // does not show.
            let spent: u64 = records.iter().map(|&r| u64::from(clamped(&LIMITS, kind, r, 0).1)).sum();
            assert!(spent <= u64::from(LIMITS.budget));
            let Some(got) = run(kind, stride, records) else {
                eprintln!("no host D3D12 device; skipped");
                return;
            };
            for (i, (&record, got)) in records.iter().zip(got).enumerate() {
                let want = clamped(&LIMITS, kind, record, 0).0;
                assert_eq!(got, want, "kind {kind} record {i} {record:x?}");
            }
        }
        // More records than one thread group, with their own values.
        let many: Vec<[u32; 3]> = (0..200).map(|i| [i, i % 7, BIG - i]).collect();
        let got = run(DRAW, 16, &many).unwrap();
        for (&record, got) in many.iter().zip(got) {
            assert_eq!(got, clamped(&LIMITS, DRAW, record, 0).0);
        }
    }

    /// Records past the list's budget run empty: a draw keeps its layout
    /// with no vertices.
    #[test]
    fn records_past_the_budget_run_empty() {
        // Each draw clamps to the most one may ask for; two fill the budget.
        let most = LIMITS.vertex_instances;
        assert_eq!(LIMITS.budget, 2 * most);
        let Some(got) = run(DRAW, 16, &[[1 << 24, u32::MAX, 0]; 6]) else {
            eprintln!("no host D3D12 device; skipped");
            return;
        };
        let instances = most >> 24;
        // Which two ran is the threads' order.
        assert_eq!(got.iter().filter(|r| **r == [1 << 24, instances, 0]).count(), 2, "{got:x?}");
        assert_eq!(got.iter().filter(|r| **r == [0, instances, 0]).count(), 4, "{got:x?}");
    }
}
