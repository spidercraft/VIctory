use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, VIEW_STRIDES, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

#[test]
fn a_heap_buffer_read_uses_the_bound_slots_stride() {
    // s0 selects the input from the whole view heap, s2 names the output
    // table. Each lane copies the second dword of one input element.
    let code = [
        0x8F04_8500u32, // s_lshl_b32 s4, s0, 5
        0xBE86_0301,    // s_mov_b32 s6, s1
        0xBE87_0380,    // s_mov_b32 s7, 0
        0xF408_0203,
        0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
        0xBE8C_0302, // s_mov_b32 s12, s2
        0xBE8D_0380, // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE030_2004,
        0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let param = |root, user_data| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: code.iter().flat_map(|word| word.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let res = resource::track(&p, &native.layout, native.stage).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let st = structure::structure(&p).unwrap();
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .unwrap();
    assert_eq!(lowered.root_srvs, [VIEW_STRIDES]);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: vec![],
        samplers: vec![],
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator.validate(&mut shader.dxil.clone()).unwrap();
    }

    let Ok(host) = Host::new(false) else {
        eprintln!("no host SM 6.6 device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let first: Vec<_> = (0..32)
        .flat_map(|lane| [100 + lane, 200 + lane, 300 + lane])
        .collect();
    let second: Vec<_> = (0..32)
        .flat_map(|lane| [400 + lane, 500 + lane, 600 + lane, 700 + lane, 800 + lane])
        .collect();
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &first,
            stride: Stride::Structured(12),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(20),
        },
        View::Buffer {
            data: &[900, 901],
            stride: Stride::Raw,
        },
    ];
    let cases = [
        (1, (200..232).collect::<Vec<_>>()),
        (2, (500..532).collect()),
        (3, vec![901; 32]),
        (0, vec![0; 32]),
        (4, vec![0; 32]),
    ];
    for (slot, expected) in cases {
        let mut user_data = vec![0; shader.user_data_dwords() as usize];
        for entry in &shader.user_data {
            if entry.root == RootType::Constants {
                user_data[entry.index as usize] = slot;
            }
        }
        let results = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(user_data)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 0, 13, 21, 1],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        assert_eq!(results[0], expected, "view heap slot {slot}");
    }
}
