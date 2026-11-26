//! Compose the title's display planes before presenting the host back buffer.

use super::*;
use windows::Win32::Foundation::{E_FAIL, RECT};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

pub(crate) struct Shown<'a> {
    pub(crate) texture: &'a ID3D12Resource,
    pub(crate) state: u32,
    pub(crate) source: Option<[i32; 4]>,
    pub(crate) destination: Option<[f32; 4]>,
}

const MAX_PLANES: u32 = 4;
const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)),RootConstants(num32BitConstants=4,b0)," \
           "StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_LINEAR,addressU=TEXTURE_ADDRESS_CLAMP," \
           "addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> image : register(t0);
SamplerState clamped : register(s0);
cbuffer Source : register(b0) { float4 source; };
struct Corner { float4 position : SV_Position; float2 uv : TEXCOORD0; };
Corner corner(uint i : SV_VertexID) {
    Corner c;
    c.uv = float2(float(i & 1) * 2.0, float(i >> 1) * 2.0);
    c.position = float4(c.uv.x * 2.0 - 1.0, 1.0 - c.uv.y * 2.0, 0.0, 1.0);
    return c;
}
[RootSignature(RS)]
float4 compose(Corner c) : SV_Target {
    return image.SampleLevel(clamped, lerp(source.xy, source.zw, c.uv), 0);
}
"#;

struct Pass {
    vs: Vec<u8>,
    ps: Vec<u8>,
    root: ID3D12RootSignature,
    views: ID3D12DescriptorHeap,
    step: u32,
    pipelines: std::collections::HashMap<(i32, bool), Option<ID3D12PipelineState>>,
}

// SAFETY: D3D12 objects are free-threaded. The slots are held until execute
// finishes through PASS's lock.
unsafe impl Send for Pass {}

static PASS: OnceLock<Mutex<Option<Pass>>> = OnceLock::new();

impl Pass {
    fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let compile = |entry, target| {
            let result = victory_shader::hostlib::compile_hlsl(SOURCE, entry, target);
            victory_loader::linux::signals::reclaim();
            result.map_err(|detail| windows::core::Error::new(E_FAIL, detail))
        };
        let vs = compile("corner", "vs_6_0")?;
        let ps = compile("compose", "ps_6_0")?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: the pixel shader carries the root signature and the heap
        // description holds only values.
        unsafe {
            let root = device.CreateRootSignature(0, &ps)?;
            let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: MAX_PLANES,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            Ok(Self {
                vs,
                ps,
                root,
                views,
                step: device.GetDescriptorHandleIncrementSize(kind),
                pipelines: Default::default(),
            })
        }
    }

    fn pipeline(
        &mut self,
        device: &ID3D12Device,
        format: DXGI_FORMAT,
        blend: bool,
    ) -> Option<ID3D12PipelineState> {
        self.pipelines
            .entry((format.0, blend))
            .or_insert_with(|| {
                let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                };
                let mut blending = D3D12_BLEND_DESC::default();
                blending.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: blend.into(),
                    SrcBlend: D3D12_BLEND_SRC_ALPHA,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    RenderTargetWriteMask: 0xf,
                    ..Default::default()
                };
                let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
                formats[0] = format;
                // SAFETY: referenced bytecode and root signature live through
                // pipeline creation. The description borrows the root pointer.
                unsafe {
                    device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&self.root),
                        VS: code(&self.vs),
                        PS: code(&self.ps),
                        BlendState: blending,
                        SampleMask: u32::MAX,
                        RasterizerState: D3D12_RASTERIZER_DESC {
                            FillMode: D3D12_FILL_MODE_SOLID,
                            CullMode: D3D12_CULL_MODE_NONE,
                            DepthClipEnable: true.into(),
                            ..Default::default()
                        },
                        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                        NumRenderTargets: 1,
                        RTVFormats: formats,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        ..Default::default()
                    })
                }
                .inspect_err(|error| {
                    tracing::warn!(%error, format = format.0, blend,
                "could not create a display-plane pipeline")
                })
                .ok()
            })
            .clone()
    }
}

fn view_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(DXGI_FORMAT(match format.0 {
        27 => 28,
        90 => 87,
        23 => 24,
        9 => 10,
        10 | 11 | 13 | 24 | 28 | 29 | 31 | 87 | 91 => format.0,
        _ => return None,
    }))
}

fn target_format(plane: DXGI_FORMAT, back: DXGI_FORMAT) -> DXGI_FORMAT {
    match (plane, back) {
        (DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM)
        | (DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM) => plane,
        _ => back,
    }
}

fn source_fractions(source: Option<[i32; 4]>, (width, height): (u32, u32)) -> [f32; 4] {
    let (w, h) = (width as f32, height as f32);
    match source {
        Some([l, t, r, b])
            if 0 <= l && l < r && r as u32 <= width && 0 <= t && t < b && b as u32 <= height =>
        {
            [l as f32 / w, t as f32 / h, r as f32 / w, b as f32 / h]
        }
        _ => [0.0, 0.0, 1.0, 1.0],
    }
}

fn placement(destination: Option<[f32; 4]>, (width, height): (u32, u32)) -> [f32; 4] {
    let (w, h) = (width as f32, height as f32);
    match destination {
        Some([l, t, r, b]) if 0.0 <= l && l < r && r <= 1.0 && 0.0 <= t && t < b && b <= 1.0 => {
            [l * w, t * h, r * w, b * h]
        }
        _ => [0.0, 0.0, w, h],
    }
}

struct Draw<'a> {
    plane: &'a Shown<'a>,
    view: DXGI_FORMAT,
    target: DXGI_FORMAT,
    pipeline: ID3D12PipelineState,
    source: [f32; 4],
    destination: [f32; 4],
}

impl Pass {
    fn prepare<'a>(
        &mut self,
        device: &ID3D12Device,
        format: DXGI_FORMAT,
        size: (u32, u32),
        planes: &'a [Shown<'a>],
    ) -> Option<Vec<Draw<'a>>> {
        if planes.len() > MAX_PLANES as usize {
            return None;
        }
        planes
            .iter()
            .enumerate()
            .map(|(i, plane)| {
                // SAFETY: the caller holds the texture through presentation.
                let desc = unsafe { plane.texture.GetDesc() };
                if desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0
                    || desc.SampleDesc.Count != 1
                    || desc.DepthOrArraySize != 1
                    || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
                {
                    return None;
                }
                let view = view_format(desc.Format)?;
                let target = target_format(view, format);
                Some(Draw {
                    plane,
                    view,
                    target,
                    pipeline: self.pipeline(device, target, i != 0)?,
                    source: source_fractions(plane.source, (desc.Width as u32, desc.Height)),
                    destination: placement(plane.destination, size),
                })
            })
            .collect()
    }

    /// Draws onto a target already in RENDER_TARGET, returning each source
    /// to its former state. The caller holds the pass through submission.
    fn record(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        draws: &[Draw<'_>],
    ) {
        const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list, and distinct live input textures and target.
        unsafe {
            list.ClearRenderTargetView(host.rtv(back, None), &[0.0, 0.0, 0.0, 1.0], None);
            list.SetDescriptorHeaps(&[Some(self.views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let cpu = self.views.GetCPUDescriptorHandleForHeapStart().ptr;
            let gpu = self.views.GetGPUDescriptorHandleForHeapStart().ptr;
            for (i, draw) in draws.iter().enumerate() {
                let texture = draw.plane.texture;
                let state = D3D12_RESOURCE_STATES(draw.plane.state as i32);
                if state != READ {
                    list.ResourceBarrier(&[transition(texture, state, READ)]);
                }
                let offset = i * self.step as usize;
                host.device.CreateShaderResourceView(
                    texture,
                    Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: draw.view,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MipLevels: 1,
                                ..Default::default()
                            },
                        },
                    }),
                    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + offset },
                );
                let target = host.rtv(
                    back,
                    Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: draw.target,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        ..Default::default()
                    }),
                );
                let [left, top, right, bottom] = draw.destination;
                list.OMSetRenderTargets(1, Some(&target), false, None);
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: left,
                    TopLeftY: top,
                    Width: right - left,
                    Height: bottom - top,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: left.floor() as i32,
                    top: top.floor() as i32,
                    right: right.ceil() as i32,
                    bottom: bottom.ceil() as i32,
                }]);
                list.SetPipelineState(&draw.pipeline);
                list.SetGraphicsRootDescriptorTable(
                    0,
                    D3D12_GPU_DESCRIPTOR_HANDLE {
                        ptr: gpu + offset as u64,
                    },
                );
                list.SetGraphicsRoot32BitConstants(1, 4, draw.source.as_ptr().cast(), 0);
                list.DrawInstanced(3, 1, 0, 0);
                if state != READ {
                    list.ResourceBarrier(&[transition(texture, READ, state)]);
                }
            }
        }
    }
}

impl Host {
    pub(crate) fn present_planes(&self, planes: &[Shown<'_>], frame: u64) -> Option<bool> {
        let first = planes.first()?;
        let largest = planes.iter().max_by_key(|plane| {
            // SAFETY: a live input texture.
            let desc = unsafe { plane.texture.GetDesc() };
            desc.Width * u64::from(desc.Height)
        })?;
        // SAFETY: a live input texture.
        let desc = unsafe { largest.texture.GetDesc() };
        let size = (desc.Width as u32, desc.Height);
        let format = swap_format(desc.Format)?;
        let mut pass = PASS
            .get_or_init(|| {
                Mutex::new(Pass::new(&self.device)
            .inspect_err(|error| tracing::warn!(%error, "could not create the display-plane pass"))
            .ok())
            })
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(pass) = pass.as_mut()
            && let Some(draws) = pass.prepare(&self.device, format, size, planes)
        {
            return self.show(size, format, frame, |list, back| {
                const TARGET: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_RENDER_TARGET;
                const PRESENT: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PRESENT;
                // SAFETY: our open presentation list and its back buffer.
                unsafe { list.ResourceBarrier(&[transition(back, PRESENT, TARGET)]) };
                pass.record(self, list, back, &draws);
                #[cfg(feature = "diagnostics")]
                {
                    const COPY: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_SOURCE;
                    // SAFETY: the back buffer has been rendered to above.
                    unsafe { list.ResourceBarrier(&[transition(back, TARGET, COPY)]) };
                    let seen = self.present_capture(list, back, frame);
                    // SAFETY: the state was COPY_SOURCE while recording the capture.
                    unsafe { list.ResourceBarrier(&[transition(back, COPY, PRESENT)]) };
                    seen
                }
                #[cfg(not(feature = "diagnostics"))]
                // SAFETY: the back buffer has been rendered to above.
                unsafe {
                    list.ResourceBarrier(&[transition(back, TARGET, PRESENT)])
                };
            });
        }
        drop(pass);
        self.present(first.texture, first.state, frame)
    }
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
#[ignore = "requires a native D3D12 device and DXC"]
fn blends_and_places_display_planes() {
    let host = host().expect("D3D12 device");
    let format = DXGI_FORMAT_R8G8B8A8_UNORM;
    const TARGET: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_RENDER_TARGET;
    const COPY: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let texture = |width, height| {
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: valid texture dimensions, format and initial state.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: width,
                        Height: height,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: format,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    TARGET,
                    None,
                    &mut texture,
                )
                .unwrap();
        }
        texture.unwrap()
    };
    let (base, overlay, output) = (texture(2, 2), texture(4, 4), texture(8, 4));
    let mut pass = Pass::new(&host.device).unwrap();
    let planes = [
        Shown {
            texture: &base,
            state: TARGET.0 as u32,
            source: None,
            destination: None,
        },
        Shown {
            texture: &overlay,
            state: TARGET.0 as u32,
            source: Some([2, 0, 4, 4]),
            destination: Some([0.5, 0.25, 1.0, 0.75]),
        },
    ];
    let draws = pass.prepare(&host.device, format, (8, 4), &planes).unwrap();
    let list = host.new_list().unwrap();
    let rtv = host.rtv(&overlay, None);
    // SAFETY: our open list and render targets in RENDER_TARGET.
    unsafe {
        list.list
            .ClearRenderTargetView(host.rtv(&base, None), &[1.0, 0.0, 0.0, 1.0], None);
        list.list
            .ClearRenderTargetView(rtv, &[0.0, 1.0, 0.0, 1.0], None);
        list.list.ClearRenderTargetView(
            rtv,
            &[0.0, 0.0, 1.0, 0.5],
            Some(&[RECT {
                left: 2,
                top: 0,
                right: 4,
                bottom: 4,
            }]),
        );
    }
    pass.record(host, &list.list, &output, &draws);
    // SAFETY: the pass leaves the output in RENDER_TARGET.
    unsafe {
        list.list
            .ResourceBarrier(&[transition(&output, TARGET, COPY)])
    };
    let (readback, footprint) =
        crate::capture::read_back(&host.device, &list.list, &output).unwrap();
    list.close();
    host.execute(&[&list]);
    // SAFETY: the data has been copied, at the row pitch returned by D3D12.
    unsafe {
        let mut pointer = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut pointer)).unwrap();
        let read = |x: usize, y: usize| {
            let at = footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize + 4 * x;
            pointer.cast::<[u8; 4]>().byte_add(at).read()
        };
        assert_eq!(read(0, 2), [255, 0, 0, 255]);
        assert_eq!(read(7, 0), [255, 0, 0, 255]);
        let blended = read(7, 2);
        assert!(
            blended[0].abs_diff(127) <= 1
                && blended[1] == 0
                && blended[2].abs_diff(128) <= 1
                && blended[3] == 255,
            "{blended:?}"
        );
        readback.Unmap(0, None);
    }
}
