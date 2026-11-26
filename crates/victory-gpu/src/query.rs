//! Query heaps. An occlusion heap measures on the host GPU (the draws that
//! mark it run there), so the title's own culling and dynamic resolution
//! read real counts; the other heap types are still answered at resolve
//! time.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE_OCCLUSION, D3D12_QUERY_TYPE,
    D3D12_QUERY_TYPE_BINARY_OCCLUSION, D3D12_QUERY_TYPE_OCCLUSION, D3D12_RANGE,
    ID3D12GraphicsCommandList, ID3D12QueryHeap, ID3D12Resource,
};

use crate::host::Host;
use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

pub(crate) const OCCLUSION: u32 = 0;
pub(crate) const BINARY: u32 = 1;
const HEAP_OCCLUSION: u32 = 0;
const CHUNK_SIZE: usize = 256;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    samples: Arc<Samples>,
}

/// Last completed value at a title heap index, updated in submission order.
#[derive(Default)]
struct Samples(Mutex<HashMap<u32, (u32, u64)>>);

/// Device method.
pub(crate) extern "win64" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        samples: Arc::default(),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

fn samples(this: This, first: u32, count: u32) -> Option<Arc<Samples>> {
    // SAFETY: the title passes a heap it created here, or null.
    let heap = unsafe { this.cast::<Heap>().as_ref() }?;
    (heap.desc.kind == HEAP_OCCLUSION && first.checked_add(count)? <= heap.desc.count)
        .then(|| heap.samples.clone())
}

fn host_type(kind: u32) -> D3D12_QUERY_TYPE {
    if kind == BINARY {
        D3D12_QUERY_TYPE_BINARY_OCCLUSION
    } else {
        D3D12_QUERY_TYPE_OCCLUSION
    }
}

/// A host query slot for one interval with an unchanged set of title queries.
#[derive(Clone, Copy)]
struct Slot {
    kind: u32,
    number: usize,
}

struct OpenQuery {
    kind: u32,
    first: usize,
    missed: u64,
}

struct Chunk {
    heap: ID3D12QueryHeap,
    readback: ID3D12Resource,
    values: [u64; CHUNK_SIZE],
    available: bool,
}

impl Chunk {
    fn new(host: &Host) -> Option<Self> {
        let mut heap = None;
        // SAFETY: plain object creation.
        unsafe {
            host.device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                        Count: CHUNK_SIZE as u32,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
        }
        Some(Self {
            heap: heap?,
            readback: host.readback_buffer((CHUNK_SIZE * 8) as u64)?,
            values: [0; CHUNK_SIZE],
            available: false,
        })
    }

    fn read(&mut self, count: usize) {
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: count * 8,
        };
        // SAFETY: this chunk's resolve completed and the read range fits.
        let mapped = unsafe { self.readback.Map(0, Some(&range), Some(&mut data)) };
        self.available = mapped.is_ok() && !data.is_null();
        if self.available {
            // SAFETY: Map supplied the pointer to the checked range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.cast::<u8>(),
                    self.values.as_mut_ptr().cast(),
                    count * 8,
                );
            }
        }
        if mapped.is_ok() {
            // SAFETY: a successful map has exactly one unmap.
            unsafe { self.readback.Unmap(0, None) };
        }
    }
}

enum Event {
    Begin(Arc<Samples>, u32),
    End(Arc<Samples>, u32, u32, Range<usize>),
    Resolve(Option<Arc<Samples>>, u32, u32, u32, usize),
}

/// Real GPU queries always begin, end and resolve inside the recording that
/// measured them. The title's resolves observe those samples in execution
/// order, including when it records the resolve on a different thread/list.
#[derive(Default)]
pub(crate) struct Recording {
    open: HashMap<(usize, u32), OpenQuery>,
    precise_queries: usize,
    active: Option<Slot>,
    slots: Vec<Slot>,
    events: Vec<Event>,
    chunks: [Vec<Chunk>; 2],
    used: [usize; 2],
    missed: u64,
}

impl Recording {
    pub(crate) fn mark(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        heap: This,
        kind: u32,
        index: u32,
        begin: bool,
    ) {
        if !matches!(kind, OCCLUSION | BINARY) {
            return;
        }
        let Some(samples) = samples(heap, index, 1) else {
            return;
        };
        let key = (Arc::as_ptr(&samples) as usize, index);
        if begin {
            if self.open.contains_key(&key) {
                return;
            }
            self.end_segment(list);
            self.events.push(Event::Begin(samples, index));
            self.open.insert(
                key,
                OpenQuery {
                    kind,
                    first: self.slots.len(),
                    missed: self.missed,
                },
            );
            self.precise_queries += usize::from(kind == OCCLUSION);
            self.begin_segment(list);
        } else if self.open.get(&key).is_some_and(|slot| slot.kind == kind) {
            let query = self.open.remove(&key).unwrap();
            self.end_segment(list);
            if self.missed == query.missed {
                self.events.push(Event::End(
                    samples,
                    index,
                    kind,
                    query.first..self.slots.len(),
                ));
            }
            self.precise_queries -= usize::from(kind == OCCLUSION);
            self.begin_segment(list);
        }
    }

    fn begin_segment(&mut self, list: &ID3D12GraphicsCommandList) {
        if self.open.is_empty() {
            return;
        }
        // Vulkan allows one active occlusion query. A count also answers
        // any binary queries open over the same interval.
        let kind = if self.precise_queries != 0 {
            OCCLUSION
        } else {
            BINARY
        };
        let used = &mut self.used[kind as usize];
        let chunks = &mut self.chunks[kind as usize];
        let chunk = *used / CHUNK_SIZE;
        if chunk == chunks.len() {
            let Some(new) = crate::host::host().and_then(Chunk::new) else {
                self.missed += 1;
                return;
            };
            chunks.push(new);
        }
        let slot = Slot {
            kind,
            number: *used,
        };
        // SAFETY: the previous segment ended and this list owns the new slot.
        unsafe {
            list.BeginQuery(
                &chunks[chunk].heap,
                host_type(kind),
                (*used % CHUNK_SIZE) as u32,
            )
        };
        *used += 1;
        self.slots.push(slot);
        self.active = Some(slot);
    }

    fn end_segment(&mut self, list: &ID3D12GraphicsCommandList) {
        let Some(slot) = self.active.take() else {
            return;
        };
        let chunk = &self.chunks[slot.kind as usize][slot.number / CHUNK_SIZE];
        // SAFETY: the host begin for this slot was recorded on this list.
        unsafe {
            list.EndQuery(
                &chunk.heap,
                host_type(slot.kind),
                (slot.number % CHUNK_SIZE) as u32,
            )
        };
    }

    pub(crate) fn missed_draw(&mut self) {
        self.missed += 1;
    }

    pub(crate) fn resolve(&mut self, heap: This, kind: u32, first: u32, count: u32, dst: *mut u8) {
        self.events.push(Event::Resolve(
            samples(heap, first, count),
            kind,
            first,
            count,
            dst as usize,
        ));
    }

    pub(crate) fn close(&mut self, list: &ID3D12GraphicsCommandList) {
        // Complete any last segment for queries the title left open.
        // No End event means the title sees the visible fallback for it.
        self.end_segment(list);
        self.open.clear();
        self.precise_queries = 0;
        for (kind, chunks) in self.chunks.iter().enumerate() {
            let mut remaining = self.used[kind];
            for chunk in chunks.iter().take(remaining.div_ceil(CHUNK_SIZE)) {
                let count = remaining.min(CHUNK_SIZE);
                // SAFETY: all slots in this range ended on this list, and
                // the readback buffer holds the entire range.
                unsafe {
                    list.ResolveQueryData(
                        &chunk.heap,
                        host_type(kind as u32),
                        0,
                        count as u32,
                        &chunk.readback,
                        0,
                    );
                }
                remaining -= count;
            }
        }
    }

    pub(crate) fn reset(&mut self) -> Self {
        Self {
            chunks: std::mem::take(&mut self.chunks),
            ..Self::default()
        }
    }

    pub(crate) fn complete(&mut self, waited: bool) {
        if waited {
            for (kind, chunks) in self.chunks.iter_mut().enumerate() {
                let mut remaining = self.used[kind];
                for chunk in chunks.iter_mut().take(remaining.div_ceil(CHUNK_SIZE)) {
                    let count = remaining.min(CHUNK_SIZE);
                    chunk.read(count);
                    remaining -= count;
                }
            }
        }

        let mut measured = 0u64;
        let mut fallback = 0u64;
        for event in &self.events {
            match event {
                Event::Begin(samples, index) => {
                    samples
                        .0
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(index);
                }
                Event::End(samples, index, kind, slots) if waited => {
                    let measured =
                        self.slots[slots.clone()]
                            .iter()
                            .try_fold(0u64, |total, slot| {
                                let chunk =
                                    &self.chunks[slot.kind as usize][slot.number / CHUNK_SIZE];
                                chunk.available.then(|| {
                                    total.saturating_add(chunk.values[slot.number % CHUNK_SIZE])
                                })
                            });
                    if let Some(value) = measured {
                        let value = if *kind == BINARY {
                            u64::from(value != 0)
                        } else {
                            value
                        };
                        samples
                            .0
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .insert(*index, (*kind, value));
                    }
                }
                Event::End(..) => {}
                Event::Resolve(samples, kind, first, count, dst) => {
                    let values = samples
                        .as_ref()
                        .map(|s| s.0.lock().unwrap_or_else(|e| e.into_inner()));
                    for i in 0..*count {
                        let value = values
                            .as_ref()
                            .and_then(|values| values.get(&(first + i)))
                            .filter(|(query_kind, _)| query_kind == kind);
                        let value = match value {
                            Some((_, value)) => {
                                measured += 1;
                                *value
                            }
                            None => {
                                fallback += 1;
                                if *kind == BINARY { 1 } else { 1 << 20 }
                            }
                        };
                        Written::Value(value).write((dst + i as usize * 8) as *mut u8, 8);
                    }
                }
            }
        }
        if measured + fallback != 0 {
            tracing::debug!(measured, fallback, "occlusion query results");
        }
    }
}

extern "win64" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "win64" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// An occlusion heap made the way the device makes one, for the tests:
/// `None` when creation fails.
#[cfg(test)]
pub(crate) fn occlusion_for_test(kind: u32, count: u32) -> Option<This> {
    let desc = Desc {
        kind,
        count,
        nodes: 0,
    };
    let mut out: This = std::ptr::null_mut();
    let hr = create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut out);
    (hr == object::S_OK && !out.is_null()).then_some(out)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};
