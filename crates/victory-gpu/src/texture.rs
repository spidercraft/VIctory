//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
    pending_upload: AtomicBool,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Prepares a title-written texture for the list's first use. The prelude
/// copies refreshed staging into the host texture before that use.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    if let Some(copy) = crate::resource::host_copy(resource) {
        #[cfg(feature = "diagnostics")]
        for (i, footprint) in staging.footprints.iter().enumerate() {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture staging {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        list.defer_texture_upload(&copy, staging);
    }
}

/// A copy footprint may leave the final row's trailing padding outside the
/// allocation returned by GetCopyableFootprints.
fn staging_subresource(data: &mut [u8], offset: u64, padded_bytes: usize) -> Option<&mut [u8]> {
    let offset = usize::try_from(offset).ok()?;
    let end = offset.saturating_add(padded_bytes).min(data.len());
    data.get_mut(offset..end)
}

impl Staging {
    pub(crate) fn needs_upload(&self) -> bool {
        self.pending_upload.load(Relaxed)
    }

    /// Copies the staged subresources into a texture in COPY_DEST.
    pub(crate) fn record_copy(&self, list: &ID3D12GraphicsCommandList, texture: &ID3D12Resource) {
        for (i, footprint) in self.footprints.iter().enumerate() {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
            };
            // SAFETY: an open list and both resources held through execution.
            unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        self.pending_upload.store(false, Relaxed);
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        // SAFETY: the mapped upload buffer is at least this large and is
        // written only by the staging refresh.
        let data = unsafe { std::slice::from_raw_parts_mut(self.data, self.size as usize) };
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let depth_pitch = pitch * rows as usize;
            let bytes = depth_pitch.saturating_mul(f.Depth as usize);
            let Some(linear) = staging_subresource(data, footprint.Offset, bytes) else {
                tracing::warn!(mip, slice, "texture footprint starts beyond its staging buffer");
                return;
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, depth_pitch)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
        self.pending_upload.store(true, Relaxed);
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
#[cfg(not(windows))]
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    crate::host::readable(at, len)
}

#[cfg(windows)]
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
        pending_upload: AtomicBool::new(true),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}

#[cfg(test)]
mod staging_bounds_tests {
    use super::staging_subresource;

    #[test]
    fn final_staging_row_does_not_include_unallocated_padding() {
        let mut data = [0xff; 48];
        let last = staging_subresource(&mut data, 16, 256).unwrap();
        assert_eq!(last.len(), 32);
        last.fill(0);
        assert_eq!(&data[..16], &[0xff; 16]);
        assert_eq!(&data[16..], &[0; 32]);
        assert!(staging_subresource(&mut data, 49, 256).is_none());
    }

    #[test]
    #[ignore = "requires a D3D12 device with BC7 support"]
    fn staged_bc7_mips_read_back_without_tail_padding() {
        use super::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_BC7_UNORM_SRGB, DXGI_SAMPLE_DESC,
        };

        const MIPS: usize = 8;
        let host = host().expect("native D3D12 device");
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024,
            Height: 512,
            DepthOrArraySize: 1,
            MipLevels: MIPS as u16,
            Format: DXGI_FORMAT_BC7_UNORM_SRGB,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); MIPS];
        let mut rows = [0; MIPS];
        let mut row_sizes = [0; MIPS];
        let mut bytes = 0;
        // SAFETY: the outputs each have one entry per mip in desc.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc, 0, MIPS as u32, 0, Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()), Some(row_sizes.as_mut_ptr()), Some(&mut bytes),
            );
        }
        let last = footprints[MIPS - 1];
        assert_eq!(bytes - last.Offset, row_sizes[MIPS - 1]);
        assert!(row_sizes[MIPS - 1] < u64::from(last.Footprint.RowPitch));
        // SAFETY: these words are the console's heap and resource descriptors.
        let (heap, title_desc) = unsafe {
            (
                std::mem::transmute::<[u32; 5], crate::resource::HeapProperties>([1, 0, 0, 1, 1]),
                std::mem::transmute::<[u32; 14], crate::resource::ResourceDesc>([
                    3, 0, 0, 0, 1024, 0, 512, 8 << 16 | 1, 99, 1, 0, 0, 0, 0,
                ]),
            )
        };
        let iid = crate::object::guid(
            0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut resource = std::ptr::null();
        assert_eq!(crate::resource::create_committed(
            std::ptr::null(), &heap, 0, &title_desc, 0, std::ptr::null(), &iid, &mut resource,
        ), crate::object::S_OK);
        let texture = crate::resource::host_resource(resource).expect("host texture");
        let staged = staging(resource, texture).expect("texture staging");
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let fill_title = |seed: u8| {
            // SAFETY: the resource owns this allocation and no submission is running.
            let tiled = unsafe { std::slice::from_raw_parts_mut(memory.cast_mut(), staged.layout.size as usize) };
            for (i, footprint) in footprints.iter().enumerate() {
                let pitch = footprint.Footprint.RowPitch as usize;
                let size = pitch * rows[i] as usize;
                let linear = vec![seed + i as u8; size];
                staged.layout.tile(tiled, i as u32, 0, &linear, pitch, size).unwrap();
            }
        };

        let read_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: desc.SampleDesc,
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: a readback heap accepts a buffer in COPY_DEST.
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
                D3D12_HEAP_FLAG_NONE, &read_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback,
            ).unwrap();
        }
        let readback = readback.unwrap();
        let check = |seed: u8| {
            let list = host.new_list().unwrap();
            let second = host.new_list().unwrap();
            for next in [&list, &second] {
                upload(next, resource, texture);
                crate::list::ensure_state(next, resource, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            }
            for (i, footprint) in footprints.iter().enumerate() {
                // SAFETY: the same footprints and live resources are used for readback.
                unsafe {
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
                    };
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&readback),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
                    };
                    list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
            }
            list.close();
            second.close();
            host.execute(&[&list, &second]);
            // SAFETY: execute waited for the copies before mapping the live buffer.
            let copied = unsafe {
                let mut mapped = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut mapped)).unwrap();
                std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize)
            };
            for (i, footprint) in footprints.iter().enumerate() {
                for row in 0..rows[i] as usize {
                    let start = footprint.Offset as usize + row * footprint.Footprint.RowPitch as usize;
                    assert!(copied[start..start + row_sizes[i] as usize].iter().all(|&b| b == seed + i as u8),
                        "unexpected bytes in mip {i}, row {row}");
                }
            }
            // SAFETY: the contents were read while no submission was active.
            unsafe { readback.Unmap(0, None) };
        };

        fill_title(0x11);
        check(0x11);
        // A redundant copy would replace the texture with these sentinel bytes.
        // SAFETY: the staging buffer is mapped, idle and writable.
        unsafe { staged.data.write_bytes(0xff, staged.size as usize) };
        check(0x11);
        fill_title(0x22);
        check(0x22);

        let release = crate::object::slot::<extern "win64" fn(This) -> u32>(resource, 0x10);
        assert_eq!(release(resource), 0);
    }
}
