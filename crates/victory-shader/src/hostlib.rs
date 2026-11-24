//! Host libraries on Linux, loaded at run time: the host D3D12
//! (vkd3d-proton built for Linux: `libvkd3d-proton-d3d12.so` and the
//! `libvkd3d-proton-d3d12core.so` it opens) and anything else we only
//! call through its C API. Their COM methods are SysV, which is what
//! `extern "system"` (and so the `windows` crate's vtables) means here.
//!
//! `VICTORY_VKD3D` names the directory holding vkd3d-proton's libraries,
//! or `libvkd3d-proton-d3d12.so` itself; unset, the dynamic loader's
//! search path is used (`docs/linux.md`, GPU).

use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const D3D12: &str = "libvkd3d-proton-d3d12.so";
const CORE: &str = "libvkd3d-proton-d3d12core.so";

/// A library opened for good (never closed).
#[derive(Clone, Copy)]
pub struct Library(usize);

impl Library {
    /// `path` (a file, or a name the dynamic loader searches for).
    pub fn open(path: &Path) -> Result<Library, String> {
        let name = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| format!("{}: NUL in path", path.display()))?;
        // SAFETY: a NUL-terminated path.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(last_error());
        }
        Ok(Library(handle as usize))
    }

    /// The address of export `name`.
    pub fn symbol(self, name: &CStr) -> Result<*mut c_void, String> {
        // SAFETY: an open library and a NUL-terminated name.
        let address = unsafe { libc::dlsym(self.0 as *mut c_void, name.as_ptr()) };
        if address.is_null() {
            return Err(format!("no {}", name.to_string_lossy()));
        }
        Ok(address)
    }
}

fn last_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated message.
    let text = unsafe { libc::dlerror() };
    if text.is_null() {
        return "not found".into();
    }
    // SAFETY: as above.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}

/// vkd3d-proton's `libvkd3d-proton-d3d12.so`, opened once.
pub fn d3d12() -> Result<Library, &'static str> {
    static LIBRARY: OnceLock<Result<Library, String>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let (core, d3d12) = match std::env::var_os("VICTORY_VKD3D").map(PathBuf::from) {
                Some(p) if p.is_dir() => (p.join(CORE), p.join(D3D12)),
                Some(p) => (p.with_file_name(CORE), p),
                None => (CORE.into(), D3D12.into()),
            };
            // The d3d12 library opens the core by its soname, which an
            // installed copy (no run path) only finds already loaded.
            let opened = Library::open(&core).and_then(|_| Library::open(&d3d12));
            opened.map_err(|e| {
                format!(
                    "no host D3D12: cannot load vkd3d-proton ({e}); build it for Linux and \
                     set VICTORY_VKD3D to its library directory (docs/linux.md, GPU)"
                )
            })
        })
        .as_ref()
        .copied()
        .map_err(String::as_str)
}

/// The address of `name` in the host D3D12.
pub fn d3d12_export(name: &CStr) -> Result<*mut c_void, String> {
    d3d12()?.symbol(name)
}

/// `D3D12CreateDevice` on the default adapter: its HRESULT, or why the
/// host D3D12 is missing.
///
/// # Safety
///
/// `iid` and `device` as `D3D12CreateDevice` takes them.
pub unsafe fn create_device(
    level: i32,
    iid: *const c_void,
    device: *mut *mut c_void,
) -> Result<i32, String> {
    type Create =
        unsafe extern "system" fn(*mut c_void, i32, *const c_void, *mut *mut c_void) -> i32;
    let f = d3d12_export(c"D3D12CreateDevice")?;
    // SAFETY: the export has this signature; the caller's pointers.
    Ok(unsafe {
        std::mem::transmute::<*mut c_void, Create>(f)(std::ptr::null_mut(), level, iid, device)
    })
}

/// `D3D12SerializeVersionedRootSignature`: its HRESULT, or why the host
/// D3D12 is missing.
///
/// # Safety
///
/// The arguments as `D3D12SerializeVersionedRootSignature` takes them.
pub unsafe fn serialize_versioned_root_signature(
    desc: *const c_void,
    blob: *mut *mut c_void,
    error: *mut *mut c_void,
) -> Result<i32, String> {
    type Serialize =
        unsafe extern "system" fn(*const c_void, *mut *mut c_void, *mut *mut c_void) -> i32;
    let f = d3d12_export(c"D3D12SerializeVersionedRootSignature")?;
    // SAFETY: the export has this signature; the caller's pointers.
    Ok(unsafe { std::mem::transmute::<*mut c_void, Serialize>(f)(desc, blob, error) })
}
