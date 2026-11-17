//! Thin dynamic bindings to the cuda driver api (nvcuda.dll / libcuda.so) and
//! nvrtc. Nothing is linked at build time: no cuda toolkit needed to build,
//! only a driver and the nvrtc library at run time.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::path::{Path, PathBuf};

use libloading::Library;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUdeviceptr = u64;
pub type NvrtcProgram = *mut c_void;

macro_rules! api {
    ($name:ident { $($f:ident : fn($($a:ty),*) -> $r:ty;)* }) => {
        pub struct $name {
            _lib: Library,
            $(pub $f: unsafe extern "C" fn($($a),*) -> $r,)*
        }
        impl $name {
            /// # Safety
            /// The library must export these symbols with these signatures.
            pub unsafe fn from_lib(lib: Library) -> Result<Self, String> {
                $(
                    let $f = *lib
                        .get::<unsafe extern "C" fn($($a),*) -> $r>(concat!(stringify!($f), "\0").as_bytes())
                        .map_err(|e| format!("missing symbol {}: {}", stringify!($f), e))?;
                )*
                Ok(Self { _lib: lib, $($f,)* })
            }
        }
    };
}

api!(Driver {
    cuInit: fn(u32) -> CUresult;
    cuDeviceGet: fn(*mut CUdevice, i32) -> CUresult;
    cuDeviceGetName: fn(*mut i8, i32, CUdevice) -> CUresult;
    cuDeviceGetAttribute: fn(*mut i32, i32, CUdevice) -> CUresult;
    cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice) -> CUresult;
    cuCtxSetCurrent: fn(CUcontext) -> CUresult;
    cuCtxSynchronize: fn() -> CUresult;
    cuModuleLoadData: fn(*mut CUmodule, *const c_void) -> CUresult;
    cuModuleUnload: fn(CUmodule) -> CUresult;
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const i8) -> CUresult;
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize) -> CUresult;
    cuMemFree_v2: fn(CUdeviceptr) -> CUresult;
    cuMemsetD32_v2: fn(CUdeviceptr, u32, usize) -> CUresult;
    cuMemcpyHtoD_v2: fn(CUdeviceptr, *const c_void, usize) -> CUresult;
    cuMemcpyDtoH_v2: fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
    cuMemGetInfo_v2: fn(*mut usize, *mut usize) -> CUresult;
    cuLaunchKernel: fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult;
    cuGetErrorString: fn(CUresult, *mut *const i8) -> CUresult;
});

api!(Nvrtc {
    nvrtcVersion: fn(*mut i32, *mut i32) -> i32;
    nvrtcCreateProgram: fn(*mut NvrtcProgram, *const i8, *const i8, i32, *const *const i8, *const *const i8) -> i32;
    nvrtcDestroyProgram: fn(*mut NvrtcProgram) -> i32;
    nvrtcCompileProgram: fn(NvrtcProgram, i32, *const *const i8) -> i32;
    nvrtcGetProgramLogSize: fn(NvrtcProgram, *mut usize) -> i32;
    nvrtcGetProgramLog: fn(NvrtcProgram, *mut i8) -> i32;
    nvrtcGetPTXSize: fn(NvrtcProgram, *mut usize) -> i32;
    nvrtcGetPTX: fn(NvrtcProgram, *mut i8) -> i32;
    nvrtcGetCUBINSize: fn(NvrtcProgram, *mut usize) -> i32;
    nvrtcGetCUBIN: fn(NvrtcProgram, *mut i8) -> i32;
    nvrtcGetErrorString: fn(i32) -> *const i8;
});

pub const ATTR_SM_COUNT: i32 = 16;
pub const ATTR_CC_MAJOR: i32 = 75;
pub const ATTR_CC_MINOR: i32 = 76;

pub fn load_driver() -> Result<Driver, String> {
    let name = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
    let lib = unsafe { Library::new(name) }.map_err(|e| format!("could not load the nvidia driver ({name}): {e}"))?;
    unsafe { Driver::from_lib(lib) }
}

fn is_nvrtc(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("nvrtc64_") && name.ends_with(".dll") && !name.contains(".alt.")
    } else {
        name.starts_with("libnvrtc.so") && !name.contains("builtins")
    }
}

fn find_in(dir: &Path) -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(is_nvrtc))
        .collect();
    hits.sort();
    hits.pop()
}

/// Search order: $KASUMI_NVRTC, the project's `.cuda` folder (pip-installed
/// nvidia-cuda-nvrtc), next to the executable, $CUDA_PATH/bin, $PATH.
pub fn find_nvrtc() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("KASUMI_NVRTC") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        if let Some(f) = find_in(&p) {
            return Some(f);
        }
    }
    let mut dirs: Vec<PathBuf> = Vec::new();
    let rel = Path::new(".cuda").join("nvidia").join("cuda_nvrtc").join(if cfg!(windows) { "bin" } else { "lib" });
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join(&rel));
    }
    dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join(&rel));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
            dirs.push(d.join("..").join("..").join(&rel));
        }
    }
    if let Ok(cp) = std::env::var("CUDA_PATH") {
        dirs.push(PathBuf::from(&cp).join("bin"));
        dirs.push(PathBuf::from(&cp).join("lib64"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    if !cfg!(windows) {
        dirs.push("/usr/local/cuda/lib64".into());
    }
    dirs.iter().find_map(|d| find_in(d))
}

pub fn load_nvrtc() -> Result<(Nvrtc, PathBuf), String> {
    let path = find_nvrtc().ok_or_else(|| {
        "nvrtc not found. install it with:\n  python -m pip install --target .cuda nvidia-cuda-nvrtc-cu12\nor set KASUMI_NVRTC to the nvrtc library".to_string()
    })?;
    // nvrtc loads its builtins library lazily by name; make its folder searchable
    if let Some(dir) = path.parent() {
        let old = std::env::var_os("PATH").unwrap_or_default();
        let mut parts = vec![dir.to_path_buf()];
        parts.extend(std::env::split_paths(&old));
        if let Ok(joined) = std::env::join_paths(parts) {
            std::env::set_var("PATH", joined);
        }
    }
    let lib = unsafe { Library::new(&path) }.map_err(|e| format!("could not load {}: {e}", path.display()))?;
    let api = unsafe { Nvrtc::from_lib(lib)? };
    Ok((api, path))
}
