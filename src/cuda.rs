//! The CUDA driver and NVRTC, loaded at run time, so building kasumi needs no
//! CUDA toolkit.

#![allow(non_snake_case)]

use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;

type CUresult = i32;
type Handle = *mut c_void;
type DevicePtr = u64;

macro_rules! api {
    ($name:ident { $($f:ident : fn($($a:ty),*) -> $r:ty;)* }) => {
        pub struct $name {
            _lib: Library,
            $($f: unsafe extern "C" fn($($a),*) -> $r,)*
        }
        impl $name {
            unsafe fn load(lib: Library) -> Result<Self> {
                $(let $f = *lib.get::<unsafe extern "C" fn($($a),*) -> $r>(concat!(stringify!($f), "\0").as_bytes())?;)*
                Ok(Self { _lib: lib, $($f,)* })
            }
        }
    };
}

api!(Driver {
    cuInit: fn(u32) -> CUresult;
    cuDeviceGet: fn(*mut i32, i32) -> CUresult;
    cuDeviceGetName: fn(*mut i8, i32, i32) -> CUresult;
    cuDeviceGetAttribute: fn(*mut i32, i32, i32) -> CUresult;
    cuDevicePrimaryCtxRetain: fn(*mut Handle, i32) -> CUresult;
    cuCtxSetCurrent: fn(Handle) -> CUresult;
    cuCtxSynchronize: fn() -> CUresult;
    cuModuleLoadData: fn(*mut Handle, *const c_void) -> CUresult;
    cuModuleUnload: fn(Handle) -> CUresult;
    cuModuleGetFunction: fn(*mut Handle, Handle, *const i8) -> CUresult;
    cuMemAlloc_v2: fn(*mut DevicePtr, usize) -> CUresult;
    cuMemFree_v2: fn(DevicePtr) -> CUresult;
    cuMemsetD32_v2: fn(DevicePtr, u32, usize) -> CUresult;
    cuMemcpyHtoD_v2: fn(DevicePtr, *const c_void, usize) -> CUresult;
    cuMemcpyDtoH_v2: fn(*mut c_void, DevicePtr, usize) -> CUresult;
    cuMemGetInfo_v2: fn(*mut usize, *mut usize) -> CUresult;
    cuLaunchKernel: fn(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut Handle) -> CUresult;
    cuGetErrorString: fn(CUresult, *mut *const i8) -> CUresult;
});

api!(Nvrtc {
    nvrtcCreateProgram: fn(*mut Handle, *const i8, *const i8, i32, *const *const i8, *const *const i8) -> i32;
    nvrtcDestroyProgram: fn(*mut Handle) -> i32;
    nvrtcCompileProgram: fn(Handle, i32, *const *const i8) -> i32;
    nvrtcGetProgramLogSize: fn(Handle, *mut usize) -> i32;
    nvrtcGetProgramLog: fn(Handle, *mut i8) -> i32;
    nvrtcGetPTXSize: fn(Handle, *mut usize) -> i32;
    nvrtcGetPTX: fn(Handle, *mut i8) -> i32;
    nvrtcGetErrorString: fn(i32) -> *const i8;
});

const ATTR_SM_COUNT: i32 = 16;
const ATTR_CC_MAJOR: i32 = 75;
const ATTR_CC_MINOR: i32 = 76;

fn is_nvrtc(name: &str) -> bool {
    if cfg!(windows) {
        name.starts_with("nvrtc64_") && name.ends_with(".dll") && !name.contains(".alt.")
    } else {
        name.starts_with("libnvrtc.so") && !name.contains("builtins")
    }
}

fn nvrtc_in(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path());
    entries.filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(is_nvrtc)).max()
}

/// Looks in $KASUMI_NVRTC, the project's `.cuda` folder, $CUDA_PATH and $PATH.
fn find_nvrtc() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("KASUMI_NVRTC").map(PathBuf::from) {
        return if p.is_file() { Some(p) } else { nvrtc_in(&p) };
    }
    let local = Path::new(".cuda/nvidia/cuda_nvrtc").join(if cfg!(windows) { "bin" } else { "lib" });
    let mut dirs = vec![local.clone()];
    if let Some(exe_dir) = std::env::current_exe().ok().as_deref().and_then(Path::parent) {
        dirs.push(exe_dir.join("../..").join(&local));
    }
    if let Some(cuda) = std::env::var_os("CUDA_PATH").map(PathBuf::from) {
        dirs.extend([cuda.join("bin"), cuda.join("lib64")]);
    }
    dirs.extend(std::env::var_os("PATH").iter().flat_map(std::env::split_paths));
    dirs.iter().find_map(|d| nvrtc_in(d))
}

fn load_nvrtc() -> Result<(Nvrtc, PathBuf)> {
    let path = find_nvrtc().context("nvrtc not found. install it with:\n  python -m pip install --target .cuda nvidia-cuda-nvrtc-cu12")?;
    // nvrtc loads its builtins library by name, so its folder must be searchable
    let dir = path.parent().context("nvrtc path has no folder")?.to_path_buf();
    let old = std::env::var_os("PATH").unwrap_or_default();
    std::env::set_var("PATH", std::env::join_paths(std::iter::once(dir).chain(std::env::split_paths(&old)))?);
    let api = unsafe { Nvrtc::load(Library::new(&path)?)? };
    Ok((api, path))
}

/// Device memory, freed on drop.
pub struct Buf {
    pub ptr: DevicePtr,
    bytes: usize,
    driver: Arc<Driver>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { (self.driver.cuMemFree_v2)(self.ptr) };
    }
}

pub struct Module {
    raw: Handle,
    driver: Arc<Driver>,
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { (self.driver.cuModuleUnload)(self.raw) };
    }
}

#[derive(Clone, Copy)]
pub struct Kernel(Handle);

pub struct Gpu {
    driver: Arc<Driver>,
    nvrtc: Nvrtc,
    pub nvrtc_path: PathBuf,
    pub name: String,
    pub sm_count: u32,
    compute_capability: (i32, i32),
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let library = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };
        let driver = unsafe { Driver::load(Library::new(library).context("could not load the nvidia driver")?)? };
        let (nvrtc, nvrtc_path) = load_nvrtc()?;
        let ok = |r: CUresult, what: &str| check(&driver, r, what);
        unsafe {
            ok((driver.cuInit)(0), "cuInit")?;
            let mut device = 0;
            ok((driver.cuDeviceGet)(&mut device, 0), "cuDeviceGet")?;
            let mut name = [0i8; 256];
            ok((driver.cuDeviceGetName)(name.as_mut_ptr(), 256, device), "cuDeviceGetName")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let attribute = |a: i32| -> Result<i32> {
                let mut v = 0;
                ok((driver.cuDeviceGetAttribute)(&mut v, a, device), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            let sm_count = attribute(ATTR_SM_COUNT)? as u32;
            let compute_capability = (attribute(ATTR_CC_MAJOR)?, attribute(ATTR_CC_MINOR)?);
            let mut context = std::ptr::null_mut();
            ok((driver.cuDevicePrimaryCtxRetain)(&mut context, device), "cuDevicePrimaryCtxRetain")?;
            ok((driver.cuCtxSetCurrent)(context), "cuCtxSetCurrent")?;
            Ok(Gpu { driver: Arc::new(driver), nvrtc, nvrtc_path, name, sm_count, compute_capability })
        }
    }

    fn ok(&self, r: CUresult, what: &str) -> Result<()> {
        check(&self.driver, r, what)
    }

    fn nvrtc_error(&self, r: i32) -> String {
        unsafe { CStr::from_ptr((self.nvrtc.nvrtcGetErrorString)(r)) }.to_string_lossy().into_owned()
    }

    /// Compile CUDA C to PTX. Going through PTX and the driver JIT measured
    /// 2 to 3 times faster than NVRTC's own cubin path, which hot reload needs.
    pub fn compile(&self, src: &str, fast_math: bool, strict: bool) -> Result<Vec<u8>> {
        let (a, b) = self.compute_capability;
        let mut args = vec![format!("--gpu-architecture=compute_{a}{b}"), "--std=c++17".into(), "-default-device".into()];
        if fast_math {
            args.push("--use_fast_math".into());
        }
        if strict {
            args.push("--fmad=false".into());
        }
        let args: Vec<CString> = args.into_iter().map(CString::new).collect::<Result<_, _>>()?;
        let arg_ptrs: Vec<*const i8> = args.iter().map(|c| c.as_ptr()).collect();
        let (src, file) = (CString::new(src)?, CString::new("kasumi.cu")?);
        let api = &self.nvrtc;
        unsafe {
            let mut program = std::ptr::null_mut();
            let r = (api.nvrtcCreateProgram)(&mut program, src.as_ptr(), file.as_ptr(), 0, std::ptr::null(), std::ptr::null());
            if r != 0 {
                bail!("nvrtc: {}", self.nvrtc_error(r));
            }
            let r = (api.nvrtcCompileProgram)(program, arg_ptrs.len() as i32, arg_ptrs.as_ptr());
            if r != 0 {
                let mut n = 0;
                (api.nvrtcGetProgramLogSize)(program, &mut n);
                let mut log = vec![0i8; n.max(1)];
                (api.nvrtcGetProgramLog)(program, log.as_mut_ptr());
                let log = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
                (api.nvrtcDestroyProgram)(&mut program);
                bail!("nvrtc: {}\n{}", self.nvrtc_error(r), log.trim());
            }
            let mut n = 0;
            (api.nvrtcGetPTXSize)(program, &mut n);
            let mut ptx = vec![0u8; n];
            (api.nvrtcGetPTX)(program, ptx.as_mut_ptr() as *mut i8);
            (api.nvrtcDestroyProgram)(&mut program);
            Ok(ptx)
        }
    }

    pub fn load(&self, image: &[u8]) -> Result<Module> {
        let mut raw = std::ptr::null_mut();
        self.ok(unsafe { (self.driver.cuModuleLoadData)(&mut raw, image.as_ptr() as *const c_void) }, "cuModuleLoadData")?;
        Ok(Module { raw, driver: self.driver.clone() })
    }

    pub fn kernel(&self, module: &Module, name: &str) -> Result<Kernel> {
        let c_name = CString::new(name)?;
        let mut f = std::ptr::null_mut();
        self.ok(unsafe { (self.driver.cuModuleGetFunction)(&mut f, module.raw, c_name.as_ptr()) }, name)?;
        Ok(Kernel(f))
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let bytes = bytes.max(4);
        let mut ptr = 0;
        self.ok(unsafe { (self.driver.cuMemAlloc_v2)(&mut ptr, bytes) }, "cuMemAlloc")?;
        Ok(Buf { ptr, bytes, driver: self.driver.clone() })
    }

    pub fn zero(&self, b: &Buf) -> Result<()> {
        self.fill32(b, b.bytes / 4)
    }

    /// Zero the first `count` 32-bit words.
    pub fn fill32(&self, b: &Buf, count: usize) -> Result<()> {
        self.ok(unsafe { (self.driver.cuMemsetD32_v2)(b.ptr, 0, count) }, "cuMemsetD32")
    }

    pub fn upload<T: Copy>(&self, b: &Buf, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= b.bytes, "upload overflows the buffer");
        self.ok(unsafe { (self.driver.cuMemcpyHtoD_v2)(b.ptr, data.as_ptr() as *const c_void, bytes) }, "cuMemcpyHtoD")
    }

    pub fn download<T: Copy>(&self, b: &Buf, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= b.bytes, "download overflows the buffer");
        self.ok(unsafe { (self.driver.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, b.ptr, bytes) }, "cuMemcpyDtoH")
    }

    /// Launch a kernel whose one parameter is the struct `args`.
    pub fn launch<A>(&self, k: Kernel, grid: (u32, u32, u32), block: (u32, u32, u32), shared_bytes: u32, args: &A) -> Result<()> {
        let mut params = [args as *const A as *mut c_void];
        let (stream, extra) = (std::ptr::null_mut(), std::ptr::null_mut());
        let r = unsafe {
            (self.driver.cuLaunchKernel)(k.0, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared_bytes, stream, params.as_mut_ptr(), extra)
        };
        self.ok(r, "cuLaunchKernel")
    }

    pub fn sync(&self) -> Result<()> {
        self.ok(unsafe { (self.driver.cuCtxSynchronize)() }, "cuCtxSynchronize")
    }

    /// Free and total device memory in bytes.
    pub fn memory(&self) -> (usize, usize) {
        let (mut free, mut total) = (0, 0);
        unsafe { (self.driver.cuMemGetInfo_v2)(&mut free, &mut total) };
        (free, total)
    }
}

fn check(driver: &Driver, r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut p = std::ptr::null();
    unsafe { (driver.cuGetErrorString)(r, &mut p) };
    let msg = if p.is_null() { "unknown".into() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy() };
    Err(anyhow!("cuda error {r} in {what}: {msg}"))
}
