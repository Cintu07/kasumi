use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use crate::ffi::*;

pub struct Gpu {
    pub drv: Arc<Driver>,
    pub rtc: Nvrtc,
    pub nvrtc_path: PathBuf,
    pub nvrtc_version: (i32, i32),
    pub ctx: CUcontext,
    pub name: String,
    pub sm_count: u32,
    pub cc: (i32, i32),
}

pub fn check(drv: &Driver, r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut p: *const i8 = std::ptr::null();
    unsafe { (drv.cuGetErrorString)(r, &mut p) };
    let msg = if p.is_null() { "unknown".to_string() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    Err(anyhow!("cuda error {r} in {what}: {msg}"))
}

/// Device memory. Freed on drop.
pub struct Buf {
    pub ptr: CUdeviceptr,
    pub bytes: usize,
    drv: Arc<Driver>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { (self.drv.cuMemFree_v2)(self.ptr) };
        }
    }
}

pub struct Module {
    pub raw: CUmodule,
    drv: Arc<Driver>,
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { (self.drv.cuModuleUnload)(self.raw) };
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Func(pub CUfunction);

pub struct Compiled {
    pub image: Vec<u8>,
    pub log: String,
    pub ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompileOpts {
    pub fast_math: bool,
    /// no fused multiply-add contraction: bit-for-bit closer to the cpu
    pub strict: bool,
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        let drv = load_driver().map_err(|e| anyhow!(e))?;
        let (rtc, nvrtc_path) = load_nvrtc().map_err(|e| anyhow!(e))?;
        unsafe {
            check(&drv, (drv.cuInit)(0), "cuInit")?;
            let mut dev = 0;
            check(&drv, (drv.cuDeviceGet)(&mut dev, 0), "cuDeviceGet")?;
            let mut name = [0i8; 256];
            check(&drv, (drv.cuDeviceGetName)(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let attr = |a: i32| -> Result<i32> {
                let mut v = 0;
                check(&drv, (drv.cuDeviceGetAttribute)(&mut v, a, dev), "cuDeviceGetAttribute")?;
                Ok(v)
            };
            let sm_count = attr(ATTR_SM_COUNT)? as u32;
            let cc = (attr(ATTR_CC_MAJOR)?, attr(ATTR_CC_MINOR)?);
            let mut ctx = std::ptr::null_mut();
            check(&drv, (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            check(&drv, (drv.cuCtxSetCurrent)(ctx), "cuCtxSetCurrent")?;
            let (mut a, mut b) = (0, 0);
            (rtc.nvrtcVersion)(&mut a, &mut b);
            Ok(Gpu { drv: Arc::new(drv), rtc, nvrtc_path, nvrtc_version: (a, b), ctx, name, sm_count, cc })
        }
    }

    fn rtc_err(&self, r: i32) -> String {
        let p = unsafe { (self.rtc.nvrtcGetErrorString)(r) };
        if p.is_null() {
            format!("nvrtc error {r}")
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }

    fn arch(&self, real: bool) -> String {
        let (a, b) = self.cc;
        if real {
            format!("--gpu-architecture=sm_{a}{b}")
        } else {
            format!("--gpu-architecture=compute_{a}{b}")
        }
    }

    /// Compile cuda c to a cubin (or ptx text when `ptx` is set).
    pub fn compile_with(&self, src: &str, opts: CompileOpts, ptx: bool) -> Result<Compiled> {
        let t0 = Instant::now();
        let csrc = CString::new(src)?;
        let name = CString::new("kasumi.cu")?;
        let mut prog: NvrtcProgram = std::ptr::null_mut();
        unsafe {
            let r = (self.rtc.nvrtcCreateProgram)(&mut prog, csrc.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null());
            if r != 0 {
                bail!("nvrtcCreateProgram: {}", self.rtc_err(r));
            }
        }
        let mut args = vec![self.arch(!ptx), "--std=c++17".to_string(), "-default-device".to_string()];
        if opts.fast_math {
            args.push("--use_fast_math".into());
        }
        if opts.strict {
            args.push("--fmad=false".into());
        }
        let cargs: Vec<CString> = args.iter().map(|a| CString::new(a.as_str()).unwrap()).collect();
        let ptrs: Vec<*const i8> = cargs.iter().map(|c| c.as_ptr()).collect();
        let res = unsafe { (self.rtc.nvrtcCompileProgram)(prog, ptrs.len() as i32, ptrs.as_ptr()) };
        let log = unsafe {
            let mut n = 0usize;
            (self.rtc.nvrtcGetProgramLogSize)(prog, &mut n);
            let mut buf = vec![0i8; n.max(1)];
            (self.rtc.nvrtcGetProgramLog)(prog, buf.as_mut_ptr());
            CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        };
        if res != 0 {
            unsafe { (self.rtc.nvrtcDestroyProgram)(&mut prog) };
            bail!("nvrtc: {}\n{}", self.rtc_err(res), log.trim());
        }
        let image = unsafe {
            let mut n = 0usize;
            if ptx {
                (self.rtc.nvrtcGetPTXSize)(prog, &mut n);
                let mut buf = vec![0u8; n];
                (self.rtc.nvrtcGetPTX)(prog, buf.as_mut_ptr() as *mut i8);
                buf
            } else {
                (self.rtc.nvrtcGetCUBINSize)(prog, &mut n);
                let mut buf = vec![0u8; n];
                (self.rtc.nvrtcGetCUBIN)(prog, buf.as_mut_ptr() as *mut i8);
                buf
            }
        };
        unsafe { (self.rtc.nvrtcDestroyProgram)(&mut prog) };
        Ok(Compiled { image, log, ms: t0.elapsed().as_secs_f64() * 1e3 })
    }

    pub fn load(&self, image: &[u8]) -> Result<Module> {
        let mut m = std::ptr::null_mut();
        unsafe { check(&self.drv, (self.drv.cuModuleLoadData)(&mut m, image.as_ptr() as *const c_void), "cuModuleLoadData")? };
        Ok(Module { raw: m, drv: self.drv.clone() })
    }

    pub fn func(&self, m: &Module, name: &str) -> Result<Func> {
        let c = CString::new(name)?;
        let mut f = std::ptr::null_mut();
        unsafe { check(&self.drv, (self.drv.cuModuleGetFunction)(&mut f, m.raw, c.as_ptr()), name)? };
        Ok(Func(f))
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let mut p = 0;
        unsafe { check(&self.drv, (self.drv.cuMemAlloc_v2)(&mut p, bytes.max(4)), "cuMemAlloc")? };
        Ok(Buf { ptr: p, bytes: bytes.max(4), drv: self.drv.clone() })
    }

    pub fn zero(&self, b: &Buf) -> Result<()> {
        self.fill32(b.ptr, 0, b.bytes / 4)
    }

    pub fn fill32(&self, ptr: CUdeviceptr, v: u32, count: usize) -> Result<()> {
        unsafe { check(&self.drv, (self.drv.cuMemsetD32_v2)(ptr, v, count), "cuMemsetD32") }
    }

    pub fn upload<T: Copy>(&self, b: &Buf, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= b.bytes, "upload overflows buffer");
        unsafe { check(&self.drv, (self.drv.cuMemcpyHtoD_v2)(b.ptr, data.as_ptr() as *const c_void, bytes), "cuMemcpyHtoD") }
    }

    pub fn download<T: Copy>(&self, b: &Buf, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        assert!(bytes <= b.bytes, "download overflows buffer");
        unsafe { check(&self.drv, (self.drv.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, b.ptr, bytes), "cuMemcpyDtoH") }
    }

    /// Launch a kernel whose single parameter is the repr(C) struct `p`.
    pub fn launch<P>(&self, f: Func, grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32, p: &P) -> Result<()> {
        // kernelParams[k] points at the k-th argument's value
        let mut args = [p as *const P as *mut c_void];
        unsafe {
            check(
                &self.drv,
                (self.drv.cuLaunchKernel)(f.0, grid.0, grid.1, grid.2, block.0, block.1, block.2, smem, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()),
                "cuLaunchKernel",
            )
        }
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { check(&self.drv, (self.drv.cuCtxSynchronize)(), "cuCtxSynchronize") }
    }

    pub fn mem_info(&self) -> (usize, usize) {
        let (mut free, mut total) = (0, 0);
        unsafe { (self.drv.cuMemGetInfo_v2)(&mut free, &mut total) };
        (free, total)
    }
}
