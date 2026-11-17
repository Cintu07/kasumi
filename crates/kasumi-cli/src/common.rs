use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use kasumi_gpu::{CompileOpts, FormModule, Gpu, Look, Program};
use kasumi_lang::{lower, parse, Form};

/// `--key value` flags plus positional arguments.
pub struct Args {
    pub pos: Vec<String>,
    pub flags: HashMap<String, String>,
}

impl Args {
    pub fn parse(raw: &[String]) -> Args {
        let mut pos = Vec::new();
        let mut flags = HashMap::new();
        let mut k = 0;
        while k < raw.len() {
            let a = &raw[k];
            if let Some(key) = a.strip_prefix("--") {
                if let Some((kk, v)) = key.split_once('=') {
                    flags.insert(kk.to_string(), v.to_string());
                } else if k + 1 < raw.len() && !raw[k + 1].starts_with("--") {
                    flags.insert(key.to_string(), raw[k + 1].clone());
                    k += 1;
                } else {
                    flags.insert(key.to_string(), "true".to_string());
                }
            } else {
                pos.push(a.clone());
            }
            k += 1;
        }
        Args { pos, flags }
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(|s| s.as_str())
    }
    pub fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    pub fn f32_or(&self, k: &str, d: f32) -> Result<f32> {
        match self.get(k) {
            Some(v) => v.parse().map_err(|_| anyhow!("--{k}: expected a number, got '{v}'")),
            None => Ok(d),
        }
    }
    pub fn u32_or(&self, k: &str, d: u32) -> Result<u32> {
        match self.get(k) {
            Some(v) => Ok(parse_count(v)? as u32),
            None => Ok(d),
        }
    }
    pub fn count_or(&self, k: &str, d: u64) -> Result<u64> {
        match self.get(k) {
            Some(v) => parse_count(v),
            None => Ok(d),
        }
    }
}

/// `20m`, `1.5b`, `500k`, `12345`
pub fn parse_count(s: &str) -> Result<u64> {
    match kasumi_lang::parser::parse_set_value(s) {
        Some(kasumi_lang::SetVal::Num(v)) if v >= 0.0 => Ok(v as u64),
        _ => bail!("expected a count like 20m, got '{s}'"),
    }
}

/// `4k`, `1080p`, `720p`, `1920x1080`
pub fn parse_res(s: &str) -> Result<(u32, u32)> {
    Ok(match s {
        "8k" => (7680, 4320),
        "4k" | "2160p" => (3840, 2160),
        "1440p" => (2560, 1440),
        "1080p" => (1920, 1080),
        "720p" => (1280, 720),
        "square" => (1080, 1080),
        "vertical" | "reel" => (1080, 1920),
        _ => {
            let (a, b) = s.split_once('x').ok_or_else(|| anyhow!("resolution like 1920x1080 or 4k, got '{s}'"))?;
            (a.parse()?, b.parse()?)
        }
    })
}

pub struct Loaded {
    pub path: PathBuf,
    pub form: Form,
    pub prog: Program,
    pub look: Look,
}

pub fn load_form(path: &Path) -> Result<Loaded> {
    let src = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let form = parse(&src).map_err(|e| anyhow!("{}\n{}", path.display(), e.render(&src)))?;
    let ir = lower(&form);
    let prog = Program::new(&form, ir);
    let look = Look::from_form(&form);
    Ok(Loaded { path: path.to_path_buf(), form, prog, look })
}

pub fn opts_for(form: &Form, force_fast: bool) -> CompileOpts {
    CompileOpts { fast_math: force_fast || form.flag("fastmath"), strict: false }
}

pub fn build_one(gpu: &Gpu, l: &Loaded, fast: bool) -> Result<FormModule> {
    FormModule::build(gpu, &[&l.prog.ir], opts_for(&l.form, fast))
}

pub fn write_png_rgb8(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).ok();
    }
    let f = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header()?;
    wr.write_image_data(rgb)?;
    Ok(())
}

pub fn u32_to_rgb8(px: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(px.len() * 3);
    for &p in px {
        v.push((p >> 16) as u8);
        v.push((p >> 8) as u8);
        v.push(p as u8);
    }
    v
}

pub fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("form").to_string()
}

pub fn gpts(points: u64, ms: f64) -> f64 {
    points as f64 / (ms.max(1e-6) * 1e-3) / 1e9
}

pub fn human(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}b", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}m", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

pub fn find_ffmpeg() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "ffmpeg.exe" } else { "ffmpeg" };
    if let Ok(p) = std::env::var("KASUMI_FFMPEG") {
        return Some(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(exe)).find(|p| p.is_file())) {
        return Some(p);
    }
    // winget installs land here before a new shell picks up PATH
    let base = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("Microsoft").join("WinGet");
    let link = base.join("Links").join(exe);
    if link.is_file() {
        return Some(link);
    }
    let pkgs = std::fs::read_dir(base.join("Packages")).ok()?;
    for pkg in pkgs.flatten().filter(|e| e.file_name().to_string_lossy().contains("FFmpeg")) {
        for build in std::fs::read_dir(pkg.path()).ok()?.flatten() {
            let cand = build.path().join("bin").join(exe);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

pub fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("{secs}")
}
