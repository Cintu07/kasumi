//! Command line flags and small helpers the commands share.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::ast::SetVal;
use crate::parser::parse_set_value;

/// `--key value` flags plus positional arguments.
pub struct Args {
    pub pos: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    pub fn parse(raw: &[String]) -> Args {
        let (mut pos, mut flags) = (Vec::new(), HashMap::new());
        let mut k = 0;
        while k < raw.len() {
            match raw[k].strip_prefix("--") {
                None => pos.push(raw[k].clone()),
                Some(key) => {
                    if let Some((name, value)) = key.split_once('=') {
                        flags.insert(name.to_string(), value.to_string());
                    } else if raw.get(k + 1).is_some_and(|v| !v.starts_with("--")) {
                        k += 1;
                        flags.insert(key.to_string(), raw[k].clone());
                    } else {
                        flags.insert(key.to_string(), "true".to_string());
                    }
                }
            }
            k += 1;
        }
        Args { pos, flags }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.flags.get(key).map(|s| s.as_str())
    }

    pub fn f32_or(&self, key: &str, default: f32) -> Result<f32> {
        self.get(key).map_or(Ok(default), |v| v.parse().map_err(|_| anyhow!("--{key}: expected a number, got '{v}'")))
    }

    pub fn count_or(&self, key: &str, default: u64) -> Result<u64> {
        self.get(key).map_or(Ok(default), parse_count)
    }

    pub fn u32_or(&self, key: &str, default: u32) -> Result<u32> {
        Ok(self.count_or(key, default as u64)? as u32)
    }

    /// The path in positional slot `k`, or an error showing the usage.
    pub fn path(&self, k: usize, usage: &str) -> Result<PathBuf> {
        self.pos.get(k).map(PathBuf::from).with_context(|| format!("usage: {usage}"))
    }
}

/// `20m`, `1.5b`, `500k`, `12345`
pub fn parse_count(s: &str) -> Result<u64> {
    match parse_set_value(s) {
        Some(SetVal::Num(v)) if v >= 0.0 => Ok(v as u64),
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
        _ => {
            let (w, h) = s.split_once('x').with_context(|| format!("resolution like 1920x1080 or 4k, got '{s}'"))?;
            (w.parse()?, h.parse()?)
        }
    })
}

pub fn write_png(path: &Path, w: u32, h: u32, rgb: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgb)?;
    Ok(())
}

/// 0x00RRGGBB pixels to tightly packed RGB bytes.
pub fn to_rgb(px: &[u32]) -> Vec<u8> {
    px.iter().flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8]).collect()
}

pub fn stem(p: &Path) -> String {
    p.file_stem().and_then(|s| s.to_str()).unwrap_or("form").to_string()
}

/// Billions of points per second.
pub fn gpts(points: u64, ms: f64) -> f64 {
    points as f64 / (ms.max(1e-6) * 1e-3) / 1e9
}

pub fn human(n: u64) -> String {
    match n {
        1_000_000_000.. => format!("{:.2}b", n as f64 / 1e9),
        1_000_000.. => format!("{:.1}m", n as f64 / 1e6),
        1000.. => format!("{:.0}k", n as f64 / 1e3),
        _ => n.to_string(),
    }
}

/// Seconds since 1970, for file names.
pub fn timestamp() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// The `.ks` files in a folder, sorted.
pub fn forms_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "ks")).collect();
    files.sort();
    Ok(files)
}
