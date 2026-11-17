//! Cheap image features from a rendered thumbnail: used for the junk filters,
//! the novelty distance and the taste model.

/// Luminance in [0, 1] from 0x00RRGGBB pixels.
pub fn luma(px: &[u32]) -> Vec<f32> {
    px.iter()
        .map(|&p| {
            let r = ((p >> 16) & 255) as f32;
            let g = ((p >> 8) & 255) as f32;
            let b = (p & 255) as f32;
            (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct Feat {
    /// fraction of pixels lit
    pub lit: f32,
    /// mean luminance
    pub mean: f32,
    /// coarse-scale contrast / fine-scale contrast (low = noise)
    pub structure: f32,
    /// mean abs change between two frames
    pub motion: f32,
    /// centroid distance of light from the centre, 0..1
    pub spread: f32,
    /// lit pixels / area of their bounding box (near 1 = a solid slab)
    pub fill: f32,
    /// fraction of the frame's outer border that is lit (high = cut off)
    pub border: f32,
    /// descriptor for novelty and taste
    pub vec: Vec<f32>,
}

fn block_means(l: &[f32], w: usize, h: usize, g: usize) -> Vec<f32> {
    let mut out = vec![0f32; g * g];
    let mut cnt = vec![0f32; g * g];
    for y in 0..h {
        for x in 0..w {
            let k = (y * g / h) * g + (x * g / w);
            out[k] += l[y * w + x];
            cnt[k] += 1.0;
        }
    }
    for (o, c) in out.iter_mut().zip(&cnt) {
        *o /= c.max(1.0);
    }
    out
}

fn std(v: &[f32]) -> f32 {
    let m = v.iter().sum::<f32>() / v.len().max(1) as f32;
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

/// `a` and `b` are luminance images of the same form a moment apart.
pub fn features(a: &[f32], b: &[f32], w: usize, h: usize) -> Feat {
    let n = (w * h) as f32;
    let lit = a.iter().filter(|&&v| v > 0.04).count() as f32 / n;
    let mean = a.iter().sum::<f32>() / n;
    let motion = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / n;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    let mut nlit = 0usize;
    for y in 0..h {
        for x in 0..w {
            if a[y * w + x] > 0.04 {
                nlit += 1;
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    let mut border_lit = 0usize;
    let mut border_n = 0usize;
    for y in 0..h {
        for x in 0..w {
            if x < 2 || y < 2 || x + 2 >= w || y + 2 >= h {
                border_n += 1;
                border_lit += (a[y * w + x] > 0.04) as usize;
            }
        }
    }
    let border = border_lit as f32 / border_n.max(1) as f32;
    let fill = if nlit == 0 { 0.0 } else { nlit as f32 / ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32 };

    let g8 = block_means(a, w, h, 8);
    let g32 = block_means(a, w, h, 32);
    let structure = std(&g8) / (std(a) + 1e-4);

    // radial profile (8 rings) and centroid spread
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let rmax = (cx * cx + cy * cy).sqrt();
    let mut rings = [0f32; 8];
    let mut rc = [0f32; 8];
    let mut wsum = 0f32;
    let mut dsum = 0f32;
    // gradient orientation histogram (8 bins)
    let mut orient = [0f32; 8];
    // mirror symmetries
    let (mut lr, mut ud, mut rot) = (0f32, 0f32, 0f32);
    for y in 0..h {
        for x in 0..w {
            let v = a[y * w + x];
            let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() / rmax;
            let k = ((d * 8.0) as usize).min(7);
            rings[k] += v;
            rc[k] += 1.0;
            wsum += v;
            dsum += v * d;
            lr += (v - a[y * w + (w - 1 - x)]).abs();
            ud += (v - a[(h - 1 - y) * w + x]).abs();
            rot += (v - a[(h - 1 - y) * w + (w - 1 - x)]).abs();
            if x + 1 < w && y + 1 < h {
                let gx = a[y * w + x + 1] - v;
                let gy = a[(y + 1) * w + x] - v;
                let m = (gx * gx + gy * gy).sqrt();
                if m > 0.02 {
                    let ang = gy.atan2(gx).rem_euclid(std::f32::consts::PI);
                    orient[((ang / std::f32::consts::PI * 8.0) as usize).min(7)] += m;
                }
            }
        }
    }
    for k in 0..8 {
        rings[k] /= rc[k].max(1.0);
    }
    let osum: f32 = orient.iter().sum::<f32>().max(1e-6);
    let spread = dsum / wsum.max(1e-6);
    let norm = (mean * n).max(1e-3);

    let mut vec = Vec::with_capacity(32 * 32 + 30);
    // coarse layout (sqrt to tame bright areas)
    vec.extend(g32.iter().map(|v| v.sqrt()));
    let scalars = [lit, mean, structure, motion * 20.0, spread, 1.0 - lr / norm, 1.0 - ud / norm, 1.0 - rot / norm];
    // scalar summary goes first in `taste_vec`, so keep it at a fixed offset
    vec.extend_from_slice(&scalars);
    vec.extend_from_slice(&rings);
    vec.extend(orient.iter().map(|o| o / osum));
    vec.extend(g8.iter().map(|v| v.sqrt()));
    Feat { lit, mean, structure, motion, spread, fill, border, vec }
}

/// The compact part of the descriptor the taste model is trained on
/// (scalars, rings, orientation, 8x8 layout): position-light, style-heavy.
pub fn taste_vec(f: &Feat) -> Vec<f32> {
    f.vec[32 * 32..].to_vec()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Junk {
    Blank,
    BlownOut,
    Frozen,
    Noise,
    Flat,
    Cropped,
}

/// The automatic filters: throw away junk before a human sees it.
pub fn junk(f: &Feat) -> Option<Junk> {
    if f.lit < 0.02 {
        Some(Junk::Blank)
    } else if f.lit > 0.6 && f.mean > 0.45 {
        Some(Junk::BlownOut)
    } else if f.motion < 0.0015 {
        Some(Junk::Frozen)
    } else if f.structure < 0.18 && f.lit > 0.12 {
        Some(Junk::Noise)
    } else if f.fill > 0.9 && f.lit > 0.02 && f.lit < 0.6 {
        Some(Junk::Flat)
    } else if f.border > 0.12 && f.lit < 0.5 {
        // a form that is not meant to fill the frame but runs off its edges
        Some(Junk::Cropped)
    } else {
        None
    }
}

pub fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}
