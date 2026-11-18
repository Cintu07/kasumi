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
    /// the most lit of the four frame edges, as a fraction of that edge
    pub edge: f32,
    /// width / length of the light, along its own axis so diagonals count (low = a sliver)
    pub aspect: f32,
    /// bounding box of the light / frame area (low = a speck)
    pub extent: f32,
    /// mean saturation of the lit pixels, set by the caller from the colour frame
    pub chroma: f32,
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
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for y in 0..h {
        for x in 0..w {
            if a[y * w + x] > 0.04 {
                nlit += 1;
                let (px, py) = (x as f32, y as f32);
                sx += px;
                sy += py;
                sxx += px * px;
                syy += py * py;
                sxy += px * py;
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
    let (aspect, extent) = if nlit == 0 {
        (0.0, 0.0)
    } else {
        let (bw, bh) = ((x1 - x0 + 1) as f32, (y1 - y0 + 1) as f32);
        // a diagonal line fills a square bounding box, so also compare the
        // spread of the light along its two principal axes
        let c = nlit as f32;
        let (mx, my) = (sx / c, sy / c);
        let (cxx, cyy, cxy) = (sxx / c - mx * mx, syy / c - my * my, sxy / c - mx * my);
        let mid = (cxx + cyy) * 0.5;
        let off = (((cxx - cyy) * 0.5).powi(2) + cxy * cxy).sqrt();
        let axes = ((mid - off).max(0.0) / (mid + off).max(1e-6)).sqrt();
        ((bw.min(bh) / bw.max(bh)).min(axes), bw * bh / n)
    };
    let on = |x: usize, y: usize| (a[y * w + x] > 0.04) as usize as f32;
    let top = (0..w).map(|x| on(x, 0)).sum::<f32>() / w as f32;
    let bottom = (0..w).map(|x| on(x, h - 1)).sum::<f32>() / w as f32;
    let left = (0..h).map(|y| on(0, y)).sum::<f32>() / h as f32;
    let right = (0..h).map(|y| on(w - 1, y)).sum::<f32>() / h as f32;
    let edge = top.max(bottom).max(left).max(right);

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
    vec.extend_from_slice(&scalars);
    vec.extend_from_slice(&rings);
    vec.extend(orient.iter().map(|o| o / osum));
    vec.extend(g8.iter().map(|v| v.sqrt()));
    Feat { lit, mean, structure, motion, spread, fill, border, edge, aspect, extent, chroma: 0.0, vec }
}

/// Mean saturation of the lit pixels of a 0x00RRGGBB frame, 0 for pure grey.
pub fn chroma(px: &[u32]) -> f32 {
    let (mut sum, mut count) = (0f32, 0usize);
    for &p in px {
        let (r, g, b) = (((p >> 16) & 255) as f32, ((p >> 8) & 255) as f32, (p & 255) as f32);
        let hi = r.max(g).max(b);
        if hi > 10.0 {
            sum += (hi - r.min(g).min(b)) / hi;
            count += 1;
        }
    }
    sum / count.max(1) as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Junk {
    Blank,
    BlownOut,
    Frozen,
    Noise,
    Sliver,
    Small,
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
    } else if f.aspect < 0.45 {
        Some(Junk::Sliver)
    } else if f.extent < 0.25 {
        Some(Junk::Small)
    } else if f.fill > 0.8 && f.lit > 0.02 && f.lit < 0.6 {
        Some(Junk::Flat)
    } else if (f.border > 0.12 && f.lit < 0.5) || (f.edge > 0.3 && f.lit < 0.7) {
        // a form that is not meant to fill the frame but runs off its edges
        Some(Junk::Cropped)
    } else {
        None
    }
}

pub fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 64;

    fn verdict(lit: impl Fn(f32, f32) -> bool) -> Option<Junk> {
        let a: Vec<f32> = (0..S * S).map(|k| lit((k % S) as f32 - 31.5, (k / S) as f32 - 31.5) as u8 as f32).collect();
        let b: Vec<f32> = a.iter().map(|v| v * 0.5).collect();
        junk(&features(&a, &b, S, S))
    }

    fn ring(x: f32, y: f32, inner: f32, outer: f32) -> bool {
        let r = (x * x + y * y).sqrt();
        r >= inner && r < outer
    }

    #[test]
    fn a_ring_inside_the_frame_is_kept() {
        assert_eq!(verdict(|x, y| ring(x, y, 21.0, 24.0)), None);
    }

    #[test]
    fn a_thin_bar_is_a_sliver() {
        assert_eq!(verdict(|x, y| x.abs() < 2.0 && y.abs() < 25.0), Some(Junk::Sliver));
    }

    #[test]
    fn a_diagonal_line_is_a_sliver() {
        assert_eq!(verdict(|x, y| (x - y).abs() < 3.0 && x.abs() < 28.0), Some(Junk::Sliver));
    }

    #[test]
    fn a_speck_is_small() {
        assert_eq!(verdict(|x, y| x.abs() < 6.0 && y.abs() < 6.0), Some(Junk::Small));
    }

    #[test]
    fn a_ring_running_off_the_frame_is_cropped() {
        assert_eq!(verdict(|x, y| ring(x, y, 33.0, 39.0)), Some(Junk::Cropped));
    }

    #[test]
    fn chroma_separates_white_from_colour() {
        assert_eq!(chroma(&[0x00ff_ffff, 0x0080_8080, 0]), 0.0);
        assert!(chroma(&[0x00ff_4020]) > 0.5);
    }
}
