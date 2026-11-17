//! Tiny bitmap text for the live window (font8x8, drawn straight into the
//! frame buffer).

use font8x8::legacy::BASIC_LEGACY;

pub fn text(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, s: &str, color: u32, scale: i32) {
    let mut cx = x;
    for ch in s.chars() {
        if ch == '\n' {
            continue;
        }
        let c = ch as usize;
        let glyph = if c < 128 { BASIC_LEGACY[c] } else { BASIC_LEGACY[b'?' as usize] };
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..8 {
                if bits & (1 << col) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cx + col * scale + dx;
                            let py = y + row as i32 * scale + dy;
                            if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                                buf[py as usize * w + px as usize] = color;
                            }
                        }
                    }
                }
            }
        }
        cx += 8 * scale;
    }
}

/// Text with a 1px dark shadow so it reads over bright forms.
pub fn label(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, s: &str, color: u32, scale: i32) {
    text(buf, w, h, x + scale, y + scale, s, 0x000000, scale);
    text(buf, w, h, x, y, s, color, scale);
}

pub fn dim_rect(buf: &mut [u32], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32) {
    for y in y0.max(0)..y1.min(h as i32) {
        for x in x0.max(0)..x1.min(w as i32) {
            let p = &mut buf[y as usize * w + x as usize];
            let r = ((*p >> 16) & 255) / 4;
            let g = ((*p >> 8) & 255) / 4;
            let b = (*p & 255) / 4;
            *p = (r << 16) | (g << 8) | b;
        }
    }
}

pub fn rect(buf: &mut [u32], w: usize, h: usize, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, thick: i32) {
    for y in y0.max(0)..y1.min(h as i32) {
        for x in x0.max(0)..x1.min(w as i32) {
            let edge = x < x0 + thick || x >= x1 - thick || y < y0 + thick || y >= y1 - thick;
            if edge {
                buf[y as usize * w + x as usize] = color;
            }
        }
    }
}

pub const WHITE: u32 = 0xE8E8E8;
pub const GREY: u32 = 0x8A8A8A;
pub const RED: u32 = 0xFF5A5A;
pub const AMBER: u32 = 0xFFC861;
