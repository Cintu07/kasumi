//! Text and boxes drawn straight into a window's pixels.

use font8x8::legacy::BASIC_LEGACY;

pub const WHITE: u32 = 0xE8E8E8;
pub const GREY: u32 = 0x8A8A8A;
pub const RED: u32 = 0xFF5A5A;
pub const AMBER: u32 = 0xFFC861;

pub struct Canvas<'a> {
    pub px: &'a mut [u32],
    pub w: usize,
    pub h: usize,
}

impl Canvas<'_> {
    fn each(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, mut f: impl FnMut(i32, i32, &mut u32)) {
        for y in y0.max(0)..y1.min(self.h as i32) {
            for x in x0.max(0)..x1.min(self.w as i32) {
                f(x, y, &mut self.px[y as usize * self.w + x as usize]);
            }
        }
    }

    pub fn fill(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        self.each(x0, y0, x1, y1, |_, _, p| *p = color);
    }

    /// Darken a region so text reads on top of it.
    pub fn dim(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        self.each(x0, y0, x1, y1, |_, _, p| *p = (*p >> 2) & 0x3F3F3F);
    }

    pub fn outline(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32, thick: i32) {
        self.each(x0, y0, x1, y1, |x, y, p| {
            if x < x0 + thick || x >= x1 - thick || y < y0 + thick || y >= y1 - thick {
                *p = color;
            }
        });
    }

    pub fn text(&mut self, x: i32, y: i32, s: &str, color: u32, scale: i32) {
        for (k, ch) in s.chars().filter(|c| *c != '\n').enumerate() {
            let glyph = BASIC_LEGACY[if ch.is_ascii() { ch as usize } else { b'?' as usize }];
            let left = x + k as i32 * 8 * scale;
            for (row, bits) in glyph.iter().enumerate() {
                for col in (0..8).filter(|c| bits & (1 << c) != 0) {
                    let (px, py) = (left + col * scale, y + row as i32 * scale);
                    self.fill(px, py, px + scale, py + scale, color);
                }
            }
        }
    }

    /// Text with a dark shadow, so it reads over bright forms.
    pub fn label(&mut self, x: i32, y: i32, s: &str, color: u32, scale: i32) {
        self.text(x + scale, y + scale, s, 0, scale);
        self.text(x, y, s, color, scale);
    }
}
