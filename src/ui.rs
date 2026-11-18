//! Text, lines and dots drawn straight into a window's pixels. Text is IBM Plex Mono.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use fontdue::{Font, FontSettings};

pub const WHITE: u32 = 0xE8E8E8;
pub const GREY: u32 = 0x8A8A8A;
pub const RED: u32 = 0xFF5A5A;
pub const AMBER: u32 = 0xFFC861;

/// Plex Mono advances every character by this share of the font size.
const ADVANCE: f32 = 0.6;

struct Glyph {
    width: usize,
    height: usize,
    left: i32,
    top: i32,
    alpha: Vec<u8>,
}

struct Type {
    regular: Font,
    medium: Font,
    glyphs: HashMap<(bool, char, u32), Rc<Glyph>>,
}

impl Type {
    fn new() -> Type {
        let load = |bytes: &[u8]| Font::from_bytes(bytes, FontSettings::default()).expect("the bundled font parses");
        Type {
            regular: load(include_bytes!("fonts/IBMPlexMono-Regular.ttf")),
            medium: load(include_bytes!("fonts/IBMPlexMono-Medium.ttf")),
            glyphs: HashMap::new(),
        }
    }

    fn glyph(&mut self, medium: bool, ch: char, size: f32) -> Rc<Glyph> {
        let font = if medium { &self.medium } else { &self.regular };
        let ascent = font.horizontal_line_metrics(size).map_or(size, |m| m.ascent);
        self.glyphs
            .entry((medium, ch, size.to_bits()))
            .or_insert_with(|| {
                let (m, alpha) = font.rasterize(ch, size);
                let top = ascent.round() as i32 - m.height as i32 - m.ymin;
                Rc::new(Glyph { width: m.width, height: m.height, left: m.xmin, top, alpha })
            })
            .clone()
    }
}

thread_local! {
    static TYPE: RefCell<Type> = RefCell::new(Type::new());
}

/// How wide a string is at this size.
pub fn text_width(s: &str, size: f32) -> i32 {
    (s.chars().count() as f32 * size * ADVANCE).round() as i32
}

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

    /// Mix `color` over one pixel, `alpha` from 0 to 1.
    fn blend(&mut self, x: i32, y: i32, color: u32, alpha: f32) {
        self.each(x, y, x + 1, y + 1, |_, _, p| {
            let mix = |shift: u32| {
                let (under, over) = ((*p >> shift & 255) as f32, (color >> shift & 255) as f32);
                ((under + (over - under) * alpha) as u32) << shift
            };
            *p = mix(16) | mix(8) | mix(0);
        });
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

    /// A smooth round dot. With `ring` only its edge is drawn.
    pub fn dot(&mut self, cx: f32, cy: f32, radius: f32, color: u32, ring: bool) {
        let reach = radius.ceil() as i32 + 1;
        for y in cy as i32 - reach..=cy as i32 + reach {
            for x in cx as i32 - reach..=cx as i32 + reach {
                let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
                let outer = (radius + 0.5 - d).clamp(0.0, 1.0);
                let inner = if ring { (d - radius + 1.0).clamp(0.0, 1.0) } else { 1.0 };
                self.blend(x, y, color, outer * inner);
            }
        }
    }

    fn write(&mut self, x: i32, y: i32, s: &str, color: u32, size: f32, medium: bool) {
        for (k, ch) in s.chars().filter(|c| *c != '\n').enumerate() {
            let glyph = TYPE.with(|t| t.borrow_mut().glyph(medium, ch, size));
            let left = x + (k as f32 * size * ADVANCE).round() as i32 + glyph.left;
            for row in 0..glyph.height {
                for col in 0..glyph.width {
                    let alpha = glyph.alpha[row * glyph.width + col];
                    if alpha > 0 {
                        self.blend(left + col as i32, y + glyph.top + row as i32, color, alpha as f32 / 255.0);
                    }
                }
            }
        }
    }

    /// Text with its top at `y`.
    pub fn text(&mut self, x: i32, y: i32, s: &str, color: u32, size: f32) {
        self.write(x, y, s, color, size, false);
    }

    /// Heavier text, for values and titles.
    pub fn strong(&mut self, x: i32, y: i32, s: &str, color: u32, size: f32) {
        self.write(x, y, s, color, size, true);
    }

    /// Text with a dark shadow, so it reads over bright forms.
    pub fn label(&mut self, x: i32, y: i32, s: &str, color: u32, size: f32) {
        self.text(x + 1, y + 1, s, 0, size);
        self.text(x, y, s, color, size);
    }
}
