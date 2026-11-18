//! How a form is shown: palette, exposure, camera and effects, read from its @settings.

use crate::ast::Form;

pub const PALETTES: &[(&str, [f32; 3], [f32; 3])] = &[
    ("mono", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
    ("ember", [1.0, 0.32, 0.08], [1.0, 0.82, 0.45]),
    ("ice", [0.22, 0.42, 1.0], [0.75, 0.95, 1.0]),
    ("sakura", [0.95, 0.38, 0.62], [1.0, 0.9, 0.93]),
    ("moss", [0.3, 0.72, 0.4], [0.95, 0.95, 0.7]),
    ("ink", [0.32, 0.36, 0.95], [0.98, 0.92, 0.84]),
    ("dusk", [0.55, 0.28, 0.95], [1.0, 0.6, 0.38]),
    ("jade", [0.08, 0.75, 0.65], [0.9, 1.0, 0.95]),
];

#[derive(Clone, Debug)]
pub struct Look {
    pub palette: String,
    pub low: [f32; 3],
    pub high: [f32; 3],
    pub gamma: f32,
    pub vibrancy: f32,
    pub heat: f32,
    pub exposure: f32,
    pub zoom: f32,
    pub cx: f32,
    pub cy: f32,
    pub dof: f32,
    pub focus: f32,
    pub trail: f32,
    pub shutter: f32,
    pub bloom: f32,
    /// Widest density-estimation blur in accumulation pixels, 0 for off.
    pub de: f32,
}

impl Look {
    pub fn from_form(f: &Form) -> Look {
        let num = |key: &str, default: f64| f.num_or(key, default) as f32;
        let mut look = Look {
            palette: String::new(),
            low: [1.0; 3],
            high: [1.0; 3],
            gamma: num("gamma", 2.2),
            vibrancy: num("vibrancy", 1.0),
            heat: num("heat", 0.2),
            exposure: num("exposure", 1.0),
            zoom: num("zoom", 1.0),
            cx: num("cx", 0.0),
            cy: num("cy", 0.0),
            dof: num("dof", 0.0),
            focus: num("focus", 0.0),
            trail: num("trail", 0.0),
            shutter: num("shutter", 0.5),
            bloom: num("bloom", 0.0),
            de: num("de", 0.0),
        };
        look.palette = f.word("palette").unwrap_or("mono").to_string();
        look.set_palette(&look.palette.clone());
        look
    }

    pub fn set_palette(&mut self, name: &str) {
        if let Some((_, low, high)) = PALETTES.iter().find(|p| p.0 == name) {
            self.palette = name.to_string();
            (self.low, self.high) = (*low, *high);
        }
    }
}
