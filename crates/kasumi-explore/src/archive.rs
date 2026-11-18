//! The archive: a MAP-Elites grid over two style axes (how much of the frame
//! is lit, and how far the light sits from the centre), keeping the best form
//! per cell. Score = taste + novelty, so evolution spreads across styles
//! instead of collapsing into one look.

use serde::{Deserialize, Serialize};

use crate::features::{dist, Feat};

pub const BINS: usize = 8;
/// Mean saturation above which a form counts as coloured.
pub const COLOUR_CHROMA: f32 = 0.15;
/// Gallery picks must be this far apart, in units of the archive's typical spacing.
pub const GALLERY_SPACING: f32 = 1.3;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Elite {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub taste: f32,
    pub novelty: f32,
    pub lit: f32,
    pub spread: f32,
    pub generation: u32,
    #[serde(skip)]
    pub vec: Vec<f32>,
}

#[derive(Default)]
pub struct Archive {
    pub cells: Vec<Option<Elite>>,
    /// median k-nn distance in the archive; novelty is measured in these units
    pub scale: f32,
}

fn knn(d: &mut [f32]) -> f32 {
    if d.is_empty() {
        return f32::NAN;
    }
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = d.len().min(4);
    d[..k].iter().sum::<f32>() / k as f32
}

pub fn cell_of(f: &Feat) -> usize {
    // lit is roughly 0.02..0.6 -> log-ish bins
    let l = ((f.lit.max(0.02).ln() - 0.02f32.ln()) / (0.6f32.ln() - 0.02f32.ln()) * BINS as f32) as usize;
    let s = ((f.spread / 0.7).clamp(0.0, 0.999) * BINS as f32) as usize;
    // colour forms get their own layer: the descriptor only sees luminance, so
    // without it a white form and its coloured twin fight over one cell
    let layer = (f.chroma > COLOUR_CHROMA) as usize;
    layer * BINS * BINS + l.min(BINS - 1) * BINS + s.min(BINS - 1)
}

impl Archive {
    pub fn new() -> Archive {
        Archive { cells: vec![None; 2 * BINS * BINS], scale: 0.0 }
    }

    pub fn elites(&self) -> impl Iterator<Item = &Elite> {
        self.cells.iter().flatten()
    }

    pub fn len(&self) -> usize {
        self.elites().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Mean distance to the 4 nearest archived forms, relative to the
    /// archive's typical spacing (1.0 = as novel as an average elite).
    pub fn novelty(&self, v: &[f32]) -> f32 {
        let mut d: Vec<f32> = self.elites().filter(|e| !e.vec.is_empty()).map(|e| dist(&e.vec, v)).collect();
        let raw = knn(&mut d);
        if raw.is_nan() || self.scale <= 0.0 {
            return 1.0;
        }
        (raw / self.scale).min(2.0)
    }

    /// Recompute every elite's novelty against the current archive (excluding
    /// itself). Early elites were scored against an almost empty archive, so
    /// without this their novelty stays inflated and nothing can replace them.
    pub fn rescore(&mut self, novelty_w: f32) {
        let vecs: Vec<(usize, Vec<f32>)> = self.cells.iter().enumerate().filter_map(|(k, c)| c.as_ref().map(|e| (k, e.vec.clone()))).collect();
        let raws: Vec<f32> = vecs
            .iter()
            .map(|(k, v)| {
                let mut d: Vec<f32> = vecs.iter().filter(|(j, _)| j != k).map(|(_, w)| dist(v, w)).collect();
                knn(&mut d)
            })
            .collect();
        let mut sorted: Vec<f32> = raws.iter().copied().filter(|r| !r.is_nan()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.scale = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        for ((k, _), raw) in vecs.iter().zip(raws) {
            let nov = if raw.is_nan() || self.scale <= 0.0 { 1.0 } else { (raw / self.scale).min(2.0) };
            if let Some(e) = self.cells[*k].as_mut() {
                e.novelty = nov;
                e.score = e.taste + novelty_w * nov;
            }
        }
    }

    /// Insert if the cell is empty or this one scores higher. Returns true if kept.
    pub fn offer(&mut self, e: Elite, f: &Feat) -> bool {
        let c = cell_of(f);
        match &self.cells[c] {
            Some(old) if old.score >= e.score => false,
            _ => {
                self.cells[c] = Some(e);
                true
            }
        }
    }

    pub fn ranked(&self) -> Vec<&Elite> {
        let mut v: Vec<&Elite> = self.elites().collect();
        v.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        v
    }

    /// The best forms, best first, skipping any that look like one already
    /// picked. Forms reloaded from disk have no descriptor and are all kept.
    pub fn diverse(&self, top: usize, spacing: f32) -> Vec<&Elite> {
        let min = self.scale * spacing;
        let mut picked: Vec<&Elite> = Vec::new();
        for e in self.ranked() {
            if picked.len() == top {
                break;
            }
            if e.vec.is_empty() || picked.iter().all(|p| dist(&p.vec, &e.vec) >= min) {
                picked.push(e);
            }
        }
        picked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elite(id: &str, score: f32, vec: Vec<f32>) -> Elite {
        Elite { id: id.into(), text: String::new(), score, taste: 0.0, novelty: 0.0, lit: 0.0, spread: 0.0, generation: 0, vec }
    }

    #[test]
    fn colour_forms_do_not_compete_with_their_white_twins() {
        let white = Feat { lit: 0.2, spread: 0.3, ..Feat::default() };
        let colour = Feat { chroma: 0.6, ..white.clone() };
        assert_ne!(cell_of(&white), cell_of(&colour));
        assert!(cell_of(&colour) < Archive::new().cells.len());
    }

    #[test]
    fn gallery_skips_near_copies_of_better_forms() {
        let mut a = Archive::new();
        a.scale = 1.0;
        a.cells[0] = Some(elite("best", 3.0, vec![0.0, 0.0]));
        a.cells[1] = Some(elite("copy", 2.0, vec![0.1, 0.0]));
        a.cells[2] = Some(elite("other", 1.0, vec![2.0, 0.0]));
        let ids: Vec<&str> = a.diverse(10, GALLERY_SPACING).iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["best", "other"]);
    }
}
