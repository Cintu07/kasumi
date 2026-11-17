//! The archive: a MAP-Elites grid over two style axes (how much of the frame
//! is lit, and how far the light sits from the centre), keeping the best form
//! per cell. Score = taste + novelty, so evolution spreads across styles
//! instead of collapsing into one look.

use serde::{Deserialize, Serialize};

use crate::features::{dist, Feat};

pub const BINS: usize = 8;

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
    l.min(BINS - 1) * BINS + s.min(BINS - 1)
}

impl Archive {
    pub fn new() -> Archive {
        Archive { cells: vec![None; BINS * BINS], scale: 0.0 }
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
}
