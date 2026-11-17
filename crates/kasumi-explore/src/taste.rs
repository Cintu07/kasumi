//! The taste model: logistic regression on image features, trained from the
//! likes and skips logged by the grid. Evaluated as the probability a held-out
//! like ranks above a held-out skip (that is the PRD's 75% bar, i.e. AUC).

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Taste {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub w: Vec<f32>,
    pub b: f32,
    pub auc: f32,
    pub n_like: usize,
    pub n_skip: usize,
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

impl Taste {
    pub fn score(&self, x: &[f32]) -> f32 {
        let mut z = self.b;
        for k in 0..self.w.len().min(x.len()) {
            z += self.w[k] * (x[k] - self.mean[k]) / self.std[k];
        }
        sigmoid(z)
    }

    fn fit(xs: &[Vec<f32>], ys: &[f32], l2: f32) -> Taste {
        let d = xs[0].len();
        let n = xs.len() as f32;
        let mut mean = vec![0f32; d];
        for x in xs {
            for k in 0..d {
                mean[k] += x[k] / n;
            }
        }
        let mut std = vec![0f32; d];
        for x in xs {
            for k in 0..d {
                std[k] += (x[k] - mean[k]).powi(2) / n;
            }
        }
        for s in std.iter_mut() {
            *s = s.sqrt().max(1e-3);
        }
        let z: Vec<Vec<f32>> = xs.iter().map(|x| (0..d).map(|k| (x[k] - mean[k]) / std[k]).collect()).collect();
        // class-balanced weights
        let pos = ys.iter().sum::<f32>().max(1.0);
        let neg = (n - pos).max(1.0);
        let mut w = vec![0f32; d];
        let mut b = 0f32;
        let lr = 0.2;
        for _ in 0..400 {
            let mut gw = vec![0f32; d];
            let mut gb = 0f32;
            for (x, &y) in z.iter().zip(ys) {
                let p = sigmoid(b + x.iter().zip(&w).map(|(a, c)| a * c).sum::<f32>());
                let cw = if y > 0.5 { n / (2.0 * pos) } else { n / (2.0 * neg) };
                let e = (p - y) * cw;
                for k in 0..d {
                    gw[k] += e * x[k];
                }
                gb += e;
            }
            for k in 0..d {
                w[k] -= lr * (gw[k] / n + l2 * w[k]);
            }
            b -= lr * gb / n;
        }
        Taste { mean, std, w, b, auc: 0.0, n_like: pos as usize, n_skip: neg as usize }
    }

    /// Train on (features, liked). Reports AUC on a held-out 25%.
    pub fn train(data: &[(Vec<f32>, bool)], seed: u64) -> Option<Taste> {
        let likes = data.iter().filter(|d| d.1).count();
        let skips = data.len() - likes;
        if likes < 4 || skips < 4 {
            return None;
        }
        let mut idx: Vec<usize> = (0..data.len()).collect();
        let mut rng = Rng::new(seed);
        for k in (1..idx.len()).rev() {
            idx.swap(k, rng.below(k + 1));
        }
        let cut = data.len() * 3 / 4;
        let (tr, te) = idx.split_at(cut);
        let xs: Vec<Vec<f32>> = tr.iter().map(|&i| data[i].0.clone()).collect();
        let ys: Vec<f32> = tr.iter().map(|&i| if data[i].1 { 1.0 } else { 0.0 }).collect();
        let held = if xs.iter().any(|_| true) && ys.iter().any(|&y| y > 0.5) && ys.iter().any(|&y| y < 0.5) {
            let m = Taste::fit(&xs, &ys, 0.02);
            let scored: Vec<(f32, bool)> = te.iter().map(|&i| (m.score(&data[i].0), data[i].1)).collect();
            auc(&scored)
        } else {
            0.5
        };
        // final model on everything
        let xs: Vec<Vec<f32>> = data.iter().map(|d| d.0.clone()).collect();
        let ys: Vec<f32> = data.iter().map(|d| if d.1 { 1.0 } else { 0.0 }).collect();
        let mut m = Taste::fit(&xs, &ys, 0.02);
        m.auc = held;
        m.n_like = likes;
        m.n_skip = skips;
        Some(m)
    }
}

/// P(score of a random like > score of a random skip).
pub fn auc(scored: &[(f32, bool)]) -> f32 {
    let pos: Vec<f32> = scored.iter().filter(|s| s.1).map(|s| s.0).collect();
    let neg: Vec<f32> = scored.iter().filter(|s| !s.1).map(|s| s.0).collect();
    if pos.is_empty() || neg.is_empty() {
        return 0.5;
    }
    let mut win = 0f32;
    for p in &pos {
        for q in &neg {
            win += if p > q { 1.0 } else if p == q { 0.5 } else { 0.0 };
        }
    }
    win / (pos.len() * neg.len()) as f32
}
