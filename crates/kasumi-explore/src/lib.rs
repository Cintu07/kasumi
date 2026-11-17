//! kasumi-explore: the parts that invent forms. Mutation on syntax trees,
//! junk filters, image features, the taste model and the style archive.
//! Rendering lives in kasumi-gpu; the explorer only sees pixels and forms.

pub mod archive;
pub mod features;
pub mod mutate;
pub mod rng;
pub mod taste;

use serde::{Deserialize, Serialize};

/// One line of `explore/clicks.jsonl`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Click {
    pub id: String,
    pub liked: bool,
    pub form: String,
    pub thumb: String,
    pub feat: Vec<f32>,
    pub ts: u64,
}

pub fn read_clicks(path: &std::path::Path) -> Vec<Click> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().filter_map(|l| serde_json::from_str(l).ok()).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kasumi_lang::parse;

    #[test]
    fn mutants_are_valid_and_different() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../forms/silk.ks")).unwrap();
        let f = parse(&src).unwrap();
        let mut rng = rng::Rng::new(7);
        let mut distinct = std::collections::HashSet::new();
        for _ in 0..300 {
            let m = mutate::mutate(&f, &mut rng, None).expect("mutation failed");
            let text = kasumi_lang::print_form(&m);
            assert!(parse(&text).is_ok(), "{text}");
            distinct.insert(text);
        }
        assert!(distinct.len() > 250, "only {} distinct mutants", distinct.len());
    }

    #[test]
    fn taste_learns_a_simple_rule() {
        let mut r = rng::Rng::new(3);
        let data: Vec<(Vec<f32>, bool)> = (0..400)
            .map(|_| {
                let x: Vec<f32> = (0..6).map(|_| r.normal() as f32).collect();
                let liked = x[0] + 0.5 * x[2] + 0.3 * r.normal() as f32 > 0.0;
                (x, liked)
            })
            .collect();
        let t = taste::Taste::train(&data, 1).unwrap();
        assert!(t.auc > 0.85, "auc {}", t.auc);
    }
}
