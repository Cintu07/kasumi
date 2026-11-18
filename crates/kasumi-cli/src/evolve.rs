//! Overnight evolution, the morning gallery, and taste-model training.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use kasumi_explore::archive::{Archive, Elite, GALLERY_SPACING};
use kasumi_explore::features::{junk, taste_vec};
use kasumi_explore::mutate::mutate;
use kasumi_explore::rng::Rng;
use kasumi_explore::taste::Taste;
use kasumi_gpu::{Gpu, Statics, Target};

use crate::common::*;
use crate::explore_rt::*;

const THUMB: u32 = 192;

/// Taste: the built-in feature model, or the SigLIP / CLIP sidecar
/// (`taste/siglip.py serve`), which scores thumbnails a batch at a time.
enum Scorer {
    Builtin(Option<Taste>),
    Embed {
        child: std::process::Child,
        stdin: std::process::ChildStdin,
        stdout: std::io::BufReader<std::process::ChildStdout>,
        dir: PathBuf,
        /// the sidecar's score is the taste score (otherwise it only supplies embeddings)
        taste_from_sidecar: bool,
        builtin: Option<Taste>,
        embeddings: bool,
    },
}

impl Scorer {
    fn new(kind: &str, novelty: &str, out: &Path) -> Result<Scorer> {
        let sidecar_taste = kind == "siglip" || kind == "embed";
        let embeddings = novelty == "embed";
        if !sidecar_taste && !embeddings {
            return Ok(Scorer::Builtin(load_taste(Path::new("explore/taste.json"))));
        }
        let py = if cfg!(windows) { ".venv/Scripts/python.exe" } else { ".venv/bin/python" };
        anyhow::ensure!(Path::new(py).is_file(), "--taste siglip needs the python venv ({py}); see README");
        anyhow::ensure!(!sidecar_taste || Path::new("explore/taste_embed.pkl").is_file(), "no embedding taste model yet: run `{py} taste/siglip.py train`");
        let mut sidecar_args = vec!["taste/siglip.py", "serve"];
        if embeddings {
            sidecar_args.push("--embed");
        }
        let mut child = std::process::Command::new(py)
            .args(&sidecar_args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .context("starting the taste sidecar")?;
        let stdin = child.stdin.take().unwrap();
        let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        std::io::BufRead::read_line(&mut stdout, &mut line)?;
        anyhow::ensure!(line.trim() == "ready", "taste sidecar failed to start");
        let dir = out.join("thumbs");
        std::fs::create_dir_all(&dir)?;
        let builtin = load_taste(Path::new("explore/taste.json"));
        Ok(Scorer::Embed { child, stdin, stdout, dir, taste_from_sidecar: sidecar_taste, builtin, embeddings })
    }

    fn describe(&self) -> String {
        match self {
            Scorer::Builtin(Some(t)) => format!("features (held-out auc {:.2}, {} likes / {} skips)", t.auc, t.n_like, t.n_skip),
            Scorer::Builtin(None) => "off (score = novelty; run `kasumi grid` then `kasumi taste`)".into(),
            Scorer::Embed { taste_from_sidecar: true, embeddings, .. } => format!("image embeddings{}", if *embeddings { ", novelty on embeddings" } else { "" }),
            Scorer::Embed { builtin: Some(t), .. } => format!("features (held-out auc {:.2}), novelty on embeddings", t.auc),
            Scorer::Embed { .. } => "off, novelty on image embeddings".into(),
        }
    }

    /// One taste score per (features, pixels, id), plus the image embedding
    /// when novelty is measured on embeddings.
    fn scores(&mut self, items: &[(&kasumi_explore::features::Feat, &[u32], &str)]) -> Result<Vec<(f32, Option<Vec<f32>>)>> {
        match self {
            Scorer::Builtin(t) => Ok(items.iter().map(|(f, _, _)| (t.as_ref().map(|m| m.score(&taste_vec(f))).unwrap_or(0.5), None)).collect()),
            Scorer::Embed { stdin, stdout, dir, taste_from_sidecar, builtin, embeddings, .. } => {
                if items.is_empty() {
                    return Ok(vec![]);
                }
                let mut req = String::new();
                for (_, px, id) in items {
                    let p = dir.join(format!("{id}.png"));
                    write_png_rgb8(&p, THUMB, THUMB, &u32_to_rgb8(px))?;
                    req.push_str(&format!("{}\n", std::fs::canonicalize(&p)?.display()));
                }
                req.push('\n'); // blank line: end of batch
                stdin.write_all(req.as_bytes())?;
                stdin.flush()?;
                let mut out = Vec::with_capacity(items.len());
                for (f, _, _) in items {
                    let mut line = String::new();
                    std::io::BufRead::read_line(stdout, &mut line)?;
                    let mut parts = line.split_whitespace();
                    let side: f32 = parts.next().context("empty line from taste sidecar")?.parse().context("bad score from taste sidecar")?;
                    let t = if *taste_from_sidecar { side } else { builtin.as_ref().map(|m| m.score(&taste_vec(f))).unwrap_or(0.5) };
                    let e = if *embeddings { Some(parts.map(|x| x.parse::<f32>()).collect::<Result<Vec<f32>, _>>().context("bad embedding from taste sidecar")?) } else { None };
                    out.push((t, e));
                }
                Ok(out)
            }
        }
    }
}

impl Drop for Scorer {
    fn drop(&mut self) {
        if let Scorer::Embed { child, .. } = self {
            let _ = child.kill();
        }
    }
}

pub fn evolve(args: &Args) -> Result<()> {
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let minutes = args.f32_or("minutes", 480.0)?;
    let max_gens = args.u32_or("gens", u32::MAX)?;
    let batch = args.u32_or("batch", 24)? as usize;
    let pts = args.count_or("points", 1_500_000)?;
    let top = args.u32_or("top", 100)? as usize;
    let novelty_w = args.f32_or("novelty", 0.6)?;
    let out = PathBuf::from(args.get("out").map(String::from).unwrap_or(format!("runs/{}", timestamp())));
    std::fs::create_dir_all(out.join("forms"))?;
    let mut taste = Scorer::new(args.get("taste").unwrap_or("features"), args.get("novelty-on").unwrap_or("features"), &out)?;
    let mut rng = Rng::from_time();
    let mut scratch = Target::new(&gpu, THUMB, THUMB, 1, 0)?;
    let mut archive = Archive::new();
    let mut log = std::fs::File::create(out.join("log.txt"))?;

    println!(
        "kasumi evolve -> {} | {} min | batch {} | {} pts/thumb | taste model: {}",
        out.display(),
        minutes,
        batch,
        human(pts),
        taste.describe()
    );

    // generation 0: the seeds
    let seeds = read_seeds(&args.pos)?;
    anyhow::ensure!(!seeds.is_empty(), "no seed forms");
    let mut seed_evals = Vec::new();
    for l in compile(&gpu, seeds.into_iter().map(Cand::new).collect()) {
        let (f, px) = evaluate(&gpu, &st, &mut scratch, &l, pts, 2.0)?;
        println!(
            "  seed {} {} | lit {:.2} fill {:.2} aspect {:.2} extent {:.2} edge {:.2} chroma {:.2} structure {:.2} motion {:.4} | {}",
            l.cand.form.header.first().map(|h| h.split(':').next().unwrap_or("").trim().to_string()).unwrap_or_default(),
            l.cand.id,
            f.lit,
            f.fill,
            f.aspect,
            f.extent,
            f.edge,
            f.chroma,
            f.structure,
            f.motion,
            junk(&f).map(|j| format!("{j:?}")).unwrap_or("ok".into())
        );
        seed_evals.push((l, f, px));
    }
    offer_all(&mut archive, &mut taste, seed_evals, 0, novelty_w)?;

    archive.rescore(novelty_w);
    let started = Instant::now();
    let (mut evaluated, mut junked, mut kept) = (0usize, 0usize, 0usize);
    let mut gen = 0u32;
    while started.elapsed().as_secs_f32() < minutes * 60.0 && gen < max_gens {
        gen += 1;
        let elites: Vec<Elite> = archive.elites().cloned().collect();
        let mut cands = Vec::with_capacity(batch);
        let mut guard = 0;
        while cands.len() < batch && guard < batch * 10 {
            guard += 1;
            // tournament of 3 for the parent, any elite as crossover donor
            let mut p = &elites[rng.below(elites.len())];
            for _ in 0..2 {
                let q = &elites[rng.below(elites.len())];
                if q.score > p.score {
                    p = q;
                }
            }
            let donor = &elites[rng.below(elites.len())];
            let (Ok(pf), Ok(df)) = (kasumi_lang::parse(&p.text), kasumi_lang::parse(&donor.text)) else { continue };
            if let Some(m) = mutate(&pf, &mut rng, Some(&df)) {
                cands.push(Cand::new(m));
            }
        }
        let mut good = Vec::new();
        for l in compile(&gpu, cands) {
            evaluated += 1;
            let (f, px) = evaluate(&gpu, &st, &mut scratch, &l, pts, 2.0)?;
            if junk(&f).is_some() {
                junked += 1;
                continue;
            }
            good.push((l, f, px));
        }
        kept += offer_all(&mut archive, &mut taste, good, gen, novelty_w)?;
        let el = started.elapsed().as_secs_f32();
        let line = format!(
            "gen {gen:>5} | {:>6.1} min | evaluated {evaluated} | junk {:.0}% | archive {}/{} cells | improvements {kept} | best {:.3}",
            el / 60.0,
            junked as f64 / evaluated.max(1) as f64 * 100.0,
            archive.len(),
            archive.cells.len(),
            archive.ranked().first().map(|e| e.score).unwrap_or(0.0)
        );
        print!("\r{line}   ");
        std::io::stdout().flush().ok();
        writeln!(log, "{line}")?;
        archive.rescore(novelty_w);
        if gen % 25 == 0 {
            save_archive(&archive, &out)?;
        }
    }
    println!();
    save_archive(&archive, &out)?;
    gallery_from(&gpu, &st, &archive, &out, top, GALLERY_SPACING)?;
    Ok(())
}

/// Score a batch for taste (one sidecar round trip), then novelty, and offer
/// each to the archive. Returns how many were kept.
fn offer_all(archive: &mut Archive, taste: &mut Scorer, evals: Vec<(Live, kasumi_explore::features::Feat, Vec<u32>)>, gen: u32, novelty_w: f32) -> Result<usize> {
    let items: Vec<(&kasumi_explore::features::Feat, &[u32], &str)> = evals.iter().map(|(l, f, px)| (f, px.as_slice(), l.cand.id.as_str())).collect();
    let ts = taste.scores(&items)?;
    let mut kept = 0;
    for ((l, f, _), (t, emb)) in evals.iter().zip(ts) {
        let vec = emb.unwrap_or_else(|| f.vec.clone());
        let n = archive.novelty(&vec);
        let e = Elite { id: l.cand.id.clone(), text: l.cand.text.clone(), score: t + novelty_w * n, taste: t, novelty: n, lit: f.lit, spread: f.spread, generation: gen, vec };
        if archive.offer(e, f) {
            kept += 1;
        }
    }
    Ok(kept)
}

fn save_archive(a: &Archive, out: &Path) -> Result<()> {
    for e in a.elites() {
        std::fs::write(out.join("forms").join(format!("{}.ks", e.id)), &e.text)?;
    }
    let ranked: Vec<&Elite> = a.ranked();
    std::fs::write(out.join("archive.json"), serde_json::to_string_pretty(&ranked)?)?;
    Ok(())
}

/// Render looping sprite strips for the best forms and write the gallery page.
fn gallery_from(gpu: &Gpu, st: &Statics, a: &Archive, out: &Path, top: usize, spacing: f32) -> Result<()> {
    let ranked = a.diverse(top, spacing);
    let dir = out.join("gallery");
    std::fs::create_dir_all(dir.join("sprites"))?;
    const FRAMES: usize = 32;
    const S: u32 = 256;
    let mut tgt = Target::new(gpu, S, S, 2, 0)?;
    let mut cards = String::new();
    for (k, e) in ranked.iter().enumerate() {
        print!("\rgallery {}/{}  ", k + 1, ranked.len());
        std::io::stdout().flush().ok();
        let Ok(form) = kasumi_lang::parse(&e.text) else { continue };
        let lives = compile(gpu, vec![Cand::new(form)]);
        let Some(l) = lives.first() else { continue };
        let mut strip = vec![0u8; (S as usize * FRAMES) * S as usize * 3];
        tgt.reset_exposure();
        for f in 0..FRAMES {
            let px = render_frame(gpu, st, &mut tgt, l, 6_000_000, 1.0 + f as f32 / 12.0, true)?;
            for y in 0..S as usize {
                for x in 0..S as usize {
                    let p = px[y * S as usize + x];
                    let o = (y * S as usize * FRAMES + f * S as usize + x) * 3;
                    strip[o] = (p >> 16) as u8;
                    strip[o + 1] = (p >> 8) as u8;
                    strip[o + 2] = p as u8;
                }
            }
        }
        write_png_rgb8(&dir.join("sprites").join(format!("{}.png", e.id)), S * FRAMES as u32, S, &strip)?;
        cards.push_str(&format!(
            "<figure><div class=\"clip\" style=\"background-image:url(sprites/{id}.png)\"></div><figcaption><span>#{rank} &middot; {id}</span><span>score {s:.2} &middot; gen {g}</span></figcaption><details><summary>formula</summary><pre>{text}</pre></details></figure>\n",
            id = e.id,
            rank = k + 1,
            s = e.score,
            g = e.generation,
            text = html_escape(&e.text)
        ));
    }
    println!();
    let html = GALLERY.replace("{{COUNT}}", &ranked.len().to_string()).replace("{{CARDS}}", &cards);
    std::fs::write(dir.join("index.html"), html)?;
    println!("gallery: {}", dir.join("index.html").display());
    Ok(())
}

pub fn gallery(args: &Args) -> Result<()> {
    let run = PathBuf::from(args.pos.first().context("usage: kasumi gallery <runs/dir>")?);
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let elites: Vec<Elite> = serde_json::from_str(&std::fs::read_to_string(run.join("archive.json"))?)?;
    let mut scratch = Target::new(&gpu, THUMB, THUMB, 1, 0)?;
    let mut a = Archive::new();
    for (k, mut e) in elites.into_iter().enumerate() {
        // descriptors are not saved with the archive, so measure each form again
        let Ok(form) = kasumi_lang::parse(&e.text) else { continue };
        let lives = compile(&gpu, vec![Cand::new(form)]);
        let Some(l) = lives.first() else { continue };
        let (f, _) = evaluate(&gpu, &st, &mut scratch, l, 1_500_000, 2.0)?;
        if junk(&f).is_some() {
            continue;
        }
        e.vec = f.vec;
        let n = a.cells.len();
        a.cells[k % n] = Some(e);
    }
    a.rescore(args.f32_or("novelty", 0.6)?);
    gallery_from(&gpu, &st, &a, &run, args.u32_or("top", 100)? as usize, args.f32_or("spacing", GALLERY_SPACING)?)
}

pub fn taste_cmd(_args: &Args) -> Result<()> {
    let clicks = kasumi_explore::read_clicks(Path::new("explore/clicks.jsonl"));
    let data: Vec<(Vec<f32>, bool)> = clicks.iter().map(|c| (c.feat.clone(), c.liked)).collect();
    let likes = data.iter().filter(|d| d.1).count();
    println!("{} judgements ({} likes, {} skips)", data.len(), likes, data.len() - likes);
    match Taste::train(&data, 42) {
        Some(t) => {
            println!(
                "held-out ranking accuracy (auc): {:.1}%  -> {}",
                t.auc * 100.0,
                if t.auc >= 0.75 { "meets the 75% bar" } else { "below the 75% bar; keep clicking (target ~500)" }
            );
            std::fs::write("explore/taste.json", serde_json::to_string(&t)?)?;
            println!("saved explore/taste.json (used by grid and evolve)");
        }
        None => println!("need at least 4 likes and 4 skips; open `kasumi grid <form>` and click"),
    }
    Ok(())
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

const GALLERY: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>kasumi gallery</title>
<style>
:root{--bg:#070707;--fg:#d8d4cc;--dim:#77736c;--line:#1c1b19}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{padding:40px 24px 8px;max-width:1400px;margin:auto}h1{font-weight:400;letter-spacing:.3em;margin:0 0 6px}
header p{color:var(--dim);margin:0}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(256px,1fr));gap:20px;padding:24px;max-width:1400px;margin:auto}
figure{margin:0;border:1px solid var(--line);background:#000}
.clip{width:100%;aspect-ratio:1;background-size:3200% 100%;background-repeat:no-repeat;animation:play 2.6s steps(31) infinite alternate}
@keyframes play{from{background-position:0 0}to{background-position:100% 0}}
figcaption{display:flex;justify-content:space-between;padding:8px 10px;color:var(--dim);font-size:12px}
details{padding:0 10px 10px}summary{cursor:pointer;color:var(--dim);font-size:12px}
pre{white-space:pre-wrap;font-size:11px;color:var(--fg);margin:8px 0 0}
</style></head><body>
<header><h1>kasumi &middot; 霞</h1><p>{{COUNT}} forms from an overnight run. every frame is millions of points, every form is a few lines of math.</p></header>
<main>
{{CARDS}}
</main></body></html>
"#;
