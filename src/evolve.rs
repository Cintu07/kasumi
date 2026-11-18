//! Evolution: breed forms, keep the best per style in the archive, and write a
//! gallery page at the end.

use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use anyhow::{ensure, Context, Result};

use crate::archive::{family, Archive, Elite, GALLERY_SPACING};
use crate::cli::*;
use crate::explore::*;
use crate::features::{junk, Feat};
use crate::mutate::mutate;
use crate::parser::parse;
use crate::render::{Engine, Format, Program, Target};
use crate::rng::Rng;

const THUMB: u32 = 192;
const THUMB_POINTS: u64 = 1_500_000;
const GALLERY_PAGE: &str = include_str!("gallery.html");

/// `taste/clip.py serve`, which scores thumbnails a batch at a time.
struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    thumbs: PathBuf,
    use_taste: bool,
    use_embeddings: bool,
}

impl Sidecar {
    fn start(use_taste: bool, use_embeddings: bool, out: &Path) -> Result<Sidecar> {
        let python = if cfg!(windows) { ".venv/Scripts/python.exe" } else { ".venv/bin/python" };
        ensure!(Path::new(python).is_file(), "the taste model needs the python venv ({python}); see the readme");
        let trained = Path::new("explore/taste_embed.pkl").is_file();
        ensure!(!use_taste || trained, "no taste model yet: run `{python} taste/clip.py train`");
        let mut command = Command::new(python);
        command.args(["taste/clip.py", "serve"]).stdin(Stdio::piped()).stdout(Stdio::piped());
        if use_embeddings {
            command.arg("--embed");
        }
        let mut child = command.spawn().context("starting taste/clip.py")?;
        let stdin = child.stdin.take().context("sidecar stdin")?;
        let mut stdout = BufReader::new(child.stdout.take().context("sidecar stdout")?);
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        ensure!(line.trim() == "ready", "taste/clip.py failed to start");
        let thumbs = out.join("thumbs");
        std::fs::create_dir_all(&thumbs)?;
        Ok(Sidecar { child, stdin, stdout, thumbs, use_taste, use_embeddings })
    }

    /// A taste score and an optional image embedding for each (id, pixels).
    fn score(&mut self, items: &[(&str, &[u32])]) -> Result<Vec<Score>> {
        if items.is_empty() {
            return Ok(vec![]);
        }
        let mut request = String::new();
        for (id, px) in items {
            let path = self.thumbs.join(format!("{id}.png"));
            write_png(&path, THUMB, THUMB, &to_rgb(px))?;
            request.push_str(&format!("{}\n", std::fs::canonicalize(&path)?.display()));
        }
        request.push('\n'); // a blank line ends the batch
        self.stdin.write_all(request.as_bytes())?;
        self.stdin.flush()?;
        let mut scores = Vec::with_capacity(items.len());
        for (id, _) in items {
            let mut line = String::new();
            self.stdout.read_line(&mut line)?;
            std::fs::remove_file(self.thumbs.join(format!("{id}.png"))).ok();
            let mut fields = line.split_whitespace().map(str::parse::<f32>);
            let taste = fields.next().context("empty line from taste/clip.py")??;
            let embedding = if self.use_embeddings { Some(fields.collect::<Result<Vec<f32>, _>>()?) } else { None };
            scores.push(Score { taste: if self.use_taste { taste } else { NO_TASTE }, embedding });
        }
        Ok(scores)
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.child.kill().ok();
    }
}

struct Score {
    taste: f32,
    embedding: Option<Vec<f32>>,
}

/// The score of every form when there is no taste model.
const NO_TASTE: f32 = 0.5;

type Measured = (Compiled, Feat, Vec<u32>);

/// Score a batch, then offer each form to the archive. Returns how many were kept.
fn offer_all(archive: &mut Archive, sidecar: &mut Option<Sidecar>, batch: Vec<Measured>, generation: u32, novelty_weight: f32) -> Result<usize> {
    let items: Vec<(&str, &[u32])> = batch.iter().map(|(c, _, px)| (c.program.id.as_str(), px.as_slice())).collect();
    let scores = match sidecar {
        Some(s) => s.score(&items)?,
        None => items.iter().map(|_| Score { taste: NO_TASTE, embedding: None }).collect(),
    };
    let mut kept = 0;
    for ((c, f, _), score) in batch.iter().zip(scores) {
        let vec = score.embedding.unwrap_or_else(|| f.vec.clone());
        let novelty = archive.novelty(&vec);
        let elite = Elite {
            id: c.program.id.clone(),
            text: c.program.text.clone(),
            score: score.taste + novelty_weight * novelty,
            taste: score.taste,
            novelty,
            lit: f.lit,
            spread: f.spread,
            generation,
            vec,
        };
        kept += archive.offer(elite, f) as usize;
    }
    Ok(kept)
}

/// Every seed family breeds equally often, or the strongest takes over the run.
/// The parent wins a tournament of 3 inside its family; any elite can donate.
fn breed(elites: &[Elite], rng: &mut Rng, want: usize) -> Vec<Program> {
    let mut families: Vec<&str> = elites.iter().map(|e| family(&e.text)).collect();
    families.sort_unstable();
    families.dedup();
    let mut children = Vec::with_capacity(want);
    for _ in 0..want * 10 {
        if children.len() == want {
            break;
        }
        let name = families[rng.below(families.len())];
        let pool: Vec<&Elite> = elites.iter().filter(|e| family(&e.text) == name).collect();
        let mut parent = pool[rng.below(pool.len())];
        for _ in 0..2 {
            let rival = pool[rng.below(pool.len())];
            if rival.score > parent.score {
                parent = rival;
            }
        }
        let donor = &elites[rng.below(elites.len())];
        let (Ok(form), Ok(donor)) = (parse(&parent.text), parse(&donor.text)) else { continue };
        if let Some(child) = mutate(&form, rng, Some(&donor)) {
            children.push(Program::new(child));
        }
    }
    children
}

fn save_archive(archive: &Archive, out: &Path) -> Result<()> {
    for e in archive.elites() {
        std::fs::write(out.join("forms").join(format!("{}.ks", e.id)), &e.text)?;
    }
    std::fs::write(out.join("archive.json"), serde_json::to_string_pretty(&archive.ranked())?)?;
    Ok(())
}

pub fn evolve(args: &Args) -> Result<()> {
    let engine = Engine::new()?;
    let minutes = args.f32_or("minutes", 480.0)?;
    let max_generations = args.u32_or("gens", u32::MAX)?;
    let batch = args.u32_or("batch", 24)? as usize;
    let top = args.u32_or("top", 100)? as usize;
    let novelty_weight = args.f32_or("novelty", 0.6)?;
    let out = args.get("out").map_or(PathBuf::from(format!("runs/{}", timestamp())), PathBuf::from);
    std::fs::create_dir_all(out.join("forms"))?;
    let use_taste = args.get("taste").is_some();
    let use_embeddings = args.get("novelty-on") == Some("embed");
    let mut sidecar = if use_taste || use_embeddings { Some(Sidecar::start(use_taste, use_embeddings, &out)?) } else { None };
    let mut rng = Rng::from_time();
    let mut scratch = Target::new(&engine.gpu, THUMB, THUMB, 1, Format::Window)?;
    let mut archive = Archive::new();
    let mut log = std::fs::File::create(out.join("log.txt"))?;
    println!("kasumi evolve -> {} | {minutes} min | batch {batch} | taste {use_taste} | embeddings {use_embeddings}", out.display());

    let seeds = read_seeds(&args.pos)?;
    ensure!(!seeds.is_empty(), "no seed forms");
    let mut measured = Vec::new();
    for c in compile_batch(&engine, seeds) {
        let (f, px) = evaluate(&engine, &mut scratch, &c, THUMB_POINTS)?;
        measured.push((c, f, px));
    }
    offer_all(&mut archive, &mut sidecar, measured, 0, novelty_weight)?;
    archive.rescore(novelty_weight);

    let started = Instant::now();
    let (mut evaluated, mut junked, mut kept) = (0, 0, 0);
    let mut generation = 0;
    while started.elapsed().as_secs_f32() < minutes * 60.0 && generation < max_generations {
        generation += 1;
        let elites: Vec<Elite> = archive.elites().cloned().collect();
        let mut good = Vec::new();
        for c in compile_batch(&engine, breed(&elites, &mut rng, batch)) {
            evaluated += 1;
            let (f, px) = evaluate(&engine, &mut scratch, &c, THUMB_POINTS)?;
            if junk(&f).is_some() {
                junked += 1;
            } else {
                good.push((c, f, px));
            }
        }
        kept += offer_all(&mut archive, &mut sidecar, good, generation, novelty_weight)?;
        let line = format!(
            "gen {generation:>5} | {:>6.1} min | evaluated {evaluated} | junk {:.0}% | archive {}/{} cells | improvements {kept} | best {:.3}",
            started.elapsed().as_secs_f32() / 60.0,
            junked as f64 / evaluated.max(1) as f64 * 100.0,
            archive.len(),
            archive.cells.len(),
            archive.ranked().first().map_or(0.0, |e| e.score)
        );
        print!("\r{line}   ");
        std::io::stdout().flush().ok();
        writeln!(log, "{line}")?;
        archive.rescore(novelty_weight);
        if generation % 25 == 0 {
            save_archive(&archive, &out)?;
        }
    }
    println!();
    save_archive(&archive, &out)?;
    write_gallery(&engine, &archive, &out, top, GALLERY_SPACING)
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Render a looping sprite strip for each of the best forms and write the page.
fn write_gallery(engine: &Engine, archive: &Archive, out: &Path, top: usize, spacing: f32) -> Result<()> {
    const FRAMES: usize = 32;
    const SIZE: usize = 256;
    let dir = out.join("gallery");
    std::fs::create_dir_all(dir.join("sprites"))?;
    let mut target = Target::new(&engine.gpu, SIZE as u32, SIZE as u32, 2, Format::Window)?;
    let picks = archive.diverse(top, spacing);
    let mut cards = String::new();
    for (k, elite) in picks.iter().enumerate() {
        print!("\rgallery {}/{}  ", k + 1, picks.len());
        std::io::stdout().flush().ok();
        let Ok(form) = parse(&elite.text) else { continue };
        let Some(c) = compile_batch(engine, vec![Program::new(form)]).pop() else { continue };
        let mut strip = vec![0u8; SIZE * FRAMES * SIZE * 3];
        target.reset_exposure();
        for f in 0..FRAMES {
            let px = draw(engine, &mut target, &c, 6_000_000, 1.0 + f as f32 / 12.0, true)?;
            for (y, row) in px.chunks_exact(SIZE).enumerate() {
                let start = (y * FRAMES + f) * SIZE * 3;
                strip[start..start + SIZE * 3].copy_from_slice(&to_rgb(row));
            }
        }
        write_png(&dir.join("sprites").join(format!("{}.png", elite.id)), (SIZE * FRAMES) as u32, SIZE as u32, &strip)?;
        cards.push_str(&format!(
            "<figure><div class=\"clip\" style=\"background-image:url(sprites/{id}.png)\"></div><figcaption><span>#{rank} &middot; {id}</span><span>score {score:.2} &middot; gen {generation}</span></figcaption><details><summary>formula</summary><pre>{text}</pre></details></figure>\n",
            id = elite.id,
            rank = k + 1,
            score = elite.score,
            generation = elite.generation,
            text = html_escape(&elite.text)
        ));
    }
    println!();
    let page = GALLERY_PAGE.replace("{{COUNT}}", &picks.len().to_string()).replace("{{CARDS}}", &cards);
    std::fs::write(dir.join("index.html"), page)?;
    println!("gallery: {}", dir.join("index.html").display());
    Ok(())
}

/// Pick a finished run's gallery again.
pub fn gallery(args: &Args) -> Result<()> {
    let run = args.path(0, "kasumi gallery <runs/dir>")?;
    let engine = Engine::new()?;
    let elites: Vec<Elite> = serde_json::from_str(&std::fs::read_to_string(run.join("archive.json"))?)?;
    let mut scratch = Target::new(&engine.gpu, THUMB, THUMB, 1, Format::Window)?;
    let mut archive = Archive::new();
    for (k, mut elite) in elites.into_iter().enumerate() {
        // descriptors are not saved with the archive, so measure each form again
        let Ok(form) = parse(&elite.text) else { continue };
        let Some(c) = compile_batch(&engine, vec![Program::new(form)]).pop() else { continue };
        let (f, _) = evaluate(&engine, &mut scratch, &c, THUMB_POINTS)?;
        if junk(&f).is_none() {
            elite.vec = f.vec;
            let cells = archive.cells.len();
            archive.cells[k % cells] = Some(elite);
        }
    }
    archive.rescore(args.f32_or("novelty", 0.6)?);
    write_gallery(&engine, &archive, &run, args.u32_or("top", 100)? as usize, args.f32_or("spacing", GALLERY_SPACING)?)
}
