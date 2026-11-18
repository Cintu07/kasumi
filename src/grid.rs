//! The judgement screen: 4 x 4 variations of a form next to a panel of
//! measurements. Left click selects a sample, right click breeds from it, and
//! every selection or skip is logged for the taste model.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};
use serde_json::json;

use crate::archive::family;
use crate::cli::*;
use crate::explore::*;
use crate::features::{dist, junk, Feat};
use crate::mutate::mutate;
use crate::render::{Engine, Format, Program, Target};
use crate::rng::Rng;
use crate::ui::Canvas;

const TILE: usize = 192;
const N: usize = 4;
const GRID: usize = TILE * N;
const PANEL: usize = 340;
const WIDTH: usize = GRID + PANEL;
const HEIGHT: usize = GRID;
const CLICKS: &str = "explore/clicks.jsonl";
/// Seconds a batch must stay up before its verdict counts.
const MIN_LOOK: f32 = 2.0;
/// Tiles closer than this, relative to how far the batch strays from its
/// parent, count as the same picture.
const MIN_TILE_GAP: f32 = 0.3;

const PANEL_BG: u32 = 0x101012;
const INK: u32 = 0xD4D4D0;
const MUTED: u32 = 0x7C7C7A;
const RULE: u32 = 0x2C2C30;
const ACCENT: u32 = 0x7FB2D9;
const PX: i32 = GRID as i32 + 20;
const PW: i32 = PANEL as i32 - 40;
const ROW: i32 = 21;

#[derive(Clone, Copy)]
struct Button(i32, i32, i32, i32);

impl Button {
    fn contains(self, x: f32, y: f32) -> bool {
        x >= self.0 as f32 && x < self.2 as f32 && y >= self.1 as f32 && y < self.3 as f32
    }
}

const NEXT: Button = Button(PX, 612, PX + PW, 668);
const FINISH: Button = Button(PX, 682, PX + PW, 738);

/// What has been logged so far.
struct Dataset {
    samples: usize,
    selected: usize,
}

fn read_dataset() -> Dataset {
    let text = std::fs::read_to_string(CLICKS).unwrap_or_default();
    let rows: Vec<serde_json::Value> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    Dataset { samples: rows.len(), selected: rows.iter().filter(|r| r["liked"] == true).count() }
}

struct Litter {
    tiles: Vec<Compiled>,
    features: Vec<Feat>,
    liked: Vec<bool>,
    born: Instant,
    tested: usize,
    rejected: usize,
}

impl Litter {
    fn ready(&self) -> bool {
        self.born.elapsed().as_secs_f32() >= MIN_LOOK
    }

    fn selected(&self) -> usize {
        self.liked.iter().skip(1).filter(|&&l| l).count()
    }
}

/// Up to `want` forms, each visibly different from the parent and from every form picked so far.
fn pick_different(vectors: &[Vec<f32>], parent: &[f32], want: usize) -> Vec<usize> {
    let mut spreads: Vec<f32> = vectors.iter().map(|v| dist(v, parent)).collect();
    spreads.sort_by(f32::total_cmp);
    // the upper quartile, because a batch can hold many near copies of the parent
    let scale = spreads.get(spreads.len() * 3 / 4).copied().unwrap_or(1.0).max(1e-3);
    let mut chosen: Vec<&[f32]> = vec![parent];
    let mut picks = Vec::new();
    while picks.len() < want {
        let gap = |v: &[f32]| chosen.iter().map(|c| dist(c, v)).fold(f32::MAX, f32::min) / scale;
        let open = (0..vectors.len()).filter(|j| !picks.contains(j) && gap(&vectors[*j]) >= MIN_TILE_GAP);
        let Some(best) = open.max_by(|a, b| gap(&vectors[*a]).min(2.0).total_cmp(&gap(&vectors[*b]).min(2.0))) else { break };
        chosen.push(&vectors[best]);
        picks.push(best);
    }
    picks
}

/// The parent in the first tile, then 15 different mutants that are not junk.
fn breed(e: &Engine, scratch: &mut Target, parent: &Program, rng: &mut Rng, points: u64) -> Result<Litter> {
    let mut tiles = compile_batch(e, vec![parent.clone()]);
    let (parent_features, _) = evaluate(e, scratch, &tiles[0], points)?;
    let (mut kept, mut vectors) = (Vec::new(), Vec::new());
    let (mut tested, mut rejected) = (0, 0);
    let mut picks = Vec::new();
    for round in 0..6 {
        let want = if round == 0 { 28 } else { 20 };
        let mut batch = Vec::new();
        while batch.len() < want {
            // chain 1 to 3 mutations so a batch spans more than small nudges
            let mut form = parent.form.clone();
            for _ in 0..1 + rng.below(3) {
                form = mutate(&form, rng, None).unwrap_or(form);
            }
            if form != parent.form {
                batch.push(Program::new(form));
            }
        }
        for c in compile_batch(e, batch) {
            tested += 1;
            let (f, _) = evaluate(e, scratch, &c, points)?;
            if junk(&f).is_some() {
                rejected += 1;
            } else {
                vectors.push(f.vec.clone());
                kept.push(Some((c, f)));
            }
        }
        picks = pick_different(&vectors, &parent_features.vec, N * N - 1);
        if picks.len() == N * N - 1 {
            break;
        }
    }
    let mut features = vec![parent_features];
    for (c, f) in picks.into_iter().filter_map(|j| kept[j].take()) {
        tiles.push(c);
        features.push(f);
    }
    let liked = vec![false; tiles.len()];
    Ok(Litter { tiles, features, liked, born: Instant::now(), tested, rejected })
}

/// Log a batch. Skips count only when you moved on from it or selected something in it.
fn log_litter(e: &Engine, scratch: &mut Target, litter: &Litter, points: u64, log: &mut File, data: &mut Dataset, moved_on: bool) -> Result<()> {
    if !litter.ready() || !(moved_on || litter.selected() > 0) {
        return Ok(());
    }
    std::fs::create_dir_all("explore/thumbs")?;
    for (c, liked) in litter.tiles.iter().zip(&litter.liked).skip(1) {
        let (_, px) = evaluate(e, scratch, c, points)?;
        let thumb = format!("explore/thumbs/{}.png", c.program.id);
        write_png(Path::new(&thumb), scratch.w, scratch.h, &to_rgb(&px))?;
        let click = json!({ "id": c.program.id, "liked": liked, "form": c.program.text, "thumb": thumb, "ts": timestamp() });
        writeln!(log, "{click}")?;
    }
    data.samples += litter.tiles.len() - 1;
    data.selected += litter.selected();
    Ok(())
}

fn centred(c: &mut Canvas, b: Button, y: i32, s: &str, color: u32, scale: i32) {
    c.text(b.0 + (b.2 - b.0 - s.len() as i32 * 8 * scale) / 2, y, s, color, scale);
}

fn heading(c: &mut Canvas, y: i32, title: &str) {
    c.text(PX, y, title, MUTED, 1);
    c.fill(PX, y + 12, PX + PW, y + 13, RULE);
}

/// A label on the left and its value lined up on the right.
fn row(c: &mut Canvas, y: i32, key: &str, value: &str) {
    c.text(PX, y, key, MUTED, 2);
    c.text(PX + PW - value.len() as i32 * 16, y, value, INK, 2);
}

fn button(c: &mut Canvas, b: Button, label: &str, enabled: bool, hovered: bool) {
    let line = if !enabled { RULE } else if hovered { ACCENT } else { MUTED };
    c.outline(b.0, b.1, b.2, b.3, line, 1);
    centred(c, b, b.1 + 20, label, if enabled { INK } else { MUTED }, 2);
}

struct Panel<'a> {
    data: &'a Dataset,
    batch: usize,
    litter: Option<&'a Litter>,
    hover: Option<usize>,
    over_next: bool,
    over_finish: bool,
    status: &'a str,
}

fn draw_panel(c: &mut Canvas, p: &Panel) {
    c.fill(GRID as i32, 0, WIDTH as i32, HEIGHT as i32, PANEL_BG);
    c.fill(GRID as i32, 0, GRID as i32 + 1, HEIGHT as i32, RULE);
    c.text(PX, 16, "JUDGEMENT", INK, 2);
    let batch = format!("batch {:04}", p.batch);
    c.text(PX + PW - batch.len() as i32 * 8, 20, &batch, MUTED, 1);

    heading(c, 44, "DATASET");
    let rate = 100.0 * p.data.selected as f64 / p.data.samples.max(1) as f64;
    row(c, 62, "samples", &p.data.samples.to_string());
    row(c, 62 + ROW, "selected", &p.data.selected.to_string());
    row(c, 62 + 2 * ROW, "rate", &format!("{rate:.1}%"));

    heading(c, 134, "BATCH");
    let count = |f: fn(&Litter) -> usize| p.litter.map_or("-".to_string(), |l| f(l).to_string());
    row(c, 152, "tested", &count(|l| l.tested));
    row(c, 152 + ROW, "rejected", &count(|l| l.rejected));
    row(c, 152 + 2 * ROW, "shown", &count(|l| l.tiles.len() - 1));
    row(c, 152 + 3 * ROW, "selected", &count(Litter::selected));

    let sample = p.hover.zip(p.litter);
    let title = p.hover.map_or("SAMPLE".to_string(), |k| if k == 0 { "SAMPLE REF".to_string() } else { format!("SAMPLE {k:02}") });
    heading(c, 245, &title);
    let readings = sample.map(|(k, l)| {
        let f = &l.features[k];
        let name = family(&l.tiles[k].program.text);
        [l.tiles[k].program.id.clone(), name.to_string(), format!("{:.3}", f.lit), format!("{:.3}", f.spread), format!("{:.4}", f.motion), format!("{:.3}", f.aspect), format!("{:.3}", f.chroma)]
    });
    for (n, key) in ["id", "family", "lit", "spread", "motion", "aspect", "chroma"].iter().enumerate() {
        row(c, 263 + n as i32 * ROW, key, readings.as_ref().map_or("-", |r| r[n].as_str()));
    }

    heading(c, 420, "CONTROLS");
    for (n, (key, action)) in [("l click", "select"), ("r click", "breed"), ("space", "next batch"), ("s", "save picks")].iter().enumerate() {
        row(c, 438 + n as i32 * ROW, key, action);
    }
    c.text(PX, 536, p.status, ACCENT, 1);

    let ready = p.litter.is_some_and(Litter::ready);
    button(c, NEXT, if p.litter.is_none() { "COMPUTING" } else { "NEXT BATCH" }, ready, p.over_next);
    button(c, FINISH, "SAVE AND QUIT", true, p.over_finish);
}

fn show_computing(window: &mut Window, buf: &mut [u32], data: &Dataset, batch: usize) -> Result<()> {
    let mut c = Canvas { px: buf, w: WIDTH, h: HEIGHT };
    c.fill(0, 0, GRID as i32, HEIGHT as i32, 0);
    centred(&mut c, Button(0, 0, GRID as i32, HEIGHT as i32), HEIGHT as i32 / 2 - 8, "computing batch", MUTED, 2);
    draw_panel(&mut c, &Panel { data, batch, litter: None, hover: None, over_next: false, over_finish: false, status: "" });
    window.update_with_buffer(buf, WIDTH, HEIGHT)?;
    Ok(())
}

fn draw_tile(c: &mut Canvas, k: usize, px: &[u32], selected: bool, hovered: bool) {
    let (left, top) = ((k % N) * TILE, (k / N) * TILE);
    for y in 0..TILE {
        c.px[(top + y) * WIDTH + left..(top + y) * WIDTH + left + TILE].copy_from_slice(&px[y * TILE..(y + 1) * TILE]);
    }
    let (x0, y0, x1, y1) = (left as i32, top as i32, (left + TILE) as i32, (top + TILE) as i32);
    if k == 0 {
        c.dim(x0, y0, x1, y1);
    }
    c.label(x0 + 6, y0 + 6, &if k == 0 { "REF".to_string() } else { format!("{k:02}") }, MUTED, 1);
    if selected {
        c.outline(x0, y0, x1, y1, ACCENT, 2);
        c.fill(x1 - 14, y0 + 6, x1 - 6, y0 + 14, ACCENT);
    } else if hovered {
        c.outline(x0, y0, x1, y1, MUTED, 1);
    }
}

fn tile_at(x: f32, y: f32, tiles: usize) -> Option<usize> {
    if x < 0.0 || y < 0.0 || x >= GRID as f32 || y >= GRID as f32 {
        return None;
    }
    let k = (y as usize / TILE) * N + x as usize / TILE;
    (k < tiles).then_some(k)
}

fn save_liked(litter: &Litter) -> Result<String> {
    std::fs::create_dir_all("forms/kept")?;
    let liked: Vec<&Compiled> = litter.tiles.iter().zip(&litter.liked).filter(|(_, l)| **l).map(|(c, _)| c).collect();
    for c in &liked {
        std::fs::write(format!("forms/kept/{}.ks", c.program.id), &c.program.text)?;
    }
    Ok(format!("saved {} to forms/kept", liked.len()))
}

pub fn grid(args: &Args) -> Result<()> {
    let mut parent = Program::load(&args.path(0, "kasumi grid <form.ks>")?)?;
    let engine = Engine::new()?;
    let gpu = &engine.gpu;
    let points = args.count_or("points", 2_000_000)?;
    let mut rng = Rng::from_time();
    std::fs::create_dir_all("explore")?;
    let mut data = read_dataset();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(CLICKS)?;

    let size = TILE as u32;
    let mut scratch = Target::new(gpu, size, size, 2, Format::Window)?;
    let mut targets = (0..N * N).map(|_| Target::new(gpu, size, size, 2, Format::Window)).collect::<Result<Vec<_>>>()?;
    let mut buf = vec![0u32; WIDTH * HEIGHT];
    // a fixed size keeps mouse positions and tiles lined up
    let options = WindowOptions { resize: false, scale_mode: ScaleMode::UpperLeft, ..WindowOptions::default() };
    let mut window = Window::new("kasumi - judgement", WIDTH, HEIGHT, options)?;
    window.set_target_fps(60);

    let mut batch = 1;
    show_computing(&mut window, &mut buf, &data, batch)?;
    let mut litter = breed(&engine, &mut scratch, &parent, &mut rng, points)?;
    let start = Instant::now();
    let (mut was_left, mut was_right) = (false, false);
    let mut status = String::new();
    let mut mouse: Option<(f32, f32)> = None;
    let max_frames = args.get("frames").map(str::parse::<u64>).transpose()?;
    let mut frames = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = start.elapsed().as_secs_f32() + 1.0;
        let hover = mouse.and_then(|(x, y)| tile_at(x, y, litter.tiles.len()));
        let mut drawn = Vec::with_capacity(litter.tiles.len());
        for (k, c) in litter.tiles.iter().enumerate() {
            drawn.push(draw(&engine, &mut targets[k], c, points, t, true)?);
        }
        let mut canvas = Canvas { px: &mut buf, w: WIDTH, h: HEIGHT };
        canvas.fill(0, 0, GRID as i32, HEIGHT as i32, 0);
        for (k, px) in drawn.iter().enumerate() {
            draw_tile(&mut canvas, k, px, litter.liked[k], hover == Some(k));
        }
        let over = |b: Button| mouse.is_some_and(|(x, y)| b.contains(x, y));
        let panel = Panel { data: &data, batch, litter: Some(&litter), hover, over_next: over(NEXT), over_finish: over(FINISH), status: &status };
        draw_panel(&mut canvas, &panel);
        window.update_with_buffer(&buf, WIDTH, HEIGHT)?;

        frames += 1;
        if max_frames.is_some_and(|n| frames >= n) {
            if let Some(shot) = args.get("shot") {
                write_png(Path::new(shot), WIDTH as u32, HEIGHT as u32, &to_rgb(&buf))?;
            }
            return Ok(());
        }

        let (left, right) = (window.get_mouse_down(MouseButton::Left), window.get_mouse_down(MouseButton::Right));
        let (left_click, right_click) = (left && !was_left, right && !was_right);
        (was_left, was_right) = (left, right);
        mouse = window.get_mouse_pos(MouseMode::Discard);
        let mut next = false;
        let mut breed_from = None;
        if let Some((x, y)) = mouse {
            let tile = tile_at(x, y, litter.tiles.len()).filter(|&k| k > 0);
            if left_click {
                if let Some(k) = tile {
                    litter.liked[k] = !litter.liked[k];
                }
                if FINISH.contains(x, y) {
                    break;
                }
                next = NEXT.contains(x, y);
            }
            if right_click {
                breed_from = tile;
            }
        }
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => next = true,
                Key::S => status = save_liked(&litter)?,
                _ => {}
            }
        }
        if !litter.ready() {
            continue;
        }
        if let Some(k) = breed_from {
            litter.liked[k] = true; // breeding from a form counts as selecting it
            parent = litter.tiles[k].program.clone();
            next = true;
        }
        if next {
            batch += 1;
            show_computing(&mut window, &mut buf, &data, batch)?;
            log_litter(&engine, &mut scratch, &litter, points, &mut log, &mut data, true)?;
            litter = breed(&engine, &mut scratch, &parent, &mut rng, points)?;
            status.clear();
            targets.iter_mut().for_each(Target::reset_exposure);
        }
    }
    log_litter(&engine, &mut scratch, &litter, points, &mut log, &mut data, false)?;
    println!("{} judgements in {CLICKS}; run `python taste/clip.py train` to retrain", data.samples);
    Ok(())
}
