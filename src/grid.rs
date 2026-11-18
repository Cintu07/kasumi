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
use crate::ui::{text_width, Canvas};

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

const PANEL_BG: u32 = 0x0B0C0E;
const INK: u32 = 0xD6DAE0;
const MUTED: u32 = 0x737A84;
const POINT: u32 = 0x9AA3AF;
const EDGE: u32 = 0x3A3F47;
const RULE: u32 = 0x24282E;
const FAINT: u32 = 0x15171B;
const ACCENT: u32 = 0x8AB8E0;
const PX: i32 = GRID as i32 + 24;
const PW: i32 = PANEL as i32 - 48;
const PLOT_TOP: i32 = 326;
const PLOT_HEIGHT: i32 = 232;

#[derive(Clone, Copy)]
struct Button(i32, i32, i32, i32);

impl Button {
    fn contains(self, x: f32, y: f32) -> bool {
        x >= self.0 as f32 && x < self.2 as f32 && y >= self.1 as f32 && y < self.3 as f32
    }
}

const NEXT: Button = Button(PX, 648, PX + 140, 692);
const FINISH: Button = Button(PX + 152, 648, PX + PW, 692);

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

fn centred(c: &mut Canvas, b: Button, y: i32, s: &str, color: u32, size: f32) {
    c.text(b.0 + (b.2 - b.0 - text_width(s, size)) / 2, y, s, color, size);
}

fn right(c: &mut Canvas, y: i32, s: &str, color: u32, size: f32) {
    c.text(PX + PW - text_width(s, size), y, s, color, size);
}

fn rule(c: &mut Canvas, y: i32) {
    c.fill(PX, y, PX + PW, y + 1, RULE);
}

/// A small label with its value underneath.
fn stat(c: &mut Canvas, x: i32, y: i32, key: &str, value: &str) {
    c.text(x, y, key, MUTED, 10.5);
    c.strong(x, y + 15, value, INK, 19.0);
}

/// One measurement: its name, a hairline scale and the number.
fn reading(c: &mut Canvas, y: i32, key: &str, value: Option<(f32, String)>) {
    let (left, end) = (PX + 62, PX + PW - 58);
    c.text(PX, y, key, MUTED, 11.5);
    c.fill(left, y + 8, end, y + 9, RULE);
    match value {
        Some((share, number)) => {
            let at = left + ((end - left) as f32 * share.clamp(0.0, 1.0)) as i32;
            c.fill(left, y + 8, at, y + 9, ACCENT);
            c.fill(at, y + 5, at + 1, y + 12, ACCENT);
            right(c, y, &number, INK, 11.5);
        }
        None => right(c, y, "-", MUTED, 11.5),
    }
}

/// Where a sample sits on the style axes the archive uses: lit on a log scale, and spread.
fn feature_space(f: &Feat) -> (f32, f32) {
    let (low, high) = (0.02f32.ln(), 0.6f32.ln());
    (((f.lit.max(0.02).ln() - low) / (high - low)).clamp(0.0, 1.0), (f.spread / 0.7).clamp(0.0, 1.0))
}

fn draw_plot(c: &mut Canvas, p: &Panel) {
    let (x0, y0, x1, y1) = (PX, PLOT_TOP, PX + PW, PLOT_TOP + PLOT_HEIGHT);
    c.outline(x0, y0, x1, y1, RULE, 1);
    for k in 1..4 {
        let (x, y) = (x0 + (x1 - x0) * k / 4, y0 + (y1 - y0) * k / 4);
        c.fill(x, y0 + 1, x + 1, y1 - 1, FAINT);
        c.fill(x0 + 1, y, x1 - 1, y + 1, FAINT);
    }
    c.text(x0 + 6, y0 + 5, "spread", MUTED, 10.0);
    let lit = "lit (log)";
    c.text(x1 - text_width(lit, 10.0) - 6, y1 - 16, lit, MUTED, 10.0);
    let Some(l) = p.litter else { return };
    let spot = |f: &Feat| {
        let (u, v) = feature_space(f);
        (x0 as f32 + 8.0 + u * (PW as f32 - 16.0), y1 as f32 - 8.0 - v * (PLOT_HEIGHT as f32 - 16.0))
    };
    for (k, f) in l.features.iter().enumerate() {
        let (x, y) = spot(f);
        match (k == 0, l.liked[k]) {
            (true, _) => c.dot(x, y, 4.5, MUTED, true),
            (_, true) => c.dot(x, y, 3.5, ACCENT, false),
            _ => c.dot(x, y, 2.5, POINT, false),
        }
    }
    if let Some(k) = p.hover {
        let (x, y) = spot(&l.features[k]);
        c.dot(x, y, 7.0, INK, true);
    }
}

fn button(c: &mut Canvas, b: Button, label: &str, enabled: bool, hovered: bool) {
    let line = if !enabled { RULE } else if hovered { ACCENT } else { EDGE };
    c.outline(b.0, b.1, b.2, b.3, line, 1);
    centred(c, b, b.1 + 13, label, if enabled { INK } else { MUTED }, 13.0);
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
    for k in 1..N as i32 {
        c.fill(k * TILE as i32, 0, k * TILE as i32 + 1, GRID as i32, FAINT);
        c.fill(0, k * TILE as i32, GRID as i32, k * TILE as i32 + 1, FAINT);
    }
    c.fill(GRID as i32, 0, WIDTH as i32, HEIGHT as i32, PANEL_BG);
    c.fill(GRID as i32, 0, GRID as i32 + 1, HEIGHT as i32, RULE);
    c.strong(PX, 20, "kasumi", INK, 14.0);
    c.text(PX + text_width("kasumi ", 14.0), 20, "judgement", MUTED, 14.0);
    right(c, 22, &format!("batch {:04}", p.batch), MUTED, 11.0);
    rule(c, 46);

    let rate = 100.0 * p.data.selected as f64 / p.data.samples.max(1) as f64;
    stat(c, PX, 58, "samples", &p.data.samples.to_string());
    stat(c, PX + 100, 58, "selected", &p.data.selected.to_string());
    stat(c, PX + 200, 58, "rate", &format!("{rate:.1}%"));

    let count = |f: fn(&Litter) -> usize| p.litter.map_or("-".to_string(), |l| f(l).to_string());
    stat(c, PX, 110, "tested", &count(|l| l.tested));
    stat(c, PX + 75, 110, "rejected", &count(|l| l.rejected));
    stat(c, PX + 150, 110, "shown", &count(|l| l.tiles.len() - 1));
    stat(c, PX + 225, 110, "picked", &count(Litter::selected));
    rule(c, 160);

    let sample = p.hover.zip(p.litter);
    let title = match p.hover {
        Some(0) => "reference".to_string(),
        Some(k) => format!("sample {k:02}"),
        None => "no sample under cursor".to_string(),
    };
    c.strong(PX, 171, &title, if p.hover.is_some() { INK } else { MUTED }, 12.0);
    if let Some((k, l)) = sample {
        let program = &l.tiles[k].program;
        right(c, 172, &format!("{}  {}", family(&program.text), program.id), MUTED, 10.5);
    }
    let f = sample.map(|(k, l)| &l.features[k]);
    let measure = |get: fn(&Feat) -> f32, full: f32, digits: usize| f.map(|f| (get(f) / full, format!("{:.*}", digits, get(f))));
    reading(c, 194, "lit", measure(|f| f.lit, 0.6, 3));
    reading(c, 213, "spread", measure(|f| f.spread, 0.7, 3));
    reading(c, 232, "motion", measure(|f| f.motion, 0.05, 4));
    reading(c, 251, "aspect", measure(|f| f.aspect, 1.0, 3));
    reading(c, 270, "chroma", measure(|f| f.chroma, 1.0, 3));
    rule(c, 296);

    c.text(PX, 307, "feature space", MUTED, 11.0);
    draw_plot(c, p);

    c.text(PX, 572, "click select   right click breed", MUTED, 11.0);
    c.text(PX, 590, "space next     s save picked", MUTED, 11.0);
    c.text(PX, 618, p.status, ACCENT, 11.0);

    let ready = p.litter.is_some_and(Litter::ready);
    button(c, NEXT, if p.litter.is_none() { "computing" } else { "next batch" }, ready, p.over_next);
    button(c, FINISH, "save and quit", true, p.over_finish);
}

fn show_computing(window: &mut Window, buf: &mut [u32], data: &Dataset, batch: usize) -> Result<()> {
    let mut c = Canvas { px: buf, w: WIDTH, h: HEIGHT };
    c.fill(0, 0, GRID as i32, HEIGHT as i32, 0);
    centred(&mut c, Button(0, 0, GRID as i32, HEIGHT as i32), HEIGHT as i32 / 2 - 8, "computing batch", MUTED, 14.0);
    draw_panel(&mut c, &Panel { data, batch, litter: None, hover: None, over_next: false, over_finish: false, status: "" });
    window.update_with_buffer(buf, WIDTH, HEIGHT)?;
    Ok(())
}

/// Corner brackets, the way a measured region is marked on a plot.
fn brackets(c: &mut Canvas, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
    const ARM: i32 = 16;
    for (x, dx) in [(x0 + 5, 1), (x1 - 6, -1)] {
        for (y, dy) in [(y0 + 5, 1), (y1 - 6, -1)] {
            c.fill(x.min(x + dx * ARM), y, x.max(x + dx * ARM) + 1, y + 1, color);
            c.fill(x, y.min(y + dy * ARM), x + 1, y.max(y + dy * ARM) + 1, color);
        }
    }
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
    let name = if k == 0 { "ref".to_string() } else { format!("{k:02}") };
    c.label(x0 + 10, y0 + 9, &name, if selected { ACCENT } else { MUTED }, 11.0);
    if selected {
        brackets(c, x0, y0, x1, y1, ACCENT);
    } else if hovered {
        brackets(c, x0, y0, x1, y1, MUTED);
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
