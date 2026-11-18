//! The clicking screen: 4 x 4 variations of a form, all animating, next to a
//! panel that says what to do. Left click likes a form, right click breeds from
//! it, and every like or skip is logged for the taste model.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};
use serde_json::json;

use crate::cli::*;
use crate::explore::*;
use crate::features::{dist, junk};
use crate::mutate::mutate;
use crate::render::{Engine, Format, Program, Target};
use crate::rng::Rng;
use crate::ui::*;

const TILE: usize = 192;
const N: usize = 4;
const GRID: usize = TILE * N;
const PANEL: usize = 340;
const WIDTH: usize = GRID + PANEL;
const HEIGHT: usize = GRID;
const CLICKS: &str = "explore/clicks.jsonl";
/// Judgements the taste model needs before it is worth training.
const GOAL: usize = 500;
/// Seconds a screen must stay up before its verdict counts.
const MIN_LOOK: f32 = 2.0;
/// Tiles closer than this, relative to how far the litter strays from its
/// parent, count as the same picture.
const MIN_TILE_GAP: f32 = 0.3;

const GREEN: u32 = 0x2FA84F;
const PANEL_BG: u32 = 0x101012;
const BUTTON_OFF: u32 = 0x2A2A2E;
const PX: i32 = GRID as i32 + 24;
const PW: i32 = PANEL as i32 - 48;

#[derive(Clone, Copy)]
struct Button(i32, i32, i32, i32);

impl Button {
    fn contains(self, x: f32, y: f32) -> bool {
        x >= self.0 as f32 && x < self.2 as f32 && y >= self.1 as f32 && y < self.3 as f32
    }
}

const NEXT: Button = Button(PX, 430, PX + PW, 520);
const FINISH: Button = Button(PX, 690, PX + PW, 744);

struct Litter {
    tiles: Vec<Compiled>,
    liked: Vec<bool>,
    born: Instant,
}

impl Litter {
    fn ready(&self) -> bool {
        self.born.elapsed().as_secs_f32() >= MIN_LOOK
    }

    fn likes(&self) -> usize {
        self.liked.iter().skip(1).filter(|&&l| l).count()
    }
}

/// Up to `want` forms, each visibly different from the parent and from every form picked so far.
fn pick_different(vectors: &[Vec<f32>], parent: &[f32], want: usize) -> Vec<usize> {
    let mut spreads: Vec<f32> = vectors.iter().map(|v| dist(v, parent)).collect();
    spreads.sort_by(f32::total_cmp);
    // the upper quartile, because a litter can hold many near copies of the parent
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
    let mut picks = Vec::new();
    for round in 0..6 {
        let want = if round == 0 { 28 } else { 20 };
        let mut batch = Vec::new();
        while batch.len() < want {
            // chain 1 to 3 mutations so a litter spans more than small nudges
            let mut form = parent.form.clone();
            for _ in 0..1 + rng.below(3) {
                form = mutate(&form, rng, None).unwrap_or(form);
            }
            if form != parent.form {
                batch.push(Program::new(form));
            }
        }
        for c in compile_batch(e, batch) {
            let (f, _) = evaluate(e, scratch, &c, points)?;
            if junk(&f).is_none() {
                kept.push(Some(c));
                vectors.push(f.vec);
            }
        }
        picks = pick_different(&vectors, &parent_features.vec, N * N - 1);
        if picks.len() == N * N - 1 {
            break;
        }
    }
    tiles.extend(picks.into_iter().filter_map(|j| kept[j].take()));
    let liked = vec![false; tiles.len()];
    Ok(Litter { tiles, liked, born: Instant::now() })
}

/// Log a screen. Skips count only when you moved on from it or liked something on it.
fn log_litter(e: &Engine, scratch: &mut Target, litter: &Litter, points: u64, log: &mut File, moved_on: bool) -> Result<usize> {
    if !litter.ready() || !(moved_on || litter.likes() > 0) {
        return Ok(0);
    }
    std::fs::create_dir_all("explore/thumbs")?;
    for (c, liked) in litter.tiles.iter().zip(&litter.liked).skip(1) {
        let (_, px) = evaluate(e, scratch, c, points)?;
        let thumb = format!("explore/thumbs/{}.png", c.program.id);
        write_png(Path::new(&thumb), scratch.w, scratch.h, &to_rgb(&px))?;
        let click = json!({ "id": c.program.id, "liked": liked, "form": c.program.text, "thumb": thumb, "ts": timestamp() });
        writeln!(log, "{click}")?;
    }
    Ok(litter.tiles.len() - 1)
}

fn centred(c: &mut Canvas, b: Button, y: i32, s: &str, color: u32, scale: i32) {
    c.text(b.0 + (b.2 - b.0 - s.len() as i32 * 8 * scale) / 2, y, s, color, scale);
}

struct Panel<'a> {
    done: usize,
    liked: usize,
    ready: bool,
    busy: bool,
    over_next: bool,
    over_finish: bool,
    toast: &'a str,
}

fn draw_panel(c: &mut Canvas, p: &Panel) {
    c.fill(GRID as i32, 0, WIDTH as i32, HEIGHT as i32, PANEL_BG);
    c.text(PX, 24, "TEACH KASUMI", WHITE, 3);
    c.text(PX, 58, "what looks good to you", GREY, 1);

    c.text(PX, 92, "1", AMBER, 3);
    c.text(PX + 36, 90, "CLICK EVERY", WHITE, 2);
    c.text(PX + 36, 110, "FORM YOU LIKE", WHITE, 2);
    c.text(PX, 150, "2", AMBER, 3);
    c.text(PX + 36, 154, "PRESS NEXT", WHITE, 2);
    c.text(PX, 192, "forms you do not click count as skips", GREY, 1);

    c.text(PX, 232, "YOUR PROGRESS", GREY, 1);
    c.text(PX, 250, &format!("{} / {GOAL}", p.done), AMBER, 4);
    c.fill(PX, 296, PX + PW, 312, BUTTON_OFF);
    c.fill(PX, 296, PX + (PW as usize * p.done.min(GOAL) / GOAL) as i32, 312, AMBER);
    if p.done >= GOAL {
        c.text(PX, 326, "GOAL REACHED!", GREEN, 2);
        c.text(PX, 352, "press finish to train", WHITE, 1);
    } else {
        let per_screen = N * N - 1;
        c.text(PX, 326, &format!("screen {} of {}", p.done / per_screen + 1, GOAL.div_ceil(per_screen)), WHITE, 2);
        c.text(PX, 352, "your clicks are saved after every screen", GREY, 1);
    }
    c.text(PX, 386, &format!("liked on this screen: {}", p.liked), if p.liked > 0 { AMBER } else { GREY }, 1);

    let (background, label, ink) = match (p.busy, p.ready, p.over_next) {
        (true, ..) => (BUTTON_OFF, "MAKING...", GREY),
        (_, false, _) => (BUTTON_OFF, "LOOK FIRST", GREY),
        (_, _, true) => (0x39C460, "NEXT SCREEN", 0),
        _ => (GREEN, "NEXT SCREEN", 0),
    };
    c.fill(NEXT.0, NEXT.1, NEXT.2, NEXT.3, background);
    centred(c, NEXT, NEXT.1 + 26, label, ink, 3);
    centred(c, NEXT, NEXT.1 + 62, "or press space", ink, 1);

    c.text(PX, 546, "WANT MORE OF ONE LOOK?", WHITE, 1);
    c.text(PX, 562, "right-click that form", GREY, 1);
    c.text(PX, 596, p.toast, AMBER, 1);

    let line = if p.over_finish { WHITE } else { GREY };
    c.outline(FINISH.0, FINISH.1, FINISH.2, FINISH.3, line, 2);
    centred(c, FINISH, FINISH.1 + 19, "SAVE AND FINISH", line, 2);
}

fn show_making(window: &mut Window, buf: &mut [u32], done: usize) -> Result<()> {
    let mut c = Canvas { px: buf, w: WIDTH, h: HEIGHT };
    c.dim(0, 0, GRID as i32, HEIGHT as i32);
    centred(&mut c, Button(0, 0, GRID as i32, HEIGHT as i32), HEIGHT as i32 / 2 - 8, "making new forms...", AMBER, 2);
    draw_panel(&mut c, &Panel { done, liked: 0, ready: false, busy: true, over_next: false, over_finish: false, toast: "" });
    window.update_with_buffer(buf, WIDTH, HEIGHT)?;
    Ok(())
}

fn draw_tile(c: &mut Canvas, k: usize, px: &[u32], liked: bool, hovered: bool) {
    let (left, top) = ((k % N) * TILE, (k / N) * TILE);
    for y in 0..TILE {
        c.px[(top + y) * WIDTH + left..(top + y) * WIDTH + left + TILE].copy_from_slice(&px[y * TILE..(y + 1) * TILE]);
    }
    let (x0, y0, x1, y1) = (left as i32, top as i32, (left + TILE) as i32, (top + TILE) as i32);
    if k == 0 {
        c.dim(x0, y0, x1, y1);
        c.label(x0 + 8, y0 + 8, "ORIGINAL", GREY, 2);
        c.label(x0 + 8, y0 + 30, "not counted", GREY, 1);
    } else if liked {
        c.outline(x0, y0, x1, y1, AMBER, 5);
        c.fill(x0, y0, x0 + 92, y0 + 28, AMBER);
        c.text(x0 + 8, y0 + 6, "LIKED", 0, 2);
    } else if hovered {
        c.outline(x0, y0, x1, y1, WHITE, 2);
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
    let mut done = std::fs::read_to_string(CLICKS).map_or(0, |s| s.lines().count());
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(CLICKS)?;

    let size = TILE as u32;
    let mut scratch = Target::new(gpu, size, size, 2, Format::Window)?;
    let mut targets = (0..N * N).map(|_| Target::new(gpu, size, size, 2, Format::Window)).collect::<Result<Vec<_>>>()?;
    let mut buf = vec![0u32; WIDTH * HEIGHT];
    // a fixed size keeps mouse positions and tiles lined up
    let options = WindowOptions { resize: false, scale_mode: ScaleMode::UpperLeft, ..WindowOptions::default() };
    let mut window = Window::new("kasumi - teach it your taste", WIDTH, HEIGHT, options)?;
    window.set_target_fps(60);

    show_making(&mut window, &mut buf, done)?;
    let mut litter = breed(&engine, &mut scratch, &parent, &mut rng, points)?;
    let start = Instant::now();
    let (mut was_left, mut was_right) = (false, false);
    let mut toast = String::new();
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
        let panel = Panel { done, liked: litter.likes(), ready: litter.ready(), busy: false, over_next: over(NEXT), over_finish: over(FINISH), toast: &toast };
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
                Key::S => toast = save_liked(&litter)?,
                _ => {}
            }
        }
        if !litter.ready() {
            continue;
        }
        if let Some(k) = breed_from {
            litter.liked[k] = true; // breeding from a form counts as liking it
            parent = litter.tiles[k].program.clone();
            next = true;
        }
        if next {
            show_making(&mut window, &mut buf, done)?;
            done += log_litter(&engine, &mut scratch, &litter, points, &mut log, true)?;
            litter = breed(&engine, &mut scratch, &parent, &mut rng, points)?;
            toast.clear();
            targets.iter_mut().for_each(Target::reset_exposure);
        }
    }
    done += log_litter(&engine, &mut scratch, &litter, points, &mut log, false)?;
    println!("{done} judgements in {CLICKS}; run `python taste/clip.py train` to retrain");
    Ok(())
}
