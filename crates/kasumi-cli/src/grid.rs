//! The live grid: 4 x 4 variations of the current form, all animating, next
//! to a panel that says what to do and how many judgements are logged.
//! left click = like, right click = breed from it, space or NEXT = new litter,
//! s = save liked forms, esc or FINISH = quit. Every like / skip is logged for
//! the taste model.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use kasumi_explore::features::{dist, junk, taste_vec};
use kasumi_explore::mutate::mutate;
use kasumi_explore::rng::Rng;
use kasumi_explore::Click;
use kasumi_gpu::{Gpu, Statics, Target};
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, ScaleMode, Window, WindowOptions};

use crate::common::*;
use crate::explore_rt::*;
use crate::hud::*;

const TILE: usize = 192;
const N: usize = 4;
const GRID: usize = TILE * N;
const PANEL: usize = 340;
/// Judgements the taste model needs before it is worth training.
const GOAL: usize = 500;
/// Seconds a litter must stay on screen before its verdict counts.
const MIN_LOOK: f32 = 2.0;
/// Tiles closer than this, relative to how far the litter strays from its
/// parent, are the same picture and only one of them is shown.
const MIN_TILE_GAP: f32 = 0.3;

const GREEN: u32 = 0x2FA84F;
const PANEL_BG: u32 = 0x101012;
const PX: i32 = GRID as i32 + 24;
const PW: i32 = PANEL as i32 - 48;
const NEXT_BTN: (i32, i32, i32, i32) = (PX, 430, PX + PW, 520);
const FINISH_BTN: (i32, i32, i32, i32) = (PX, 690, PX + PW, 744);

struct Litter {
    tiles: Vec<Live>,
    liked: Vec<bool>,
    born: Instant,
}

impl Litter {
    fn ready(&self) -> bool {
        self.born.elapsed().as_secs_f32() >= MIN_LOOK
    }
}

/// Indices of up to `want` forms: good ones first, each visibly different from
/// the parent and from every form already picked.
fn pick(keep: &[(f32, Live, Vec<f32>)], parent: &[f32], want: usize) -> Vec<usize> {
    let mut d: Vec<f32> = keep.iter().map(|k| dist(&k.2, parent)).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // the upper quartile, because a litter can hold many near copies of the parent
    let scale = d.get(d.len() * 3 / 4).copied().unwrap_or(1.0).max(1e-3);
    let mut chosen: Vec<&[f32]> = vec![parent];
    let mut out: Vec<usize> = Vec::new();
    while out.len() < want {
        let mut best: Option<(f32, usize)> = None;
        for (j, k) in keep.iter().enumerate() {
            if out.contains(&j) {
                continue;
            }
            let near = chosen.iter().map(|c| dist(c, &k.2)).fold(f32::MAX, f32::min) / scale;
            if near < MIN_TILE_GAP {
                continue;
            }
            let s = k.0 + 0.5 * near.min(2.0);
            if best.map_or(true, |b| s > b.0) {
                best = Some((s, j));
            }
        }
        let Some((_, j)) = best else { break };
        chosen.push(&keep[j].2);
        out.push(j);
    }
    out
}

/// Parent in slot 0, then the best 15 different non-junk mutants (by taste if trained).
fn breed(gpu: &Gpu, st: &Statics, scratch: &mut Target, parent: &Cand, rng: &mut Rng, taste: Option<&kasumi_explore::taste::Taste>, pts: u64) -> Result<Litter> {
    let mut keep: Vec<(f32, Live, Vec<f32>)> = Vec::new();
    let first = compile(gpu, vec![parent.clone()]);
    let (pf, _) = evaluate(gpu, st, scratch, &first[0], pts, 2.0)?;
    let mut picks = Vec::new();
    for round in 0..6 {
        let want = if round == 0 { 28 } else { 20 };
        let mut cands = Vec::new();
        while cands.len() < want {
            // chain 1-3 mutations so the litter spans more than small nudges
            let mut f = parent.form.clone();
            for _ in 0..1 + rng.below(3) {
                if let Some(m) = mutate(&f, rng, None) {
                    f = m;
                }
            }
            if f != parent.form {
                cands.push(Cand::new(f));
            }
        }
        for l in compile(gpu, cands) {
            let (f, _) = evaluate(gpu, st, scratch, &l, pts, 2.0)?;
            if junk(&f).is_some() {
                continue;
            }
            let s = taste.map(|t| t.score(&taste_vec(&f))).unwrap_or(0.5);
            keep.push((s, l, f.vec.clone()));
        }
        picks = pick(&keep, &pf.vec, N * N - 1);
        if picks.len() == N * N - 1 {
            break;
        }
    }
    let mut tiles = first;
    let mut slots: Vec<Option<Live>> = keep.into_iter().map(|k| Some(k.1)).collect();
    for j in picks {
        tiles.extend(slots[j].take());
    }
    let n = tiles.len();
    Ok(Litter { tiles, liked: vec![false; n], born: Instant::now() })
}

/// `judged`: you rerolled or bred (an explicit verdict on the litter). On
/// quit, a litter only counts if something in it was liked.
fn log_litter(gpu: &Gpu, st: &Statics, scratch: &mut Target, lit: &Litter, pts: u64, log: &mut std::fs::File, judged: bool) -> Result<usize> {
    let any_like = lit.liked.iter().skip(1).any(|&l| l);
    if !lit.ready() || !(judged || any_like) {
        return Ok(0);
    }
    std::fs::create_dir_all("explore/thumbs")?;
    let mut n = 0;
    for (k, l) in lit.tiles.iter().enumerate().skip(1) {
        let (f, px) = evaluate(gpu, st, scratch, l, pts, 2.0)?;
        let thumb = format!("explore/thumbs/{}.png", l.cand.id);
        write_png_rgb8(&PathBuf::from(&thumb), scratch.w, scratch.h, &u32_to_rgb8(&px))?;
        let c = Click { id: l.cand.id.clone(), liked: lit.liked[k], form: l.cand.text.clone(), thumb, feat: taste_vec(&f), ts: timestamp().parse().unwrap_or(0) };
        writeln!(log, "{}", serde_json::to_string(&c)?)?;
        n += 1;
    }
    Ok(n)
}

fn inside(b: (i32, i32, i32, i32), x: f32, y: f32) -> bool {
    x >= b.0 as f32 && x < b.2 as f32 && y >= b.1 as f32 && y < b.3 as f32
}

fn centred(buf: &mut [u32], w: usize, h: usize, b: (i32, i32, i32, i32), y: i32, s: &str, color: u32, scale: i32) {
    let x = b.0 + (b.2 - b.0 - s.len() as i32 * 8 * scale) / 2;
    text(buf, w, h, x, y, s, color, scale);
}

struct PanelState<'a> {
    done: usize,
    liked: usize,
    ready: bool,
    busy: bool,
    over_next: bool,
    over_finish: bool,
    toast: &'a str,
}

fn draw_panel(buf: &mut [u32], w: usize, h: usize, p: &PanelState) {
    fill(buf, w, h, GRID as i32, 0, w as i32, h as i32, PANEL_BG);
    text(buf, w, h, PX, 24, "TEACH KASUMI", WHITE, 3);
    text(buf, w, h, PX, 58, "what looks good to you", GREY, 1);

    text(buf, w, h, PX, 92, "1", AMBER, 3);
    text(buf, w, h, PX + 36, 90, "CLICK EVERY", WHITE, 2);
    text(buf, w, h, PX + 36, 110, "FORM YOU LIKE", WHITE, 2);
    text(buf, w, h, PX, 150, "2", AMBER, 3);
    text(buf, w, h, PX + 36, 154, "PRESS NEXT", WHITE, 2);
    text(buf, w, h, PX, 192, "forms you do not click count as skips", GREY, 1);

    text(buf, w, h, PX, 232, "YOUR PROGRESS", GREY, 1);
    text(buf, w, h, PX, 250, &format!("{} / {GOAL}", p.done), AMBER, 4);
    fill(buf, w, h, PX, 296, PX + PW, 312, 0x2A2A2E);
    let bar = (PW as usize * p.done.min(GOAL) / GOAL) as i32;
    fill(buf, w, h, PX, 296, PX + bar, 312, AMBER);
    if p.done >= GOAL {
        text(buf, w, h, PX, 326, "GOAL REACHED!", GREEN, 2);
        text(buf, w, h, PX, 352, "press finish to train", WHITE, 1);
    } else {
        let per_screen = N * N - 1;
        text(buf, w, h, PX, 326, &format!("screen {} of {}", p.done / per_screen + 1, GOAL.div_ceil(per_screen)), WHITE, 2);
        text(buf, w, h, PX, 352, "your clicks are saved after every screen", GREY, 1);
    }
    text(buf, w, h, PX, 386, &format!("liked on this screen: {}", p.liked), if p.liked > 0 { AMBER } else { GREY }, 1);

    let (bg, label_text, fg) = if p.busy {
        (0x2A2A2E, "MAKING...", GREY)
    } else if !p.ready {
        (0x2A2A2E, "LOOK FIRST", GREY)
    } else if p.over_next {
        (0x39C460, "NEXT SCREEN", 0x000000)
    } else {
        (GREEN, "NEXT SCREEN", 0x000000)
    };
    fill(buf, w, h, NEXT_BTN.0, NEXT_BTN.1, NEXT_BTN.2, NEXT_BTN.3, bg);
    centred(buf, w, h, NEXT_BTN, NEXT_BTN.1 + 26, label_text, fg, 3);
    centred(buf, w, h, NEXT_BTN, NEXT_BTN.1 + 62, "or press space", fg, 1);

    text(buf, w, h, PX, 546, "WANT MORE OF ONE LOOK?", WHITE, 1);
    text(buf, w, h, PX, 562, "right-click that form", GREY, 1);
    if !p.toast.is_empty() {
        text(buf, w, h, PX, 596, p.toast, AMBER, 1);
    }

    let line = if p.over_finish { WHITE } else { GREY };
    rect(buf, w, h, FINISH_BTN.0, FINISH_BTN.1, FINISH_BTN.2, FINISH_BTN.3, line, 2);
    centred(buf, w, h, FINISH_BTN, FINISH_BTN.1 + 19, "SAVE AND FINISH", line, 2);
}

pub fn grid(args: &Args) -> Result<()> {
    let path = PathBuf::from(args.pos.first().context("usage: kasumi grid <form.ks>")?);
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let pts = args.count_or("points", 2_000_000)?;
    let mut rng = Rng::from_time();
    let taste = load_taste(&PathBuf::from("explore/taste.json"));
    std::fs::create_dir_all("explore")?;
    let mut done = kasumi_explore::read_clicks(Path::new("explore/clicks.jsonl")).len();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open("explore/clicks.jsonl")?;

    let seed = kasumi_lang::parse(&std::fs::read_to_string(&path)?).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut parent = Cand::new(seed);
    let mut scratch = Target::new(&gpu, TILE as u32, TILE as u32, 2, 0)?;
    let mut tgts: Vec<Target> = (0..N * N).map(|_| Target::new(&gpu, TILE as u32, TILE as u32, 2, 0)).collect::<Result<_>>()?;

    let (w, h) = (GRID + PANEL, GRID);
    let mut buf = vec![0u32; w * h];
    // a fixed size keeps mouse positions and tiles lined up
    let mut window = Window::new("kasumi - teach it your taste", w, h, WindowOptions { resize: false, scale_mode: ScaleMode::UpperLeft, ..WindowOptions::default() })?;
    window.set_target_fps(60);

    let splash = |window: &mut Window, buf: &mut Vec<u32>, done: usize| -> Result<()> {
        crate::hud::dim_rect(buf, w, h, 0, 0, GRID as i32, h as i32);
        centred(buf, w, h, (0, 0, GRID as i32, h as i32), h as i32 / 2 - 8, "making new forms...", AMBER, 2);
        draw_panel(buf, w, h, &PanelState { done, liked: 0, ready: false, busy: true, over_next: false, over_finish: false, toast: "" });
        window.update_with_buffer(buf, w, h)?;
        Ok(())
    };
    splash(&mut window, &mut buf, done)?;
    let mut litter = breed(&gpu, &st, &mut scratch, &parent, &mut rng, taste.as_ref(), pts)?;
    let t0 = Instant::now();
    let (mut was_l, mut was_r) = (false, false);
    let mut toast = String::new();
    let mut mouse: Option<(f32, f32)> = None;

    let max_frames = args.get("frames").map(|v| v.parse::<u64>()).transpose()?;
    let mut frame_no = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = t0.elapsed().as_secs_f32() + 1.0;
        let tile_at = |mx: f32, my: f32| -> Option<usize> {
            if mx < 0.0 || my < 0.0 || mx >= GRID as f32 || my >= GRID as f32 {
                return None;
            }
            let k = (my as usize / TILE) * N + mx as usize / TILE;
            (k < litter.tiles.len()).then_some(k)
        };
        let hover = mouse.and_then(|(x, y)| tile_at(x, y));
        fill(&mut buf, w, h, 0, 0, GRID as i32, h as i32, 0);
        for (k, l) in litter.tiles.iter().enumerate() {
            let px = render_frame(&gpu, &st, &mut tgts[k], l, pts, t, true)?;
            let (ox, oy) = ((k % N) * TILE, (k / N) * TILE);
            for y in 0..TILE {
                buf[(oy + y) * w + ox..(oy + y) * w + ox + TILE].copy_from_slice(&px[y * TILE..(y + 1) * TILE]);
            }
            let (x0, y0, x1, y1) = (ox as i32, oy as i32, (ox + TILE) as i32, (oy + TILE) as i32);
            if k == 0 {
                dim_rect(&mut buf, w, h, x0, y0, x1, y1);
                label(&mut buf, w, h, x0 + 8, y0 + 8, "ORIGINAL", GREY, 2);
                label(&mut buf, w, h, x0 + 8, y0 + 30, "not counted", GREY, 1);
            } else if litter.liked[k] {
                rect(&mut buf, w, h, x0, y0, x1, y1, AMBER, 5);
                fill(&mut buf, w, h, x0, y0, x0 + 92, y0 + 28, AMBER);
                text(&mut buf, w, h, x0 + 8, y0 + 6, "LIKED", 0x000000, 2);
            } else if hover == Some(k) {
                rect(&mut buf, w, h, x0, y0, x1, y1, WHITE, 2);
            }
        }
        let liked = litter.liked.iter().skip(1).filter(|&&l| l).count();
        let over = |b| mouse.is_some_and(|(x, y)| inside(b, x, y));
        draw_panel(&mut buf, w, h, &PanelState { done, liked, ready: litter.ready(), busy: false, over_next: over(NEXT_BTN), over_finish: over(FINISH_BTN), toast: &toast });
        window.update_with_buffer(&buf, w, h)?;

        frame_no += 1;
        if max_frames.is_some_and(|n| frame_no >= n) {
            if let Some(p) = args.get("shot") {
                write_png_rgb8(&PathBuf::from(p), w as u32, h as u32, &u32_to_rgb8(&buf))?;
            }
            break;
        }

        let l_down = window.get_mouse_down(MouseButton::Left);
        let r_down = window.get_mouse_down(MouseButton::Right);
        mouse = window.get_mouse_pos(MouseMode::Discard);
        let (l_click, r_click) = (l_down && !was_l, r_down && !was_r);
        (was_l, was_r) = (l_down, r_down);
        let mut breed_from: Option<usize> = None;
        let mut reroll = false;
        let mut finish = false;
        if let Some((mx, my)) = mouse {
            if l_click {
                match tile_at(mx, my) {
                    Some(k) if k > 0 => litter.liked[k] = !litter.liked[k],
                    _ => {}
                }
                reroll = inside(NEXT_BTN, mx, my);
                finish = inside(FINISH_BTN, mx, my);
            }
            if r_click {
                breed_from = tile_at(mx, my).filter(|&k| k > 0);
            }
        }
        if finish {
            break;
        }
        for k in window.get_keys_pressed(KeyRepeat::No) {
            match k {
                Key::Space => reroll = true,
                Key::S => {
                    std::fs::create_dir_all("forms/kept")?;
                    let mut n = 0;
                    for (k, l) in litter.tiles.iter().enumerate() {
                        if litter.liked[k] {
                            std::fs::write(format!("forms/kept/{}.ks", l.cand.id), &l.cand.text)?;
                            n += 1;
                        }
                    }
                    toast = format!("saved {n} to forms/kept");
                }
                _ => {}
            }
        }
        if !litter.ready() {
            continue;
        }
        if let Some(k) = breed_from {
            litter.liked[k] = true; // breeding from it counts as a like
            parent = litter.tiles[k].cand.clone();
            reroll = true;
        }
        if reroll {
            splash(&mut window, &mut buf, done)?;
            done += log_litter(&gpu, &st, &mut scratch, &litter, pts, &mut log, true)?;
            litter = breed(&gpu, &st, &mut scratch, &parent, &mut rng, taste.as_ref(), pts)?;
            toast.clear();
            for t in tgts.iter_mut() {
                t.reset_exposure();
            }
        }
    }
    if max_frames.is_none() {
        done += log_litter(&gpu, &st, &mut scratch, &litter, pts, &mut log, false)?;
    }
    println!("{done} judgements in explore/clicks.jsonl; run `kasumi taste` to retrain");
    Ok(())
}
