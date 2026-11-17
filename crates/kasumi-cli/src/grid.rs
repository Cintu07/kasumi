//! The live grid: 4 x 4 variations of the current form, all animating.
//! left click = like, right click = breed from it, space = new litter,
//! s = save liked forms, esc = quit. Every like / skip is logged for the
//! taste model.

use std::io::Write as _;
use std::path::PathBuf;
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

const TILE: usize = 256;
const N: usize = 4;
const BAR: usize = 28;

struct Litter {
    tiles: Vec<Live>,
    liked: Vec<bool>,
    born: Instant,
}

/// Parent in slot 0, then the best 15 non-junk mutants (by taste if trained).
fn breed(gpu: &Gpu, st: &Statics, scratch: &mut Target, parent: &Cand, rng: &mut Rng, taste: Option<&kasumi_explore::taste::Taste>, pts: u64) -> Result<(Litter, usize, usize)> {
    let mut keep: Vec<(f32, Live, Vec<f32>)> = Vec::new();
    let mut tried = 0;
    let mut junked = 0;
    let first = compile(gpu, vec![parent.clone()]);
    for round in 0..4 {
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
            tried += 1;
            let (f, _) = evaluate(gpu, st, scratch, &l, pts, 2.0)?;
            if junk(&f).is_some() {
                junked += 1;
                continue;
            }
            let s = taste.map(|t| t.score(&taste_vec(&f))).unwrap_or(0.5);
            keep.push((s, l, f.vec.clone()));
        }
        if keep.len() >= N * N - 1 {
            break;
        }
    }
    // greedy pick: taste + distance to what is already on the grid, so the
    // sixteen tiles are sixteen different ideas
    let (pf, _) = evaluate(gpu, st, scratch, &first[0], pts, 2.0)?;
    let mut chosen: Vec<Vec<f32>> = vec![pf.vec];
    let mut tiles: Vec<Live> = first;
    let scale = {
        let mut d: Vec<f32> = keep.iter().map(|k| dist(&k.2, &chosen[0])).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d.get(d.len() / 2).copied().unwrap_or(1.0).max(1e-3)
    };
    while tiles.len() < N * N && !keep.is_empty() {
        let mut best = (f32::MIN, 0);
        for (j, k) in keep.iter().enumerate() {
            let near = chosen.iter().map(|c| dist(c, &k.2)).fold(f32::MAX, f32::min) / scale;
            let s = k.0 + 0.5 * near.min(2.0);
            if s > best.0 {
                best = (s, j);
            }
        }
        let k = keep.swap_remove(best.1);
        chosen.push(k.2);
        tiles.push(k.1);
    }
    let n = tiles.len();
    Ok((Litter { tiles, liked: vec![false; n], born: Instant::now() }, tried, junked))
}

/// `judged`: you rerolled or bred (an explicit verdict on the litter). On
/// quit, a litter only counts if something in it was liked.
fn log_litter(gpu: &Gpu, st: &Statics, scratch: &mut Target, lit: &Litter, pts: u64, log: &mut std::fs::File, judged: bool) -> Result<usize> {
    let any_like = lit.liked.iter().skip(1).any(|&l| l);
    if lit.born.elapsed().as_secs_f32() < 2.0 || !(judged || any_like) {
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

pub fn grid(args: &Args) -> Result<()> {
    let path = PathBuf::from(args.pos.first().context("usage: kasumi grid <form.ks>")?);
    let gpu = Gpu::new()?;
    let st = Statics::new(&gpu)?;
    let pts = args.count_or("points", 2_000_000)?;
    let mut rng = Rng::from_time();
    let taste = load_taste(&PathBuf::from("explore/taste.json"));
    std::fs::create_dir_all("explore")?;
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open("explore/clicks.jsonl")?;

    let seed = kasumi_lang::parse(&std::fs::read_to_string(&path)?).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut parent = Cand::new(seed);
    let mut scratch = Target::new(&gpu, TILE as u32, TILE as u32, 2, 0)?;
    let mut tgts: Vec<Target> = (0..N * N).map(|_| Target::new(&gpu, TILE as u32, TILE as u32, 2, 0)).collect::<Result<_>>()?;

    let (w, h) = (TILE * N, TILE * N + BAR);
    let mut buf = vec![0u32; w * h];
    let mut window = Window::new("kasumi grid", w, h, WindowOptions { resize: true, scale_mode: ScaleMode::AspectRatioStretch, ..WindowOptions::default() })?;
    window.set_target_fps(60);

    let splash = |window: &mut Window, buf: &mut Vec<u32>, msg: &str| -> Result<()> {
        crate::hud::dim_rect(buf, w, h, 0, 0, w as i32, h as i32);
        label(buf, w, h, 24, (h / 2) as i32, msg, AMBER, 2);
        window.update_with_buffer(buf, w, h)?;
        Ok(())
    };
    splash(&mut window, &mut buf, "breeding...")?;
    let (mut litter, mut tried, mut junked) = breed(&gpu, &st, &mut scratch, &parent, &mut rng, taste.as_ref(), pts)?;
    let mut gen = 0u32;
    let mut logged_total = 0usize;
    let t0 = Instant::now();
    let (mut was_l, mut was_r) = (false, false);
    let mut toast = String::new();

    let max_frames = args.get("frames").map(|v| v.parse::<u64>()).transpose()?;
    let mut frame_no = 0u64;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = t0.elapsed().as_secs_f32() + 1.0;
        for (k, l) in litter.tiles.iter().enumerate() {
            let px = render_frame(&gpu, &st, &mut tgts[k], l, pts, t, true)?;
            let (ox, oy) = ((k % N) * TILE, (k / N) * TILE + BAR);
            for y in 0..TILE {
                buf[(oy + y) * w + ox..(oy + y) * w + ox + TILE].copy_from_slice(&px[y * TILE..(y + 1) * TILE]);
            }
            if k == 0 {
                rect(&mut buf, w, h, ox as i32, oy as i32, (ox + TILE) as i32, (oy + TILE) as i32, 0x3A3A3A, 2);
                label(&mut buf, w, h, ox as i32 + 6, oy as i32 + 6, "parent", GREY, 1);
            }
            if litter.liked[k] {
                rect(&mut buf, w, h, ox as i32, oy as i32, (ox + TILE) as i32, (oy + TILE) as i32, AMBER, 3);
            }
        }
        for p in buf[..BAR * w].iter_mut() {
            *p = 0x0B0B0B;
        }
        let status = format!(
            "gen {gen} | {} kept of {tried} ({junked} junk filtered) | {} | logged {logged_total} | click like | right-click breed | space reroll | s save liked | esc  {toast}",
            litter.tiles.len() - 1,
            if taste.is_some() { "taste model on" } else { "no taste model yet" },
        );
        label(&mut buf, w, h, 8, 10, &status, GREY, 1);
        window.update_with_buffer(&buf, w, h)?;

        frame_no += 1;
        if max_frames.is_some_and(|n| frame_no >= n) {
            if let Some(p) = args.get("shot") {
                write_png_rgb8(&PathBuf::from(p), w as u32, h as u32, &u32_to_rgb8(&buf))?;
            }
            break;
        }

        // mouse -> tile
        let (ww, wh) = window.get_size();
        let tile_at = |mx: f32, my: f32| -> Option<usize> {
            let bx = (mx / ww.max(1) as f32 * w as f32) as usize;
            let by = (my / wh.max(1) as f32 * h as f32) as usize;
            if by < BAR {
                return None;
            }
            let k = ((by - BAR) / TILE) * N + bx / TILE;
            (k < litter.tiles.len()).then_some(k)
        };
        let l_down = window.get_mouse_down(MouseButton::Left);
        let r_down = window.get_mouse_down(MouseButton::Right);
        let pos = window.get_mouse_pos(MouseMode::Discard);
        let mut breed_from: Option<usize> = None;
        if let Some((mx, my)) = pos {
            if l_down && !was_l {
                if let Some(k) = tile_at(mx, my) {
                    litter.liked[k] = !litter.liked[k];
                }
            }
            if r_down && !was_r {
                breed_from = tile_at(mx, my);
            }
        }
        was_l = l_down;
        was_r = r_down;
        let mut reroll = false;
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
                    toast = format!("| saved {n} to forms/kept");
                }
                _ => {}
            }
        }
        if let Some(k) = breed_from {
            litter.liked[k] = true; // breeding from it counts as a like
            parent = litter.tiles[k].cand.clone();
            reroll = true;
        }
        if reroll {
            splash(&mut window, &mut buf, "breeding...")?;
            logged_total += log_litter(&gpu, &st, &mut scratch, &litter, pts, &mut log, true)?;
            let r = breed(&gpu, &st, &mut scratch, &parent, &mut rng, taste.as_ref(), pts)?;
            (litter, tried, junked) = r;
            gen += 1;
            for t in tgts.iter_mut() {
                t.reset_exposure();
            }
        }
    }
    if max_frames.is_none() {
        logged_total += log_litter(&gpu, &st, &mut scratch, &litter, pts, &mut log, false)?;
    }
    println!("logged {logged_total} judgements to explore/clicks.jsonl; run `kasumi taste` to retrain");
    Ok(())
}
