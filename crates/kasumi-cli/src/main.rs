mod bench;
mod common;
mod evolve;
mod explore_rt;
mod export;
mod grid;
mod hud;
mod live;

use anyhow::Result;
use common::Args;

const USAGE: &str = "kasumi - gpu engine for generative point art

usage:
  kasumi live   <form.ks> [--res 1080p] [--points 20m] [--mode warp]   live window, hot reload on save
  kasumi render <form.ks> [--res 4k] [--points 200m] [--sub 8] [--out f.png]
  kasumi render <form.ks> --seconds 10 [--fps 60] [--out f.mp4]      video (ffmpeg + nvenc)
  kasumi check  <form.ks>...                                        gpu vs cpu reference interpreter
  kasumi bench  [forms...] [--points 100m] [--res 4k]               naive vs warp vs tile binning
  kasumi ptx    <form.ks>                                           dump generated cuda + ptx
  kasumi grid   <form.ks>                                           4x4 live mutations: click like, right-click breed
  kasumi taste                                                      train the taste model on your clicks
  kasumi evolve [seeds...] [--minutes 480] [--batch 24]             overnight evolution -> runs/<id>/gallery
  kasumi gallery <runs/id>                                          rebuild a run's gallery page
  kasumi info                                                       gpu / nvrtc info
";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw[0] == "help" || raw[0] == "--help" || raw[0] == "-h" {
        print!("{USAGE}");
        return;
    }
    let args = Args::parse(&raw[1..]);
    let r: Result<()> = match raw[0].as_str() {
        "live" => live::live(&args),
        "render" => export::render(&args),
        "check" => bench::check_cmd(&args),
        "bench" => bench::bench(&args),
        "ptx" => bench::ptx(&args),
        "info" => bench::info(),
        "grid" => grid::grid(&args),
        "evolve" => evolve::evolve(&args),
        "gallery" => evolve::gallery(&args),
        "taste" => evolve::taste_cmd(&args),
        other => {
            eprintln!("unknown command '{other}'\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
