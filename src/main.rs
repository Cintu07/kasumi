use anyhow::Result;
use kasumi::cli::Args;
use kasumi::{bench, evolve, export, grid, live};

const USAGE: &str = "kasumi - gpu engine for generative point art

usage:
  kasumi live   <form.ks> [--res 1080p] [--points 20m] [--mode warp]   live window, hot reload on save
  kasumi render <form.ks> [--res 4k] [--points 200m] [--sub 8] [--out f.png]
  kasumi render <form.ks> --seconds 10 [--fps 60] [--out f.mp4]        video (ffmpeg + nvenc)
  kasumi check  <form.ks>...                                           gpu vs cpu reference interpreter
  kasumi bench  [forms...] [--points 100m] [--res 4k]                  naive vs warp vs tile binning
  kasumi ptx    <form.ks>                                              dump generated cuda + ptx
  kasumi grid   <form.ks>                                              click the forms you like
  kasumi evolve [seeds...] [--minutes 480] [--taste clip] [--novelty-on embed]
  kasumi gallery <runs/id>                                             rebuild a run's gallery page
  kasumi info                                                          gpu / nvrtc info
";

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let command = raw.first().map_or("help", |s| s.as_str());
    let args = Args::parse(raw.get(1..).unwrap_or(&[]));
    let result: Result<()> = match command {
        "live" => live::live(&args),
        "render" => export::render(&args),
        "check" => bench::check_forms(&args),
        "bench" => bench::bench(&args),
        "ptx" => bench::ptx(&args),
        "info" => bench::info(),
        "grid" => grid::grid(&args),
        "evolve" => evolve::evolve(&args),
        "gallery" => evolve::gallery(&args),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            Ok(())
        }
        other => {
            eprintln!("unknown command '{other}'\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(err) = result {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
