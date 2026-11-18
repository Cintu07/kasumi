//! Kernel arguments. Each struct is declared once here and generated for both
//! Rust and CUDA, so the two sides cannot drift apart.

macro_rules! rust_type {
    (f32) => { f32 };
    (i32) => { i32 };
    (u64) => { u64 };
    (f32p) => { u64 };
    (u32p) => { u64 };
    (voidp) => { u64 };
    (f32x3) => { [f32; 3] };
    (i32x4) => { [i32; 4] };
    (f32p4) => { [u64; 4] };
}

macro_rules! kernel_args {
    ($($name:ident { $($field:ident: $ty:ident,)* })*) => {
        $(
            #[repr(C)]
            #[derive(Clone, Copy, Default)]
            pub struct $name { $(pub $field: rust_type!($ty),)* }
        )*
        const STRUCTS_CU: &str = concat!($(
            "struct ", stringify!($name), " {", $(" ", stringify!($ty), " ", stringify!($field), ";",)* " };\n",
        )*);
    };
}

kernel_args! {
    SplatArgs {
        uniforms: f32p, density: f32p, hue: f32p,
        staged_key: u32p, staged_weight: f32p, staged_tile: u32p, tile_count: u32p,
        probe: f32p,
        base: u64, chains: u64, stride: u64,
        inv_n: f32, n_uniforms: i32,
        w: i32, h: i32,
        scale_x: f32, scale_y: f32, offset_x: f32, offset_y: f32,
        dof: f32, focus: f32,
        iters: i32, warmup: i32,
        tiles_x: i32, tiles: i32,
        subframe: i32, mode: i32,
    }
    BinArgs {
        keys: u32p, weights: f32p,
        tile_count: u32p, tile_offset: u32p, tile_cursor: u32p, slice_prefix: u32p, work: u32p,
        density: f32p, hue: f32p,
        w: i32, h: i32, tiles_x: i32, tiles: i32, has_b: i32, has_h: i32,
    }
    ScatterArgs {
        staged_tile: u32p, staged_key: u32p, staged_weight: f32p,
        keys: u32p, weights: f32p, tile_cursor: u32p,
        n: u64, tiles: i32, has_b: i32,
    }
    HistArgs { density: f32p, hist: u32p, n: u64, }
    ToneArgs {
        density: f32p, hue: f32p, out: voidp,
        w: i32, h: i32, ss: i32, format: i32,
        inv_log_ref: f32, gamma_inv: f32, vibrancy: f32, heat: f32,
        low: f32x3, high: f32x3,
        has_h: i32,
        glow: f32p, glow_w: i32, glow_h: i32, glow_gain: f32,
    }
    GlowArgs {
        density: f32p, hue: f32p, out: f32p,
        w: i32, h: i32, cell: i32, glow_w: i32, glow_h: i32, has_h: i32,
        inv_log_ref: f32, low: f32x3, high: f32x3,
    }
    BlurArgs { src: f32p, dst: f32p, w: i32, h: i32, dx: i32, dy: i32, }
    DownArgs {
        src_density: f32p, src_hue: f32p, dst_density: f32p, dst_hue: f32p,
        src_w: i32, src_h: i32, dst_w: i32, dst_h: i32, has_h: i32,
    }
    ResolveArgs {
        density: f32p, hue: f32p, out_density: f32p, out_hue: f32p,
        level_density: f32p4, level_hue: f32p4,
        w: i32, h: i32, level_w: i32x4, level_h: i32x4,
        max_radius: f32, curve: f32, has_h: i32,
    }
    DecayArgs { density: f32p, hue: f32p, n: u64, factor: f32, }
}

const TYPES_CU: &str = "typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;
typedef float f32;
typedef f32* f32p;
typedef u32* u32p;
typedef void* voidp;
struct f32x3 { f32 r, g, b; };
struct i32x4 { i32 v[4]; };
struct f32p4 { f32p v[4]; };
";

/// Tiles one block of the count kernel can track before it falls back to a full flush.
pub const TOUCH_COUNT: u32 = 2048;
/// Tiles one block of the scatter kernel handles per round.
pub const TOUCH_SCATTER: u32 = 1024;

/// The header every CUDA module starts with.
pub fn header() -> String {
    format!("{TYPES_CU}#define KS_TOUCH_C {TOUCH_COUNT}
#define KS_TOUCH_S {TOUCH_SCATTER}
{STRUCTS_CU}")
}
