//! Spike item 0.5.9: throughput at the real klein block dimensions.
//!
//! Builds the FLUX.2-klein transformer stack shape (5 double-stream +
//! 20 single-stream blocks, hidden 3072, 24 heads x 128) with random weights
//! and runs bf16 forward+backward+AdamW steps on LoRA factors at 512px and
//! 1024px image token counts, reporting ms/step and peak VRAM in use (NVML
//! sampler thread). If a configuration runs out of memory it is reported
//! cleanly and the next one still runs (the failed run's CUDA context is
//! released when its scope unwinds).
//!
//! The stack is laid out to match the real klein block structure as far as
//! `RESEARCH.md` pins it down (section 2): mlp_ratio 3.0, three modulation
//! linears (double-stream image, double-stream text, single-stream; each
//! shared by all blocks of its kind), and a single-stream block whose Q/K/V and
//! SwiGLU feed come from ONE fused in-projection and whose two branches are
//! joined into ONE fused out-projection. That lands at ~3.7B params against
//! the real ~4B, so the VRAM numbers are close to the real model's rather than
//! pessimistic.
//!
//! Every size the block functions use comes from `Cfg`, not from the consts:
//! `Cfg::real` is the config measured on the GPU, `Cfg::tiny` is the CPU test
//! at the bottom of this file, which builds and trains the *same* code at
//! fixture dimensions. Both shape bugs found on 2026-10-28 (the fused
//! out-projection width, and a gate that could not broadcast into the 9216-wide
//! MLP activation) were only visible as GPU errors; the tiny test is what makes
//! that class of bug fail in `cargo test` instead. Two limits on how small
//! "tiny" can go: `head_dim` stays 128 (`AttnBwd`'s softmax scale and the lib's
//! `ROPE_AXES` are written for it, so dim is 2 heads x 128 = 256), and the CPU
//! test runs fp32 because candle's CPU backend has no bf16 matmul kernel.
//!
//! Assumptions (printed at startup; the real config is not fully pinned down
//! yet — see `MODEL_SUPPORT.md` open items):
//! - token counts: 16x total pixel->token downscale (VAE + patchify), so
//!   512px -> 32x32 = 1024 image tokens, 1024px -> 64x64 = 4096; txt 512.
//! - single-stream block: in-projection 3072 -> 27648 (Q, K, V of 3072 each
//!   plus a SwiGLU gate and up of 9216 each = 9 dim-wide chunks), then one
//!   fused out-projection 12288 -> 3072 over the attention output joined with
//!   the activated MLP. The adaLN gates are applied to the two branches
//!   *before* the join rather than to halves of the projection output after
//!   it: identical by linearity, and it keeps the fused matmul whole.
//! - every linear is bias-free, as in klein (diffusers creates all block,
//!   modulation, embedder and output linears with `bias=False`).
//! - LoRA: rank 4, alpha 4, so scale alpha/rank = 1.0, on exactly the modules
//!   the real klein LoRA file adapts (`RESEARCH.md` section 3.1) — the
//!   double-stream *image* to_q/to_k/to_v/to_out (the reference's `to_out.0`)
//!   and the single-stream fused in- and out-projections. The text stream and
//!   the double-stream MLPs carry no LoRA, as in that file. The lib's
//!   `LORA_SCALE` (2.0, the fixture's alpha 8 / rank 4) is deliberately not
//!   used here; every call goes through `lin` below.
//! - RoPE: the lib's 4-axis [32,32,32,32] tables (head_dim 128 matches the
//!   real model, so `AttnBwd` applies unchanged; axis dims are still open per
//!   `RESEARCH.md` and throughput-neutral).
//!
//! Run with Strata unloaded (CLAUDE.md testing rule): NVML peak includes any
//! other GPU users. Writes nothing; random weights, no checkpoint downloads.

use candle_core::{backprop::GradStore, DType, Device, Result, Tensor, Var, D};
use foundry_spike::{
    adamw_step, apply_rope, checkpointed_backward, get, layer_norm_mod, linear_scaled, mse,
    rms_norm, rope_tables_dtype, swiglu, AdamW, AttnBwd, RopeTables, Stage, Weights, NORM_EPS,
};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Real klein transformer dims (MODEL_SUPPORT.md). These are `Cfg::real`'s
// defaults; the block functions read their sizes from `Cfg` instead, so the
// tiny CPU test can build the same code smaller.
const DIM: usize = 3072;
const HEADS: usize = 24;
const HEAD_DIM: usize = 128;
const DOUBLE_BLOCKS: usize = 5;
const SINGLE_BLOCKS: usize = 20;
// Open item; 512 is the common caption-token budget. Throughput-relevant but
// small next to the image stream.
const TXT_LEN: usize = 512;
// mlp_ratio 3.0 (RESEARCH.md section 2), not the FLUX.1-style 4x.
const MLP_RATIO: usize = 3;
const MLP_HIDDEN: usize = MLP_RATIO * DIM;
// The single-stream block's fused linears: the in-projection emits Q, K, V
// (DIM each) plus the SwiGLU gate and up (MLP_HIDDEN each) = 3 + 2*3 = 9
// DIM-wide chunks = 27648; the out-projection consumes the attention output
// joined with the activated MLP = 3072 + 9216 = 12288.
const IN_CHUNKS: usize = 3 + 2 * MLP_RATIO;
const IN_PROJ: usize = IN_CHUNKS * DIM;
const OUT_PROJ_IN: usize = DIM + MLP_HIDDEN;
// Three modulation linears, as in diffusers' `Flux2Transformer2DModel`:
// `double_stream_modulation_img` and `_txt` (2 shift/scale/gate sets = 6*DIM
// each, shared by all double blocks) and `single_stream_modulation` (1 set =
// 3*DIM, shared by all single blocks). The image and text streams of a double
// block are modulated differently.
const MOD_IMG_KEY: &str = "mod_img";
const MOD_TXT_KEY: &str = "mod_txt";
const MOD_SINGLE_KEY: &str = "mod_single";
// The real klein LoRA header (RESEARCH.md section 3.1): rank 4, alpha 4,
// `use_rslora` false — hence scale alpha/rank = 1.0.
const LORA_RANK: usize = 4;
const LORA_ALPHA: f64 = 4.0;
const LORA_SCALE: f64 = LORA_ALPHA / LORA_RANK as f64;
// Trainable LoRA values in the real-size stack: `rank * (in + out)` per adapted
// module (A is [rank, in], B is [out, rank]). The double-stream term is the four
// *image* projections only — the text stream and the MLPs carry no LoRA.
const LORA_VALUES: usize = DOUBLE_BLOCKS * 4 * LORA_RANK * 2 * DIM
    + SINGLE_BLOCKS * LORA_RANK * (DIM + IN_PROJ)
    + SINGLE_BLOCKS * LORA_RANK * (OUT_PROJ_IN + DIM);
// Pixels per image-token side (VAE 8x + patchify 2x). If the real VAE/patch
// split differs, the token counts below still hold; only this constant moves.
const TOKEN_DOWNSCALE: usize = 16;
const WARMUP_STEPS: usize = 2;
const TIMED_STEPS: usize = 5;
const OPT: AdamW = AdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
};

// `AttnBwd`'s softmax scale is 1/sqrt(foundry_spike::HEAD_DIM); the real
// klein head_dim is also 128, so the lib's custom op applies unchanged.
const _: () = assert!(HEAD_DIM == foundry_spike::HEAD_DIM);
// The fused single-stream widths, spelled out: 27648 in, 12288 out, 9216 MLP.
const _: () = assert!(MLP_HIDDEN == 9216 && IN_PROJ == 27648 && OUT_PROJ_IN == 12288);
// The real LoRA scale, spelled out: alpha 4 at rank 4 is exactly 1.0.
const _: () = assert!(LORA_SCALE == 1.0);
// The target set is right only if the trainable count matches the real file's
// header exactly: 120 tensors, 4,177,920 values (`RESEARCH.md` section 3.1).
// Drop `to_k`/`to_out` from the double image stream and this stops compiling.
const _: () = assert!(LORA_VALUES == 4_177_920);

/// NVML errors into Candle's error type so `?` works throughout.
fn nvml_err(e: NvmlError) -> candle_core::Error {
    candle_core::Error::Msg(format!("nvml: {e}"))
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Bytes per element, for the param-count print. Only the bf16 GPU path prints
/// it; the fp32 CPU test never does.
fn dtype_bytes(dt: DType) -> usize {
    match dt {
        DType::BF16 | DType::F16 => 2,
        _ => 4,
    }
}

/// Stack shape and compute dtype: the block functions' only source of sizes.
#[derive(Clone, Copy)]
struct Cfg {
    dim: usize,
    heads: usize,
    head_dim: usize,
    mlp_ratio: usize,
    double_blocks: usize,
    single_blocks: usize,
    txt_len: usize,
    img_seq: usize,
    img_grid: usize,
    /// Compute dtype for weights, inputs and rope tables. The GPU run is bf16;
    /// the CPU test is fp32 because candle's CPU backend has no bf16 matmul
    /// kernel ("unsupported dtype BF16 for op matmul").
    dtype: DType,
}

impl Cfg {
    /// The config measured on the GPU.
    fn real(img_grid: usize) -> Self {
        Cfg {
            dim: DIM,
            heads: HEADS,
            head_dim: HEAD_DIM,
            mlp_ratio: MLP_RATIO,
            double_blocks: DOUBLE_BLOCKS,
            single_blocks: SINGLE_BLOCKS,
            txt_len: TXT_LEN,
            img_seq: img_grid * img_grid,
            img_grid,
            dtype: DType::BF16,
        }
    }

    /// The CPU test's stand-in: same structure, fixture dimensions. `head_dim`
    /// is the one size that cannot shrink — `AttnBwd`'s softmax scale and the
    /// lib's `ROPE_AXES` are written for 128 — so "tiny" means dim 256
    /// (2 heads x 128, the fixture's own shape), one double and one single
    /// block, 16 image + 8 text tokens, fp32 (no CPU bf16 matmul). Test-only:
    /// the GPU run only ever uses `Cfg::real`.
    #[cfg(test)]
    fn tiny() -> Self {
        Cfg {
            dim: foundry_spike::DIM,
            heads: foundry_spike::HEADS,
            head_dim: foundry_spike::HEAD_DIM,
            mlp_ratio: MLP_RATIO,
            double_blocks: 1,
            single_blocks: 1,
            txt_len: foundry_spike::TXT_LEN,
            img_seq: foundry_spike::IMG_SEQ,
            img_grid: foundry_spike::IMG_GRID,
            dtype: DType::F32,
        }
    }

    fn mlp_hidden(&self) -> usize {
        self.dim * self.mlp_ratio
    }
    fn in_chunks(&self) -> usize {
        3 + 2 * self.mlp_ratio
    }
    fn in_proj(&self) -> usize {
        self.in_chunks() * self.dim
    }
    fn out_proj_in(&self) -> usize {
        self.dim + self.mlp_hidden()
    }
    fn all_seq(&self) -> usize {
        self.img_seq + self.txt_len
    }

    /// The relations the block functions assume. Cheap, and a wrong one of
    /// these otherwise shows up as an opaque candle broadcast error.
    fn check(&self) -> Result<()> {
        if self.heads * self.head_dim != self.dim {
            return Err(candle_core::Error::Msg(format!(
                "cfg: heads {} x head_dim {} != dim {}",
                self.heads, self.head_dim, self.dim
            )));
        }
        if self.head_dim != foundry_spike::HEAD_DIM {
            return Err(candle_core::Error::Msg(format!(
                "cfg: head_dim {} is not what `AttnBwd`'s scale and the rope axes assume ({})",
                self.head_dim,
                foundry_spike::HEAD_DIM
            )));
        }
        if self.img_grid * self.img_grid != self.img_seq {
            return Err(candle_core::Error::Msg(format!(
                "cfg: img_grid {} squared != img_seq {}",
                self.img_grid, self.img_seq
            )));
        }
        Ok(())
    }
}

/// Every linear in this harness takes the real LoRA scale (1.0). The lib's
/// `linear` applies the fixture's 2.0, which no key here should pick up by
/// accident; the double-stream MLPs go through `swiglu`, which carries no LoRA
/// at all (as in the real LoRA file), so its scale is never applied.
fn lin(w: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
    linear_scaled(w, name, x, LORA_SCALE)
}

/// Per-block handle: weight map, sequence config, key prefix ("double.0").
struct Blk<'a> {
    w: &'a Weights,
    cfg: &'a Cfg,
    p: String,
}

struct Rts {
    img: RopeTables,
    txt: RopeTables,
    joint: RopeTables,
}

/// The six adaLN modulation parts of one double-block stream (attention set
/// then MLP set). Computed ONCE per run: `temb_silu` is constant across steps
/// and the modulation linears are not trainable (LoRA trains the projections).
struct Mod {
    sh1: Tensor,
    sc1: Tensor,
    g1: Tensor,
    sh2: Tensor,
    sc2: Tensor,
    g2: Tensor,
}

/// The single-block modulation: one shift/scale/gate set.
struct Mod3 {
    sh: Tensor,
    sc: Tensor,
    g: Tensor,
}

/// All three modulations of the stack.
struct Mods {
    img: Mod,
    txt: Mod,
    single: Mod3,
}

fn mod_parts(w: &Weights, cfg: &Cfg, temb_silu: &Tensor) -> Result<Mods> {
    let d = cfg.dim;
    let split = |key: &str, sets: usize| -> Result<Vec<Tensor>> {
        let out = lin(w, key, temb_silu)?.reshape(3 * sets * d)?;
        (0..3 * sets).map(|i| out.narrow(0, i * d, d)).collect()
    };
    let six = |key: &str| -> Result<Mod> {
        let p = split(key, 2)?;
        Ok(Mod {
            sh1: p[0].clone(),
            sc1: p[1].clone(),
            g1: p[2].clone(),
            sh2: p[3].clone(),
            sc2: p[4].clone(),
            g2: p[5].clone(),
        })
    };
    let s = split(MOD_SINGLE_KEY, 1)?;
    Ok(Mods {
        img: six(MOD_IMG_KEY)?,
        txt: six(MOD_TXT_KEY)?,
        single: Mod3 {
            sh: s[0].clone(),
            sc: s[1].clone(),
            g: s[2].clone(),
        },
    })
}

struct Inputs {
    img: Tensor,
    txt: Tensor,
    temb_silu: Tensor,
    target_img: Tensor,
    target_txt: Tensor,
}

struct Session {
    w: Weights,
    vars: Vec<(String, Var)>,
    m: HashMap<String, Tensor>,
    v: HashMap<String, Tensor>,
}

fn rand_w(dev: &Device, dtype: DType, shape: (usize, usize)) -> Result<Tensor> {
    Tensor::randn(0.0f32, 0.02, shape, dev)?
        .to_dtype(dtype)?
        .contiguous()
}

/// Add a bias-free linear (klein has no linear biases). Optionally its LoRA
/// A/B factors: A randn, B zero — standard LoRA
/// init, contribution starts at zero. Returns the base parameter count.
fn add_linear(
    w: &mut Weights,
    dev: &Device,
    cfg: &Cfg,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    lora: bool,
) -> Result<usize> {
    let dt = cfg.dtype;
    w.insert(
        format!("weights/{name}.weight"),
        rand_w(dev, dt, (out_dim, in_dim))?,
    );
    if lora {
        w.insert(
            format!("weights/{name}.lora_a"),
            Tensor::randn(0.0f32, 0.02, (LORA_RANK, in_dim), dev)?
                .to_dtype(dt)?
                .contiguous()?,
        );
        w.insert(
            format!("weights/{name}.lora_b"),
            // (out, rank): the lib's `linear` multiplies by b.t().
            Tensor::zeros((out_dim, LORA_RANK), dt, dev)?,
        );
    }
    Ok(out_dim * in_dim)
}

fn add_mlp(w: &mut Weights, dev: &Device, cfg: &Cfg, name: &str) -> Result<usize> {
    let (d, mh) = (cfg.dim, cfg.mlp_hidden());
    let mut n = 0;
    for (t, out_d, in_d) in [("gate", mh, d), ("up", mh, d), ("down", d, mh)] {
        n += add_linear(w, dev, cfg, &format!("{name}.{t}"), out_d, in_d, false)?;
    }
    Ok(n)
}

fn add_qk_norms(
    w: &mut Weights,
    dev: &Device,
    cfg: &Cfg,
    q_key: &str,
    k_key: &str,
) -> Result<usize> {
    let (hd, dt) = (cfg.head_dim, cfg.dtype);
    w.insert(
        format!("weights/{q_key}.weight"),
        Tensor::ones((hd,), dt, dev)?,
    );
    w.insert(
        format!("weights/{k_key}.weight"),
        Tensor::ones((hd,), dt, dev)?,
    );
    Ok(2 * hd)
}

/// Build the whole block stack with random weights in `cfg.dtype` at `cfg`'s
/// sizes. Key names follow the lib's
/// `weights/{name}.{weight|bias|lora_a|lora_b}` pattern so
/// `linear_scaled`/`swiglu` work unchanged.
fn build_stack(dev: &Device, cfg: &Cfg) -> Result<(Weights, usize)> {
    cfg.check()?;
    let d = cfg.dim;
    let mut w = Weights::new();
    let mut params = 0usize;
    // Three modulation linears for the entire stack, not one per block.
    params += add_linear(&mut w, dev, cfg, MOD_IMG_KEY, 6 * d, d, false)?;
    params += add_linear(&mut w, dev, cfg, MOD_TXT_KEY, 6 * d, d, false)?;
    params += add_linear(&mut w, dev, cfg, MOD_SINGLE_KEY, 3 * d, d, false)?;
    for i in 0..cfg.double_blocks {
        let p = format!("double.{i}");
        for s in ["img", "txt"] {
            // The real LoRA file adapts the double-stream image stream only
            // (`attn.to_q`, `to_k`, `to_v`, `to_out.0`) — never the text
            // stream, and never the MLPs (`RESEARCH.md` section 3.1).
            let lora = s == "img";
            for t in ["to_q", "to_k", "to_v", "to_out"] {
                params += add_linear(&mut w, dev, cfg, &format!("{p}.{s}.{t}"), d, d, lora)?;
            }
            params += add_qk_norms(
                &mut w,
                dev,
                cfg,
                &format!("{p}.{s}_q_norm"),
                &format!("{p}.{s}_k_norm"),
            )?;
            params += add_mlp(&mut w, dev, cfg, &format!("{p}.{s}.mlp"))?;
        }
    }
    for i in 0..cfg.single_blocks {
        let p = format!("single.{i}");
        // The QKV and the SwiGLU gate/up are one linear; the attention output
        // and the activated MLP feed one out-projection. No separate MLP
        // linears, and no per-block modulation. Both fused linears are the
        // real file's LoRA targets for the single stream.
        params += add_linear(
            &mut w,
            dev,
            cfg,
            &format!("{p}.to_in"),
            cfg.in_proj(),
            d,
            true,
        )?;
        params += add_linear(
            &mut w,
            dev,
            cfg,
            &format!("{p}.to_out"),
            d,
            cfg.out_proj_in(),
            true,
        )?;
        params += add_qk_norms(
            &mut w,
            dev,
            cfg,
            &format!("{p}.q_norm"),
            &format!("{p}.k_norm"),
        )?;
    }
    Ok((w, params))
}

/// One double-stream block (same wiring as the lib's fixture-sized
/// `double_block`, except that the image and text streams take their own
/// modulation parts, precomputed once per run from the stack's shared
/// image and text modulation linears).
fn double_block(
    b: &Blk,
    img: &Tensor,
    txt: &Tensor,
    mi: &Mod,
    mt: &Mod,
    rt_i: &RopeTables,
    rt_t: &RopeTables,
) -> Result<(Tensor, Tensor)> {
    let (seq_i, seq_t) = (b.cfg.img_seq, b.cfg.txt_len);
    let (h, hd, d) = (b.cfg.heads, b.cfg.head_dim, b.cfg.dim);
    let ih = layer_norm_mod(img, &mi.sh1, &mi.sc1, NORM_EPS)?;
    let th = layer_norm_mod(txt, &mt.sh1, &mt.sc1, NORM_EPS)?;

    let q_i = {
        let h = lin(b.w, &format!("{}.img.to_q", b.p), &ih)?.reshape((seq_i, h, hd))?;
        apply_rope(
            &rms_norm(
                &h,
                get(b.w, &format!("weights/{}.img_q_norm.weight", b.p))?,
                NORM_EPS,
            )?,
            &rt_i.cos,
            &rt_i.sin,
        )?
    };
    let k_i = {
        let h = lin(b.w, &format!("{}.img.to_k", b.p), &ih)?.reshape((seq_i, h, hd))?;
        apply_rope(
            &rms_norm(
                &h,
                get(b.w, &format!("weights/{}.img_k_norm.weight", b.p))?,
                NORM_EPS,
            )?,
            &rt_i.cos,
            &rt_i.sin,
        )?
    };
    let v_i = lin(b.w, &format!("{}.img.to_v", b.p), &ih)?.reshape((seq_i, h, hd))?;
    let q_t = {
        let h = lin(b.w, &format!("{}.txt.to_q", b.p), &th)?.reshape((seq_t, h, hd))?;
        apply_rope(
            &rms_norm(
                &h,
                get(b.w, &format!("weights/{}.txt_q_norm.weight", b.p))?,
                NORM_EPS,
            )?,
            &rt_t.cos,
            &rt_t.sin,
        )?
    };
    let k_t = {
        let h = lin(b.w, &format!("{}.txt.to_k", b.p), &th)?.reshape((seq_t, h, hd))?;
        apply_rope(
            &rms_norm(
                &h,
                get(b.w, &format!("weights/{}.txt_k_norm.weight", b.p))?,
                NORM_EPS,
            )?,
            &rt_t.cos,
            &rt_t.sin,
        )?
    };
    let v_t = lin(b.w, &format!("{}.txt.to_v", b.p), &th)?.reshape((seq_t, h, hd))?;

    let q = Tensor::cat(&[&q_t, &q_i], 0)?;
    let k = Tensor::cat(&[&k_t, &k_i], 0)?;
    let v = Tensor::cat(&[&v_t, &v_i], 0)?;
    let attn = q.apply_op3(&k, &v, AttnBwd)?;
    let a_t = attn.narrow(0, 0, seq_t)?;
    let a_i = attn.narrow(0, seq_t, seq_i)?;

    let img = (img
        + mi.g1.broadcast_mul(&lin(
            b.w,
            &format!("{}.img.to_out", b.p),
            &a_i.contiguous()?.reshape((seq_i, d))?,
        )?)?)?;
    let txt = (txt
        + mt.g1.broadcast_mul(&lin(
            b.w,
            &format!("{}.txt.to_out", b.p),
            &a_t.contiguous()?.reshape((seq_t, d))?,
        )?)?)?;

    let ih2 = layer_norm_mod(&img, &mi.sh2, &mi.sc2, NORM_EPS)?;
    let th2 = layer_norm_mod(&txt, &mt.sh2, &mt.sc2, NORM_EPS)?;
    let img = (img
        + mi.g2
            .broadcast_mul(&swiglu(b.w, &format!("{}.img.mlp", b.p), &ih2)?)?)?;
    let txt = (txt
        + mt.g2
            .broadcast_mul(&swiglu(b.w, &format!("{}.txt.mlp", b.p), &th2)?)?)?;
    Ok((img, txt))
}

/// One single-stream block in the real klein layout: ONE fused in-projection
/// (`to_in`, dim -> in_proj) giving Q, K, V of dim each plus a SwiGLU gate and
/// up of mlp_hidden each, then ONE fused out-projection (`to_out`, out_proj_in
/// -> dim) over the attention output joined with the activated MLP.
///
/// One adaLN gate scales the whole out-projection output before the residual
/// add (the diffusers single-block structure as recorded in `RESEARCH.md`;
/// treat the exact gate count as unverified until the source is re-read).
fn single_block(b: &Blk, x: &Tensor, m: &Mod3, rt: &RopeTables) -> Result<Tensor> {
    let (all, h, hd, d) = (b.cfg.all_seq(), b.cfg.heads, b.cfg.head_dim, b.cfg.dim);
    let (mr, mh) = (b.cfg.mlp_ratio, b.cfg.mlp_hidden());
    let nh = layer_norm_mod(x, &m.sh, &m.sc, NORM_EPS)?;
    let fused = lin(b.w, &format!("{}.to_in", b.p), &nh)?.reshape((all, b.cfg.in_chunks(), d))?;
    // chunks 0..3 = Q, K, V; 3..6 = SwiGLU gate; 6..9 = SwiGLU up
    let split = |which: usize| -> Result<Tensor> {
        fused
            .narrow(1, which, 1)?
            .contiguous()?
            .reshape((all, h, hd))
    };
    let wide = |which: usize| -> Result<Tensor> {
        fused.narrow(1, which, mr)?.contiguous()?.reshape((all, mh))
    };
    let q = apply_rope(
        &rms_norm(
            &split(0)?,
            get(b.w, &format!("weights/{}.q_norm.weight", b.p))?,
            NORM_EPS,
        )?,
        &rt.cos,
        &rt.sin,
    )?;
    let k = apply_rope(
        &rms_norm(
            &split(1)?,
            get(b.w, &format!("weights/{}.k_norm.weight", b.p))?,
            NORM_EPS,
        )?,
        &rt.cos,
        &rt.sin,
    )?;
    let v = split(2)?;
    let attn = q
        .apply_op3(&k, &v, AttnBwd)?
        .contiguous()?
        .reshape((all, d))?;
    let act = wide(3)?.silu()?.mul(&wide(3 + mr)?)?;
    let joined = Tensor::cat(&[&attn, &act], D::Minus1)?;
    let out = lin(b.w, &format!("{}.to_out", b.p), &joined)?;
    // One gate (dim wide) on the fused out-projection's output. The earlier
    // per-branch gating multiplied a dim-wide gate into the mlp_hidden-wide MLP
    // activation, which cannot broadcast (second harness bug, 2026-10-28).
    x + m.g.broadcast_mul(&out)?
}

fn forward(
    w: &Weights,
    cfg: &Cfg,
    inp: &Inputs,
    modp: &Mods,
    rts: &Rts,
) -> Result<(Tensor, Tensor)> {
    let mut img = inp.img.clone();
    let mut txt = inp.txt.clone();
    for i in 0..cfg.double_blocks {
        let b = Blk {
            w,
            cfg,
            p: format!("double.{i}"),
        };
        (img, txt) = double_block(&b, &img, &txt, &modp.img, &modp.txt, &rts.img, &rts.txt)?;
    }
    let mut joint = Tensor::cat(&[&txt, &img], 0)?;
    for i in 0..cfg.single_blocks {
        let b = Blk {
            w,
            cfg,
            p: format!("single.{i}"),
        };
        joint = single_block(&b, &joint, &modp.single, &rts.joint)?;
    }
    let final_txt = joint.narrow(0, 0, cfg.txt_len)?;
    let final_img = joint.narrow(0, cfg.txt_len, cfg.img_seq)?;
    Ok((final_img, final_txt))
}

/// The checkpoint stages: the double blocks — the last one folds the txt/img
/// join into one joint tensor so every later stage has arity 1 — then the
/// single blocks. Each closure is the block itself, so the recomputed graph
/// during backward is exactly the graph the forward pass ran.
fn build_stages<'a>(cfg: &'a Cfg, modp: &'a Mods, rts: &'a Rts) -> Vec<Stage<'a>> {
    let mut stages: Vec<Stage<'a>> = Vec::with_capacity(cfg.double_blocks + cfg.single_blocks);
    for i in 0..cfg.double_blocks {
        let p = format!("double.{i}");
        let join = i + 1 == cfg.double_blocks;
        stages.push(Stage {
            run: Box::new(move |w: &Weights, ins: &[Tensor]| {
                let b = Blk {
                    w,
                    cfg,
                    p: p.clone(),
                };
                let (img, txt) = double_block(
                    &b, &ins[0], &ins[1], &modp.img, &modp.txt, &rts.img, &rts.txt,
                )?;
                if join {
                    Ok(vec![Tensor::cat(&[&txt, &img], 0)?])
                } else {
                    Ok(vec![img, txt])
                }
            }),
        });
    }
    for i in 0..cfg.single_blocks {
        let p = format!("single.{i}");
        stages.push(Stage {
            run: Box::new(move |w: &Weights, ins: &[Tensor]| {
                let b = Blk {
                    w,
                    cfg,
                    p: p.clone(),
                };
                Ok(vec![single_block(&b, &ins[0], &modp.single, &rts.joint)?])
            }),
        });
    }
    stages
}

/// Random inputs at `cfg`'s sizes (the harness measures throughput, it does not
/// train anything real).
fn make_inputs(dev: &Device, cfg: &Cfg) -> Result<Inputs> {
    let cast = |t: Result<Tensor>| -> Result<Tensor> { t?.to_dtype(cfg.dtype)?.contiguous() };
    Ok(Inputs {
        img: cast(Tensor::randn(0.0f32, 0.02, (cfg.img_seq, cfg.dim), dev))?,
        txt: cast(Tensor::randn(0.0f32, 0.02, (cfg.txt_len, cfg.dim), dev))?,
        temb_silu: cast(Tensor::randn(0.0f32, 0.02, (1, cfg.dim), dev))?,
        target_img: cast(Tensor::randn(0.0f32, 1.0, (cfg.img_seq, cfg.dim), dev))?,
        target_txt: cast(Tensor::randn(0.0f32, 1.0, (cfg.txt_len, cfg.dim), dev))?,
    })
}

/// RoPE tables for the image, text and joint streams, in the compute dtype.
///
/// Axes are (T, H, W, L) as in klein (RESEARCH.md section 3): image latent ids
/// are `[0, h, w, 0]`, text ids are `[0, 0, 0, l]`. Throughput-neutral, but the
/// harness should use the real assignment so nothing copied from it carries the
/// fixture's simplified axes into Phase 4.
fn make_rope(dev: &Device, cfg: &Cfg) -> Result<Rts> {
    let pos_img: Vec<[f32; 4]> = (0..cfg.img_seq)
        .map(|i| {
            [
                0.0,
                (i / cfg.img_grid) as f32,
                (i % cfg.img_grid) as f32,
                0.0,
            ]
        })
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..cfg.txt_len)
        .map(|i| [0.0, 0.0, 0.0, i as f32])
        .collect();
    // The joint stream is `cat(&[txt, img])` in `forward`, so its positions
    // must be txt-then-img too. The harness had them image-then-text since it
    // was written — throughput-neutral (same shapes, same ops), but the tiny
    // CPU test makes the mismatch worth fixing.
    let pos_all: Vec<[f32; 4]> = pos_txt.iter().chain(pos_img.iter()).copied().collect();
    Ok(Rts {
        img: rope_tables_dtype(&pos_img, dev, cfg.dtype)?,
        txt: rope_tables_dtype(&pos_txt, dev, cfg.dtype)?,
        joint: rope_tables_dtype(&pos_all, dev, cfg.dtype)?,
    })
}

/// Turn the LoRA factors of `base` into `Var` leaves plus fp32 Adam moments.
/// `Var::from_tensor` copies storage, so the returned session owns its LoRA
/// values and the weight map is repointed at them (that is what `Var::set`
/// then updates).
fn make_session(base: &Weights, dev: &Device) -> Result<Session> {
    let mut w = base.clone();
    let mut lora_keys: Vec<String> = w
        .keys()
        .filter(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b"))
        .cloned()
        .collect();
    lora_keys.sort();
    let mut vars: Vec<(String, Var)> = Vec::new();
    for key in lora_keys {
        let var = Var::from_tensor(get(&w, &key)?)?;
        w.insert(key.clone(), var.as_tensor().clone());
        vars.push((key, var));
    }
    let mut m: HashMap<String, Tensor> = HashMap::new();
    let mut v: HashMap<String, Tensor> = HashMap::new();
    for (key, var) in &vars {
        m.insert(
            key.clone(),
            Tensor::zeros(var.as_tensor().shape(), DType::F32, dev)?,
        );
        v.insert(
            key.clone(),
            Tensor::zeros(var.as_tensor().shape(), DType::F32, dev)?,
        );
    }
    Ok(Session { w, vars, m, v })
}

impl Session {
    /// One full training step: forward graph, MSE loss, backward, AdamW on
    /// the LoRA factors (identical structure to `stability.rs`). The two
    /// phases are tagged so an OOM says where it happened.
    fn step(
        &mut self,
        step: usize,
        cfg: &Cfg,
        inp: &Inputs,
        modp: &Mods,
        rts: &Rts,
    ) -> Result<f32> {
        let (final_img, final_txt) = forward(&self.w, cfg, inp, modp, rts)
            .map_err(|e| candle_core::Error::Msg(format!("died in forward: {e:?}")))?;
        let loss = mse(&final_img, &inp.target_img)?.add(&mse(&final_txt, &inp.target_txt)?)?;
        let grads = loss
            .backward()
            .map_err(|e| candle_core::Error::Msg(format!("died in backward: {e:?}")))?;
        self.adamw(&grads, step)?;
        loss.to_vec0::<f32>()
    }

    /// The same step with per-block gradient checkpointing: only the stage
    /// inputs stay alive, each block's graph is rebuilt during the reverse
    /// walk. `seed` is the loss over the final stage's outputs.
    fn step_ckpt<F>(
        &mut self,
        step: usize,
        stages: &[Stage<'_>],
        inputs: &[Tensor],
        seed: &F,
    ) -> Result<f32>
    where
        F: Fn(&[Tensor]) -> Result<Tensor>,
    {
        // The input gradients (img/txt) are computed because reaching the
        // first stage's LoRA factors requires walking through them; this bin
        // has no use for them, so they are dropped.
        let (loss, _input_grads, grads) =
            checkpointed_backward(&self.w, stages, inputs, seed, &self.vars)?;
        self.adamw(&grads, step)?;
        Ok(loss)
    }

    /// Bias-corrected AdamW over a gradient store (both step paths).
    fn adamw(&mut self, grads: &GradStore, step: usize) -> Result<()> {
        for (key, var) in &self.vars {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
            let mm = self.m.get_mut(key).unwrap();
            let vv = self.v.get_mut(key).unwrap();
            adamw_step(var, g, mm, vv, step, &OPT)?;
        }
        Ok(())
    }
}

fn train_burst(
    s: &mut Session,
    cfg: &Cfg,
    inp: &Inputs,
    modp: &Mods,
    rts: &Rts,
    stages: Option<&[Stage<'_>]>,
) -> Result<(Vec<f32>, Duration)> {
    // Both paths return the step's loss, so the timed loop is shared.
    let one_step = |s: &mut Session, step: usize| -> Result<f32> {
        match stages {
            None => s.step(step, cfg, inp, modp, rts),
            Some(st) => {
                let inputs = [inp.img.clone(), inp.txt.clone()];
                let seed = |outs: &[Tensor]| -> Result<Tensor> {
                    let joint = &outs[0];
                    let final_txt = joint.narrow(0, 0, cfg.txt_len)?;
                    let final_img = joint.narrow(0, cfg.txt_len, cfg.img_seq)?;
                    mse(&final_img, &inp.target_img)?.add(&mse(&final_txt, &inp.target_txt)?)
                };
                s.step_ckpt(step, st, &inputs, &seed)
            }
        }
    };
    for step in 1..=WARMUP_STEPS {
        one_step(s, step)?;
    }
    let started = Instant::now();
    let mut losses = Vec::with_capacity(TIMED_STEPS);
    for step in (WARMUP_STEPS + 1)..=(WARMUP_STEPS + TIMED_STEPS) {
        losses.push(one_step(s, step)?);
    }
    Ok((losses, started.elapsed()))
}

/// NVML sampler: peak in-use = total - min free, polled while the burst
/// runs. `Nvml` is not `Send`, so the thread initializes its own handle.
fn spawn_sampler(
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<std::result::Result<(u64, u64), NvmlError>> {
    std::thread::spawn(move || {
        let nvml = Nvml::init()?;
        let gpu = nvml.device_by_index(0)?;
        let total = gpu.memory_info()?.total;
        let mut min_free = u64::MAX;
        while !stop.load(Ordering::Relaxed) {
            if let Ok(info) = gpu.memory_info() {
                min_free = min_free.min(info.free);
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        Ok((min_free, total))
    })
}

fn run_config(label: &str, cfg: &Cfg, checkpoint: bool) -> Result<()> {
    println!(
        "--- {label}: dim {}, heads {}x{}, {} double + {} single blocks, img_seq {} \
         (grid {}x{}), txt {}, joint {}, dtype {:?} ---",
        cfg.dim,
        cfg.heads,
        cfg.head_dim,
        cfg.double_blocks,
        cfg.single_blocks,
        cfg.img_seq,
        cfg.img_grid,
        cfg.img_grid,
        cfg.txt_len,
        cfg.all_seq(),
        cfg.dtype
    );
    let started = Instant::now();
    let dev = Device::new_cuda(0)?;
    let (base, params) = build_stack(&dev, cfg)?;
    println!(
        "{label}: built {params} base params ({:.2} GB in {:?}) in {:?}",
        gb((params * dtype_bytes(cfg.dtype)) as u64),
        cfg.dtype,
        started.elapsed()
    );

    let mut s = make_session(&base, &dev)?;
    let trainable: usize = s
        .vars
        .iter()
        .map(|(_, v)| v.as_tensor().shape().elem_count())
        .sum();
    println!(
        "{label}: {} trainable LoRA tensors, {trainable} params",
        s.vars.len()
    );

    let inp = make_inputs(&dev, cfg)?;
    let rts = make_rope(&dev, cfg)?;

    // The modulation parts are frozen (that linear carries no LoRA) and the
    // temb input never changes, so they are computed once for the whole run.
    let modp = mod_parts(&s.w, cfg, &inp.temb_silu)?;
    let stages = if checkpoint {
        Some(build_stages(cfg, &modp, &rts))
    } else {
        None
    };

    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_sampler(Arc::clone(&stop));
    let result = train_burst(&mut s, cfg, &inp, &modp, &rts, stages.as_deref());
    stop.store(true, Ordering::Relaxed);
    let (min_free, total) = sampler
        .join()
        .map_err(|_| candle_core::Error::Msg("vram sampler thread panicked".into()))?
        .map_err(nvml_err)?;
    // The peak line prints BEFORE the error is propagated: an OOM run is
    // exactly the case where the peak reading is the interesting number.
    if min_free == u64::MAX {
        println!("{label}: NVML sampler took no samples; peak VRAM not recorded");
    } else {
        println!(
            "{label}: peak VRAM in use ~{:.2} GB (min free {:.2} GB of {:.2} GB total)",
            gb(total - min_free),
            gb(min_free),
            gb(total)
        );
    }
    let (losses, elapsed) = result?;
    let ms = elapsed.as_secs_f64() * 1e3 / TIMED_STEPS as f64;
    println!("{label}: {TIMED_STEPS} timed steps in {elapsed:?} -> {ms:.0} ms/step");
    println!("{label}: timed-step losses {losses:?}");
    Ok(())
}

fn main() {
    let checkpoint = std::env::args().any(|a| a == "--checkpoint");
    println!("assumptions (real config not fully pinned; see MODEL_SUPPORT.md open items):");
    println!(
        "  tokens: {TOKEN_DOWNSCALE}x pixel->token downscale => 512px = {} img tokens, \
         1024px = {} img tokens; txt stream {TXT_LEN}",
        (512 / TOKEN_DOWNSCALE).pow(2),
        (1024 / TOKEN_DOWNSCALE).pow(2)
    );
    println!(
        "  mlp hidden {MLP_HIDDEN} ({}x dim), three modulation linears (double img/txt, single), \
         single block = fused in-projection {DIM} -> {IN_PROJ} + fused out-projection \
         {OUT_PROJ_IN} -> {DIM}; stack ~3.7B params against the real ~4B",
        MLP_RATIO
    );
    println!(
        "  LoRA rank {LORA_RANK} alpha {LORA_ALPHA} => scale {LORA_SCALE}, on the double-stream \
         image to_q/to_k/to_v/to_out and the single-stream fused in/out projections \
         ({LORA_VALUES} trainable values = the real klein LoRA file's count, RESEARCH.md 3.1); \
         text stream and MLPs unadapted; \
         rope axes {:?} (throughput-neutral); random weights; nothing written to disk",
        foundry_spike::ROPE_AXES
    );
    println!(
        "  backward: {}",
        if checkpoint {
            "per-block gradient checkpointing (each block's graph rebuilt during backward)"
        } else {
            "naive full-graph autograd (every intermediate of all 25 blocks retained)"
        }
    );
    if let Ok(nvml) = Nvml::init() {
        if let Ok(gpu) = nvml.device_by_index(0) {
            if let (Ok(name), Ok(info)) = (gpu.name(), gpu.memory_info()) {
                println!(
                    "NVML at start: {name}, free {:.2} GB of {:.2} GB \
                     (run with Strata unloaded for valid numbers)",
                    gb(info.free),
                    gb(info.total)
                );
            }
        }
    }

    let mut measured = 0;
    for (label, px) in [("512px", 512usize), ("1024px", 1024)] {
        let cfg = Cfg::real(px / TOKEN_DOWNSCALE);
        match run_config(label, &cfg, checkpoint) {
            Ok(()) => measured += 1,
            Err(e) => {
                let msg = format!("{e:?}").to_lowercase();
                // The phase tags are attached where the error is raised
                // (`Session::step` / `checkpointed_backward`), so the OOM
                // message itself says which half of the step blew up.
                let phase = if msg.contains("died in backward") {
                    "backward"
                } else if msg.contains("died in forward") {
                    "forward"
                } else {
                    "unknown phase"
                };
                if msg.contains("out of memory")
                    || msg.contains("out_of_memory")
                    || msg.contains("outofmemory")
                {
                    println!(
                        "{label}: OUT OF MEMORY in {phase} — not usable at this size on this \
                         GPU (the peak-VRAM line above is where it died)"
                    );
                } else {
                    println!("{label}: FAILED in {phase}: {e:?}");
                }
            }
        }
    }
    println!(
        "RESULT: {measured} of 2 configurations measured ({}); \
         the usable-or-not judgement goes to BUILD_LOG.md",
        if checkpoint {
            "checkpointed"
        } else {
            "full-graph"
        }
    );
}

#[cfg(test)]
mod tests {
    //! CPU shape + gradient test for the throughput stack (the reason `Cfg`
    //! exists). It builds and trains the *same* `build_stack` / `forward` /
    //! `build_stages` / `Session` code the GPU run uses, at tiny dimensions, in
    //! both backward modes, and checks the two modes still agree.
    //!
    //! Both harness shape bugs of 2026-10-28 — the fused out-projection width
    //! and the adaLN gate that could not broadcast into the MLP activation —
    //! were only visible as GPU errors, after minutes of weight building. This
    //! test catches that class in `cargo test`, on CPU, in well under a second.
    //! No GPU, no fixtures, nothing read from or written to disk. Tolerances are
    //! `foundry_spike::compare`'s, unchanged.

    use super::*;
    use foundry_spike::compare;

    /// The trainable set the real klein LoRA file has (`RESEARCH.md` section
    /// 3.1): the double-stream image to_q/to_k/to_v/to_out and the
    /// single-stream fused in/out projections, A and B each.
    fn expected_lora_keys(cfg: &Cfg) -> Vec<String> {
        let mut keys = Vec::new();
        for i in 0..cfg.double_blocks {
            for t in ["to_q", "to_k", "to_v", "to_out"] {
                keys.push(format!("weights/double.{i}.img.{t}.lora_a"));
                keys.push(format!("weights/double.{i}.img.{t}.lora_b"));
            }
        }
        for i in 0..cfg.single_blocks {
            for t in ["to_in", "to_out"] {
                keys.push(format!("weights/single.{i}.{t}.lora_a"));
                keys.push(format!("weights/single.{i}.{t}.lora_b"));
            }
        }
        keys.sort();
        keys
    }

    #[test]
    // candle's `Result` alias has one error type, so the test's own result is
    // spelled out in full.
    fn tiny_stack_trains_in_both_backward_modes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let cfg = Cfg::tiny();
        cfg.check()?;
        let (base, params) = build_stack(&dev, &cfg)?;

        // The fused single-stream widths at tiny dims, spelled out the same way
        // the real ones are pinned by the `const _: () = assert!` above:
        // 9 dim-wide chunks in (3*256 QKV + 2*3*256 gate/up), attention +
        // activation out.
        assert_eq!(cfg.mlp_hidden(), 768);
        assert_eq!(cfg.in_chunks(), 9);
        assert_eq!(cfg.in_proj(), 2304);
        assert_eq!(cfg.out_proj_in(), 1024);
        assert_eq!(
            get(&base, "weights/single.0.to_in.weight")?.dims(),
            &[2304, 256]
        );
        assert_eq!(
            get(&base, "weights/single.0.to_out.weight")?.dims(),
            &[256, 1024]
        );
        assert_eq!(
            get(&base, "weights/single.0.to_in.lora_b")?.dims(),
            &[2304, LORA_RANK]
        );
        assert_eq!(
            get(&base, "weights/single.0.to_in.lora_a")?.dims(),
            &[LORA_RANK, 256]
        );
        assert_eq!(
            get(&base, "weights/double.0.img.to_out.lora_a")?.dims(),
            &[LORA_RANK, 256]
        );

        let mut full = make_session(&base, &dev)?;
        let mut ckpt = make_session(&base, &dev)?;

        // LoRA targets exactly as `RESEARCH.md` 3.1 lists them: 4 image
        // projections in the one double block plus the two fused projections in
        // the one single block, A and B each. Nothing on the text stream,
        // nothing on the MLPs.
        let want = expected_lora_keys(&cfg);
        assert_eq!(want.len(), 12, "1 double + 1 single block");
        let got: Vec<String> = full.vars.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(got, want, "trainable LoRA set does not match 3.1");
        let got_ckpt: Vec<String> = ckpt.vars.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(got_ckpt, want);

        let inp = make_inputs(&dev, &cfg)?;
        let rts = make_rope(&dev, &cfg)?;
        let modp = mod_parts(&base, &cfg, &inp.temb_silu)?;

        // Two steps, not one: at step 1 the LoRA B factors are still zero, so
        // the A gradients are exactly zero and step 1 would not exercise the A
        // path at all.
        let steps = 2;
        let loss_full = {
            let mut l = 0.0f32;
            for s in 1..=steps {
                l = full.step(s, &cfg, &inp, &modp, &rts)?;
            }
            l
        };
        let stages = build_stages(&cfg, &modp, &rts);
        assert_eq!(stages.len(), cfg.double_blocks + cfg.single_blocks);
        let inputs = [inp.img.clone(), inp.txt.clone()];
        let seed = |outs: &[Tensor]| -> Result<Tensor> {
            let joint = &outs[0];
            let final_txt = joint.narrow(0, 0, cfg.txt_len)?;
            let final_img = joint.narrow(0, cfg.txt_len, cfg.img_seq)?;
            mse(&final_img, &inp.target_img)?.add(&mse(&final_txt, &inp.target_txt)?)
        };
        let mut loss_ckpt = 0.0f32;
        for s in 1..=steps {
            loss_ckpt = ckpt.step_ckpt(s, &stages, &inputs, &seed)?;
        }

        assert!(
            loss_full.is_finite() && loss_ckpt.is_finite(),
            "loss not finite: full {loss_full} / checkpointed {loss_ckpt}"
        );
        // Same weights, same inputs, same step count: the two paths must land on
        // the same step loss.
        assert!(
            (loss_full - loss_ckpt).abs() < 1e-3,
            "step-{steps} losses disagree: full {loss_full} vs checkpointed {loss_ckpt}"
        );

        // And the LoRA factors they updated to must match, which is the real
        // check: a stage wired wrong in `build_stages` updates the wrong
        // tensors or misses a gradient.
        let mut failed: Vec<String> = Vec::new();
        for ((key, a), (key2, b)) in full.vars.iter().zip(ckpt.vars.iter()) {
            assert_eq!(key, key2);
            if !compare(key, a.as_tensor(), b.as_tensor())? {
                failed.push(key.clone());
            }
        }
        assert!(
            failed.is_empty(),
            "checkpointed and full-graph LoRA factors disagree: {}",
            failed.join(", ")
        );

        println!(
            "RESULT: PASS — tiny stack (dim {}, heads {}x{}, {} double + {} single, \
             {} img + {} txt tokens, {params} base params, {} LoRA tensors) ran \
             {steps} forward+backward+AdamW steps in both modes; step-{steps} loss \
             {loss_full:.5} full vs {loss_ckpt:.5} checkpointed",
            cfg.dim,
            cfg.heads,
            cfg.head_dim,
            cfg.double_blocks,
            cfg.single_blocks,
            cfg.img_seq,
            cfg.txt_len,
            full.vars.len()
        );
        Ok(())
    }
}
