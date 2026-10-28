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
//! Assumptions (printed at startup; the real config is not fully pinned down
//! yet — see `MODEL_SUPPORT.md` open items):
//! - token counts: 16x total pixel->token downscale (VAE + patchify), so
//!   512px -> 32x32 = 1024 image tokens, 1024px -> 64x64 = 4096; txt 512.
//! - mlp hidden = 4 x dim (FLUX convention). The resulting stack is ~5.9B
//!   params, heavier than the real ~4B, so VRAM numbers are conservative
//!   (pessimistic) and step times are per-op-shape representative.
//! - LoRA rank 16 (scale 2.0) on double img to_q/to_v and single to_qkv —
//!   the fixture's target pattern; real target modules/rank still open.
//! - RoPE: the lib's 4-axis [32,32,32,32] tables (head_dim 128 matches the
//!   real model, so `AttnBwd` applies unchanged; axis dims are still open per
//!   `RESEARCH.md` and throughput-neutral).
//!
//! Run with Strata unloaded (CLAUDE.md testing rule): NVML peak includes any
//! other GPU users. Writes nothing; random weights, no checkpoint downloads.

use candle_core::{DType, Device, Result, Tensor, Var};
use foundry_spike::{
    adamw_step, apply_rope, get, layer_norm_mod, linear, mse, rms_norm, rope_tables_dtype, swiglu,
    AdamW, AttnBwd, RopeTables, Weights, NORM_EPS,
};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Real klein transformer dims (MODEL_SUPPORT.md).
const DIM: usize = 3072;
const HEADS: usize = 24;
const HEAD_DIM: usize = 128;
const DOUBLE_BLOCKS: usize = 5;
const SINGLE_BLOCKS: usize = 20;
// Open item; 512 is the common caption-token budget. Throughput-relevant but
// small next to the image stream.
const TXT_LEN: usize = 512;
const MLP_HIDDEN: usize = 4 * DIM;
const LORA_RANK: usize = 16;
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

/// NVML errors into Candle's error type so `?` works throughout.
fn nvml_err(e: NvmlError) -> candle_core::Error {
    candle_core::Error::Msg(format!("nvml: {e}"))
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

struct Cfg {
    img_seq: usize,
    img_grid: usize,
}

impl Cfg {
    fn all_seq(&self) -> usize {
        self.img_seq + TXT_LEN
    }
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

fn rand_w(dev: &Device, shape: (usize, usize)) -> Result<Tensor> {
    Tensor::randn(0.0f32, 0.02, shape, dev)?
        .to_dtype(DType::BF16)?
        .contiguous()
}

/// Add a linear (weight + zero bias; the lib's `linear` requires the bias
/// key). Optionally its LoRA A/B factors: A randn, B zero — standard LoRA
/// init, contribution starts at zero. Returns the base parameter count.
fn add_linear(
    w: &mut Weights,
    dev: &Device,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    lora: bool,
) -> Result<usize> {
    w.insert(
        format!("weights/{name}.weight"),
        rand_w(dev, (out_dim, in_dim))?,
    );
    w.insert(
        format!("weights/{name}.bias"),
        Tensor::zeros((out_dim,), DType::BF16, dev)?,
    );
    if lora {
        w.insert(
            format!("weights/{name}.lora_a"),
            Tensor::randn(0.0f32, 0.02, (LORA_RANK, in_dim), dev)?
                .to_dtype(DType::BF16)?
                .contiguous()?,
        );
        w.insert(
            format!("weights/{name}.lora_b"),
            // (out, rank): the lib's `linear` multiplies by b.t().
            Tensor::zeros((out_dim, LORA_RANK), DType::BF16, dev)?,
        );
    }
    Ok(out_dim * in_dim + out_dim)
}

fn add_mlp(w: &mut Weights, dev: &Device, name: &str) -> Result<usize> {
    let mut n = 0;
    for (t, out_d, in_d) in [
        ("gate", MLP_HIDDEN, DIM),
        ("up", MLP_HIDDEN, DIM),
        ("down", DIM, MLP_HIDDEN),
    ] {
        n += add_linear(w, dev, &format!("{name}.{t}"), out_d, in_d, false)?;
    }
    Ok(n)
}

fn add_qk_norms(w: &mut Weights, dev: &Device, q_key: &str, k_key: &str) -> Result<usize> {
    w.insert(
        format!("weights/{q_key}.weight"),
        Tensor::ones((HEAD_DIM,), DType::BF16, dev)?,
    );
    w.insert(
        format!("weights/{k_key}.weight"),
        Tensor::ones((HEAD_DIM,), DType::BF16, dev)?,
    );
    Ok(2 * HEAD_DIM)
}

/// Build the whole 5+20 block stack with random bf16 weights. Key names
/// follow the lib's `weights/{name}.{weight|bias|lora_a|lora_b}` pattern so
/// `linear`/`swiglu` work unchanged.
fn build_stack(dev: &Device) -> Result<(Weights, usize)> {
    let mut w = Weights::new();
    let mut params = 0usize;
    for i in 0..DOUBLE_BLOCKS {
        let p = format!("double.{i}");
        params += add_linear(&mut w, dev, &format!("{p}.mod"), 6 * DIM, DIM, false)?;
        for s in ["img", "txt"] {
            for t in ["to_q", "to_k", "to_v", "to_out"] {
                let lora = s == "img" && (t == "to_q" || t == "to_v");
                params += add_linear(&mut w, dev, &format!("{p}.{s}.{t}"), DIM, DIM, lora)?;
            }
            params += add_qk_norms(
                &mut w,
                dev,
                &format!("{p}.{s}_q_norm"),
                &format!("{p}.{s}_k_norm"),
            )?;
            params += add_mlp(&mut w, dev, &format!("{p}.{s}.mlp"))?;
        }
    }
    for i in 0..SINGLE_BLOCKS {
        let p = format!("single.{i}");
        params += add_linear(&mut w, dev, &format!("{p}.mod"), 6 * DIM, DIM, false)?;
        params += add_linear(&mut w, dev, &format!("{p}.to_qkv"), 3 * DIM, DIM, true)?;
        params += add_linear(&mut w, dev, &format!("{p}.to_out"), DIM, DIM, false)?;
        params += add_qk_norms(&mut w, dev, &format!("{p}.q_norm"), &format!("{p}.k_norm"))?;
        params += add_mlp(&mut w, dev, &format!("{p}.mlp"))?;
    }
    Ok((w, params))
}

/// One double-stream block at the real dims (same wiring as the lib's
/// fixture-sized `double_block`; temb_silu is silu(temb), precomputed once
/// per step since every block shares the same adaLN input).
fn double_block(
    b: &Blk,
    img: &Tensor,
    txt: &Tensor,
    temb_silu: &Tensor,
    rt_i: &RopeTables,
    rt_t: &RopeTables,
) -> Result<(Tensor, Tensor)> {
    let seq_i = b.cfg.img_seq;
    let modout = linear(b.w, &format!("{}.mod", b.p), temb_silu)?.reshape(6 * DIM)?;
    let part = |i: usize| modout.narrow(0, i * DIM, DIM);
    let (sh1, sc1, g1, sh2, sc2, g2) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);

    let ih = layer_norm_mod(img, &sh1, &sc1, NORM_EPS)?;
    let th = layer_norm_mod(txt, &sh1, &sc1, NORM_EPS)?;

    let q_i = {
        let h =
            linear(b.w, &format!("{}.img.to_q", b.p), &ih)?.reshape((seq_i, HEADS, HEAD_DIM))?;
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
        let h =
            linear(b.w, &format!("{}.img.to_k", b.p), &ih)?.reshape((seq_i, HEADS, HEAD_DIM))?;
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
    let v_i = linear(b.w, &format!("{}.img.to_v", b.p), &ih)?.reshape((seq_i, HEADS, HEAD_DIM))?;
    let q_t = {
        let h =
            linear(b.w, &format!("{}.txt.to_q", b.p), &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;
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
        let h =
            linear(b.w, &format!("{}.txt.to_k", b.p), &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;
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
    let v_t =
        linear(b.w, &format!("{}.txt.to_v", b.p), &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;

    let q = Tensor::cat(&[&q_t, &q_i], 0)?;
    let k = Tensor::cat(&[&k_t, &k_i], 0)?;
    let v = Tensor::cat(&[&v_t, &v_i], 0)?;
    let attn = q.apply_op3(&k, &v, AttnBwd)?;
    let a_t = attn.narrow(0, 0, TXT_LEN)?;
    let a_i = attn.narrow(0, TXT_LEN, seq_i)?;

    let img = (img
        + g1.broadcast_mul(&linear(
            b.w,
            &format!("{}.img.to_out", b.p),
            &a_i.contiguous()?.reshape((seq_i, DIM))?,
        )?)?)?;
    let txt = (txt
        + g1.broadcast_mul(&linear(
            b.w,
            &format!("{}.txt.to_out", b.p),
            &a_t.contiguous()?.reshape((TXT_LEN, DIM))?,
        )?)?)?;

    let ih2 = layer_norm_mod(&img, &sh2, &sc2, NORM_EPS)?;
    let th2 = layer_norm_mod(&txt, &sh2, &sc2, NORM_EPS)?;
    let img = (img + g2.broadcast_mul(&swiglu(b.w, &format!("{}.img.mlp", b.p), &ih2)?)?)?;
    let txt = (txt + g2.broadcast_mul(&swiglu(b.w, &format!("{}.txt.mlp", b.p), &th2)?)?)?;
    Ok((img, txt))
}

/// One single-stream block at the real dims (same wiring as the lib's
/// fixture-sized `single_block`, including the fused-MLP-on-normed-input
/// shortcut).
fn single_block(b: &Blk, x: &Tensor, temb_silu: &Tensor, rt: &RopeTables) -> Result<Tensor> {
    let all = b.cfg.all_seq();
    let modout = linear(b.w, &format!("{}.mod", b.p), temb_silu)?.reshape(6 * DIM)?;
    let part = |i: usize| modout.narrow(0, i * DIM, DIM);
    let (sh1, sc1, g_at, _sh2, _sc2, g_mlp) =
        (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);
    let nh = layer_norm_mod(x, &sh1, &sc1, NORM_EPS)?;
    let qkv = linear(b.w, &format!("{}.to_qkv", b.p), &nh)?.reshape((all, 3, DIM))?;
    let split = |which: usize| -> Result<Tensor> {
        qkv.narrow(1, which, 1)?
            .contiguous()?
            .reshape((all, HEADS, HEAD_DIM))
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
        .reshape((all, DIM))?;
    let mlp = swiglu(b.w, &format!("{}.mlp", b.p), &nh)?;
    x + g_at.broadcast_mul(&attn)? + g_mlp.broadcast_mul(&mlp)?
}

fn forward(w: &Weights, cfg: &Cfg, inp: &Inputs, rts: &Rts) -> Result<(Tensor, Tensor)> {
    let mut img = inp.img.clone();
    let mut txt = inp.txt.clone();
    for i in 0..DOUBLE_BLOCKS {
        let b = Blk {
            w,
            cfg,
            p: format!("double.{i}"),
        };
        (img, txt) = double_block(&b, &img, &txt, &inp.temb_silu, &rts.img, &rts.txt)?;
    }
    let mut joint = Tensor::cat(&[&txt, &img], 0)?;
    for i in 0..SINGLE_BLOCKS {
        let b = Blk {
            w,
            cfg,
            p: format!("single.{i}"),
        };
        joint = single_block(&b, &joint, &inp.temb_silu, &rts.joint)?;
    }
    let final_txt = joint.narrow(0, 0, TXT_LEN)?;
    let final_img = joint.narrow(0, TXT_LEN, cfg.img_seq)?;
    Ok((final_img, final_txt))
}

impl Session {
    /// One full training step: forward graph, MSE loss, backward, AdamW on
    /// the LoRA factors (identical structure to `stability.rs`).
    fn step(&mut self, step: usize, cfg: &Cfg, inp: &Inputs, rts: &Rts) -> Result<f32> {
        let (final_img, final_txt) = forward(&self.w, cfg, inp, rts)?;
        let loss = mse(&final_img, &inp.target_img)?.add(&mse(&final_txt, &inp.target_txt)?)?;
        let grads = loss.backward()?;
        for (key, var) in &self.vars {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
            let mm = self.m.get_mut(key).unwrap();
            let vv = self.v.get_mut(key).unwrap();
            adamw_step(var, g, mm, vv, step, &OPT)?;
        }
        loss.to_vec0::<f32>()
    }
}

fn train_burst(
    s: &mut Session,
    cfg: &Cfg,
    inp: &Inputs,
    rts: &Rts,
) -> Result<(Vec<f32>, Duration)> {
    for step in 1..=WARMUP_STEPS {
        s.step(step, cfg, inp, rts)?;
    }
    let started = Instant::now();
    let mut losses = Vec::with_capacity(TIMED_STEPS);
    for step in (WARMUP_STEPS + 1)..=(WARMUP_STEPS + TIMED_STEPS) {
        losses.push(s.step(step, cfg, inp, rts)?);
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

fn run_config(label: &str, cfg: &Cfg) -> Result<()> {
    println!(
        "--- {label}: img_seq {} (grid {}x{}), txt {TXT_LEN}, joint {} ---",
        cfg.img_seq,
        cfg.img_grid,
        cfg.img_grid,
        cfg.all_seq()
    );
    let started = Instant::now();
    let dev = Device::new_cuda(0)?;
    let (mut w, params) = build_stack(&dev)?;
    println!(
        "{label}: built {params} base params ({:.2} GB bf16) in {:?}",
        gb((params * 2) as u64),
        started.elapsed()
    );

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
    let trainable: usize = vars
        .iter()
        .map(|(_, v)| v.as_tensor().shape().elem_count())
        .sum();
    let mut m: HashMap<String, Tensor> = HashMap::new();
    let mut v: HashMap<String, Tensor> = HashMap::new();
    for (key, var) in &vars {
        m.insert(
            key.clone(),
            Tensor::zeros(var.as_tensor().shape(), DType::F32, &dev)?,
        );
        v.insert(
            key.clone(),
            Tensor::zeros(var.as_tensor().shape(), DType::F32, &dev)?,
        );
    }
    println!(
        "{label}: {} trainable LoRA tensors, {trainable} params",
        vars.len()
    );

    let inp = Inputs {
        img: Tensor::randn(0.0f32, 0.02, (cfg.img_seq, DIM), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?,
        txt: Tensor::randn(0.0f32, 0.02, (TXT_LEN, DIM), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?,
        temb_silu: Tensor::randn(0.0f32, 0.02, (1, DIM), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?,
        target_img: Tensor::randn(0.0f32, 1.0, (cfg.img_seq, DIM), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?,
        target_txt: Tensor::randn(0.0f32, 1.0, (TXT_LEN, DIM), &dev)?
            .to_dtype(DType::BF16)?
            .contiguous()?,
    };
    let pos_img: Vec<[f32; 4]> = (0..cfg.img_seq)
        .map(|i| {
            [
                (i / cfg.img_grid) as f32,
                (i % cfg.img_grid) as f32,
                0.0,
                0.0,
            ]
        })
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rts = Rts {
        img: rope_tables_dtype(&pos_img, &dev, DType::BF16)?,
        txt: rope_tables_dtype(&pos_txt, &dev, DType::BF16)?,
        joint: rope_tables_dtype(&pos_all, &dev, DType::BF16)?,
    };

    let mut s = Session { w, vars, m, v };
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_sampler(Arc::clone(&stop));
    let result = train_burst(&mut s, cfg, &inp, &rts);
    stop.store(true, Ordering::Relaxed);
    let (min_free, total) = sampler
        .join()
        .map_err(|_| candle_core::Error::Msg("vram sampler thread panicked".into()))?
        .map_err(nvml_err)?;
    let (losses, elapsed) = result?;
    let ms = elapsed.as_secs_f64() * 1e3 / TIMED_STEPS as f64;
    println!("{label}: {TIMED_STEPS} timed steps in {elapsed:?} -> {ms:.0} ms/step");
    println!("{label}: timed-step losses {losses:?}");
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
    Ok(())
}

fn main() {
    println!("assumptions (real config not fully pinned; see MODEL_SUPPORT.md open items):");
    println!(
        "  tokens: {TOKEN_DOWNSCALE}x pixel->token downscale => 512px = {} img tokens, \
         1024px = {} img tokens; txt stream {TXT_LEN}",
        (512 / TOKEN_DOWNSCALE).pow(2),
        (1024 / TOKEN_DOWNSCALE).pow(2)
    );
    println!(
        "  mlp hidden {} (4x dim; stack ~5.9B params, heavier than the real ~4B => VRAM \
         numbers are conservative)",
        MLP_HIDDEN
    );
    println!(
        "  LoRA rank {LORA_RANK} scale 2.0 on double img to_q/to_v + single to_qkv; \
         rope axes {:?} (throughput-neutral); random weights; nothing written to disk",
        foundry_spike::ROPE_AXES
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
        let grid = px / TOKEN_DOWNSCALE;
        let cfg = Cfg {
            img_seq: grid * grid,
            img_grid: grid,
        };
        match run_config(label, &cfg) {
            Ok(()) => measured += 1,
            Err(e) => {
                let msg = format!("{e:?}").to_lowercase();
                if msg.contains("out of memory")
                    || msg.contains("out_of_memory")
                    || msg.contains("outofmemory")
                {
                    println!("{label}: OUT OF MEMORY — not usable at this size on this GPU");
                } else {
                    println!("{label}: FAILED: {e:?}");
                }
            }
        }
    }
    println!(
        "RESULT: {measured} of 2 configurations measured; \
         the usable-or-not judgement goes to BUILD_LOG.md"
    );
}
