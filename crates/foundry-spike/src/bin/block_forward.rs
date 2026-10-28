//! Spike item 0.5.3: forward pass of the fixture block pair in Candle bf16 on the GPU,
//! compared against the committed PyTorch reference (tools/fixtures/block_bf16.safetensors,
//! semantics per tools/fixtures/PROVENANCE.md).
//! Acceptance: every recorded activation within the TESTING.md bf16 tolerance (rel L2 <= 2e-2).

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

// Must match CONFIG in tools/fixtures/gen_block.py (checked via fixture shapes).
const DIM: usize = 256;
const HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const ROPE_AXES: [usize; 4] = [32, 32, 32, 32];
const ROPE_THETA: f32 = 10000.0;
const IMG_GRID: usize = 4;
const IMG_SEQ: usize = IMG_GRID * IMG_GRID;
const TXT_LEN: usize = 8;
const ALL_SEQ: usize = IMG_SEQ + TXT_LEN;
const NORM_EPS: f32 = 1e-6;
const LORA_SCALE: f64 = 2.0; // alpha 8.0 / rank 4
const REL_L2_MAX: f64 = 2e-2;

type Weights = HashMap<String, Tensor>;

struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Tensor> {
    w.get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture tensor missing: {key}")))
}

// ----- ops (reference semantics from PROVENANCE.md) ---------------------------

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // statistics in fp32; (x*scale) cast back to compute dtype, then * weight
    let xf = x.to_dtype(DType::F32)?;
    let scale = xf
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1.0, eps as f64)?
        .sqrt()?;
    let y = xf
        .broadcast_div(&scale)?
        .to_dtype(x.dtype())?
        .broadcast_mul(weight)?;
    Ok(y)
}

fn layer_norm_mod(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
    // LayerNorm(affine=False) in fp32, then adaLN (1 + scale) * norm + shift, cast back
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let centered = xf.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = centered.broadcast_div(&var.affine(1.0, eps as f64)?.sqrt()?)?;
    let m = normed
        .broadcast_mul(&scale.to_dtype(DType::F32)?.affine(1.0, 1.0)?)?
        .broadcast_add(&shift.to_dtype(DType::F32)?)?;
    m.to_dtype(x.dtype())
}

fn rope_tables(positions: &[[f32; 4]], device: &Device) -> Result<RopeTables> {
    // fp32 tables, cast to compute dtype (the reference casts before use)
    let mut cos = Vec::with_capacity(positions.len() * HEAD_DIM);
    let mut sin = Vec::with_capacity(positions.len() * HEAD_DIM);
    for pos in positions {
        for (axis, &dim_a) in ROPE_AXES.iter().enumerate() {
            let half = dim_a / 2;
            let mut freqs = Vec::with_capacity(half);
            for j in 0..half {
                let inv = ROPE_THETA.powf(-2.0 * j as f32 / dim_a as f32);
                freqs.push(pos[axis] * inv);
            }
            for _ in 0..2 {
                for &f in &freqs {
                    cos.push(f.cos());
                    sin.push(f.sin());
                }
            }
        }
    }
    let shape = (positions.len(), HEAD_DIM);
    let cos = Tensor::from_vec(cos, shape, device)?.to_dtype(DType::BF16)?;
    let sin = Tensor::from_vec(sin, shape, device)?.to_dtype(DType::BF16)?;
    Ok(RopeTables { cos, sin })
}

fn rope_rotate_half(x: &Tensor) -> Result<Tensor> {
    // per-axis segment: [x1 | x2] -> [-x2 | x1], with a running offset
    let mut parts: Vec<Tensor> = Vec::new();
    let mut off = 0;
    for dim_a in ROPE_AXES {
        let half = dim_a / 2;
        let seg = x.narrow(D::Minus1, off, dim_a)?;
        parts.push(seg.narrow(D::Minus1, half, half)?.neg()?);
        parts.push(seg.narrow(D::Minus1, 0, half)?);
        off += dim_a;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (seq, heads, head_dim) compute dtype; cos/sin: (seq, head_dim) compute dtype
    let xf = x.to_dtype(DType::F32)?;
    let rot = rope_rotate_half(&xf)?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(1)?; // (seq, 1, head_dim)
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(1)?;
    (xf.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?)?.to_dtype(x.dtype())
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // q,k,v: (seq, heads, head_dim); scores in fp32, softmax in fp32, probs cast back
    let qt = q.contiguous()?.transpose(0, 1)?; // (heads, seq, head_dim)
    let kt = k.contiguous()?.transpose(0, 1)?;
    let vt = v.contiguous()?.transpose(0, 1)?;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    // contiguous(): candle's CUDA matmul rejects batch-transposed views (overlapping
    // batch strides); the copies do not change values.
    let scores = qt
        .to_dtype(DType::F32)?
        .matmul(&kt.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let probs = candle_nn::ops::softmax(&scores, 2)?.to_dtype(q.dtype())?;
    probs.matmul(&vt.contiguous()?)?.transpose(0, 1)
}

fn linear(w: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
    // y = W x + b + (alpha/rank) * B (A x) when LoRA tensors exist
    let y = x
        .matmul(&get(w, &format!("weights/{name}.weight"))?.t()?)?
        .broadcast_add(get(w, &format!("weights/{name}.bias"))?)?;
    let a = w.get(&format!("weights/{name}.lora_a"));
    let b = w.get(&format!("weights/{name}.lora_b"));
    if let (Some(a), Some(b)) = (a, b) {
        let lo = x
            .matmul(&a.t()?)?
            .matmul(&b.t()?)?
            .affine(LORA_SCALE, 0.0)?;
        return y.broadcast_add(&lo);
    }
    Ok(y)
}

fn swiglu(w: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&linear(w, &format!("{name}.gate"), x)?)?;
    let up = linear(w, &format!("{name}.up"), x)?;
    linear(w, &format!("{name}.down"), &gate.mul(&up)?)
}

// ----- blocks -----------------------------------------------------------------

fn double_block(
    w: &Weights,
    acts: &mut Vec<(String, Tensor)>,
    img: &Tensor,
    txt: &Tensor,
    temb: &Tensor,
    rt_i: &RopeTables,
    rt_t: &RopeTables,
) -> Result<(Tensor, Tensor)> {
    // the reference feeds silu(temb), not temb, into the modulation linear
    let modout = linear(
        w,
        "double.mod",
        &candle_nn::ops::silu(temb)?.reshape((1, DIM))?,
    )?
    .reshape(6 * DIM)?;
    let part = |i: usize| modout.narrow(0, i * DIM, DIM);
    let (sh1, sc1, g1, sh2, sc2, g2) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);

    let ih = layer_norm_mod(img, &sh1, &sc1, NORM_EPS)?;
    let th = layer_norm_mod(txt, &sh1, &sc1, NORM_EPS)?;
    acts.push(("double.img_norm1".into(), ih.clone()));
    acts.push(("double.txt_norm1".into(), th.clone()));

    // qk norm weight names use the module short name: double.img_q_norm etc.
    let q_i = {
        let h = linear(w, "double.img.to_q", &ih)?.reshape((IMG_SEQ, HEADS, HEAD_DIM))?;
        apply_rope(
            &rms_norm(&h, get(w, "weights/double.img_q_norm.weight")?, NORM_EPS)?,
            &rt_i.cos,
            &rt_i.sin,
        )?
    };
    let k_i = {
        let h = linear(w, "double.img.to_k", &ih)?.reshape((IMG_SEQ, HEADS, HEAD_DIM))?;
        apply_rope(
            &rms_norm(&h, get(w, "weights/double.img_k_norm.weight")?, NORM_EPS)?,
            &rt_i.cos,
            &rt_i.sin,
        )?
    };
    let v_i = linear(w, "double.img.to_v", &ih)?.reshape((IMG_SEQ, HEADS, HEAD_DIM))?;
    let q_t = {
        let h = linear(w, "double.txt.to_q", &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;
        apply_rope(
            &rms_norm(&h, get(w, "weights/double.txt_q_norm.weight")?, NORM_EPS)?,
            &rt_t.cos,
            &rt_t.sin,
        )?
    };
    let k_t = {
        let h = linear(w, "double.txt.to_k", &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;
        apply_rope(
            &rms_norm(&h, get(w, "weights/double.txt_k_norm.weight")?, NORM_EPS)?,
            &rt_t.cos,
            &rt_t.sin,
        )?
    };
    let v_t = linear(w, "double.txt.to_v", &th)?.reshape((TXT_LEN, HEADS, HEAD_DIM))?;
    acts.push(("double.q".into(), q_i.clone()));
    acts.push(("double.k".into(), k_i.clone()));

    let q = Tensor::cat(&[&q_t, &q_i], 0)?;
    let k = Tensor::cat(&[&k_t, &k_i], 0)?;
    let v = Tensor::cat(&[&v_t, &v_i], 0)?;
    let attn = attention(&q, &k, &v)?;
    let a_t = attn.narrow(0, 0, TXT_LEN)?;
    let a_i = attn.narrow(0, TXT_LEN, IMG_SEQ)?;
    acts.push(("double.attn_img".into(), a_i.clone()));

    let img = (img
        + g1.broadcast_mul(&linear(
            w,
            "double.img.to_out",
            &a_i.contiguous()?.reshape((IMG_SEQ, DIM))?,
        )?)?)?;
    let txt = (txt
        + g1.broadcast_mul(&linear(
            w,
            "double.txt.to_out",
            &a_t.contiguous()?.reshape((TXT_LEN, DIM))?,
        )?)?)?;
    acts.push(("double.img_post_attn".into(), img.clone()));

    let ih2 = layer_norm_mod(&img, &sh2, &sc2, NORM_EPS)?;
    let th2 = layer_norm_mod(&txt, &sh2, &sc2, NORM_EPS)?;
    let img = (img + g2.broadcast_mul(&swiglu(w, "double.img.mlp", &ih2)?)?)?;
    let txt = (txt + g2.broadcast_mul(&swiglu(w, "double.txt.mlp", &th2)?)?)?;
    acts.push(("double.img_out".into(), img.clone()));
    acts.push(("double.txt_out".into(), txt.clone()));
    Ok((img, txt))
}

fn single_block(
    w: &Weights,
    acts: &mut Vec<(String, Tensor)>,
    x: &Tensor,
    temb: &Tensor,
    rt: &RopeTables,
) -> Result<Tensor> {
    let modout = linear(
        w,
        "single.mod",
        &candle_nn::ops::silu(temb)?.reshape((1, DIM))?,
    )?
    .reshape(6 * DIM)?;
    let part = |i: usize| modout.narrow(0, i * DIM, DIM);
    // chunk order matches the reference; the single block has no second adaLN shift/scale
    let (sh1, sc1, g_at, _sh2, _sc2, g_mlp) =
        (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?, part(5)?);

    let nh = layer_norm_mod(x, &sh1, &sc1, NORM_EPS)?;
    acts.push(("single.norm1".into(), nh.clone()));

    let qkv = linear(w, "single.to_qkv", &nh)?.reshape((ALL_SEQ, 3, DIM))?;
    let split = |which: usize| -> Result<Tensor> {
        qkv.narrow(1, which, 1)?
            .contiguous()?
            .reshape((ALL_SEQ, HEADS, HEAD_DIM))
    };
    let q = apply_rope(
        &rms_norm(
            &split(0)?,
            get(w, "weights/single.q_norm.weight")?,
            NORM_EPS,
        )?,
        &rt.cos,
        &rt.sin,
    )?;
    let k = apply_rope(
        &rms_norm(
            &split(1)?,
            get(w, "weights/single.k_norm.weight")?,
            NORM_EPS,
        )?,
        &rt.cos,
        &rt.sin,
    )?;
    let v = split(2)?;
    acts.push(("single.q".into(), q.clone()));
    acts.push(("single.k".into(), k.clone()));

    let attn = attention(&q, &k, &v)?
        .contiguous()?
        .reshape((ALL_SEQ, DIM))?;
    acts.push(("single.attn".into(), attn.clone()));

    let mlp = swiglu(w, "single.mlp", &nh)?;
    acts.push(("single.mlp".into(), mlp.clone()));

    let out = (x + g_at.broadcast_mul(&attn)? + g_mlp.broadcast_mul(&mlp)?)?;
    acts.push(("single.out".into(), out.clone()));
    Ok(out)
}

// ----- driver ------------------------------------------------------------------

fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let d = a
        .to_dtype(DType::F32)?
        .broadcast_sub(&b.to_dtype(DType::F32)?)?;
    d.sqr()?.mean_all()
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let gpu_name = cudarc::driver::CudaContext::new(0)
        .and_then(|c| c.name())
        .unwrap_or_else(|_| "unknown".into());
    println!("device: {gpu_name}");

    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_bf16.safetensors");
    let w: Weights = safetensors::load(&path, &dev)?;
    println!("fixture: {} ({} tensors)", path.display(), w.len());

    let img = get(&w, "inputs/img")?;
    let txt = get(&w, "inputs/txt")?;
    let temb = get(&w, "inputs/temb")?;
    assert_eq!(img.dims(), &[IMG_SEQ, DIM]);
    assert_eq!(txt.dims(), &[TXT_LEN, DIM]);

    // 4-axis positions: img axis0=row, axis1=col; txt axis0=index; axes 2-3 zero.
    let pos_img: Vec<[f32; 4]> = (0..IMG_SEQ)
        .map(|i| [(i / IMG_GRID) as f32, (i % IMG_GRID) as f32, 0.0, 0.0])
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rt_i = rope_tables(&pos_img, &dev)?;
    let rt_t = rope_tables(&pos_txt, &dev)?;
    let rt_a = rope_tables(&pos_all, &dev)?;

    let mut acts: Vec<(String, Tensor)> = Vec::new();
    let (img2, txt2) = double_block(&w, &mut acts, img, txt, temb, &rt_i, &rt_t)?;
    let joint = Tensor::cat(&[&txt2, &img2], 0)?;
    let joint_out = single_block(&w, &mut acts, &joint, temb, &rt_a)?;
    let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
    let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;
    acts.push(("output/img".into(), final_img.clone()));
    acts.push(("output/txt".into(), final_txt.clone()));

    let loss = mse(&final_img, get(&w, "inputs/target_img")?)?
        .add(&mse(&final_txt, get(&w, "inputs/target_txt")?)?)?;
    acts.push(("loss".into(), loss.clone()));

    // Compare every recorded activation against the fixture (rel L2, f32 math).
    let mut fail = false;
    for (name, ours) in &acts {
        let ref_t = get(&w, name)?;
        let a = ours.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = ref_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if a.len() != b.len() {
            println!(
                "{name:24} SHAPE MISMATCH {} vs {}",
                ours.dims().len(),
                ref_t.dims().len()
            );
            fail = true;
            continue;
        }
        let num: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2) as f64).sum();
        let den: f64 = b.iter().map(|y| (y * y) as f64).sum::<f64>().sqrt();
        let rel = num.sqrt() / (den + 1e-12);
        let ok = rel <= REL_L2_MAX;
        fail |= !ok;
        println!(
            "{name:24} rel_l2 = {rel:.3e} {}",
            if ok { "" } else { "<-- FAIL" }
        );
    }
    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
