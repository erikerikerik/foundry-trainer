//! Shared building blocks for the later Phase 0.5 spike bins (0.5.5 onward).
//!
//! The fixture block pair replayed in Candle — ops and reference semantics are
//! identical to `bin/block_forward.rs` (item 0.5.3) — but attention is routed
//! through the hand-written `AttnBwd` custom op from `bin/attn_backward.rs`
//! (item 0.5.4), so backward through attention uses the production path.
//! The 0.5.3/0.5.4 bins keep their self-contained copies as recorded PASS
//! artifacts; this lib is where the code evolves from.

use candle_core::{
    backend::BackendStorage, backprop::GradStore, op::BackpropOp, CpuStorage, CudaStorage,
    CustomOp3, DType, Device, Layout, Result, Shape, Storage, Tensor, Var, D,
};
use std::collections::HashMap;

// Must match CONFIG in tools/fixtures/gen_block.py (checked via fixture shapes).
pub const DIM: usize = 256;
pub const HEADS: usize = 2;
pub const HEAD_DIM: usize = 128;
pub const ROPE_AXES: [usize; 4] = [32, 32, 32, 32];
pub const ROPE_THETA: f32 = 10000.0;
pub const IMG_GRID: usize = 4;
pub const IMG_SEQ: usize = IMG_GRID * IMG_GRID;
pub const TXT_LEN: usize = 8;
pub const ALL_SEQ: usize = IMG_SEQ + TXT_LEN;
pub const NORM_EPS: f32 = 1e-6;
pub const LORA_SCALE: f64 = 2.0; // alpha 8.0 / rank 4
pub const REL_L2_MAX: f64 = 2e-2;
pub const COS_MIN: f64 = 0.999;

pub type Weights = HashMap<String, Tensor>;

pub struct RopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
}

pub fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Tensor> {
    w.get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture tensor missing: {key}")))
}

// ----- ops (reference semantics from tools/fixtures/PROVENANCE.md) ------------

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
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

pub fn layer_norm_mod(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f32) -> Result<Tensor> {
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

pub fn rope_tables(positions: &[[f32; 4]], device: &Device) -> Result<RopeTables> {
    rope_tables_dtype(positions, device, DType::BF16)
}

pub fn rope_tables_dtype(
    positions: &[[f32; 4]],
    device: &Device,
    dtype: DType,
) -> Result<RopeTables> {
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
    let cos = Tensor::from_vec(cos, shape, device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin, shape, device)?.to_dtype(dtype)?;
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

pub fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: (seq, heads, head_dim) compute dtype; cos/sin: (seq, head_dim) compute dtype
    let xf = x.to_dtype(DType::F32)?;
    let rot = rope_rotate_half(&xf)?;
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(1)?; // (seq, 1, head_dim)
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(1)?;
    (xf.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?)?.to_dtype(x.dtype())
}

/// The reference attention forward, recomputed inside the custom op so the op
/// is self-contained (this is what Candle records as the op's output).
fn attn_fwd(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // q,k,v: (seq, heads, head_dim); scores/softmax in fp32, probs cast back
    let qt = q.contiguous()?.transpose(0, 1)?; // (heads, seq, head_dim)
    let kt = k.contiguous()?.transpose(0, 1)?;
    let vt = v.contiguous()?.transpose(0, 1)?;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    // .contiguous() after the cast: in bf16 to_dtype materializes the view,
    // in fp32 it is a no-op and CUDA matmul would reject the batch-transposed
    // strides
    let scores = qt
        .to_dtype(DType::F32)?
        .contiguous()?
        .matmul(&kt.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let probs = candle_nn::ops::softmax(&scores, 2)?.to_dtype(q.dtype())?;
    probs.matmul(&vt.contiguous()?)?.transpose(0, 1)
}

pub struct AttnBwd;

impl CustomOp3 for AttnBwd {
    fn name(&self) -> &'static str {
        "foundry-attn-bwd"
    }

    fn cpu_fwd(
        &self,
        s_q: &CpuStorage,
        l_q: &Layout,
        s_k: &CpuStorage,
        l_k: &Layout,
        s_v: &CpuStorage,
        l_v: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        for l in [l_q, l_k, l_v] {
            match l.contiguous_offsets() {
                Some((0, n)) if n == l.shape().elem_count() => {}
                _ => candle_core::bail!("cpu_fwd requires contiguous offset-0 inputs"),
            }
        }
        let wrap = |s: &CpuStorage, l: &Layout| {
            Tensor::from_storage(
                Storage::Cpu(s.clone()),
                l.shape().clone(),
                BackpropOp::none(),
                false,
            )
        };
        let out = attn_fwd(&wrap(s_q, l_q), &wrap(s_k, l_k), &wrap(s_v, l_v))?.contiguous()?;
        let (st, _) = out.storage_and_layout();
        match &*st {
            Storage::Cpu(s) => Ok((s.clone(), out.shape().clone())),
            _ => candle_core::bail!("attn fwd produced non-cpu storage"),
        }
    }

    fn cuda_fwd(
        &self,
        s_q: &CudaStorage,
        l_q: &Layout,
        s_k: &CudaStorage,
        l_k: &Layout,
        s_v: &CudaStorage,
        l_v: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // try_clone (BackendStorage) materializes each layout into a contiguous
        // device slice, so from_storage's contiguous-stride assumption holds.
        let wrap = |s: &CudaStorage, l: &Layout| -> Result<Tensor> {
            Ok(Tensor::from_storage(
                Storage::Cuda(s.try_clone(l)?),
                l.shape().clone(),
                BackpropOp::none(),
                false,
            ))
        };
        let out = attn_fwd(&wrap(s_q, l_q)?, &wrap(s_k, l_k)?, &wrap(s_v, l_v)?)?.contiguous()?;
        let (st, layout) = out.storage_and_layout();
        match &*st {
            Storage::Cuda(s) => Ok((s.try_clone(layout)?, out.shape().clone())),
            _ => candle_core::bail!("attn fwd produced non-cuda storage"),
        }
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // Mirror of PyTorch autograd over attn_fwd's graph. Shapes in (h,s,d)
        // after the transposes; matmul operands contiguous (CUDA requirement).
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let qt = q.contiguous()?.transpose(0, 1)?.contiguous()?;
        let kt = k.contiguous()?.transpose(0, 1)?.contiguous()?;
        let vt = v.contiguous()?.transpose(0, 1)?.contiguous()?;
        let qf = qt.to_dtype(DType::F32)?;
        let kf = kt.to_dtype(DType::F32)?;
        // recompute the fp32 softmax probabilities (same inputs -> same values)
        let probs32 = candle_nn::ops::softmax(
            &qt.to_dtype(DType::F32)?
                .matmul(&kf.transpose(1, 2)?.contiguous()?)?
                .affine(scale, 0.0)?,
            2,
        )?;
        let probs = probs32.to_dtype(q.dtype())?;

        let dout = grad_res.contiguous()?.transpose(0, 1)?.contiguous()?;
        // matmul backward in the compute dtype: dprobs = dout @ vt^T, dV = probs^T @ dout
        let dprobs = dout.matmul(&vt.transpose(1, 2)?.contiguous()?)?;
        let dvt = probs.transpose(1, 2)?.contiguous()?.matmul(&dout)?;
        // softmax backward in fp32: dscores = p * (dp - sum(dp * p)) * scale
        let dpf = dprobs.to_dtype(DType::F32)?;
        let rowsum = dpf.mul(&probs32)?.sum_keepdim(2)?;
        let dscores = probs32
            .mul(&dpf.broadcast_sub(&rowsum)?)?
            .affine(scale, 0.0)?;
        // the scores matmul ran in fp32, so dQ/dK are fp32 matmuls cast back
        let dqt = dscores.matmul(&kf)?.to_dtype(q.dtype())?;
        let dkt = dscores
            .transpose(1, 2)?
            .contiguous()?
            .matmul(&qf)?
            .to_dtype(k.dtype())?;
        Ok((
            Some(dqt.transpose(0, 1)?),
            Some(dkt.transpose(0, 1)?),
            Some(dvt.transpose(0, 1)?),
        ))
    }
}

pub fn linear(w: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
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

pub fn swiglu(w: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
    let gate = candle_nn::ops::silu(&linear(w, &format!("{name}.gate"), x)?)?;
    let up = linear(w, &format!("{name}.up"), x)?;
    linear(w, &format!("{name}.down"), &gate.mul(&up)?)
}

// ----- blocks (attention via AttnBwd, the production backward path) -----------

pub fn double_block(
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
    let attn = q.apply_op3(&k, &v, AttnBwd)?;
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

pub fn single_block(
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

    let attn = q
        .apply_op3(&k, &v, AttnBwd)?
        .contiguous()?
        .reshape((ALL_SEQ, DIM))?;
    acts.push(("single.attn".into(), attn.clone()));

    let mlp = swiglu(w, "single.mlp", &nh)?;
    acts.push(("single.mlp".into(), mlp.clone()));

    let out = (x + g_at.broadcast_mul(&attn)? + g_mlp.broadcast_mul(&mlp)?)?;
    acts.push(("single.out".into(), out.clone()));
    Ok(out)
}

pub fn mse(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let d = a
        .to_dtype(DType::F32)?
        .broadcast_sub(&b.to_dtype(DType::F32)?)?;
    d.sqr()?.mean_all()
}

/// AdamW hyperparameters (weight decay fixed at 0 for the spike).
#[derive(Clone, Copy)]
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
}

/// One bias-corrected AdamW update (weight decay 0), matching PyTorch's AdamW
/// as used by the fixture generator: step_size = lr/(1-b1^t),
/// denom = sqrt(v)/(sqrt(1-b2^t)) + eps. Moments are fp32 (the production
/// choice); the parameter is updated in place in its own dtype via `Var::set`.
pub fn adamw_step(
    var: &Var,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    step: usize,
    opt: &AdamW,
) -> Result<()> {
    let AdamW {
        lr,
        beta1,
        beta2,
        eps,
    } = *opt;
    let g = grad.to_dtype(DType::F32)?;
    let mnew = m.affine(beta1, 0.0)?.add(&g.affine(1.0 - beta1, 0.0)?)?;
    let vnew = v
        .affine(beta2, 0.0)?
        .add(&g.sqr()?.affine(1.0 - beta2, 0.0)?)?;
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    let mhat = mnew.affine(1.0 / bc1, 0.0)?;
    let denom = vnew.affine(1.0 / bc2, 0.0)?.sqrt()?.affine(1.0, eps)?;
    let delta = mhat.broadcast_div(&denom)?.affine(lr, 0.0)?;
    let pnew = var
        .as_tensor()
        .to_dtype(DType::F32)?
        .broadcast_sub(&delta)?
        .to_dtype(var.as_tensor().dtype())?;
    var.set(&pnew)?;
    *m = mnew;
    *v = vnew;
    Ok(())
}

// ----- per-block gradient checkpointing -------------------------------------

/// One stage of a checkpointed forward/backward. The *same* closure runs
/// twice: once during the forward pass (with the LoRA factors passed in as
/// detached tensors, so no backprop node is recorded anywhere) and once again
/// during backward (with them as live variable leaves, so the graph is real).
/// `run` gets the weight map and the stage's inputs, and returns its outputs.
pub type StageFn<'a> = Box<dyn Fn(&Weights, &[Tensor]) -> Result<Vec<Tensor>> + 'a>;

pub struct Stage<'a> {
    pub run: StageFn<'a>,
}

/// Forward and backward keeping only each stage's *inputs* alive.
///
/// Forward: every stage runs graph-free (LoRA factors detached — same storage,
/// so still the current values), which means each stage's intermediates are
/// freed the moment it returns; the only tensors retained are `stage_inputs`.
/// Backward: stages are walked in reverse. A stage's saved inputs become
/// variable leaves, its graph is rebuilt by the same `run` closure, and its
/// upstream gradient is injected with the seed trick proven in item 0.5.4 —
/// `sum(out_j * upstream_j)`, whose backward hands `upstream_j` to the graph
/// unchanged. The stage's input gradients are then that stage's upstream
/// gradient for the next (earlier) stage.
///
/// Returns `(loss, gradients for inputs, gradients for lora_vars)`. Every LoRA
/// factor must be used by exactly one stage (per-block factors are disjoint,
/// which is how the fixture and the klein stack are built): a stage's backward
/// only yields gradients for leaves inside its own graph.
pub fn checkpointed_backward(
    base: &Weights,
    stages: &[Stage<'_>],
    inputs: &[Tensor],
    seed: impl FnOnce(&[Tensor]) -> Result<Tensor>,
    lora_vars: &[(String, Var)],
) -> Result<(f32, Vec<Tensor>, GradStore)> {
    if stages.is_empty() {
        return Err(candle_core::Error::Msg(
            "checkpointed_backward: no stages".into(),
        ));
    }
    let mut w_det = base.clone();
    for (key, var) in lora_vars {
        w_det.insert(key.clone(), var.as_detached_tensor());
    }
    let mut stage_inputs: Vec<Vec<Tensor>> = Vec::with_capacity(stages.len());
    let mut cur = inputs.to_vec();
    for st in stages {
        stage_inputs.push(cur.clone());
        cur = (st.run)(&w_det, &cur)
            .map_err(|e| candle_core::Error::Msg(format!("died in forward: {e:?}")))?;
    }

    // The final outputs become variable leaves: backward through the seed loss
    // is what produces their upstream gradients.
    let out_vars: Vec<Var> = cur.iter().map(Var::from_tensor).collect::<Result<_>>()?;
    let out_leafs: Vec<Tensor> = out_vars.iter().map(|v| v.as_tensor().clone()).collect();
    let loss = seed(&out_leafs)?;
    let loss_val = loss.to_vec0::<f32>()?;
    let out_grads = loss
        .backward()
        .map_err(|e| candle_core::Error::Msg(format!("died in backward (loss): {e:?}")))?;
    let mut upstream: Vec<Tensor> = Vec::with_capacity(out_leafs.len());
    for (i, t) in out_leafs.iter().enumerate() {
        upstream.push(out_grads.get(t).cloned().ok_or_else(|| {
            candle_core::Error::Msg(format!("checkpointed_backward: no gradient for output {i}"))
        })?);
    }

    let mut w_live = base.clone();
    for (key, var) in lora_vars {
        w_live.insert(key.clone(), var.as_tensor().clone());
    }
    let mut lora_grads = GradStore::default();
    let mut input_grads: Vec<Tensor> = Vec::new();
    for (k, st) in stages.iter().enumerate().rev() {
        let in_vars: Vec<Var> = stage_inputs[k]
            .iter()
            .map(Var::from_tensor)
            .collect::<Result<_>>()?;
        let in_leafs: Vec<Tensor> = in_vars.iter().map(|v| v.as_tensor().clone()).collect();
        let outs = (st.run)(&w_live, &in_leafs).map_err(|e| {
            candle_core::Error::Msg(format!("died in backward (recomputing stage {k}): {e:?}"))
        })?;
        if outs.len() != upstream.len() {
            return Err(candle_core::Error::Msg(format!(
                "checkpointed_backward: stage {k} returned {} outputs, expected {}",
                outs.len(),
                upstream.len()
            )));
        }
        let mut seeded: Option<Tensor> = None;
        for (o, g) in outs.iter().zip(&upstream) {
            let t = o.mul(g)?.sum_all()?;
            seeded = Some(match seeded {
                None => t,
                Some(p) => p.add(&t)?,
            });
        }
        let seeded = seeded.ok_or_else(|| {
            candle_core::Error::Msg(format!("checkpointed_backward: stage {k} has no outputs"))
        })?;
        let grads = seeded
            .backward()
            .map_err(|e| candle_core::Error::Msg(format!("died in backward (stage {k}): {e:?}")))?;
        let next: Vec<Tensor> = in_leafs
            .iter()
            .map(|t| {
                grads.get(t).cloned().ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "checkpointed_backward: no gradient through stage {k}"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        if k == 0 {
            input_grads = next.clone();
        }
        for (_, var) in lora_vars {
            if let Some(g) = grads.get(var.as_tensor()) {
                lora_grads.insert(var.as_tensor(), g.clone());
            }
        }
        upstream = next;
    }
    Ok((loss_val, input_grads, lora_grads))
}

/// cos >= COS_MIN and rel L2 <= REL_L2_MAX, computed in f64 over f32 values.
pub fn compare(name: &str, ours: &Tensor, reference: &Tensor) -> Result<bool> {
    let a = ours
        .contiguous()?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if a.len() != b.len() {
        println!("{name:24} SHAPE MISMATCH {} vs {}", a.len(), b.len());
        return Ok(false);
    }
    let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let nb: f64 = b.iter().map(|y| (y * y) as f64).sum::<f64>().sqrt();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let num: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2) as f64).sum();
    let cos = dot / (na * nb + 1e-30);
    let rel = num.sqrt() / (nb + 1e-12);
    let ok = cos >= COS_MIN && rel <= REL_L2_MAX;
    println!(
        "{name:24} cos = {cos:.6}  rel_l2 = {rel:.3e} {}",
        if ok { "" } else { "<-- FAIL" }
    );
    Ok(ok)
}
