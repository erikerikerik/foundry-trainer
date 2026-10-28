//! Spike item 0.5.4: attention backward as a Candle custom op (bf16) on the GPU.
//!
//! The forward half of `AttnBwd` replays the reference attention (same semantics
//! as `block_forward.rs`); the backward half is hand-written to mirror PyTorch
//! autograd on that exact graph (reference in tools/fixtures/PROVENANCE.md):
//! dprobs/dV in bf16 matmuls, softmax backward in fp32, dQ/dK as fp32 matmuls
//! cast back to bf16.
//! Acceptance: gradient cosine >= 0.999 (and rel L2 <= 2e-2) on attention inputs,
//! against the `grad/attn/*` tensors in the fixture.

use candle_core::{
    backend::BackendStorage, op::BackpropOp, safetensors, CpuStorage, CudaStorage, CustomOp3,
    DType, Device, Layout, Result, Shape, Storage, Tensor, Var,
};
use std::collections::HashMap;
use std::path::Path;

// Must match CONFIG in tools/fixtures/gen_block.py.
const HEAD_DIM: usize = 128;
const REL_L2_MAX: f64 = 2e-2;
const COS_MIN: f64 = 0.999;

type Weights = HashMap<String, Tensor>;

fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Tensor> {
    w.get(key)
        .ok_or_else(|| candle_core::Error::Msg(format!("fixture tensor missing: {key}")))
}

/// The reference attention forward, recomputed inside the custom op so the op
/// is self-contained (this is what Candle will record as the op's output).
fn attn_fwd(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // q,k,v: (seq, heads, head_dim); scores/softmax in fp32, probs cast back
    let qt = q.contiguous()?.transpose(0, 1)?; // (heads, seq, head_dim)
    let kt = k.contiguous()?.transpose(0, 1)?;
    let vt = v.contiguous()?.transpose(0, 1)?;
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let scores = qt
        .to_dtype(DType::F32)?
        .matmul(&kt.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let probs = candle_nn::ops::softmax(&scores, 2)?.to_dtype(q.dtype())?;
    probs.matmul(&vt.contiguous()?)?.transpose(0, 1)
}

struct AttnBwd;

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

/// cos >= COS_MIN and rel L2 <= REL_L2_MAX, computed in f64 over f32 values.
fn compare(name: &str, ours: &Tensor, reference: &Tensor) -> Result<bool> {
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

    let mut fail = false;
    for tag in ["double", "single"] {
        // gradients are only tracked for variable leaves, so wrap the inputs
        let q = Var::from_tensor(get(&w, &format!("attn_in/{tag}.q"))?)?;
        let k = Var::from_tensor(get(&w, &format!("attn_in/{tag}.k"))?)?;
        let v = Var::from_tensor(get(&w, &format!("attn_in/{tag}.v"))?)?;
        let attn = q
            .as_tensor()
            .apply_op3(k.as_tensor(), v.as_tensor(), AttnBwd)?;
        // forward sanity: the op's output must match the recorded attention output
        fail |= !compare(
            &format!("fwd/{tag}.attn"),
            &attn,
            get(&w, &format!("attn_in/{tag}.attn"))?,
        )?;

        // seed the backward with the recorded upstream gradient into the
        // attention output: d(sum(attn * seed))/d(attn) = seed exactly
        let seed = get(&w, &format!("grad/attn/{tag}.out"))?;
        let objective = attn.mul(seed)?.sum_all()?;
        let grads = objective.backward()?;
        for (n, var) in [("q", &q), ("k", &k), ("v", &v)] {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {tag}.{n}")))?;
            fail |= !compare(
                &format!("grad/{tag}.{n}"),
                g,
                get(&w, &format!("grad/attn/{tag}.{n}"))?,
            )?;
        }
    }
    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
