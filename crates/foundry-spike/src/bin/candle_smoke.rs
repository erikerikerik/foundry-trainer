//! Split of spike item 0.5.3: prove Candle's CUDA backend builds and runs the
//! bf16 ops the block forward needs (matmul, layernorm, softmax, elementwise).

use candle_core::{DType, Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = Device::new_cuda(0)?;
    println!("candle cuda device: ok");

    let w = Tensor::randn(0.0f32, 1.0, (128, 128), &dev)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0.0f32, 1.0, (16, 128), &dev)?.to_dtype(DType::BF16)?;
    let y = x.matmul(&w.t()?)?;
    println!("bf16 matmul: ok {:?}", y.dims());

    let one = Tensor::ones(128, DType::BF16, &dev)?;
    let zero = Tensor::zeros(128, DType::BF16, &dev)?;
    let ln = candle_nn::ops::layer_norm(&y, &one, &zero, 1e-6)?;
    println!("layernorm: ok {:?}", ln.dims());

    let qkv = y.reshape((2, 8, 128))?; // (heads, seq, head_dim)
    let scores = qkv
        .contiguous()?
        .matmul(&qkv.contiguous()?.transpose(1, 2)?)?;
    let probs = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, 2)?.to_dtype(DType::BF16)?;
    let attn = probs.matmul(&qkv)?;
    println!("softmax attention: ok {:?}", attn.dims());

    let gated = candle_nn::ops::silu(&y)?.mul(&y)?;
    println!("silu gate (SwiGLU part): ok {:?}", gated.dims());

    Ok(())
}
