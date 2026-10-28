//! Spike item 0.5.8: memory teardown test with NVML.
//!
//! Reads NVML free VRAM before any CUDA context exists, runs a real training
//! burst (bf16 fixture block load + 20 AdamW steps on the six LoRA factors,
//! the same code path as `stability.rs`), then drops every Candle tensor and
//! the CUDA device and measures again. Candle 0.11 has no caching allocator
//! and no device cache: dropping the last `Device` releases the CUDA primary
//! context (`cuDevicePrimaryCtxRelease`), which frees every allocation the
//! process owns — this test verifies that on the real driver stack instead of
//! assuming it (CLAUDE.md rule 5).
//! Accept: free VRAM back within 200 MB of baseline.
//!
//! Run with Strata unloaded (CLAUDE.md testing rule): the numbers are only
//! valid on an otherwise-idle GPU. Writes nothing; touches no user data.

use candle_core::{safetensors, DType, Device, Result, Tensor, Var};
use foundry_spike::{
    adamw_step, double_block, get, mse, rope_tables_dtype, single_block, AdamW, Weights, IMG_GRID,
    IMG_SEQ, TXT_LEN,
};
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

// Same optimizer as the 0.5.6/0.5.7 runs; the burst only needs to allocate a
// realistic training graph, not to converge.
const OPT: AdamW = AdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
};
const STEPS: usize = 20;
const TOLERANCE_BYTES: u64 = 200 * 1024 * 1024;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// NVML errors into Candle's error type so `?` works throughout.
fn nvml_err(e: nvml_wrapper::error::NvmlError) -> candle_core::Error {
    candle_core::Error::Msg(format!("nvml: {e}"))
}

/// Best-effort per-process listing. Under WDDM the kernel-mode driver owns
/// GPU memory and NVML always reports `Unavailable` per process, so the
/// device-level free/used delta is the reliable signal on this machine; the
/// pid list is still printed to show which processes hold GPU work.
fn print_processes(nvml: &Nvml) {
    match nvml
        .device_by_index(0)
        .and_then(|d| d.running_compute_processes())
    {
        Ok(procs) if procs.is_empty() => println!("  compute processes: none"),
        Ok(procs) => {
            for p in procs {
                let mem = match p.used_gpu_memory {
                    UsedGpuMemory::Used(b) => format!("{:.0} MB", mb(b)),
                    UsedGpuMemory::Unavailable => "per-process bytes n/a under WDDM".into(),
                };
                println!("  compute process pid {}: {mem}", p.pid);
            }
        }
        Err(e) => println!("  compute processes: unavailable ({e})"),
    }
}

/// Load the bf16 fixture and run `steps` AdamW steps on the six LoRA factors
/// (identical loop to `stability.rs`). Returns the final loss and the NVML
/// free-VRAM reading taken after the last step while weights, LoRA vars and
/// fp32 Adam moments are still alive. Everything CUDA is owned by this
/// function's scope, so returning from it drops all of it.
fn run_burst(dev: &Device, nvml: &Nvml, steps: usize) -> Result<(f32, u64)> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_bf16.safetensors");
    let mut w: Weights = safetensors::load(&path, dev)?;

    let mut lora_keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("weights/") && (k.ends_with(".lora_a") || k.ends_with(".lora_b")))
        .cloned()
        .collect();
    lora_keys.sort();
    let mut vars: Vec<(String, Var)> = Vec::new();
    for key in lora_keys {
        let var = Var::from_tensor(&get(&w, &key)?.to_dtype(DType::BF16)?)?;
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

    let img = get(&w, "inputs/img")?.to_dtype(DType::BF16)?.contiguous()?;
    let txt = get(&w, "inputs/txt")?.to_dtype(DType::BF16)?.contiguous()?;
    let temb = get(&w, "inputs/temb")?
        .to_dtype(DType::BF16)?
        .contiguous()?;
    let target_img = get(&w, "inputs/target_img")?.clone();
    let target_txt = get(&w, "inputs/target_txt")?.clone();
    let pos_img: Vec<[f32; 4]> = (0..IMG_SEQ)
        .map(|i| [(i / IMG_GRID) as f32, (i % IMG_GRID) as f32, 0.0, 0.0])
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rt_i = rope_tables_dtype(&pos_img, dev, DType::BF16)?;
    let rt_t = rope_tables_dtype(&pos_txt, dev, DType::BF16)?;
    let rt_a = rope_tables_dtype(&pos_all, dev, DType::BF16)?;

    let mut loss = Tensor::zeros((), DType::F32, dev)?;
    for step in 1..=steps {
        let mut acts: Vec<(String, Tensor)> = Vec::new();
        let (img2, txt2) = double_block(&w, &mut acts, &img, &txt, &temb, &rt_i, &rt_t)?;
        let joint = Tensor::cat(&[&txt2, &img2], 0)?;
        let joint_out = single_block(&w, &mut acts, &joint, &temb, &rt_a)?;
        let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
        let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;
        loss = mse(&final_img, &target_img)?.add(&mse(&final_txt, &target_txt)?)?;
        let grads = loss.backward()?;
        for (key, var) in &vars {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
            let mm = m.get_mut(key).unwrap();
            let vv = v.get_mut(key).unwrap();
            adamw_step(var, g, mm, vv, step, &OPT)?;
        }
    }
    let free_during = nvml
        .device_by_index(0)
        .map_err(nvml_err)?
        .memory_info()
        .map_err(nvml_err)?
        .free;
    Ok((loss.to_vec0::<f32>()?, free_during))
}

fn main() -> Result<()> {
    // NVML only: no CUDA context before the baseline reading.
    let nvml = Nvml::init().map_err(nvml_err)?;
    let gpu = nvml.device_by_index(0).map_err(nvml_err)?;
    println!("device: {}", gpu.name().map_err(nvml_err)?);
    let base = gpu.memory_info().map_err(nvml_err)?;
    println!(
        "baseline: total {:.0} MB, used {:.0} MB, free {:.0} MB",
        mb(base.total),
        mb(base.used),
        mb(base.free)
    );
    print_processes(&nvml);

    let (final_loss, free_during);
    {
        let dev = Device::new_cuda(0)?;
        let out = run_burst(&dev, &nvml, STEPS)?;
        final_loss = out.0;
        free_during = out.1;
        drop(dev); // last CudaContext handle -> primary context release
    }
    println!("burst: {STEPS} AdamW steps, final loss {final_loss:?}");
    println!(
        "after steps, before drop: free {:.0} MB (training held >= {:.0} MB)",
        mb(free_during),
        mb(base.free.saturating_sub(free_during))
    );

    // NVML counters lag the driver; give the context release a moment to show
    // up in memory_info before the final reading.
    std::thread::sleep(Duration::from_millis(1000));
    let after = gpu.memory_info().map_err(nvml_err)?;
    println!(
        "after teardown: used {:.0} MB, free {:.0} MB",
        mb(after.used),
        mb(after.free)
    );
    print_processes(&nvml);

    // saturating_sub: if free VRAM is *above* baseline (another process
    // released memory meanwhile), the deficit is 0.
    let deficit = base.free.saturating_sub(after.free);
    println!(
        "free-VRAM deficit vs baseline: {:.1} MB (gate: <= {:.0} MB)",
        mb(deficit),
        mb(TOLERANCE_BYTES)
    );
    let ok = deficit <= TOLERANCE_BYTES;
    println!("RESULT: {}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
