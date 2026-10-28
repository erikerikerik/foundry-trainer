//! Spike item 0.5.8b: memory teardown with the worker-process model.
//!
//! 0.5.8 (`bin/mem_teardown.rs`) failed in-process: after dropping every
//! Candle tensor and the `Device`, 344 MB of the burst stayed unreleased on
//! this driver stack, three runs in a row, while a *process exit* clearly did
//! release everything (run 2 started at the baseline run 1 had ended at). This
//! bin tests the model that finding points at: training runs in a worker
//! process that exits when the job ends.
//!
//! Parent: NVML free VRAM (no CUDA context in this process at all), then
//! spawns *itself* as a child (`--child`), waits for it to exit, sleeps 1 s
//! for the NVML counters to catch up, reads NVML again. Gate: the free-VRAM
//! deficit vs baseline is <= 200 MB — the same gate as 0.5.8, unchanged.
//! Child: creates the CUDA device, runs the identical 20-step bf16 AdamW
//! burst from `bin/mem_teardown.rs`, and exits.
//!
//! Three rounds, because the plan's acceptance is "3 consecutive runs". Each
//! round takes a fresh baseline, so a leak that accumulates shows up as a
//! rising deficit.
//!
//! Run with Strata unloaded (CLAUDE.md testing rule): the numbers are only
//! valid on an otherwise-idle GPU. Writes nothing; touches no user data.

use candle_core::{safetensors, DType, Device, Result, Tensor, Var};
use foundry_spike::{
    adamw_step, double_block, get, mse, rope_tables_dtype, single_block, AdamW, Weights, IMG_GRID,
    IMG_SEQ, TXT_LEN,
};
use nvml_wrapper::Nvml;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

// Same optimizer and burst as 0.5.8 / 0.5.7; the point is the allocation
// lifetime, not convergence.
const OPT: AdamW = AdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
};
const STEPS: usize = 20;
const TOLERANCE_BYTES: u64 = 200 * 1024 * 1024;
const ROUNDS: usize = 3;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// NVML errors into Candle's error type so `?` works throughout.
fn nvml_err(e: nvml_wrapper::error::NvmlError) -> candle_core::Error {
    candle_core::Error::Msg(format!("nvml: {e}"))
}

/// The 20-step burst, byte-for-byte the loop from `bin/mem_teardown.rs`
/// (deliberate duplication: the bins are recorded artifacts, and the child
/// must be a standalone process that owns the whole CUDA lifetime).
fn run_burst(dev: &Device) -> Result<f32> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_bf16.safetensors");
    let mut w: Weights = safetensors::load(path, dev)?;

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
    for step in 1..=STEPS {
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
    loss.to_vec0::<f32>()
}

/// Child mode: own the CUDA lifetime, do the work, exit. Process exit is the
/// teardown being measured, so nothing here releases anything by hand.
fn child() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let loss = run_burst(&dev)?;
    println!(
        "child pid {}: {STEPS} AdamW steps, final loss {loss:?}",
        std::process::id()
    );
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().any(|a| a == "--child") {
        return child();
    }

    // Parent: NVML only. Never touches CUDA, so it holds no GPU memory of its
    // own and cannot contaminate the readings.
    let nvml = Nvml::init().map_err(nvml_err)?;
    let gpu = nvml.device_by_index(0).map_err(nvml_err)?;
    println!("device: {}", gpu.name().map_err(nvml_err)?);
    println!(
        "parent pid {} (no CUDA context in this process)",
        std::process::id()
    );

    let exe = std::env::current_exe()?;
    let mut worst = 0u64;
    let mut all_ok = true;
    for round in 1..=ROUNDS {
        let base = gpu.memory_info().map_err(nvml_err)?;
        println!(
            "round {round}: baseline used {:.0} MB, free {:.0} MB",
            mb(base.used),
            mb(base.free)
        );

        let status = std::process::Command::new(&exe)
            .arg("--child")
            .status()
            .map_err(|e| candle_core::Error::Msg(format!("spawn {}: {e}", exe.display())))?;
        if !status.success() {
            return Err(candle_core::Error::Msg(format!(
                "worker child exited with {status} — its output above has the reason"
            )));
        }

        // NVML counters lag the driver; give the context release a moment.
        std::thread::sleep(Duration::from_millis(1000));
        let after = gpu.memory_info().map_err(nvml_err)?;
        // saturating_sub: free above baseline (something else released memory)
        // counts as a deficit of 0.
        let deficit = base.free.saturating_sub(after.free);
        worst = worst.max(deficit);
        let ok = deficit <= TOLERANCE_BYTES;
        all_ok &= ok;
        println!(
            "round {round}: after child exit used {:.0} MB, free {:.0} MB -> deficit {:.1} MB (gate: <= {:.0} MB) {}",
            mb(after.used),
            mb(after.free),
            mb(deficit),
            mb(TOLERANCE_BYTES),
            if ok { "" } else { "<-- FAIL" }
        );
    }

    println!("worst deficit over {ROUNDS} rounds: {:.1} MB", mb(worst));
    println!("RESULT: {}", if all_ok { "PASS" } else { "FAIL" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
