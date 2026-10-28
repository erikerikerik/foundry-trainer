//! Spike item 0.5.7: determinism and 1000-iteration stability.
//! (a) Two independent 10-step AdamW runs on the bf16 fixture from identical
//!     state must produce bit-identical per-step losses and final LoRA params.
//! (b) A 1000-iteration bf16 run on fixed fixture data must keep every loss and
//!     parameter finite, end below its starting loss, and show no sustained
//!     upward drift (100-step window means strictly decreasing, bounded
//!     step-to-step rises, short rise runs only).
//! (c) Control: the same loop on the fp32 fixture, same drift gates. It shows
//!     the isolated step rises are Adam overshoot at this lr, not a bf16
//!     precision artifact (they appear in fp32 too, and fp32 descends further).

use candle_core::{safetensors, DType, Device, Result, Tensor, Var};
use foundry_spike::{
    adamw_step, double_block, get, mse, rope_tables_dtype, single_block, AdamW, Weights, IMG_GRID,
    IMG_SEQ, TXT_LEN,
};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

// Must match CONFIG in tools/fixtures/gen_block.py.
const OPT: AdamW = AdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
};
const STABILITY_STEPS: usize = 1000;
// Single-step rises up to ~6e-3 are ordinary Adam overshoot at lr 1e-3 (the
// fp32 control shows them too, so they are not a precision artifact). "No
// drift" is enforced structurally instead: 100-step window means must strictly
// decrease, rise runs must stay short, and no single step may jump by more
// than 1e-2 (a diverging run at this lr grows much faster than that).
const MAX_RISE: f32 = 1e-2;
const MAX_RISE_RUN: usize = 10;
const WINDOW: usize = 100;

/// Per-step losses, and final LoRA values as f32 (bf16 -> f32 is injective,
/// so bit comparison is exact).
type RunOut = (Vec<f32>, Vec<(String, Vec<f32>)>);

/// Run `steps` AdamW steps on a fresh load of `file` (inputs fixed, only the
/// six LoRA factors updated).
fn run_loop(dev: &Device, file: &str, pdtype: DType, steps: usize) -> Result<RunOut> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../tools/fixtures/{file}"));
    let mut w: Weights = safetensors::load(&path, dev)?;

    let mut lora_keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("weights/") && (k.ends_with(".lora_a") || k.ends_with(".lora_b")))
        .cloned()
        .collect();
    lora_keys.sort();
    let mut vars: Vec<(String, Var)> = Vec::new();
    for key in lora_keys {
        let var = Var::from_tensor(&get(&w, &key)?.to_dtype(pdtype)?)?;
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

    let img = get(&w, "inputs/img")?.to_dtype(pdtype)?.contiguous()?;
    let txt = get(&w, "inputs/txt")?.to_dtype(pdtype)?.contiguous()?;
    let temb = get(&w, "inputs/temb")?.to_dtype(pdtype)?.contiguous()?;
    let target_img = get(&w, "inputs/target_img")?.clone();
    let target_txt = get(&w, "inputs/target_txt")?.clone();
    let pos_img: Vec<[f32; 4]> = (0..IMG_SEQ)
        .map(|i| [(i / IMG_GRID) as f32, (i % IMG_GRID) as f32, 0.0, 0.0])
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rt_i = rope_tables_dtype(&pos_img, dev, pdtype)?;
    let rt_t = rope_tables_dtype(&pos_txt, dev, pdtype)?;
    let rt_a = rope_tables_dtype(&pos_all, dev, pdtype)?;

    let mut losses: Vec<f32> = Vec::with_capacity(steps);
    for step in 1..=steps {
        let mut acts: Vec<(String, Tensor)> = Vec::new();
        let (img2, txt2) = double_block(&w, &mut acts, &img, &txt, &temb, &rt_i, &rt_t)?;
        let joint = Tensor::cat(&[&txt2, &img2], 0)?;
        let joint_out = single_block(&w, &mut acts, &joint, &temb, &rt_a)?;
        let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
        let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;
        let loss = mse(&final_img, &target_img)?.add(&mse(&final_txt, &target_txt)?)?;
        let grads = loss.backward()?;
        for (key, var) in &vars {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
            let mm = m.get_mut(key).unwrap();
            let vv = v.get_mut(key).unwrap();
            adamw_step(var, g, mm, vv, step, &OPT)?;
        }
        losses.push(loss.to_vec0::<f32>()?);
    }
    let params = vars
        .into_iter()
        .map(|(key, var)| {
            Ok((
                key,
                var.as_tensor()
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((losses, params))
}

/// Print a trajectory summary; return (all finite, decreased, max rise,
/// longest run of consecutive rises, window means strictly decreasing).
fn trajectory(
    label: &str,
    losses: &[f32],
    elapsed: std::time::Duration,
) -> Result<(bool, bool, f32, usize, bool)> {
    let first = losses[0];
    let last = *losses.last().unwrap();
    let finite = losses.iter().all(|x| x.is_finite());
    let max_rise = losses
        .windows(2)
        .map(|win| win[1] - win[0])
        .fold(f32::MIN, f32::max);
    let mut rises = 0;
    let (mut run, mut longest_run) = (0, 0);
    for win in losses.windows(2) {
        if win[1] > win[0] {
            rises += 1;
            run += 1;
            longest_run = longest_run.max(run);
        } else {
            run = 0;
        }
    }
    // drift check: the mean loss of each WINDOW-step block must be strictly
    // below the previous block's mean. Isolated step rises are ordinary Adam
    // overshoot; a window mean climbing is the drift we are gating against.
    let means: Vec<f32> = losses
        .chunks(WINDOW)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    let windows_ok = means.windows(2).all(|win| win[1] < win[0]);
    let samples: Vec<String> = (0..losses.len())
        .step_by(100)
        .chain(std::iter::once(losses.len() - 1))
        .map(|i| format!("{}:{:?}", i + 1, losses[i]))
        .collect();
    println!("{label}: {} steps in {elapsed:?}", losses.len());
    println!("{label}: loss {samples:?}");
    println!(
        "{label}: first {first:?} last {last:?}; {rises} of {} steps rose, \
         max rise {max_rise:?}, longest rise run {longest_run}",
        losses.len() - 1
    );
    println!("{label}: window means {means:?} strictly decreasing: {windows_ok}");
    Ok((finite, last < first, max_rise, longest_run, windows_ok))
}

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let gpu_name = cudarc::driver::CudaContext::new(0)
        .and_then(|c| c.name())
        .unwrap_or_else(|_| "unknown".into());
    println!("device: {gpu_name}");
    let mut fail = false;

    // (a) determinism: two independent bf16 runs, same fixture, same code path
    let (la, pa) = run_loop(&dev, "block_bf16.safetensors", DType::BF16, 10)?;
    let (lb, pb) = run_loop(&dev, "block_bf16.safetensors", DType::BF16, 10)?;
    let losses_ok =
        la.len().eq(&lb.len()) && la.iter().zip(&lb).all(|(x, y)| x.to_bits() == y.to_bits());
    println!("determinism: per-step losses bit-identical: {losses_ok}");
    fail |= !losses_ok;
    for ((key, a), (_, b)) in pa.iter().zip(&pb) {
        let ok = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        println!("determinism: {key:24} final params bit-identical: {ok}");
        fail |= !ok;
    }

    // (b) 1000-iteration stability on fixed data, bf16 production path
    let started = Instant::now();
    let (losses, params) = run_loop(&dev, "block_bf16.safetensors", DType::BF16, STABILITY_STEPS)?;
    let (finite, decreased, max_rise, run, windows_ok) =
        trajectory("stability bf16", &losses, started.elapsed())?;
    let params_finite = params
        .iter()
        .all(|(_, vals)| vals.iter().all(|x| x.is_finite()));
    let ok = finite
        && params_finite
        && decreased
        && max_rise <= MAX_RISE
        && run <= MAX_RISE_RUN
        && windows_ok;
    println!(
        "stability bf16: losses finite {finite}, params finite {params_finite}, \
         decreased {decreased}, max rise <= {MAX_RISE:?} {}, rise run <= {MAX_RISE_RUN} {}, \
         windows decreasing {windows_ok}",
        max_rise <= MAX_RISE,
        run <= MAX_RISE_RUN
    );
    fail |= !ok;

    // (c) fp32 control: same loop, fp32 fixture and params. The control shows
    // the step-level rises are Adam dynamics (they appear in fp32 too, and are
    // larger there), so both runs get the same drift gates, not a smoothness
    // bound. The fp32 run must also stay finite and descend.
    let started = Instant::now();
    let (flosses, _) = run_loop(&dev, "block_fp32.safetensors", DType::F32, STABILITY_STEPS)?;
    let (f_finite, f_decreased, f_max_rise, f_run, f_windows_ok) =
        trajectory("control fp32", &flosses, started.elapsed())?;
    let ok =
        f_finite && f_decreased && f_max_rise <= MAX_RISE && f_run <= MAX_RISE_RUN && f_windows_ok;
    println!(
        "control fp32: losses finite {f_finite}, decreased {f_decreased}, \
         max rise <= {MAX_RISE:?} {} (bf16 max rise was {max_rise:?}), \
         rise run <= {MAX_RISE_RUN} {f_run}, windows decreasing {f_windows_ok}",
        f_max_rise <= MAX_RISE
    );
    fail |= !ok;

    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
