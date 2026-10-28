//! Spike item 0.5.6: AdamW on the LoRA parameters (bf16 params, fp32 optimizer
//! state), 10 steps with a freshly recomputed gradient each step, mirroring the
//! reference loop in tools/fixtures/gen_block.py (clear grads -> forward ->
//! backward -> step). Bias-corrected update, weight_decay 0.
//! Acceptance: LoRA values after steps 1 and 10 within tolerance against the
//! fixture's `opt1/*` and `opt10/*` tensors (cos >= 0.999, rel L2 <= 2e-2).

use candle_core::{safetensors, DType, Device, Result, Tensor, Var};
use foundry_spike::{
    compare, double_block, get, mse, rope_tables, single_block, Weights, IMG_GRID, IMG_SEQ, TXT_LEN,
};
use std::collections::HashMap;
use std::path::Path;

// Must match CONFIG in tools/fixtures/gen_block.py.
const LR: f64 = 1e-3;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const OPT_STEPS: usize = 10;

fn main() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let gpu_name = cudarc::driver::CudaContext::new(0)
        .and_then(|c| c.name())
        .unwrap_or_else(|_| "unknown".into());
    println!("device: {gpu_name}");

    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_bf16.safetensors");
    let mut w: Weights = safetensors::load(&path, &dev)?;
    println!("fixture: {} ({} tensors)", path.display(), w.len());

    // the six LoRA factors become variable leaves; inputs stay plain tensors
    // (the reference detaches them in the per-step re-forward)
    let mut lora_keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("weights/") && (k.ends_with(".lora_a") || k.ends_with(".lora_b")))
        .cloned()
        .collect();
    lora_keys.sort();
    let mut vars: Vec<(String, Var)> = Vec::new();
    for key in lora_keys {
        let var = Var::from_tensor(get(&w, &key)?)?;
        w.insert(key.clone(), var.as_tensor().clone());
        vars.push((key, var));
    }

    // Adam moments in fp32 (the production choice; the bf16 reference keeps
    // them in bf16, and the fp32-vs-bf16 noise floor for opt10/* is 1.64e-2)
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

    let img = get(&w, "inputs/img")?;
    let txt = get(&w, "inputs/txt")?;
    let temb = get(&w, "inputs/temb")?;
    let pos_img: Vec<[f32; 4]> = (0..IMG_SEQ)
        .map(|i| [(i / IMG_GRID) as f32, (i % IMG_GRID) as f32, 0.0, 0.0])
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rt_i = rope_tables(&pos_img, &dev)?;
    let rt_t = rope_tables(&pos_txt, &dev)?;
    let rt_a = rope_tables(&pos_all, &dev)?;

    let mut fail = false;
    for step in 1..=OPT_STEPS {
        // fresh graph each step, like the reference: clear grads, re-forward,
        // re-backward at the current parameter values
        let mut acts: Vec<(String, Tensor)> = Vec::new();
        let (img2, txt2) = double_block(&w, &mut acts, img, txt, temb, &rt_i, &rt_t)?;
        let joint = Tensor::cat(&[&txt2, &img2], 0)?;
        let joint_out = single_block(&w, &mut acts, &joint, temb, &rt_a)?;
        let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
        let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;
        let loss = mse(&final_img, get(&w, "inputs/target_img")?)?
            .add(&mse(&final_txt, get(&w, "inputs/target_txt")?)?)?;
        let grads = loss.backward()?;

        for (key, var) in &vars {
            let g = grads
                .get(var.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?
                .to_dtype(DType::F32)?;
            let mnew = m[key]
                .affine(BETA1, 0.0)?
                .add(&g.affine(1.0 - BETA1, 0.0)?)?;
            let vnew = v[key]
                .affine(BETA2, 0.0)?
                .add(&g.sqr()?.affine(1.0 - BETA2, 0.0)?)?;
            // PyTorch AdamW: step_size = lr/(1-b1^t), denom = sqrt(v)/(sqrt(1-b2^t)) + eps
            let bc1 = 1.0 - BETA1.powi(step as i32);
            let bc2 = 1.0 - BETA2.powi(step as i32);
            let mhat = mnew.affine(1.0 / bc1, 0.0)?;
            let denom = vnew.affine(1.0 / bc2, 0.0)?.sqrt()?.affine(1.0, EPS)?;
            let delta = mhat.broadcast_div(&denom)?.affine(LR, 0.0)?;
            let pnew = var
                .as_tensor()
                .to_dtype(DType::F32)?
                .broadcast_sub(&delta)?
                .to_dtype(DType::BF16)?;
            var.set(&pnew)?;
            m.insert(key.clone(), mnew);
            v.insert(key.clone(), vnew);
        }
        println!("step {step}: loss = {:?}", loss.to_vec0::<f32>()?);

        if step == 1 || step == OPT_STEPS {
            for (key, var) in &vars {
                let ref_key = key.replace("weights/", &format!("opt{step}/"));
                fail |= !compare(&ref_key, var.as_tensor(), get(&w, &ref_key)?)?;
            }
        }
    }
    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
