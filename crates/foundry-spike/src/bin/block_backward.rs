//! Spike item 0.5.5: full-block backward including LoRA A/B gradients (bf16,
//! CUDA), against the committed PyTorch reference (tools/fixtures/block_bf16.safetensors).
//!
//! The whole block pair is rebuilt as one Candle autograd graph (attention via
//! the `AttnBwd` custom op) with variable leaves for the two stream inputs and
//! all six LoRA factors; the loss is computed in Candle and `backward()` runs
//! over the real graph (no seed trick — every upstream gradient is ours).
//! Acceptance: LoRA gradients within tolerance (cos >= 0.999, rel L2 <= 2e-2),
//! plus the input gradients and a forward replay of every recorded activation.

use candle_core::{safetensors, Device, Result, Tensor, Var};
use foundry_spike::{
    compare, double_block, get, mse, rope_tables, single_block, Weights, IMG_GRID, IMG_SEQ, TXT_LEN,
};
use std::path::Path;

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

    // Candle only tracks gradients for variable leaves: make the two stream
    // inputs and every LoRA A/B factor leaves of the graph.
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
    println!(
        "variable leaves: {} LoRA tensors + inputs/img + inputs/txt",
        vars.len()
    );

    let img_in = Var::from_tensor(get(&w, "inputs/img")?)?;
    let txt_in = Var::from_tensor(get(&w, "inputs/txt")?)?;
    let temb = get(&w, "inputs/temb")?;

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
    let (img2, txt2) = double_block(
        &w,
        &mut acts,
        img_in.as_tensor(),
        txt_in.as_tensor(),
        temb,
        &rt_i,
        &rt_t,
    )?;
    let joint = Tensor::cat(&[&txt2, &img2], 0)?;
    let joint_out = single_block(&w, &mut acts, &joint, temb, &rt_a)?;
    let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
    let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;

    // same loss as the reference: mse(img_out,target_img) + mse(txt_out,target_txt)
    let loss = mse(&final_img, get(&w, "inputs/target_img")?)?
        .add(&mse(&final_txt, get(&w, "inputs/target_txt")?)?)?;
    acts.push(("loss".into(), loss.clone()));

    let grads = loss.backward()?;

    let mut fail = false;
    // forward sanity: the variable-wrapped graph must replay the same activations
    for (name, ours) in &acts {
        fail |= !compare(name, ours, get(&w, name)?)?;
    }
    // the gradients under test
    for (name, var) in [("inputs/img", &img_in), ("inputs/txt", &txt_in)] {
        let g = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {name}")))?;
        fail |= !compare(
            &format!("grad/{name}"),
            g,
            get(&w, &format!("grad/{name}"))?,
        )?;
    }
    for (key, var) in &vars {
        let g = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
        let ref_key = key.replace("weights/", "grad/");
        fail |= !compare(&ref_key, g, get(&w, &ref_key)?)?;
    }
    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
