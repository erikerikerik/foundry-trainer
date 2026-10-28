//! Spike item 0.5.5: full-block backward including LoRA A/B gradients (bf16,
//! CUDA), against the committed PyTorch reference (tools/fixtures/block_bf16.safetensors).
//!
//! The whole block pair is rebuilt as one Candle autograd graph (attention via
//! the `AttnBwd` custom op) with variable leaves for the two stream inputs and
//! all six LoRA factors; the loss is computed in Candle and `backward()` runs
//! over the real graph (no seed trick — every upstream gradient is ours).
//! Acceptance: LoRA gradients within tolerance (cos >= 0.999, rel L2 <= 2e-2),
//! plus the input gradients and a forward replay of every recorded activation.
//!
//! `--checkpoint` runs the same block pair through `lib::checkpointed_backward`
//! instead: two stages (double block + the txt/img join, then the single
//! block), only their inputs kept, each stage's graph rebuilt during the
//! reverse walk. The acceptance is deliberately identical — the same 8
//! gradients against the same fixture tensors at the same tolerances. If
//! checkpointing changed any gradient, this run would fail where the default
//! run passes. The activation replay only exists in the full-graph mode (a
//! checkpointed forward keeps no activations, which is the whole point).

use candle_core::{safetensors, Device, Result, Tensor, Var};
use foundry_spike::{
    checkpointed_backward, compare, double_block, get, mse, rope_tables, single_block, Stage,
    Weights, IMG_GRID, IMG_SEQ, TXT_LEN,
};
use std::path::Path;

struct Rts {
    img: foundry_spike::RopeTables,
    txt: foundry_spike::RopeTables,
    joint: foundry_spike::RopeTables,
}

/// Default mode: one autograd graph over the whole pair, real `backward()`.
fn full_graph(w: &Weights, vars: &[(String, Var)], rts: &Rts) -> Result<bool> {
    let img_in = Var::from_tensor(get(w, "inputs/img")?)?;
    let txt_in = Var::from_tensor(get(w, "inputs/txt")?)?;
    let temb = get(w, "inputs/temb")?;

    let mut acts: Vec<(String, Tensor)> = Vec::new();
    let (img2, txt2) = double_block(
        w,
        &mut acts,
        img_in.as_tensor(),
        txt_in.as_tensor(),
        temb,
        &rts.img,
        &rts.txt,
    )?;
    let joint = Tensor::cat(&[&txt2, &img2], 0)?;
    let joint_out = single_block(w, &mut acts, &joint, temb, &rts.joint)?;
    let final_txt = joint_out.narrow(0, 0, TXT_LEN)?;
    let final_img = joint_out.narrow(0, TXT_LEN, IMG_SEQ)?;

    // same loss as the reference: mse(img_out,target_img) + mse(txt_out,target_txt)
    let loss = mse(&final_img, get(w, "inputs/target_img")?)?
        .add(&mse(&final_txt, get(w, "inputs/target_txt")?)?)?;
    acts.push(("loss".into(), loss.clone()));

    let grads = loss.backward()?;

    let mut fail = false;
    // forward sanity: the variable-wrapped graph must replay the same activations
    for (name, ours) in &acts {
        fail |= !compare(name, ours, get(w, name)?)?;
    }
    // the gradients under test
    for (name, var) in [("inputs/img", &img_in), ("inputs/txt", &txt_in)] {
        let g = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {name}")))?;
        fail |= !compare(&format!("grad/{name}"), g, get(w, &format!("grad/{name}"))?)?;
    }
    for (key, var) in vars {
        let g = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
        let ref_key = key.replace("weights/", "grad/");
        fail |= !compare(&ref_key, g, get(w, &ref_key)?)?;
    }
    Ok(fail)
}

/// Checkpointed mode: the same pair as two stages, gradients produced by the
/// reverse walk in `checkpointed_backward`.
fn checkpointed(w: &Weights, vars: &[(String, Var)], rts: &Rts, dev: &Device) -> Result<bool> {
    // Each closure needs its own capture; a `Tensor` clone shares storage, so
    // both stages read the same temb.
    let temb = get(w, "inputs/temb")?.clone();
    let temb1 = temb.clone();
    // Stage 0 folds the txt/img join into its single output so stage 1 has
    // arity 1; the stage closures are the lib's own block functions, so the
    // graph rebuilt during backward is exactly the one the forward pass ran.
    let stages = vec![
        Stage {
            run: Box::new(move |w: &Weights, ins: &[Tensor]| {
                let mut acts = Vec::new();
                let (img2, txt2) =
                    double_block(w, &mut acts, &ins[0], &ins[1], &temb, &rts.img, &rts.txt)?;
                Ok(vec![Tensor::cat(&[&txt2, &img2], 0)?])
            }),
        },
        Stage {
            run: Box::new(move |w: &Weights, ins: &[Tensor]| {
                let mut acts = Vec::new();
                Ok(vec![single_block(
                    w, &mut acts, &ins[0], &temb1, &rts.joint,
                )?])
            }),
        },
    ];
    let inputs = [get(w, "inputs/img")?.clone(), get(w, "inputs/txt")?.clone()];
    let seed = |outs: &[Tensor]| -> Result<Tensor> {
        let final_txt = outs[0].narrow(0, 0, TXT_LEN)?;
        let final_img = outs[0].narrow(0, TXT_LEN, IMG_SEQ)?;
        mse(&final_img, get(w, "inputs/target_img")?)?
            .add(&mse(&final_txt, get(w, "inputs/target_txt")?)?)
    };

    let (loss, input_grads, grads) = checkpointed_backward(w, &stages, &inputs, seed, vars)?;
    let mut fail = !compare("loss", &Tensor::new(&[loss], dev)?, get(w, "loss")?)?;
    for (ig, name) in input_grads.iter().zip(["inputs/img", "inputs/txt"]) {
        fail |= !compare(
            &format!("grad/{name}"),
            ig,
            get(w, &format!("grad/{name}"))?,
        )?;
    }
    for (key, var) in vars {
        let g = grads
            .get(var.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {key}")))?;
        let ref_key = key.replace("weights/", "grad/");
        fail |= !compare(&ref_key, g, get(w, &ref_key)?)?;
    }
    Ok(fail)
}

fn main() -> Result<()> {
    let checkpoint = std::env::args().any(|a| a == "--checkpoint");
    let dev = Device::new_cuda(0)?;
    let gpu_name = cudarc::driver::CudaContext::new(0)
        .and_then(|c| c.name())
        .unwrap_or_else(|_| "unknown".into());
    println!("device: {gpu_name}");

    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_bf16.safetensors");
    let mut w: Weights = safetensors::load(&path, &dev)?;
    println!("fixture: {} ({} tensors)", path.display(), w.len());

    // Candle only tracks gradients for variable leaves: make every LoRA A/B
    // factor a variable. (The two stream inputs become leaves too — in the
    // full-graph mode here, and for stage 0 inside `checkpointed_backward`.)
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
    println!(
        "backward: {}",
        if checkpoint {
            "per-block checkpointing (2 stages, graphs rebuilt in reverse)"
        } else {
            "one full-graph autograd pass"
        }
    );

    // 4-axis positions: img axis0=row, axis1=col; txt axis0=index; axes 2-3 zero.
    let pos_img: Vec<[f32; 4]> = (0..IMG_SEQ)
        .map(|i| [(i / IMG_GRID) as f32, (i % IMG_GRID) as f32, 0.0, 0.0])
        .collect();
    let pos_txt: Vec<[f32; 4]> = (0..TXT_LEN).map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let pos_all: Vec<[f32; 4]> = pos_img.iter().chain(pos_txt.iter()).copied().collect();
    let rts = Rts {
        img: rope_tables(&pos_img, &dev)?,
        txt: rope_tables(&pos_txt, &dev)?,
        joint: rope_tables(&pos_all, &dev)?,
    };

    let fail = if checkpoint {
        checkpointed(&w, &vars, &rts, &dev)?
    } else {
        full_graph(&w, &vars, &rts)?
    };
    println!("RESULT: {}", if fail { "FAIL" } else { "PASS" });
    if fail {
        std::process::exit(1);
    }
    Ok(())
}
