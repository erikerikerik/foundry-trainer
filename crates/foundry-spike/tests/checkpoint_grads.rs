//! Proof for the gradient-checkpointing flag (item 0.5.9 follow-up): the
//! checkpointed backward must produce the *same* gradients as the ordinary
//! full-graph backward.
//!
//! Deliberately CPU + the fp32 fixture: a correctness proof needs no GPU run
//! (the GPU bins are the owner's, with Strata unloaded), and fp32 keeps the
//! comparison about the algorithm rather than about dtype noise. The fixture
//! is the same reference the 0.5.5 test uses, so the gradients are checked
//! three ways: full-graph vs fixture, checkpointed vs fixture, and
//! checkpointed vs full-graph. Tolerances are `foundry_spike::compare`'s
//! (cos >= 0.999, rel L2 <= 2e-2) — unchanged, as everywhere else in the
//! spike.

use candle_core::{backprop::GradStore, safetensors, Device, Result, Tensor, Var};
use foundry_spike::{
    checkpointed_backward, compare, double_block, get, mse, rope_tables, single_block, Stage,
    Weights, IMG_GRID, IMG_SEQ, TXT_LEN,
};
use std::collections::HashMap;
use std::path::Path;

struct Rts {
    img: foundry_spike::RopeTables,
    txt: foundry_spike::RopeTables,
    joint: foundry_spike::RopeTables,
}

/// `inputs/img`, `inputs/txt` and the six LoRA factors: the eight gradients
/// the 0.5.5 fixture records.
fn grad_names(vars: &[(String, Var)]) -> Vec<String> {
    let mut names: Vec<String> = vec!["grad/inputs/img".into(), "grad/inputs/txt".into()];
    names.extend(vars.iter().map(|(k, _)| k.replace("weights/", "grad/")));
    names.sort();
    names
}

fn need<'a>(g: &'a GradStore, t: &Tensor, what: &str) -> Result<&'a Tensor> {
    g.get(t)
        .ok_or_else(|| candle_core::Error::Msg(format!("no gradient for {what}")))
}

/// One autograd graph over the whole block pair; returns the eight gradients
/// keyed by their fixture names.
fn full_graph(w: &Weights, vars: &[(String, Var)], rts: &Rts) -> Result<HashMap<String, Tensor>> {
    let img_in = Var::from_tensor(get(w, "inputs/img")?)?;
    let txt_in = Var::from_tensor(get(w, "inputs/txt")?)?;
    let temb = get(w, "inputs/temb")?;
    let mut acts = Vec::new();
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
    let loss = mse(&final_img, get(w, "inputs/target_img")?)?
        .add(&mse(&final_txt, get(w, "inputs/target_txt")?)?)?;
    let grads = loss.backward()?;

    let mut out = HashMap::new();
    out.insert(
        "grad/inputs/img".into(),
        need(&grads, img_in.as_tensor(), "inputs/img")?.clone(),
    );
    out.insert(
        "grad/inputs/txt".into(),
        need(&grads, txt_in.as_tensor(), "inputs/txt")?.clone(),
    );
    for (key, var) in vars {
        out.insert(
            key.replace("weights/", "grad/"),
            need(&grads, var.as_tensor(), key)?.clone(),
        );
    }
    Ok(out)
}

/// The same pair as two checkpoint stages; returns the same eight gradients.
fn checkpointed(w: &Weights, vars: &[(String, Var)], rts: &Rts) -> Result<HashMap<String, Tensor>> {
    // Each closure needs its own capture; a `Tensor` clone shares storage.
    let temb = get(w, "inputs/temb")?.clone();
    let temb1 = temb.clone();
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
    let (_loss, input_grads, grads) = checkpointed_backward(w, &stages, &inputs, seed, vars)?;

    let mut out = HashMap::new();
    out.insert("grad/inputs/img".into(), input_grads[0].clone());
    out.insert("grad/inputs/txt".into(), input_grads[1].clone());
    for (key, var) in vars {
        out.insert(
            key.replace("weights/", "grad/"),
            need(&grads, var.as_tensor(), key)?.clone(),
        );
    }
    Ok(out)
}

#[test]
// `Result` here is candle's one-error-type alias, so the test's own result is
// spelled out in full.
fn checkpointing_leaves_gradients_unchanged() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let dev = Device::Cpu;
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fixtures/block_fp32.safetensors");
    let mut w: Weights = safetensors::load(&path, &dev)?;

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

    let full = full_graph(&w, &vars, &rts)?;
    let ckpt = checkpointed(&w, &vars, &rts)?;

    let names = grad_names(&vars);
    assert_eq!(names.len(), 8, "expected 8 gradients under test");
    assert_eq!(full.len(), 8);
    assert_eq!(ckpt.len(), 8);

    let mut failed: Vec<String> = Vec::new();
    for name in &names {
        let reference = get(&w, name)?;
        let f = full.get(name).expect("full-graph gradient");
        let c = ckpt.get(name).expect("checkpointed gradient");
        // Both against the PyTorch reference (the 0.5.5 acceptance), and
        // checkpointed directly against full-graph: "unchanged" in the strict
        // sense, not merely "both still inside tolerance".
        for (label, ok) in [
            (
                "vs fixture",
                compare(&format!("{name} full-graph"), f, reference)?,
            ),
            (
                "vs fixture",
                compare(&format!("{name} checkpointed"), c, reference)?,
            ),
            (
                "vs full-graph",
                compare(&format!("{name} checkpointed"), c, f)?,
            ),
        ] {
            if !ok {
                failed.push(format!("{name} {label}"));
            }
        }
    }
    assert!(
        failed.is_empty(),
        "checkpointed gradients disagree: {}",
        failed.join(", ")
    );
    println!(
        "RESULT: PASS — all {} gradients identical between full-graph and \
         checkpointed backward (and both vs the fixture)",
        names.len()
    );
    Ok(())
}
