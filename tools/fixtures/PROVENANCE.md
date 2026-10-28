# Fixture provenance — Phase 0.5 spike block (item 0.5.2)

Dev-only reference data generated externally under CLAUDE.md rule 2. Nothing in
`tools/` ships or runs in CI; only this data and note are committed.

## Generation

- Date: 2026-10-28 (desktop dev machine, Windows 11, CPU only — no GPU used)
- Tool: `python gen_block.py` in this directory
- Python 3.12.10, torch 2.14.1+cpu, safetensors 0.8.0
- Seed: 20261028 (`torch.use_deterministic_algorithms(True)`, 1 thread)
- fp32 pipeline run twice: bit-identical (determinism PASS, recorded in `meta.json`)

## Files

| File | Contents |
|---|---|
| `block_fp32.safetensors` | 94 tensors, fp32: inputs, weights (incl. LoRA A/B), intermediate activations, output, loss, gradients (frozen-path inputs + LoRA A/B), LoRA values after 1 and 10 AdamW steps |
| `block_bf16.safetensors` | same 94 tensors, bf16 run (weights cast from the same fp32 init) |
| `meta.json` | config, determinism result, fp32-vs-bf16 noise-floor report (rel L2 + cosine per tensor) |

## Structure (scaled-down FLUX.2-klein-style, per `docs/TESTING.md`)

`dim` 256 = 2 heads x `head_dim` 128; 4-axis RoPE `[32,32,32,32]`, theta 10000,
rotate-half convention; RMSNorm QK-norm (eps 1e-6); adaLN modulation
(`(1+scale)*LN(x)+shift`, gates multiply branches); SwiGLU MLP (inner 512);
one double-stream block (16 img tokens on a 4x4 grid + 8 txt tokens, joint
attention, txt-then-img concat order) then one single-stream block over the
concatenated sequence (fused QKV + fused SwiGLU from one normed input,
`out = x + gate_attn*attn + gate_mlp*mlp`). LoRA rank 4, alpha 8 (scale 2) on
`double.img.to_q`, `double.img.to_v`, `single.to_qkv`; A init N(0,0.02),
B init N(0,0.01) (nonzero so LoRA gradients are nonzero). Loss:
`mse(img_out,target_img) + mse(txt_out,target_txt)`. AdamW lr 1e-3,
betas (0.9,0.999), eps 1e-8, weight_decay 0.

## Reference semantics the Rust implementation must match

- matmuls: compute-dtype inputs, fp32 accumulation (hardware behavior)
- attention softmax: fp32, probabilities cast back to compute dtype
- LayerNorm/RMSNorm statistics: fp32, result cast back
- RoPE tables: fp32, cast to compute dtype

## Noise floor (PyTorch bf16 vs PyTorch fp32, same inputs)

Worst rel L2 over 37 compared tensors: 1.64e-2 (`opt10/double.img.to_q.lora_b`,
cosine 0.999869). All tensors within the initial `TESTING.md` thresholds
(rel L2 <= 2e-2, cosine >= 0.999), so the initial thresholds stand; the
per-tensor floor is in `meta.json` for tighter bounds later.

## Known simplifications vs real klein

- Single-block fusion combines attn and MLP with separate gates; exact diffusers
  `FluxSingleTransformerBlock` wiring is unverified (Phase 0 cloud item).
- RoPE axis split is equal quarters; real klein axis dims are unverified.
- Loss is synthetic (MSE vs fixed target); the real training loss/timestep
  recipe is an open item in `docs/RESEARCH.md`.
These affect op coverage, not op semantics; revisit fixtures if the cloud
reading changes the structure.

## License

Generated with PyTorch (BSD-3-Clause) as a dev tool only; no third-party source
copied. The fixture data is our own synthetic tensors.
