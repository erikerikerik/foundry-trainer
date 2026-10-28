# Fixture provenance — Phase 0.5 spike block (item 0.5.2)

Dev-only reference data generated externally under CLAUDE.md rule 2. Nothing in
`tools/` ships or runs in CI; only this data and note are committed.

## Generation

- Date: 2026-10-28 (desktop dev machine, Windows 11, CPU only — no GPU used)
- Tool: `python gen_block.py` in this directory
- Python 3.12.10, torch 2.14.1+cpu, safetensors 0.8.0
- Seed: 20261028 (`torch.use_deterministic_algorithms(True)`, 1 thread)
- fp32 pipeline run twice: bit-identical (determinism PASS, recorded in `meta.json`)
- 2026-10-28 second run: added `attn_in/*` and `grad/attn/*` tensors for spike
  item 0.5.4 (isolated attention-backward test). No RNG draws changed, so all
  previously recorded tensors are bit-identical (verified: unchanged noise-floor
  entries in `meta.json`, and `block_forward.rs` still PASSes).

## Files

| File | Contents |
|---|---|
| `block_fp32.safetensors` | 110 tensors, fp32: inputs, weights (incl. LoRA A/B), intermediate activations (incl. `attn_in/*`: each attention's q/k/v/attn graph tensors), output, loss, gradients (frozen-path inputs, LoRA A/B, and `grad/attn/*`: dL/dq, dL/dk, dL/dv plus the upstream gradient into each attention output), LoRA values after 1 and 10 AdamW steps |
| `block_bf16.safetensors` | same 110 tensors, bf16 run (weights cast from the same fp32 init) |
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

Worst rel L2 over 53 compared tensors: 1.64e-2 (`opt10/double.img.to_q.lora_b`,
cosine 0.999869). All tensors within the initial `TESTING.md` thresholds
(rel L2 <= 2e-2, cosine >= 0.999), so the initial thresholds stand; the
per-tensor floor is in `meta.json` for tighter bounds later. The attention
gradients specifically: worst rel L2 1.12e-2, worst cosine 0.99994 — the
0.5.4 acceptance (cosine >= 0.999, rel L2 <= 2e-2) sits above this floor.

## Known simplifications vs real klein

- Single-block fusion combines attn and MLP with separate gates; exact diffusers
  `FluxSingleTransformerBlock` wiring is unverified (Phase 0 cloud item).
- RoPE axis split is equal quarters; real klein axis dims are unverified.
- Loss is synthetic (MSE vs fixed target); the real training loss/timestep
  recipe is an open item in `docs/RESEARCH.md`.
These affect op coverage, not op semantics; revisit fixtures if the cloud
reading changes the structure.

## Fixture vs the faithful-klein stack in `throughput.rs` (2026-10-28)

Checked `gen_block.py` against the layout `crates/foundry-spike/src/bin/
throughput.rs` now builds, which follows `docs/RESEARCH.md` section 2 (the
klein 4B config — note there is no section 3.1; section 3 is "Not yet
researched"). The fixture block does **not** match that layout. The fixtures
were deliberately **not regenerated**: `gen_block.py` is unchanged and every
recorded tensor stays bit-identical. `throughput.rs` builds the faithful
layout with its own random weights; the fixture stays the small block whose
purpose is op-semantics verification, not shape fidelity.

| | fixture (`gen_block.py`) | real klein / `throughput.rs` |
|---|---|---|
| single-block in-projection | three linears: `single.to_qkv` (dim→3·dim), `mlp_gate`, `mlp_up` (dim→512) | ONE fused linear dim→9·dim = 27648 at dim 3072 (q,k,v of 3072, then SwiGLU gate and up of 9216) |
| single-block out-projection | none — `out = x + g_at*attn + g_mlp*mlp` adds the gated branches straight to the residual | ONE fused linear (dim+mlp)→dim = 12288→3072 over the attention output joined with the activated MLP |
| MLP width | 512 = 2·dim | `mlp_ratio` 3.0 → 9216 |
| modulation | one linear *per block*, with bias | one bias-free linear shared by all 25 blocks |
| LoRA targets | `double.img.to_q`, `double.img.to_v`, `single.to_qkv` | rank 4 on the fused single-stream in/out projections plus double `img.to_q`/`to_v`; the real target modules and rank are still an open item (`docs/RESEARCH.md` section 3, `docs/MODEL_SUPPORT.md`) |
| RoPE theta | 10000 | 2000 |
| double-stream block | separate img/txt q,k,v,out + per-stream SwiGLU: shape matches real klein apart from the shared bias-free modulation | same |

Two consequences for anyone reusing the fixture:

- The fixture exercises *no* fused-in/out-projection code path and *no*
  shared-modulation path, so those two are verified only by the throughput
  run's own forward/backward consistency, not against a PyTorch reference.
- The fixture's single-block `mlp_down` (512→dim) has no counterpart in the
  faithful layout, where the MLP output width is folded into the fused
  out-projection input.

Unchanged fixture quirks that `throughput.rs` also reproduces: the joint
sequence is concat(txt, img) while its single-block RoPE tables are built
img-then-txt, and the equal-quarter RoPE axis split.

## License

Generated with PyTorch (BSD-3-Clause) as a dev tool only; no third-party source
copied. The fixture data is our own synthetic tensors.
