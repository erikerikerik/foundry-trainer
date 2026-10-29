# STATUS

## Current milestone
Phase 0 research complete. Phase 0.5 spike **measurements complete** (2026-10-28, day 2 of the 10-day budget): numerics 0.5.1 to 0.5.7 PASS (tag `v0.0.3-spike-pass`), worker-process teardown 0.5.8b PASS (0.0 MB deficit x3; in-process 0.5.8 FAIL by 344 MB, design changed to a worker process), checkpointed gradients unchanged on GPU, throughput 0.5.9 measured: 512px usable (1.4 s/step, 15 GB peak), 1024px not usable as built (card full, 111 s/step). Backend #1 (FLUX.2-klein-base-4B) stays. Open before Phase 1: owner confirmation of the 0.5.9 judgement and a merge of `phase-0-5-spike` to `main`.

## Completed
- Repo skeleton, amended `docs/SPEC.md`, `CLAUDE.md`, public GitHub remote, tag `v0.0.1-bootstrap` (on GitHub).
- Strata verified at `http://127.0.0.1:8080` (health, models, Anthropic-style `/v1/messages`); `tools/dev/claude-strata.ps1` launches Claude Code against it. See `docs/STRATA.md`.
- Phase 0 docs: `RESEARCH.md`, `MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TESTING.md`, `IMPLEMENTATION_PLAN.md`.
- Windows toolchain installed and recorded in `docs/ARCHITECTURE.md` (Rust 1.99.0 MSVC, CUDA 13.4, VS 2022 17.14.43, driver 616.64).
- Spike crate `crates/foundry-spike` (item 0.5.1): NVRTC vector add through cudarc 0.19.10. Passes `cargo fmt`, `cargo clippy -D warnings` and `cargo test` in the cloud (no GPU there).
- 0.5.1 run on the desktop (branch `phase-0-5-spike`): prints `device: NVIDIA GeForce RTX 4090`, 1M-element vector add with 0 mismatches, `RESULT: PASS`. Output recorded in `docs/BUILD_LOG.md`.
- 0.5.2: CPU PyTorch installed with owner approval (torch 2.14.1+cpu); `tools/fixtures/gen_block.py` + fixtures (`block_fp32/bf16.safetensors`, `meta.json`, `PROVENANCE.md`) committed. Measured bf16-vs-fp32 noise floor is inside the initial `TESTING.md` thresholds.
- 0.5.3 (path A): `crates/foundry-spike/src/bin/block_forward.rs` runs the fixture block pair in Candle 0.11 CUDA bf16 on the 4090; every recorded activation within tolerance (worst rel L2 6.2e-3, noise floor 1.64e-2). `candle_smoke.rs` proves the CUDA op set. Toolchain recipe for nvcc-under-Git-Bash in `docs/BUILD_LOG.md`.
- 0.5.4 (path A): `crates/foundry-spike/src/bin/attn_backward.rs` — attention as a Candle `CustomOp3` with hand-written backward; gradients vs fixture `grad/attn/*`: cosine 1.000000, worst rel L2 1.4e-6. Fixtures extended to 110 tensors (`attn_in/*`, `grad/attn/*`); pre-existing tensors bit-identical. Details in `docs/BUILD_LOG.md`.
- 0.5.5 (path A): `crates/foundry-spike/src/lib.rs` (shared block code + `AttnBwd`, attention now routed through the custom op) and `src/bin/block_backward.rs` — whole block pair as one Candle autograd graph with `Var` leaves for `inputs/img`, `inputs/txt` and all six LoRA factors; real `loss.backward()`. All 8 gradients pass (worst rel L2 8.9e-3, cos >= 0.99996 — inside the 1.2e-2 fixture gradient noise floor); forward replay of every activation also matches. Details in `docs/BUILD_LOG.md`.
- 0.5.6 (path A): `crates/foundry-spike/src/bin/opt_step.rs` — 10-step AdamW training loop on the six LoRA `Var` parameters (fresh graph + `backward()` per step, hand-written bias-corrected AdamW, fp32 moments, `Var::set` for the in-place update). LoRA values after steps 1 and 10 match the fixture `opt1/*`/`opt10/*` (worst rel L2 1.18e-2, cos >= 0.99993 — inside the 1.64e-2 noise floor); loss falls monotonically 3.8932 -> 3.8818. Details in `docs/BUILD_LOG.md`.
- 0.5.7 (path A): `crates/foundry-spike/src/bin/stability.rs` — two independent 10-step bf16 runs are bit-identical (per-step losses + all six final LoRA tensors); 1000-step bf16 run on fixed data stays finite and drift-free (loss 3.8932 -> 2.1219, 100-step window means strictly decreasing, longest rise run 3). The fp32 control (same loop, fp32 fixture) rises on isolated steps *more* than bf16 (max 6.2e-3 vs 1.6e-3), proving step rises are Adam overshoot at lr 1e-3 rather than a precision artifact; the drift gate is structural (window means, rise-run and rise-size caps), applied to both. AdamW extracted to `lib::adamw_step`. Details in `docs/BUILD_LOG.md`.
- 0.5.8/0.5.9 measurement binaries written (NOT yet run — they must run with Strata unloaded): `src/bin/mem_teardown.rs` (NVML baseline before any CUDA context, 20-step bf16 fixture training burst, drop device + tensors, verify free VRAM returns within 200 MB; prints per-process list with the WDDM caveat) and `src/bin/throughput.rs` (random-weight 5+20 block stack at real klein dims — 3072 hidden, 24x128 heads — 1024 and 4096 image tokens, 2 warmup + 5 timed AdamW steps, ms/step + NVML-sampled peak VRAM, clean OUT-OF-MEMORY report per config). `nvml-wrapper` 0.13.0 added; licenses of all spike deps recorded in `THIRD_PARTY_NOTICES.md`.
- 0.5.8b + 0.5.9 follow-ups implemented locally (measurements still owner-run): `src/bin/worker_teardown.rs` (parent is NVML-only with no CUDA context at all, spawns its own executable as a child that runs the same 20-step burst, waits for exit, sleeps 1 s, re-reads; same 200 MB gate, 3 rounds, `RESULT: PASS/FAIL`); `throughput.rs` now faithful to klein (mlp ratio 3 → 9216, **one** modulation linear shared by all 25 blocks, single block = one fused in-projection 3072→27648 and one fused out-projection 12288→3072; stack ≈3.74B params vs ~4B real); OOM runs now print peak VRAM before the error and report whether they died in forward or backward; `lib::checkpointed_backward` + `Stage` give per-block gradient checkpointing behind `--checkpoint` in `throughput` and `block_backward`. Proof that gradients are unchanged is a CPU test, `tests/checkpoint_grads.rs`: all 8 gradients bit-identical between full-graph and checkpointed backward (rel L2 0.000e0, cos 1.000000) and both vs the fixture (worst 1.34e-3) at unchanged tolerances. `tools/fixtures/PROVENANCE.md` records how the fixture block differs from the faithful layout; fixtures were NOT regenerated. Details in `docs/BUILD_LOG.md` (2026-10-28).

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- Nothing running. Next local-agent task (needs Strata back): CPU smoke test for the throughput stack at tiny sizes; align harness LoRA targets/scale with `docs/RESEARCH.md` 3.1; then run fmt, clippy and test before the merge to `main`.

## Known issues
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3). These need live web access, so a local model cannot do them.
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation. The clock started at the first spike commit `e03d552` (2026-10-27).
- VRAM measurements (items 0.5.8, 0.5.8b, 0.5.9) are invalid while Strata is loaded, and the local agent runs on Strata. Those items must be run by the owner from a plain PowerShell window with Strata unloaded; see `docs/KICKOFF.md`.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Owner: confirm the 0.5.9 judgement (512px usable, 1024px needs attention-memory work in Phase 5); restart Strata for the local agent.
2. Local agent: throughput CPU smoke test and LoRA target alignment; `cargo fmt`, `cargo clippy -D warnings`, `cargo test`; then merge `phase-0-5-spike` to `main` and tag (push the tag from the desktop).
3. Phase 1 item 1.1 (workspace skeleton) after the merge.
4. Open research: text-layer choice (9,18,27 vs 10,20,30), whether training applies the timestep shift, whether diffusers' loader reads alpha from LoRA metadata.

## Last known working commit
See `git log` on `main`.
