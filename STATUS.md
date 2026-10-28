# STATUS

## Current milestone
Phase 0 (research) complete in draft. Phase 0.5 spike: 0.5.1 **PASSED on the desktop 4090**, 0.5.2 fixtures generated, 0.5.3 **PASSED** (Candle bf16 forward, all activations rel L2 <= 6.3e-3 vs 2e-2 tolerance), 0.5.4 **PASSED** (hand-written attention backward CustomOp; gradient cosine 1.000000, worst rel L2 1.4e-6 vs 0.999/2e-2 acceptance), 0.5.5 **PASSED** (full-block backward incl. LoRA A/B gradients; worst rel L2 8.9e-3, below the 1.2e-2 fixture noise floor), 0.5.6 **PASSED** (10 AdamW steps on LoRA params; worst rel L2 1.18e-2 vs `opt1/*`/`opt10/*`, below the 1.64e-2 noise floor), 0.5.7 **PASSED** (bit-identical repeated runs; 1000-step bf16 stability: loss 3.89 -> 2.12, all finite, window means strictly decreasing — no drift; fp32 control shows step rises are Adam overshoot, not precision noise; 2026-10-28). 0.5.10 **decision recorded: PASS on path A**, Backend #1 confirmed as FLUX.2-klein-base-4B, tagged `v0.0.3-spike-pass` (2026-10-28, day 2 of the 10-day budget). Remaining in Phase 0.5: 0.5.8/0.5.9, owner-run measurements with Strata unloaded. Phase 1 must not start until those are recorded.

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

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- 0.5.8 in-process FAIL (344 MB, reproducible; process exit releases everything; worker-process model proposed, new item 0.5.8b) and 0.5.9 OOM at 512px and 1024px with the naive full-graph backward: see `docs/BUILD_LOG.md` (2026-10-28 owner-run entry). Next: faithful dims + per-block gradient checkpointing, then re-measure. Original note: nothing else for the local agent. The 0.5.8/0.5.9 binaries are written and committed; running them is owner work with Strata unloaded (the local agent runs on Strata) — see `docs/KICKOFF.md`.

## Known issues
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3). These need live web access, so a local model cannot do them.
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation. The clock started at the first spike commit `e03d552` (2026-10-27).
- VRAM measurements (items 0.5.8, 0.5.9) are invalid while Strata is loaded, and the local agent runs on Strata. Those items must be run by the owner from a plain PowerShell window with Strata unloaded; see `docs/KICKOFF.md`.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Owner: items 0.5.8/0.5.9 — unload Strata, then in Git Bash from the repo root:
   `. tools/dev/vcvars-env.sh && cargo run -p foundry-spike --bin mem_teardown --release`
   and `cargo run -p foundry-spike --bin throughput --release`; record outputs in `BUILD_LOG.md` and tick the plan.
2. Cloud Claude: LoRA header and diffusers pipeline reading (Phase 0 leftovers); the fixture structure may need a pass once the real single-block wiring and RoPE axis dims are confirmed.
3. After 0.5.8/0.5.9 are recorded: Phase 1 item 1.1 (workspace skeleton) may start.

## Last known working commit
See `git log` on `main`.
