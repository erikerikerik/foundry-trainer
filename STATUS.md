# STATUS

## Current milestone
Phase 0 (research) complete in draft. Phase 0.5 spike: 0.5.1 **PASSED on the desktop 4090**, 0.5.2 fixtures generated, 0.5.3 **PASSED** (Candle bf16 forward, all activations rel L2 <= 6.3e-3 vs 2e-2 tolerance; 2026-10-28). Next: 0.5.4 attention backward.

## Completed
- Repo skeleton, amended `docs/SPEC.md`, `CLAUDE.md`, public GitHub remote, tag `v0.0.1-bootstrap` (on GitHub).
- Strata verified at `http://127.0.0.1:8080` (health, models, Anthropic-style `/v1/messages`); `tools/dev/claude-strata.ps1` launches Claude Code against it. See `docs/STRATA.md`.
- Phase 0 docs: `RESEARCH.md`, `MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TESTING.md`, `IMPLEMENTATION_PLAN.md`.
- Windows toolchain installed and recorded in `docs/ARCHITECTURE.md` (Rust 1.99.0 MSVC, CUDA 13.4, VS 2022 17.14.43, driver 616.64).
- Spike crate `crates/foundry-spike` (item 0.5.1): NVRTC vector add through cudarc 0.19.10. Passes `cargo fmt`, `cargo clippy -D warnings` and `cargo test` in the cloud (no GPU there).
- 0.5.1 run on the desktop (branch `phase-0-5-spike`): prints `device: NVIDIA GeForce RTX 4090`, 1M-element vector add with 0 mismatches, `RESULT: PASS`. Output recorded in `docs/BUILD_LOG.md`.
- 0.5.2: CPU PyTorch installed with owner approval (torch 2.14.1+cpu); `tools/fixtures/gen_block.py` + fixtures (`block_fp32/bf16.safetensors`, `meta.json`, `PROVENANCE.md`) committed. Measured bf16-vs-fp32 noise floor is inside the initial `TESTING.md` thresholds.
- 0.5.3 (path A): `crates/foundry-spike/src/bin/block_forward.rs` runs the fixture block pair in Candle 0.11 CUDA bf16 on the 4090; every recorded activation within tolerance (worst rel L2 6.2e-3, noise floor 1.64e-2). `candle_smoke.rs` proves the CUDA op set. Toolchain recipe for nvcc-under-Git-Bash in `docs/BUILD_LOG.md`.

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- 0.5.4 (attention backward as a Candle CustomOp, bf16) — correctness test, so Strata may stay loaded (same reasoning as 0.5.1); the unload-before-GPU-run rule still binds 0.5.8/0.5.9 for the owner.

## Known issues
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3). These need live web access, so a local model cannot do them.
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation. The clock started at the first spike commit `e03d552` (2026-10-27).
- VRAM measurements (items 0.5.8, 0.5.9) are invalid while Strata is loaded, and the local agent runs on Strata. Those items must be run by the owner from a plain PowerShell window with Strata unloaded; see `docs/KICKOFF.md`.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Local agent: 0.5.4 attention backward (CustomOp, cosine >= 0.999 vs `grad/*` fixture tensors), then 0.5.5/0.5.6, on `phase-0-5-spike`.
2. Cloud Claude: LoRA header and diffusers pipeline reading (Phase 0 leftovers); the fixture structure may need a pass once the real single-block wiring and RoPE axis dims are confirmed.
3. Owner (later, when the spike reaches them): items 0.5.8/0.5.9 from a plain PowerShell window with Strata unloaded.

## Last known working commit
See `git log` on `main`.
