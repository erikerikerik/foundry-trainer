# STATUS

## Current milestone
Phase 0 (research) complete in draft. Phase 0.5 spike: item 0.5.1 **PASSED on the desktop 4090** (2026-10-28). Blocked at 0.5.2: PyTorch is not installed on the desktop (Python 3.12.10 is).

## Completed
- Repo skeleton, amended `docs/SPEC.md`, `CLAUDE.md`, public GitHub remote, tag `v0.0.1-bootstrap` (on GitHub).
- Strata verified at `http://127.0.0.1:8080` (health, models, Anthropic-style `/v1/messages`); `tools/dev/claude-strata.ps1` launches Claude Code against it. See `docs/STRATA.md`.
- Phase 0 docs: `RESEARCH.md`, `MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TESTING.md`, `IMPLEMENTATION_PLAN.md`.
- Windows toolchain installed and recorded in `docs/ARCHITECTURE.md` (Rust 1.99.0 MSVC, CUDA 13.4, VS 2022 17.14.43, driver 616.64).
- Spike crate `crates/foundry-spike` (item 0.5.1): NVRTC vector add through cudarc 0.19.10. Passes `cargo fmt`, `cargo clippy -D warnings` and `cargo test` in the cloud (no GPU there).
- 0.5.1 run on the desktop (branch `phase-0-5-spike`): prints `device: NVIDIA GeForce RTX 4090`, 1M-element vector add with 0 mismatches, `RESULT: PASS`. Output recorded in `docs/BUILD_LOG.md`.

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- Blocked on 0.5.2 (fixture generator): needs PyTorch on the desktop, which is not installed. Per `docs/KICKOFF.md` the local agent stops and reports rather than installing it. A CPU-only wheel is sufficient for fixtures (`pip install torch --index-url https://download.pytorch.org/whl/cpu`, ~200 MB) and avoids touching the CUDA stack; owner decides.

## Known issues
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3). These need live web access, so a local model cannot do them.
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation. The clock started at the first spike commit `e03d552` (2026-10-27).
- VRAM measurements (items 0.5.8, 0.5.9) are invalid while Strata is loaded, and the local agent runs on Strata. Those items must be run by the owner from a plain PowerShell window with Strata unloaded; see `docs/KICKOFF.md`.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Owner: install PyTorch on the desktop (CPU wheel is enough for fixtures) or tell the agent to proceed differently; then the local agent continues with 0.5.2 on `phase-0-5-spike`.
2. Cloud Claude: LoRA header and diffusers pipeline reading (Phase 0 leftovers).
3. Owner (later, when the spike reaches them): items 0.5.8/0.5.9 from a plain PowerShell window with Strata unloaded.

## Last known working commit
See `git log` on `main`.
