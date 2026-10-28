# STATUS

## Current milestone
Phase 0 (research) complete in draft. Phase 0.5 (numerical feasibility spike) item 0.5.1 is written and waiting for its first run on the desktop GPU.

## Completed
- Repo skeleton, amended `docs/SPEC.md`, `CLAUDE.md`, public GitHub remote, tag `v0.0.1-bootstrap` (on GitHub).
- Strata verified at `http://127.0.0.1:8080` (health, models, Anthropic-style `/v1/messages`); `tools/dev/claude-strata.ps1` launches Claude Code against it. See `docs/STRATA.md`.
- Phase 0 docs: `RESEARCH.md`, `MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TESTING.md`, `IMPLEMENTATION_PLAN.md`.
- Windows toolchain installed and recorded in `docs/ARCHITECTURE.md` (Rust 1.99.0 MSVC, CUDA 13.4, VS 2022 17.14.43, driver 616.64).
- Spike crate `crates/foundry-spike` (item 0.5.1): NVRTC vector add through cudarc 0.19.10. Passes `cargo fmt`, `cargo clippy -D warnings` and `cargo test` in the cloud (no GPU there).

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- 0.5.1: run `cargo run -p foundry-spike` on the desktop and record the output. Not yet run on a GPU.

## Known issues
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3). These need live web access, so a local model cannot do them.
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation. The clock starts at the first spike commit.
- VRAM measurements (items 0.5.8, 0.5.9) are invalid while Strata is loaded, and the local agent runs on Strata. Those items must be run by the owner from a plain PowerShell window with Strata unloaded; see `docs/KICKOFF.md`.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Owner: `git pull`, then `cargo run -p foundry-spike`; paste the output.
2. Local agent (Strata + Claude Code): follow `docs/KICKOFF.md` for items 0.5.2 onward.
3. Cloud Claude: LoRA header and diffusers pipeline reading (Phase 0 leftovers).

## Last known working commit
See `git log` on `main`.
