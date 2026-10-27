# STATUS

## Current milestone
Phase 0 (research) complete in draft; waiting on three small items before the Phase 0.5 spike can start.

## Completed
- Repo skeleton, amended `docs/SPEC.md`, `CLAUDE.md`, public GitHub remote, tag `v0.0.1-bootstrap` (on GitHub).
- Strata verified at `http://127.0.0.1:8080` (health, models, Anthropic-style `/v1/messages`); `tools/dev/claude-strata.ps1` launches Claude Code against it. See `docs/STRATA.md`.
- Phase 0 docs: `RESEARCH.md`, `MODEL_SUPPORT.md`, `ARCHITECTURE.md`, `TESTING.md`, `IMPLEMENTATION_PLAN.md`.

## Key findings
- Leading Backend #1: FLUX.2-klein-base-4B (Apache 2.0, official BFL training guidance, fits a 4090). Alternate: Z-Image (6B, Apache 2.0).
- Candle's flash-attention is forward only; Burn's autodiff attention is a composed fallback. Hand-written attention backward is likely needed (spike path A), with cudarc kernels as fallback (path B).

## In progress
- Nothing running.

## Known issues
- Windows toolchain versions are not recorded yet (`docs/ARCHITECTURE.md`).
- LoRA tensor naming and the training loss/timestep recipe for klein are unverified (`docs/RESEARCH.md`, section 3).
- The Phase 0.5 time box in `docs/IMPLEMENTATION_PLAN.md` (10 working days, 5 failed attempts per problem) is a proposed default awaiting owner confirmation.
- Pushing tags from the cloud workspace has been unreliable; push tags from the desktop.

## Next actions
1. Owner: confirm or change the spike budget; run the toolchain version commands on the desktop.
2. Read a real klein LoRA header and the diffusers FLUX.2 pipeline source; record results.
3. Phase 0.5 spike, items 0.5.1 to 0.5.10.

## Last known working commit
See `git log` on `main`.
