# Implementation plan

Each item is sized for one focused session within a local model's context window. Every item lists inputs (what to read), an acceptance check, and its commit or tag. Phases are gated as in `CLAUDE.md`. Where an item says **Claude**, run it with a model that has web access or strong numerics; the rest suit the local model.

## Phase 0: research (this document set)
- [x] RESEARCH, MODEL_SUPPORT, ARCHITECTURE, TESTING, STRATA (partial), IMPLEMENTATION_PLAN written. Tag: `v0.0.2-research`.
- [x] Record Windows toolchain versions in `ARCHITECTURE.md`.
- [ ] **Claude:** read the header of a real FLUX.2 klein LoRA `.safetensors` and record tensor names, alpha and scale conventions in `RESEARCH.md`.
- [ ] **Claude:** read the diffusers FLUX.2 pipeline source to confirm text-encoder layer selection, timestep sampling and loss.

## Phase 0.5: numerical feasibility spike (hard gate)

**Time box (hard budget):** 10 working days from the first spike commit, and at most 5 distinct failed attempts on any one numerics problem. When either limit is hit without a PASS, write a failure report to `BUILD_LOG.md` and `STATUS.md` and stop for human review. These are proposed defaults; the owner may change them before the spike starts.

Inputs for all items: `docs/SPEC.md` section 22, `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `docs/MODEL_SUPPORT.md`. Before every GPU run, unload Strata and record NVML free VRAM.

- [x] 0.5.1 Confirm CUDA builds on the dev machine: a hello-world kernel via the chosen crate compiles and runs on the 4090. Accept: prints device name and a correct vector add. Commit.
- [x] 0.5.2 Fixture generator `tools/fixtures/gen_block.py` for the scaled-down klein-style block, fp32 and bf16, with noise-floor report. Accept: fixture files plus `PROVENANCE.md` committed.
- [x] 0.5.3 Forward pass of the block on GPU (path A), bf16. Accept: activations within `TESTING.md` tolerances.
- [x] 0.5.4 Attention backward as a custom op (bf16). Accept: gradient cosine >= 0.999 on attention inputs.
- [x] 0.5.5 Full-block backward including LoRA A/B gradients. Accept: LoRA gradients within tolerance.
- [x] 0.5.6 AdamW step on LoRA parameters. Accept: update within tolerance after 1 and 10 steps.
- [x] 0.5.7 Repeat/determinism and 1000-iteration stability run. Accept: bit-identical repeats, no drift in loss on fixed data.
- [ ] 0.5.8 Memory teardown test with NVML. Accept: free VRAM back within 200 MB of baseline.
- [ ] 0.5.9 Throughput measurement at the real klein block dimensions and 1024px token count. Accept: step time recorded; usable-or-not judgement written in `BUILD_LOG.md`.
- [ ] 0.5.10 Decision record: PASS or FAIL, chosen path (A, B or C), confirm Backend #1. Tag a passing implementation `v0.0.3-spike-pass`.

If path A fails inside its share of the budget (days 1 to 5), switch to path B for days 6 to 10.

## Phase 1: application skeleton (starts only after the spike passes)
- [ ] 1.1 Cargo workspace and crate stubs per SPEC section 5; `cargo test` and `cargo build --release` pass. Tag `v0.0.4-bootstrap-workspace`.
- [ ] 1.2 Axum server on 127.0.0.1 with static embedded page and a health endpoint. Accept: page loads; request with a foreign `Origin` to a state-changing endpoint is rejected (test).
- [ ] 1.3 Project create/open/save with relative paths (`PROJECT_FORMAT.md` first). Accept: move a project directory and reopen it.
- [ ] 1.4 Settings persistence and structured event stream (SSE) with a test client.
- [ ] 1.5 Hardware detection (NVML with graceful absence, CPU, RAM, disk) shown in the status bar.
- [ ] 1.6 Output stage validation: path, write test, free space, name conflicts, never overwrite.
- [ ] 1.7 Structured logging to `metrics.jsonl` and `training.log`.

## Phase 2: Strata integration
- [ ] 2.1 **Verify the real Strata interface**: unload, load/wake, sleeping state, busy and timeout behavior, image request format. Record in `STRATA.md` (already verified: health, models, messages).
- [ ] 2.2 `AssistantProvider` trait and `StrataProvider` health/status.
- [ ] 2.3 Caption a single image through Strata; then batch captioning with progress.
- [ ] 2.4 GPU handoff: unload, NVML verification, retry within failure budget, clear error on failure.
- [ ] 2.5 Wake path and failure recovery per SPEC sections 17 and 18.

## Phase 3: dataset system
- [ ] 3.1 Import (folder, drag and drop) for PNG/JPEG/WebP with EXIF rotation, thumbnails, dimensions.
- [ ] 3.2 Captions: editable, user-modified flag, never overwrite without confirmation.
- [ ] 3.3 Duplicate and near-duplicate detection; dataset statistics.
- [ ] 3.4 Batch operations (trigger word, pattern apply) and QA hooks.

## Phases 4 to 9
Planned per SPEC section 42 and broken down into session-sized items when the preceding gate passes: 4 first-model inference (validated against reference), 5 native LoRA training, 6 first external-compatible LoRA (`v0.1.0-first-working-trainer`), 7 full UX, 8 second model family, 9 reliability.
