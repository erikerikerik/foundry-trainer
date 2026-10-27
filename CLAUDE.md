# Foundry Trainer: Agent Rules

Native Rust, browser-UI LoRA trainer for image-generation models. Full spec: `docs/SPEC.md`.
Do NOT reload the whole spec each session. Read only the doc for the task at hand (see "Where to look").

## Hard rules

1. **No Python in the shipped runtime.** No PyTorch/LibTorch/tch-rs, Kohya, Diffusers, Accelerate, PEFT, bitsandbytes, Gradio, no subprocess calls to Python, no embedded Python, no hidden fallback.
2. **Dev-only exception:** one-time reference fixtures may be generated externally (including with Python) under `tools/fixtures/`. Commit only the resulting data plus a provenance note. `tools/` is excluded from the Cargo workspace build and from CI.
3. **Strata is optional.** Foundry must work fully with no assistant provider.
4. **Never silently** delete images, edit captions, change training parameters, overwrite outputs, or mutate user data.
5. **Never assume GPU memory was released.** Measure it with NVML.
6. Web UI binds to `127.0.0.1` only. Check `Origin`/`Host` headers on all state-changing endpoints.
7. Never log or commit secrets (HF tokens, Strata keys). Keep `.gitignore` current.
8. Before reusing third-party source or adding a model: check its license, record it in `THIRD_PARTY_NOTICES.md` / `docs/LICENSES.md`.

## Priorities (when tradeoffs occur)

1. Correct training  2. Recoverability  3. Data integrity  4. Standards compatibility
5. GPU memory safety  6. Reliability  7. Simple UX  8. Performance  9. More models  10. Polish

## Git discipline

- `main` always builds and passes `cargo test`. Never force-push `main`; never rewrite public history.
- Work on feature branches (`phase-N-topic`). Merge to `main` only when green.
- Commit after every meaningful unit of work. Push checkpoints. Tag milestones (`v0.0.1-bootstrap`, ...).
- Keep `STATUS.md` current (milestone, done, in progress, known issues, next actions, last good commit).
- Log architectural decisions and failed approaches in `docs/BUILD_LOG.md`. Not every command.

## Autonomy and stopping

Make ordinary decisions yourself: pick a sensible default, document it, implement, test, continue.

**Stop and write a note in `STATUS.md` only if:**
- credentials are needed that can't be obtained programmatically,
- a license is ambiguous in a way that affects permitted use,
- required hardware is missing,
- an operation is destructive and needs explicit permission,
- you have exhausted the time box (below) or hit an unresolvable architectural contradiction.

**Time boxes (do not loop on failed experiments):**
- Any single failing CUDA/numerics problem: max 5 distinct attempts or 1 working session, whichever is first. Then write a failure report (what was tried, results, hypotheses) to `docs/BUILD_LOG.md`, update `STATUS.md`, and stop.
- Phase 0.5 spike: hard budget set in `docs/IMPLEMENTATION_PLAN.md`. Exceeding it = stop and report, not "try harder".

## Phase order (hard gates)

0. Research (needs live web; output goes to `docs/`) -> 0.5 numerical feasibility spike (**mandatory PASS gate**; built from the leading model candidate's real ops) -> 1 app skeleton -> 2 Strata integration (verify the real API first, document in `docs/STRATA.md`) -> 3 dataset -> 4 first-model inference (validate against reference before any training) -> 5 native LoRA training -> 6 first external-compatible LoRA (`v0.1.0`) -> 7 full UX -> 8 second model family -> 9 reliability.

Do not start a phase whose gate has not passed. Do not build the full UI before inference works.

## Testing rules

- Numerical work needs golden tests with explicit tolerances (activations, loss, gradients, LoRA gradients, optimizer step). Plausible-looking output is not proof.
- Use the toy validation model for loop/checkpoint/resume/CUDA-teardown/UI tests. No multi-GB downloads in CI.
- Before any GPU test run on the dev machine: confirm the local coding LLM (Strata) is unloaded, or VRAM measurements are invalid.

## Working style (token budget)

- Read narrowly: open the specific crate/doc you need, not the whole repo.
- Small tasks. If a checklist item in `docs/IMPLEMENTATION_PLAN.md` feels larger than one focused session, split it before starting.
- Prefer a narrow, reliable vertical slice over broad unfinished scaffolding.
- Run `cargo fmt`, `cargo clippy`, and `cargo test` before each commit.

## Where to look

| Need | Read |
|---|---|
| Requirements, workflows | `docs/SPEC.md` (section you need only) |
| Backend / framework choice | `docs/RESEARCH.md`, `docs/ARCHITECTURE.md` |
| Which model, status | `docs/MODEL_SUPPORT.md` |
| Strata endpoints and lifecycle | `docs/STRATA.md` |
| Project / checkpoint / resume format | `docs/PROJECT_FORMAT.md` |
| Test strategy and fixtures | `docs/TESTING.md` |
| Windows toolchain versions | `docs/ARCHITECTURE.md` (toolchain section) |
| Current state | `STATUS.md` |
