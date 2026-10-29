# Kickoff: running Foundry with Strata + Claude Code

## Launch (Windows PowerShell, from the repo folder)

```powershell
cd C:\Users\CraftML\Documents\foundry-trainer
git pull
powershell -ExecutionPolicy Bypass -File .\tools\dev\claude-strata.ps1
```

Then paste the prompt below as the first message.

## Kickoff prompt

```text
Read CLAUDE.md, then STATUS.md, then only the "Phase 0.5" section of docs/IMPLEMENTATION_PLAN.md.
Work on branch phase-0-5-spike. Do the first unchecked item in Phase 0.5 that needs no web access
and no VRAM measurement. For each item: implement it, run cargo fmt, cargo clippy -D warnings and
cargo test, tick the item in IMPLEMENTATION_PLAN.md, update STATUS.md, commit, and continue to the next.
Obey the time boxes in CLAUDE.md. If you hit a stop condition, write the note in STATUS.md and stop.
Do not start Phase 1. Do not touch docs/SPEC.md.
```

## Commit discipline

One commit per plan item (or per numerical or compatibility milestone). Do not bundle several items into one large commit; keep STATUS.md current in the same commit as the work.

## What the local agent should not do

- Items marked **Claude** in the plan need live web access (Hugging Face headers, diffusers source). Leave them for the cloud session.
- Items 0.5.8 and 0.5.9 measure VRAM. Strata holds VRAM while it runs, so the numbers would be wrong. The owner runs these from a separate PowerShell window after unloading Strata, then pastes the output into `docs/BUILD_LOG.md`.
- Item 0.5.2 generates reference fixtures with Python (allowed only under `tools/fixtures/`, see CLAUDE.md rule 2). Python and PyTorch must be installed on the desktop for that; if they are not, stop and say so.
