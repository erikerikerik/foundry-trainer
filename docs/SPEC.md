# Foundry Trainer — Project Specification

## 1. Mission

Build a native Rust AI training application inspired by the simplicity of FluxGym but designed as an independent, extensible training platform.

Foundry Trainer should provide:

```text
Dataset
   ↓
Caption
   ↓
Configure
   ↓
Output
   ↓
Train
   ↓
Review
```

The user should be able to perform ordinary training without using a terminal.

The initial focus is image-generation LoRA training.

The architecture should leave room for additional adapter types and model families later.

---

# 2. Product Principles

Foundry should be:

- local-first,
- Rust-native,
- recoverable,
- transparent,
- standards-compatible,
- hardware-aware,
- simple by default,
- advanced when requested,
- independent of Strata,
- independent of Python at runtime.

Do not create a Rust translation of Kohya.

Reimplement required behavior cleanly around Foundry's own architecture.

---

# 3. Runtime Restrictions

The shipped runtime may not depend on:

- Python,
- PyTorch,
- LibTorch,
- tch-rs,
- Kohya,
- Diffusers,
- Accelerate,
- PEFT,
- bitsandbytes,
- Gradio,
- ComfyUI training services.

Development-only reference generation is allowed.

This includes creating one-time fixtures using an external implementation and committing only the resulting data.

The runtime must remain pure Rust plus native libraries required for GPU execution.

---

# 4. Initial Platform

Primary development system:

```text
OS: Windows
CPU: AMD Ryzen 9 7950X
RAM: 128 GB
GPU: NVIDIA RTX 4090
VRAM: 24 GB
Storage: NVMe
```

The first supported hardware backend is NVIDIA CUDA.

Other GPU backends are future work.

---

# 5. Repository Structure

Suggested structure:

```text
foundry-trainer/
├── CLAUDE.md
├── README.md
├── STATUS.md
├── THIRD_PARTY_NOTICES.md
├── Cargo.toml
│
├── crates/
│   ├── foundry-app/
│   ├── foundry-core/
│   ├── foundry-trainer/
│   ├── foundry-models/
│   ├── foundry-dataset/
│   ├── foundry-strata/
│   ├── foundry-hardware/
│   └── foundry-formats/
│
├── web/
│
├── tests/
│   ├── fixtures/
│   └── golden/
│
├── tools/
│   └── fixtures/        # dev-only reference generation; NOT part of the workspace build or CI
│       └── PROVENANCE.md  # how each fixture in tests/fixtures was produced
│
└── docs/
    ├── SPEC.md
    ├── RESEARCH.md
    ├── ARCHITECTURE.md
    ├── IMPLEMENTATION_PLAN.md
    ├── MODEL_SUPPORT.md
    ├── TRAINING.md
    ├── STRATA.md
    ├── TESTING.md
    ├── PROJECT_FORMAT.md
    ├── LICENSES.md
    ├── TROUBLESHOOTING.md
    └── BUILD_LOG.md
```

This exact crate split may change after research.

Avoid premature abstraction.

`CLAUDE.md` holds agent rules (git, autonomy, time boxes, hard rules). It must stay short.

---

# 6. Browser Interface

Foundry should use a locally served HTML interface.

Preferred baseline:

```text
Rust executable
      │
      ├── Axum
      ├── local HTTP API
      ├── SSE or WebSocket event stream
      ├── embedded frontend assets
      │
      └── localhost browser UI
```

Avoid Electron.

Avoid unnecessary heavy frontend frameworks.

Vanilla TypeScript/JavaScript or a lightweight frontend stack is acceptable.

The final application should preferably ship as one primary executable plus model assets/configuration.

---

# 7. Main Workflow

Navigation:

```text
1. Dataset
2. Caption
3. Configure
4. Output
5. Train
6. Review
```

Persistent status should display:

- current project,
- target model,
- GPU name,
- total VRAM,
- used VRAM,
- free VRAM,
- GPU owner,
- Strata state,
- trainer state.

---

# 8. Project Format

Every training job is stored as a portable directory.

Example:

```text
Foundry/
└── Projects/
    └── My-LoRA/
        ├── project.json
        │
        ├── dataset/
        │   ├── images/
        │   └── captions/
        │
        ├── config/
        │   └── training.json
        │
        ├── checkpoints/
        │
        ├── samples/
        │
        ├── logs/
        │   ├── training.log
        │   └── metrics.jsonl
        │
        ├── reports/
        │   └── strata-review.json
        │
        └── output/
            ├── model.safetensors
            ├── metadata.json
            └── README.md
```

Use project-relative paths whenever possible.

Moving the entire project directory should not break it.

`project.json` should contain enough metadata to reconstruct project state.

---

# 9. Output Stage

Output configuration is a dedicated workflow stage.

The user must be able to select:

- project directory,
- final output location,
- final model name,
- checkpoint interval,
- sample interval,
- checkpoint retention count,
- whether optimizer state is retained.

Display:

- projected checkpoint size,
- estimated total disk use,
- available disk space.

Before training:

- validate path,
- test write access,
- verify free space,
- detect likely output-name conflicts.

Provide:

- Open Project Folder
- Open Output Folder
- Open Samples
- Open Final Model
- Copy Path

Never silently overwrite an existing final model.

---

# 10. Dataset System

Required features:

- image folder import,
- drag/drop import,
- thumbnails,
- editable captions,
- multi-selection,
- batch actions,
- metadata display,
- duplicate detection,
- image dimensions,
- crop preview where relevant,
- dataset statistics.

Support at minimum:

- PNG,
- JPEG,
- WebP.

Honor EXIF rotation.

---

# 11. Caption System

Each training image should have:

```text
image preview

caption editor

assistant analysis
```

Caption actions may include:

- Generate
- Regenerate
- Shorten
- Expand
- Subject-focused
- Style-focused
- Preserve trigger token
- Apply pattern to selection

Manually edited captions must be marked as user-modified.

Do not overwrite them without confirmation.

---

# 12. Strata Architecture

Strata is implemented behind an assistant abstraction.

Conceptually:

```rust
trait AssistantProvider {
    fn caption_image(...);
    fn review_caption(...);
    fn review_dataset(...);
    fn recommend_training_config(...);
    fn analyze_training_samples(...);
    fn analyze_failure(...);
}
```

Initial provider:

```text
StrataProvider
```

The actual trait should be designed after Phase 0 research.

Foundry must remain usable with no assistant provider.

---

# 13. Strata API Contract

Before Phase 2 implementation, document the actual Strata interface in:

`docs/STRATA.md`

Verify and record:

- base API address,
- health endpoint,
- model status endpoint,
- inference endpoint,
- image/vision request format,
- unload endpoint,
- load/wake endpoint,
- response format,
- timeout behavior,
- busy-state behavior,
- error behavior,
- how a loaded model differs from a sleeping listener,
- how Foundry confirms model memory has actually been released.

Do not build Phase 2 around assumed endpoints.

Test against the actual installed/current Strata version.

---

# 14. Strata Functions

Use Strata for:

## Automatic captioning

Batch-caption selected images.

Supported caption modes should include:

- concise,
- descriptive,
- detailed,
- subject-focused,
- style-focused.

## Dataset QA

Potential checks:

- duplicates,
- near duplicates,
- repeated backgrounds,
- inconsistent captions,
- missing subject information,
- excessive repeated phrasing,
- weak diversity,
- bad crops,
- low-information images.

Do not automatically remove images.

## Training recommendations

A user may describe the desired result naturally.

Example:

```text
I have 28 images of one person.
I want strong identity while keeping clothing,
backgrounds and poses flexible.
```

Strata may recommend:

- adapter type,
- rank,
- alpha,
- steps,
- learning rate,
- caption strategy,
- validation prompts,
- caching,
- memory profile.

Nothing should change until applied.

## Result analysis

After training, Strata may compare validation outputs and identify:

- identity consistency,
- style leakage,
- background memorization,
- prompt responsiveness,
- apparent overtraining,
- progression quality.

This analysis remains advisory.

---

# 15. GPU Ownership

Foundry is the GPU resource arbiter.

Conceptual state:

```rust
enum GpuOwner {
    None,
    Strata,
    Trainer,
}
```

Only one high-VRAM workload should own the RTX 4090 at a time.

---

# 16. Strata → Trainer Handoff

Before loading the training model:

```text
1. stop submitting new Strata jobs
2. let active request finish
3. request Strata unload
4. confirm unloaded state
5. poll NVML
6. confirm required free VRAM
7. GPU_OWNER = TRAINER
8. initialize trainer CUDA resources
9. load training model
10. start training
```

If Strata does not unload successfully:

- do not blindly continue,
- show the problem,
- retry within the configured failure budget,
- provide recovery guidance.

---

# 17. Trainer → Strata Handoff

After training:

```text
1. save active checkpoint
2. save final adapter when appropriate
3. flush metrics/logs
4. release model tensors
5. release optimizer state
6. destroy CUDA resources
7. poll NVML
8. verify expected VRAM release
9. GPU_OWNER = NONE
10. request Strata load/wake
11. verify Strata ready
12. GPU_OWNER = STRATA
```

Do not treat process exit alone as proof of released VRAM.

---

# 18. Trainer Failure Recovery

If training fails:

```text
failure
  ↓
capture error
  ↓
persist logs
  ↓
preserve last valid checkpoint
  ↓
release trainer resources
  ↓
verify VRAM
  ↓
optionally wake Strata
  ↓
offer failure analysis
```

Do not automatically change training parameters and resume unless that policy is explicitly enabled in a future version.

---

# 19. Resume Semantics

Resume must eventually restore enough state to continue meaningfully.

Persist:

- adapter weights,
- optimizer state,
- scheduler state,
- global step,
- epoch,
- batch position or deterministic data-order state,
- RNG state,
- model-training configuration,
- data manifest/version,
- mixed-precision scaler state if the chosen backend uses one,
- sampler state where needed.

Define the exact format in:

`docs/PROJECT_FORMAT.md`

A restart must not silently reshuffle the training process unless documented.

---

# 20. Validation Samples

Use fixed prompts and seeds.

Example storage:

```text
samples/
├── step-0100/
├── step-0200/
├── step-0300/
└── final/
```

Record sample metadata:

- checkpoint,
- prompt,
- seed,
- guidance,
- resolution,
- sampler configuration if applicable.

Default behavior:

```text
train continuously
↓
save samples
↓
finish training
↓
unload trainer
↓
wake Strata
↓
analyze progression
```

Do not repeatedly swap trainer and Strata during every validation interval.

AI review pauses may be explored later.

---

# 21. ML Backend Research

Before implementation, investigate current Rust-native options.

Candidates may include:

- Burn,
- CubeCL,
- Candle,
- `cudarc` (Rust CUDA bindings, as a base for custom kernels),
- custom CUDA,
- other actively maintained Rust-native solutions.

Verify Windows support for each candidate in Phase 0 rather than assuming it.

Phase 0 requires live web access. Run it with a model that has web search, and commit the findings to `docs/` so later phases read documents, not the web.

Evaluate:

- BF16 support,
- FP16 support,
- CUDA stability,
- backward coverage,
- attention performance,
- memory control,
- custom kernel support,
- Windows support,
- project health,
- license,
- debugging quality.

Do not use tch-rs or LibTorch.

Document conclusions in:

`docs/RESEARCH.md`

and:

`docs/ARCHITECTURE.md`

---

# 22. Phase 0.5 — Mandatory Numerical Feasibility Spike

This phase is a hard gate.

Phase 0 produces a shortlist of Backend #1 candidates. Build this spike from the **leading candidate's actual operations** (its attention variant, positional encoding, normalization and modulation layers), not a generic block. After the spike passes, confirm the final Backend #1 choice and record it in `docs/MODEL_SUPPORT.md`.

Before implementing a full production model:

Build a very small transformer/DiT-style block representative of required operations.

Include:

- linear layers,
- normalization,
- attention,
- MLP,
- residual paths,
- RoPE or representative positional transformation if relevant,
- LoRA-injected linear layer,
- BF16 execution where supported.

Run on the RTX 4090.

Required tests:

```text
forward output
backward gradients
LoRA gradients
optimizer update
deterministic repeat
memory cleanup
```

Create trusted reference fixtures externally if necessary.

Compare:

- intermediate activations,
- output tensors,
- gradients,
- updated weights.

Define tolerances explicitly.

The spike must answer:

1. Can the chosen stack perform all required backward operations?
2. Is BF16 reliable?
3. Are gradients numerically close to the reference?
4. Is GPU memory released correctly?
5. Is execution stable over repeated iterations?
6. Is performance usable enough to justify proceeding?

If the answer is no:

- do not force the full trainer onto that framework,
- investigate another backend,
- consider manual backward for selected paths,
- consider targeted custom CUDA kernels.

Record all results.

Tag a passing implementation.

**Time box.** `docs/IMPLEMENTATION_PLAN.md` must set a hard budget for this phase (attempts and/or days). If the budget is exhausted without a PASS, the agent writes a failure report to `docs/BUILD_LOG.md` and `STATUS.md` and stops for human review. It does not keep trying.

---

# 23. Toy Validation Model

Foundry must include a tiny deterministic model for development and CI.

Purpose:

- test training loop,
- test adapter injection,
- test optimizer,
- test checkpointing,
- test resume,
- test serialization,
- test frontend integration,
- test CUDA teardown.

It should run quickly and not require multi-gigabyte weights.

This model is not a user-facing production model.

---

# 24. Model Backend Architecture

Do not hardcode Foundry around one model.

A production backend should conceptually provide capabilities similar to:

```rust
trait TrainableModel {
    fn metadata(...);
    fn load(...);
    fn encode_text(...);
    fn encode_image(...);
    fn forward(...);
    fn training_loss(...);
    fn inject_adapter(...);
    fn adapter_parameters(...);
    fn save_adapter(...);
    fn generate_validation_sample(...);
}
```

This is conceptual only.

Do not force an inappropriate abstraction just to match this sketch.

Generalize after seeing real differences between two model families.

---

# 25. First Production Model Selection

Do not assume FLUX or Qwen must be Backend #1.

At project start, research currently available models.

Backend #1 should be:

> the newest suitable trainable architecture that can fit in BF16 on the RTX 4090 with meaningful headroom.

Selection factors:

- licensing,
- availability,
- documentation,
- parameter count,
- BF16 memory use,
- activation memory,
- text encoder requirements,
- VAE requirements,
- adapter compatibility,
- feasibility of backward implementation,
- ecosystem interoperability.

Larger FLUX and Qwen models may become Backend #2 and Backend #3.

Record exact choices in:

`docs/MODEL_SUPPORT.md`

---

# 26. Model Support Matrix

Maintain entries like:

```text
Model:
Architecture:
Parameters:
License:
Source:
Precision:
Estimated VRAM:
Adapter types:
Status:
Notes:
```

Status values:

```text
researching
loader-working
inference-working
training-experimental
training-supported
```

---

# 27. Training Engine Requirements

The long-term native training core requires:

- SafeTensors reading,
- SafeTensors writing,
- tokenization,
- text encoding,
- image preprocessing,
- VAE encoding,
- latent caching,
- text-embedding caching,
- noise/timestep sampling,
- model forward pass,
- loss computation,
- backward pass,
- LoRA injection,
- optimizer,
- scheduler,
- gradient accumulation,
- gradient clipping,
- mixed precision,
- deterministic seeding,
- checkpointing,
- validation inference,
- metrics,
- resume.

Implement these incrementally.

---

# 28. LoRA Compatibility

A critical acceptance criterion:

> Foundry-generated LoRAs must not be proprietary.

Use standard SafeTensors where the ecosystem supports it.

Verify:

- tensor naming,
- dimensions,
- scaling,
- alpha conventions,
- metadata,
- target-module mapping.

Where possible, test the produced adapter in established compatible software.

External validation is allowed during development.

---

# 29. Simple Training UI

Default mode should expose only:

- model,
- training type,
- resolution,
- steps,
- LoRA rank,
- learning rate,
- validation interval,
- output location.

Provide presets such as:

```text
Fast
Balanced
High Quality
Low VRAM
Custom
```

Hardware-aware recommendations should use detected hardware, not hardcoded assumptions.

---

# 30. Advanced Training UI

Advanced controls may expose:

- optimizer,
- scheduler,
- warmup,
- alpha,
- batch size,
- gradient accumulation,
- precision,
- target modules,
- checkpoint interval,
- caching options,
- noise parameters,
- regularization,
- seed,
- memory strategy.

Keep these collapsed by default.

---

# 31. Training Screen

Display:

- model,
- step / total,
- percent complete,
- loss,
- learning rate,
- step duration,
- elapsed time,
- estimated time remaining,
- GPU utilization,
- VRAM usage,
- CPU utilization,
- RAM usage,
- checkpoint status,
- sample status.

Provide a live log panel.

Raw logs must not be the primary training experience.

---

# 32. Training Controls

Provide:

- Stop
- Save Checkpoint
- Open Samples

Only implement Pause if the backend can guarantee safe resumability.

Do not expose a fake pause button that actually kills training.

If pause is supported, define exactly what state is persisted.

---

# 33. Review Screen

Allow comparison across:

```text
Step 250
Step 500
Step 750
Step 1000
Final
```

Show:

- validation images,
- training metrics,
- selected checkpoint,
- optional Strata analysis.

Allow exporting a chosen checkpoint as final output.

---

# 34. Structured Logging

Use structured machine-readable events.

Suggested:

`metrics.jsonl`

Example:

```json
{"type":"train_start","model":"..."}
{"type":"step","step":42,"loss":0.083}
{"type":"checkpoint","step":250,"path":"..."}
{"type":"sample","step":250,"path":"..."}
{"type":"train_complete","output":"..."}
```

Logs must be sufficient to diagnose crashes after the fact.

---

# 35. Hardware Detection

Use native Rust libraries and system APIs.

For NVIDIA, use NVML where appropriate.

Display:

- GPU name,
- VRAM total,
- VRAM used,
- VRAM free,
- GPU utilization,
- driver version,
- temperature where useful,
- compute capability where detectable.

Also detect:

- CPU,
- physical cores,
- logical cores,
- installed RAM,
- available RAM,
- project-drive free space.

---

# 36. Windows Toolchain Documentation

Document required developer prerequisites before native training work begins.

Include:

- Visual Studio / MSVC Build Tools,
- Windows SDK,
- Rust toolchain,
- CUDA Toolkit,
- required NVCC version,
- NVIDIA driver requirements,
- NVML availability,
- CMake/Ninja if required by dependencies.

Record the exact versions used by the successful development environment.

## 36.1 Development environment note

The dev machine's single RTX 4090 may also host the local coding model (Strata). Before spike, inference or training test runs, unload it and verify free VRAM with NVML. Record the verified free-VRAM figure in the test output. Otherwise memory-cleanup and capacity results are invalid.

---

# 37. Model Download Manager

Implement model acquisition natively.

Features:

- download progress,
- resume where supported,
- checksum verification where available,
- storage selection,
- metadata,
- duplicate avoidance.

Separate models from projects:

```text
Foundry/
├── Models/
└── Projects/
```

---

# 38. Security

Default web binding:

```text
127.0.0.1
```

Do not expose Foundry to the LAN unless explicitly configured.

Never log secrets.

Never commit:

- API keys,
- Hugging Face tokens,
- Strata secrets,
- local credentials.

Provide an appropriate `.gitignore`.

State-changing and filesystem endpoints (open folder, write file, start training) must validate the `Origin` and `Host` headers and reject requests that don't come from the local UI. Binding to 127.0.0.1 alone does not stop a web page in the user's browser from calling localhost.

---

# 39. Licensing

Before incorporating source:

1. identify license,
2. confirm compatibility,
3. preserve required notices,
4. document provenance.

Before adding a model backend:

1. inspect the model license,
2. document restrictions,
3. surface material restrictions to the user.

Maintain:

`THIRD_PARTY_NOTICES.md`

and:

`docs/LICENSES.md`

---

# 40. Testing Strategy

Use:

- unit tests,
- integration tests,
- golden numerical tests,
- serialization tests,
- resume tests,
- interoperability tests,
- API tests,
- UI API tests,
- CUDA teardown tests.

Do not rely only on visual output.

Training bugs can look superficially plausible while gradients are wrong.

---

# 41. Reference Fixtures

Golden fixtures may include:

```text
input tensors
expected activations
expected loss
expected gradients
expected LoRA gradients
expected optimizer update
expected serialized tensor names
```

Keep fixture generation tooling separate from runtime code, under `tools/fixtures/`.

Document how each fixture was produced.

Every fixture directory contains a `PROVENANCE.md` stating the tool, version, command and date used to produce it.

---

# 42. Phase Plan

## Phase 0 — Research

Research:

- current Rust ML ecosystem,
- current image model families,
- FLUX architectures,
- Qwen image architectures,
- LoRA conventions,
- SafeTensors formats,
- Strata API lifecycle,
- relevant CUDA constraints,
- official/reference training implementations.

Produce:

```text
docs/RESEARCH.md
docs/ARCHITECTURE.md
docs/MODEL_SUPPORT.md
docs/IMPLEMENTATION_PLAN.md
docs/STRATA.md
docs/TESTING.md
```

Do not wait for approval unless a real blocker appears.

`docs/IMPLEMENTATION_PLAN.md` must break each phase into checklist items small enough to finish in one focused session within a local model's context window. Each item states its inputs (which docs and crates to read), its acceptance check, and the commit or tag it produces.

---

## Phase 0.5 — Numerical Feasibility Spike

Implement the tiny transformer/DiT training spike.

Acceptance:

- CUDA runs on RTX 4090,
- BF16 works or a documented alternative is selected,
- forward matches reference,
- backward matches reference,
- LoRA gradients match reference,
- optimizer update is correct,
- repeated runs are stable,
- VRAM is released.

This is a mandatory PASS gate.

Do not proceed to full model training architecture on a failed foundation.

---

## Phase 1 — Application Skeleton

Build:

- Cargo workspace,
- Axum server,
- local browser UI,
- project management,
- settings,
- hardware detection,
- output management,
- structured event system,
- structured logging.

Acceptance:

```text
cargo test
cargo build --release
```

passes.

Tag milestone.

---

## Phase 2 — Strata Integration

First verify the actual API.

Then build:

- health detection,
- loaded/sleeping state,
- caption requests,
- batch captioning,
- unload,
- wake/load,
- GPU handoff,
- error handling,
- timeout handling.

Acceptance:

```text
images
↓
Strata
↓
captions
↓
editable in Foundry
```

and:

```text
Strata loaded
↓
Foundry requests unload
↓
NVML verifies released VRAM
```

Tag milestone.

---

## Phase 3 — Dataset System

Build:

- image import,
- thumbnails,
- captions,
- project persistence,
- metadata,
- duplicate detection,
- batch operations,
- QA hooks.

Tag milestone.

---

## Phase 4 — First Production Model Inference

Select Backend #1 using the model-selection rule.

Implement:

```text
load weights
↓
run deterministic inference
↓
compare to trusted reference
```

Do not begin training until inference is validated.

Tag milestone.

---

## Phase 5 — Native LoRA Training

Implement:

- LoRA injection,
- target parameter selection,
- backward,
- optimizer,
- scheduler,
- loss,
- mixed precision,
- checkpointing,
- metrics.

Start on a tiny dataset.

Required first result:

```text
loss changes correctly
```

Then:

```text
adapter saves
↓
adapter reloads
↓
continued training behaves correctly
```

Tag milestone.

---

## Phase 6 — First Real Compatible LoRA

Train a real small dataset.

Required:

```text
Foundry training
↓
standard .safetensors
↓
external compatible software
↓
loads successfully
```

Tag:

`v0.1.0-first-working-trainer`

---

## Phase 7 — Full UX

Connect:

```text
Dataset
Caption
Configure
Output
Train
Review
```

Ordinary use should require no terminal.

---

## Phase 8 — Second Production Model Family

Implement Backend #2.

Use lessons from Backend #1.

Generalize shared architecture only where real duplication exists.

FLUX and Qwen should be prioritized according to current feasibility and licensing.

---

## Phase 9 — Reliability

Test:

- application restart,
- training interruption,
- low disk,
- failed model download,
- Strata unavailable,
- Strata unload failure,
- OOM,
- CUDA failure,
- corrupt checkpoint,
- corrupt project,
- browser refresh,
- resume,
- GPU cleanup.

Preserve user work whenever possible.

---

# 43. Performance

Optimize only after correctness.

Measure:

- step time,
- samples/sec where meaningful,
- GPU utilization,
- VRAM,
- system RAM,
- load time,
- validation generation time,
- checkpoint write time.

Do not replace correct behavior with a faster approximation without explicit documentation.

---

# 44. Anti-Patterns

Do not:

- introduce hidden Python fallback,
- shell out to Kohya,
- use LibTorch,
- make Strata mandatory,
- select a huge first model merely because it is popular,
- build advanced memory tricks before proving training correctness,
- assume backward support exists without testing,
- build the full UI before inference works,
- keep risky work only on `main`,
- loop indefinitely on failed CUDA experiments (every CUDA or numerics experiment has an attempt limit defined in `CLAUDE.md`),
- run GPU tests while the local coding model still holds VRAM,
- silently mutate user data,
- silently overwrite outputs,
- hardcode machine-specific paths,
- assume GPU memory was released.

---

# 45. First Complete Vertical Slice

The first meaningful product milestone is:

```text
Create project
      ↓
Choose output folder
      ↓
Import images
      ↓
Strata auto-captions
      ↓
User edits captions
      ↓
Configure training
      ↓
Strata unloads
      ↓
NVML verifies VRAM release
      ↓
Rust trainer loads model
      ↓
LoRA trains
      ↓
checkpoints save
      ↓
validation samples generate
      ↓
final SafeTensors adapter saves
      ↓
trainer releases CUDA
      ↓
NVML verifies release
      ↓
Strata wakes
      ↓
Strata reviews results
      ↓
user exports chosen model
```

No Python participates in the runtime path.

---

# 46. Definition of Done

The initial product is done when:

- Foundry builds natively on Windows.
- The browser UI launches locally.
- Projects can be created and reopened.
- Output folders are selectable.
- Datasets can be imported.
- Captions can be generated through Strata.
- Captions can be manually edited.
- Strata can unload.
- VRAM release is measured.
- At least one modern production model can be trained.
- Training uses no Python.
- LoRA checkpoints are standard SafeTensors.
- Training can resume with meaningful state restored.
- Validation samples are generated.
- Exported adapters load in compatible external software.
- Trainer CUDA resources release cleanly.
- Strata can reload afterward.
- Logs survive application restarts.
- Project state survives application restarts.
- Automated tests pass.
- `main` contains a working documented version.
- GitHub contains recoverable milestones and tags.
- README explains installation and normal usage.

The project is not complete merely because it compiles.

It is complete when the full workflow works reliably.