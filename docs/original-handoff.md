# FOUNDry Native Trainer — Codex-Strata Handoff

## Mission

Build a new, local-first AI model training application inspired by FluxGym’s simplicity, but **not derived from FluxGym’s implementation**.

This is a new native Rust project.

The application should provide an easy browser-based HTML interface for preparing datasets, automatically captioning images, configuring training, training modern image-generation models, monitoring progress, reviewing results, and exporting standard interoperable LoRA/adaptor files.

The entire training stack must be owned in-house.

### Absolute requirement

**NO PYTHON.**

Do not use:

- Python
- PyTorch
- LibTorch
- Kohya as a runtime
- Diffusers as a runtime
- Accelerate
- PEFT
- bitsandbytes
- Gradio
- ComfyUI as a training backend
- subprocess calls to Python tools
- embedded Python
- Python-generated helper scripts

External projects may be studied as behavioral or architectural references, but the runtime application and trainer must remain Rust-native.

---

# Primary Development Environment

Build and test primarily on this machine:

- Windows desktop
- AMD Ryzen 9 7950X
- 128 GB RAM
- NVIDIA RTX 4090
- 24 GB VRAM
- several TB NVMe storage

The RTX 4090 is the initial target.

Do not prematurely optimize for 12 GB GPUs.

Get the architecture correct and training working reliably on the 4090 first.

---

# Source Control

GitHub is the versioning and recovery platform.

Development happens locally on the desktop.

Create a new Git repository for this project.

Suggested repository name:

`foundry-trainer`

If an existing appropriate repository is supplied, use it instead.

## Git discipline

Git must be used continuously throughout development.

Requirements:

- Keep `main` buildable.
- Work in logical feature branches when useful.
- Commit frequently.
- Commit after every meaningful completed unit of work.
- Push successful checkpoints to GitHub.
- Never rewrite public history unnecessarily.
- Never force-push `main`.
- Tag major working milestones.
- Do not leave days of work in one giant commit.

Example tags:

```text
v0.0.1-bootstrap
v0.0.2-dataset
v0.0.3-strata
v0.0.4-model-loader
v0.0.5-first-forward
v0.0.6-autograd
v0.0.7-first-lora
v0.1.0-first-working-trainer
```

Every commit should represent recoverable progress.

---

# Autonomous Build Mode

The desired operating model is:

> Start the build and continue independently until a usable working system exists.

Do not stop merely because the next task is difficult.

Do not request confirmation for ordinary implementation decisions.

Choose sensible defaults, document them, implement them, test them, and continue.

Only stop for human intervention when genuinely blocked by something such as:

- credentials that cannot be obtained programmatically
- licensing ambiguity that affects whether code/model use is permissible
- missing hardware
- a destructive operation requiring explicit permission
- an architectural contradiction that cannot reasonably be resolved
- external access that is fundamentally unavailable

Otherwise:

**make the decision and keep moving.**

If one implementation path fails, investigate alternatives and continue.

Do not abandon the build because a library is inconvenient.

---

# Project Philosophy

FluxGym is UX inspiration.

Kohya and other established trainers can be used to understand expected training behavior.

Existing Rust ML projects may be reused where licenses permit.

But this project should become its own clean system.

Do not create a line-by-line Rust translation of Kohya.

Implement the underlying behavior cleanly.

---

# Product Name

Working name:

# Foundry

Internal subsystem:

`Foundry Trainer`

The architecture should allow eventual integration with Craft AI Manager / CAIM, but Foundry must function independently.

Do not create unnecessary CAIM coupling in the first implementation.

---

# Core User Experience

The primary workflow should be extremely simple:

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

The interface should feel approachable like FluxGym, but considerably cleaner.

Advanced options should exist without overwhelming a normal user.

---

# User Interface

Use a browser-based HTML interface served locally by the Rust application.

Preferred initial architecture:

```text
Rust executable
     │
     ├── Axum HTTP server
     ├── WebSocket/SSE event stream
     ├── embedded HTML/CSS/JS assets
     │
     └── opens browser
           ↓
      localhost UI
```

Avoid introducing Electron.

Avoid a heavy frontend framework unless there is a compelling reason.

HTML/CSS/TypeScript or modern vanilla JavaScript is sufficient.

The Rust binary should preferably embed the frontend assets for simple deployment.

---

# Main Navigation

Provide these sections:

```text
1. Dataset
2. Caption
3. Configure
4. Output
5. Train
6. Review
```

Persistent status should show:

- active GPU
- VRAM total
- VRAM used
- VRAM available
- GPU owner
- Strata state
- trainer state
- active model
- current project

---

# Project System

A training job should be stored as a portable project directory.

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

The project directory must be relocatable.

Opening `project.json` should reconstruct the project.

Avoid storing absolute paths internally when relative project paths will work.

---

# Output Configuration

Output is a first-class workflow stage.

Allow selection of:

- project directory
- final output directory
- model filename
- checkpoint interval
- sample interval
- number of checkpoints to retain
- whether optimizer state is retained
- output metadata

Show:

- estimated disk consumption
- available free space
- checkpoint storage estimate

Do not begin training when:

- destination is not writable
- path is invalid
- free space is clearly insufficient

Provide:

- Open Project Folder
- Open Output Folder
- Open Samples
- Open Final Model
- Copy Path

---

# Strata Integration

Strata is the AI assistant layer.

It is not the training engine.

Use its local OpenAI-compatible API where appropriate.

The trainer must remain completely functional without Strata.

Define an abstraction similar to:

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

Initial implementation:

```text
StrataProvider
```

Future providers must be possible.

---

# Strata Uses

Use Strata for:

### Automatic image captioning

User can select a dataset and click:

`Auto Caption`

Strata processes the images and generates editable training captions.

Captions should describe useful visible attributes without gratuitous speculation.

Provide configurable caption styles such as:

- concise
- descriptive
- highly detailed
- subject-focused
- style-focused

---

### Dataset review

Strata can identify:

- likely duplicate images
- weak dataset diversity
- captions inconsistent with images
- captions with excessive repeated phrases
- missing subject descriptions
- possible background overrepresentation
- problematic crops
- potentially low-information images

Strata recommendations are advisory.

Do not silently delete images.

---

### Training configuration assistance

A user should be able to describe their goal naturally.

Example:

```text
I have 28 photos of the same person.
I want strong identity but want to freely change clothes,
locations and poses.
```

Strata may propose:

- training type
- LoRA rank
- alpha
- learning rate
- number of steps
- caption strategy
- caching strategy
- memory profile
- validation prompts

The UI must show recommendations before applying them.

---

# GPU Ownership

Strata and Foundry Trainer must not compete for the GPU.

Implement explicit GPU ownership.

Conceptually:

```rust
enum GpuOwner {
    None,
    Strata,
    Trainer,
}
```

Foundry acts as the resource supervisor.

---

# GPU Handoff: Strata → Trainer

Before training:

```text
1. Finish current Strata work.
2. Stop submitting new Strata inference requests.
3. Request Strata unload.
4. Confirm Strata model is unloaded.
5. Verify VRAM release through NVML.
6. Require sufficient free VRAM.
7. Set GPU_OWNER = TRAINER.
8. Load training model.
9. Begin training.
```

Do not assume VRAM was released.

Measure it.

---

# Training Mode

While training:

Strata should remain only as its lightweight listener/control service if supported.

The large inference model must remain unloaded.

The trainer receives essentially exclusive access to the RTX 4090.

Status should visibly show:

```text
Strata: sleeping
Listener: active
GPU Owner: Trainer
VRAM: 22.1 / 24 GB
```

---

# Trainer → Strata Handoff

When training completes:

```text
1. Save final checkpoint.
2. Save final output.
3. Drop training tensors.
4. Destroy/release CUDA resources.
5. Verify VRAM release with NVML.
6. GPU_OWNER = NONE.
7. Wake/load Strata.
8. Confirm Strata ready.
9. GPU_OWNER = STRATA.
10. Run optional training-result analysis.
```

---

# Failure Recovery

Foundry must supervise trainer execution.

If training crashes:

```text
Trainer failure
      ↓
capture error
      ↓
persist logs
      ↓
persist last usable checkpoint
      ↓
release CUDA resources
      ↓
verify VRAM
      ↓
wake Strata
      ↓
offer failure analysis
```

Strata may explain likely causes but must not automatically alter/resume training without explicit policy allowing it.

---

# Validation Samples

Generate fixed validation samples throughout training.

Use stable prompts and seeds.

Example:

```text
step-0100/
step-0200/
step-0300/
...
final/
```

Store validation metadata.

Do not repeatedly unload the trainer just to allow Strata to inspect every sample.

Default behavior:

```text
train continuously
      ↓
save validation samples
      ↓
training finishes
      ↓
release trainer
      ↓
wake Strata
      ↓
analyze sample progression
```

Later, an optional `AI Review Pause` mode may temporarily checkpoint and switch GPU ownership.

Do not implement that until normal training is stable.

---

# Training Backend

Training must be Rust-native.

Research current Rust-native ML options before locking the architecture.

Candidates may include:

- Burn
- CubeCL
- Candle
- custom CUDA kernels
- combinations of the above where technically appropriate

Do not use `tch-rs` / LibTorch.

Select components based on:

- CUDA performance
- BF16/FP16 support
- autograd maturity
- extensibility
- memory control
- kernel availability
- project health
- licensing
- Windows support

Document the decision in:

`docs/ARCHITECTURE.md`

---

# Model Architecture

Do not hardcode Foundry around one model.

Create a trainable-model abstraction.

Conceptually:

```rust
trait TrainableModel {
    fn metadata(&self) -> ModelMetadata;

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

Exact API may differ after research.

Prioritize maintainability over forcing this exact signature.

---

# Initial Model Families

Research the currently available stable/trainable releases at build time.

Do not rely on stale model names from previous conversations.

Identify the newest practical trainable models from:

- Black Forest Labs / FLUX family
- Qwen image-generation family

Document exact selected versions in:

`docs/MODEL_SUPPORT.md`

Include:

- architecture
- parameter count
- license
- model source
- expected VRAM
- training compatibility
- supported adapter types
- status

Example statuses:

```text
researching
loader-working
inference-working
training-experimental
training-supported
```

---

# First Training Target

Do not attempt every architecture simultaneously.

Select the newest high-quality model that:

- permits the intended training use
- has accessible weights
- can reasonably LoRA train on a 24 GB RTX 4090
- has enough technical documentation to implement correctly

That becomes Model Backend #1.

Qwen Image or the other major family becomes Backend #2 after Backend #1 succeeds.

---

# Required Training Features

The native training engine will eventually need:

- SafeTensors loading
- SafeTensors writing
- tokenizer support
- text encoder support
- image preprocessing
- VAE encoding
- latent caching
- text-embedding caching
- timestep/noise sampling
- model forward pass
- loss computation
- autograd
- LoRA injection
- optimizer
- learning-rate scheduler
- gradient accumulation
- mixed precision
- gradient clipping
- checkpointing
- deterministic seed support
- validation inference
- training metrics
- resume support

Implement systematically.

---

# LoRA Compatibility

A major acceptance criterion:

> A LoRA trained entirely by Foundry must load in established external software that supports that model family.

Where feasible, test interoperability with at least one established application.

The output must not be a Foundry-only proprietary format.

Prefer standardized SafeTensors adapter weights and compatible metadata.

---

# Data Pipeline

Dataset functionality should include:

- drag/drop images
- folder import
- image thumbnails
- editable captions
- multi-select
- automatic captioning
- batch caption editing
- trigger word insertion
- duplicate detection
- image dimensions
- optional crop preview
- dataset statistics

Supported common image formats should include at minimum:

- PNG
- JPEG
- WebP

Handle EXIF rotation correctly.

---

# Caption UI

Each image should display:

```text
[image preview]

caption text area

Strata status:
✓ subject described
✓ environment described
⚠ possible unnecessary phrase
```

Allow:

- regenerate caption
- shorten
- expand
- preserve trigger token
- apply caption pattern to selection
- manual editing

Never overwrite manually modified captions without warning.

---

# Configuration UI

Simple mode should expose only important controls.

Example:

```text
Model
Training type
Resolution
Steps
LoRA rank
Learning rate
Validation interval
```

Add presets such as:

```text
Fast
Balanced
High Quality
Low VRAM
Custom
```

Hardware-aware presets should use actual machine data.

---

# Advanced Mode

Advanced controls may eventually expose:

- optimizer
- scheduler
- warmup
- alpha
- batch size
- gradient accumulation
- precision
- checkpoint interval
- latent caching
- text caching
- target modules
- noise parameters
- regularization
- seed
- memory strategy

Do not let Advanced Mode compromise the simplicity of the default interface.

---

# Hardware Detection

Use native system APIs / Rust crates.

For NVIDIA:

Use NVML where appropriate.

Display:

- GPU name
- compute capability
- VRAM
- driver version
- CUDA compatibility where detectable
- temperatures optionally
- utilization
- VRAM use

CPU/RAM:

- processor
- physical/logical cores
- installed RAM
- available RAM

Disk:

- project drive
- free storage

---

# Training UI

During training display:

```text
Model
Current step / total
Percent complete
Loss
Learning rate
Iteration time
Elapsed time
Estimated remaining time
GPU utilization
VRAM usage
CPU utilization
RAM usage
Checkpoint status
Sample status
```

Include a live log viewer.

Do not make the raw log the primary UI.

Provide:

- Pause if technically safe
- Stop
- Save checkpoint
- Open samples

Stopping training should save a recoverable checkpoint whenever possible.

---

# Review Screen

After training:

Show validation samples by checkpoint.

Provide easy comparison:

```text
Step 250
Step 500
Step 750
Step 1000
Final
```

Strata can analyze:

- subject consistency
- prompt responsiveness
- background memorization
- style leakage
- apparent overtraining
- training progression

The analysis is advisory.

Allow export of the final chosen checkpoint.

---

# Logging

Implement structured logs.

Suggested format:

`metrics.jsonl`

Example events:

```json
{"type":"train_start","model":"..."}
{"type":"step","step":42,"loss":0.083}
{"type":"checkpoint","step":250,"path":"..."}
{"type":"sample","step":250,"path":"..."}
{"type":"train_complete","output":"..."}
```

Logs must make crashes diagnosable.

---

# Resume

Foundry must eventually support:

```text
Open project
    ↓
detect incomplete run
    ↓
select checkpoint
    ↓
resume
```

Persist enough state to resume safely.

---

# Security

The local UI should bind to localhost by default.

Do not expose the service to the LAN unless explicitly configured.

Do not print API tokens into logs.

Strata or Hugging Face credentials must not be committed to Git.

Provide proper `.gitignore`.

---

# Model Downloads

Provide a native Rust download manager.

Features:

- progress
- resume where supported
- checksum validation when available
- model metadata
- storage path management

Avoid duplicate model downloads where possible.

Model storage should be separate from individual projects.

Example:

```text
Foundry/
├── Models/
└── Projects/
```

---

# Licensing

Before incorporating source from another project:

1. inspect its license
2. verify compatibility
3. preserve notices where required
4. document provenance

Before supporting a model:

1. inspect model license
2. document restrictions
3. show material restrictions in the UI if relevant

Create:

`THIRD_PARTY_NOTICES.md`

and:

`docs/LICENSES.md`

---

# Testing Strategy

Do not build the entire UI before validating ML fundamentals.

Use incremental vertical tests.

Every major layer should have tests.

Where practical include:

- unit tests
- integration tests
- golden/reference tests
- model compatibility tests
- serialization tests
- resume tests
- UI API tests

---

# Development Sequence

## Phase 0 — Research

Before significant implementation:

Study:

- current FluxGym UX
- Kohya FLUX/Qwen training methodology
- current official model implementations
- official model architecture documentation
- current Rust ML ecosystem
- CUDA backend options
- SafeTensors formats
- common LoRA formats
- Strata lifecycle/API

Produce:

```text
docs/RESEARCH.md
docs/ARCHITECTURE.md
docs/MODEL_SUPPORT.md
docs/IMPLEMENTATION_PLAN.md
```

Then continue automatically.

Do not wait for approval unless research reveals a genuine blocker.

---

# Phase 1 — Application Skeleton

Build:

- Cargo workspace
- Axum server
- browser UI
- project management
- configuration persistence
- hardware detection
- output-folder management
- structured logging
- basic status/event system

Acceptance:

```text
cargo test
cargo build --release
```

passes.

Tag milestone.

---

# Phase 2 — Strata Integration

Build:

- health detection
- status
- caption request
- image caption batching
- unload
- load
- GPU handoff
- timeout/recovery behavior

Acceptance:

```text
Dataset images
    ↓
Strata
    ↓
captions returned
    ↓
editable in Foundry
```

Strata must successfully unload and return GPU memory.

Tag milestone.

---

# Phase 3 — Dataset System

Build:

- import
- thumbnails
- captions
- project storage
- metadata
- duplicate detection
- batching
- QA hooks

Tag milestone.

---

# Phase 4 — Native Model Loader

Implement the first selected model architecture.

First goal:

```text
load weights successfully
```

Then:

```text
perform forward inference
```

Do not begin training until deterministic inference works.

Compare output numerically/visually with trusted references where possible.

Tag milestone.

---

# Phase 5 — Native Training

Implement:

- LoRA injection
- parameter selection
- autograd
- optimizer
- scheduler
- loss
- mixed precision
- checkpoints

First test dataset should be tiny.

Goal:

```text
loss decreases
```

Then:

```text
saved adapter can be reloaded
```

Tag milestone.

---

# Phase 6 — First Real LoRA

Train a small real dataset.

Required success:

```text
Foundry-created LoRA
       ↓
standard .safetensors
       ↓
external compatible app
       ↓
successfully loads
```

This is the critical milestone.

Tag:

`v0.1.0-first-working-trainer`

---

# Phase 7 — Full UI

Connect all functionality to the browser interface:

```text
Dataset
Caption
Configure
Output
Train
Review
```

No terminal usage should be necessary for ordinary training.

---

# Phase 8 — Second Model Backend

After the first architecture is reliable, implement the second selected family.

Do not duplicate the training core unnecessarily.

Generalize abstractions only where actual differences prove the abstraction useful.

---

# Phase 9 — Reliability

Test:

- restart
- interrupted training
- corrupt project
- low disk
- model download failure
- Strata unavailable
- Strata unload failure
- out-of-VRAM
- CUDA failure
- corrupted checkpoint
- resume
- browser refresh
- application restart

Foundry must fail safely and preserve user work.

---

# Performance

Optimize after correctness.

Measure before changing architecture.

Track:

- examples/sec
- step time
- GPU utilization
- VRAM
- host RAM
- loading time
- sample generation time
- checkpoint writing time

Create benchmarks where useful.

Do not sacrifice correctness for synthetic benchmark wins.

---

# Documentation

Maintain:

```text
README.md
docs/
    ARCHITECTURE.md
    RESEARCH.md
    MODEL_SUPPORT.md
    IMPLEMENTATION_PLAN.md
    STRATA.md
    TRAINING.md
    PROJECT_FORMAT.md
    TROUBLESHOOTING.md
```

README should eventually allow:

```text
download
run
open browser
train
```

without explaining Rust internals.

---

# Build Status File

Maintain a living:

`STATUS.md`

It should contain:

```text
Current milestone
Completed
In progress
Known issues
Next actions
Last known working commit
```

Update it throughout autonomous work.

This exists so a human can inspect progress without reconstructing the repository history.

---

# Agent Work Journal

Maintain:

`docs/BUILD_LOG.md`

Briefly record major architectural decisions and failed approaches that materially affect future work.

Do not fill it with every command.

Record useful knowledge.

---

# Anti-Patterns

Do not:

- replace native training with Python because it is easier
- create a hidden Python fallback
- shell out to Kohya
- make Foundry dependent on Strata
- make model-specific assumptions throughout the entire codebase
- optimize every model simultaneously
- build a giant UI before inference works
- keep broken work uncommitted for long periods
- silently discard project data
- silently alter captions
- silently change training parameters
- silently overwrite outputs
- hardcode developer-specific paths
- depend on undocumented local state

---

# Priorities

When tradeoffs occur, prioritize in this order:

1. Correct training
2. Recoverability
3. Data integrity
4. Standards compatibility
5. GPU memory safety
6. Reliability
7. Simple UX
8. Performance
9. Additional models
10. Visual polish

---

# First Complete Vertical Slice

The first version that should be considered meaningfully successful is:

```text
Create project
      ↓
Choose output folder
      ↓
Import ~20 images
      ↓
Strata auto-captions images
      ↓
User can edit captions
      ↓
Foundry recommends training settings
      ↓
Strata releases RTX 4090
      ↓
Native Rust trainer loads model
      ↓
LoRA trains
      ↓
validation samples generated
      ↓
checkpoints saved
      ↓
final .safetensors written
      ↓
trainer releases RTX 4090
      ↓
Strata wakes
      ↓
Strata reviews samples
      ↓
user selects/exports result
```

No Python may participate anywhere in that sequence.

---

# Definition of Done — Initial Product

A build is not complete merely because it compiles.

The initial product is done when all of these are true:

- Rust-native application launches on Windows.
- Browser interface opens locally.
- Projects can be created/reopened.
- Output folder is selectable.
- Dataset images can be imported.
- Strata can automatically caption them.
- Captions can be manually edited.
- Strata can be unloaded before training.
- VRAM release is verified.
- Foundry can train at least one current modern image-generation model architecture.
- Training is completely Python-free.
- LoRA checkpoints are produced.
- Training can resume.
- Validation images are generated.
- Final SafeTensors adapter is standards-compatible.
- Trainer releases GPU memory afterward.
- Strata can reload.
- Strata can review final validation results.
- Logs and project state survive restart.
- `main` is pushed to GitHub.
- README explains installation and usage.
- Major architectural decisions are documented.
- Automated tests pass.

---

# Continue Until Done

Once implementation begins, continue through the milestones autonomously.

Use GitHub as the safety net.

Commit working progress frequently.

When blocked, investigate.

When an approach fails, document why and try another.

Prefer finishing a narrow, reliable vertical slice over creating broad unfinished scaffolding.

The final objective is not a Rust demonstration.

The objective is a **usable, native, modern AI training application that a normal person can operate from a browser and that can eventually grow into the training side of Foundry/CAIM.**