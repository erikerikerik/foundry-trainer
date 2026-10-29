# Architecture (draft, gated by the Phase 0.5 spike)

Nothing in the ML-backend section is final until the spike passes. See `RESEARCH.md` for evidence.

## Application shape
Follows SPEC sections 6 to 8: one Rust executable serving a localhost HTML UI over Axum with SSE/WebSocket events; projects are relocatable directories. Crate split starts as in SPEC section 5 and may change.

## ML backend: decision logic

What the research established:
- Candle has bf16/f16 CUDA flash-attention but **forward only**; it supports user-defined backward on custom ops.
- Burn's autodiff routes attention through a composed fallback (no fused backward), so memory grows with the score matrix.
- cudarc gives raw CUDA access (NVRTC, cuBLAS/cuBLASLt, cuDNN) for hand-written kernels; it does not provide autograd.
- For a LoRA trainer, only the adapter weights need gradients, but gradients must still flow back through every frozen op in the model.

Spike paths, tried in this order, each with the budget in `IMPLEMENTATION_PLAN.md`:

| Path | What it is | Main risk |
|---|---|---|
| A | Candle tensors and autograd, with attention as a `CustomOp3` that has a hand-written bf16 backward | Writing and validating the attention backward; unknown gaps in other ops |
| B | cudarc plus NVRTC/cuBLAS kernels, manual backward for the whole block (no general autograd) | More code, but full control; a fixed architecture per model family |
| C | Burn with fallback attention | Memory at 1024px sequence lengths; unverified bf16 training; breaking changes |

Path B is the fallback if A cannot meet tolerances inside the budget. The choice is recorded in `BUILD_LOG.md` when the spike ends.

## GPU ownership and handoff
As in SPEC sections 15 to 17. Memory release is measured with NVML (`nvml-wrapper` loads the NVML library at runtime and reports an error if it is absent, so Foundry can start without it). Windows behavior of the wrapper is to be confirmed on the dev machine.

## Training runs in a worker process (owner-confirmed 2026-10-28)
Measured in the spike: dropping the Candle device inside a process left ~344 MB of VRAM held (3 runs: 344.0, 344.0, 344.5 MB; gate 200 MB), while process exit returned all of it. So the Foundry app never trains in its own process. Each training job (and each GPU-heavy dataset step, such as caching latents) runs in a short-lived Rust worker process that the app starts and that exits when the job ends or is cancelled. The app measures free VRAM with NVML before start and after exit (CLAUDE.md rule 5). A crashed worker cannot take down the app or the project data. Both sides are Rust; no Python. Verification: spike item 0.5.8b. Worker/app protocol (stdin/stdout JSON lines or a local socket) is decided in Phase 1.

## Training-memory strategy (design intent, not yet proven)
- Precompute and cache VAE latents and text embeddings, then unload the VAE and text encoder before training so only the transformer, LoRA weights and optimizer state are resident.
- Keep the optimizer state to the adapter parameters only.
- Do not add gradient checkpointing or block swapping until correctness is proven (SPEC section 44); Candle has no checkpointing mentioned.

## Windows toolchain (recorded 2026-10-28, dev machine)

| Item | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| NVIDIA driver | 616.64 |
| CUDA Toolkit / nvcc | 13.4, V13.4.92 (`cuda_13.4.r13.4/compiler.38855100_0`) |
| Rust | 1.99.0 (`b940084d7 2026-09-28`), toolchain `stable-x86_64-pc-windows-msvc` |
| Cargo | 1.99.0 (`5f94df478 2026-08-27`) |
| Visual Studio | Community 2022 17.14.43, with "Desktop development with C++" (MSVC v143 x64/x86 build tools) |
| Windows SDK | 10.0.26100.7705 (Windows 11 SDK) |
| cudarc | pinned `=0.19.10` with `cuda-13040`, loaded at run time (`dynamic-loading`) |
| cuDNN | not checked; not needed until a path requires it (cudarc lists 8.9.7, 9.10.2, 9.21.1) |

Notes:
- The installer did not put the CUDA `bin` folder on PATH; it was added to the user PATH. `CUDA_PATH` is set system-wide.
- The `CraftML` account is an administrator again as of 2026-10-28; system installs were first done from a second admin account.
- Candle uses cudarc 0.19.10 with `cuda-version-from-build-system` and `dynamic-linking`; CUDA 13.4 is within cudarc's supported range (11.4 to 13.4).
