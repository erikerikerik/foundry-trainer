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

## Training-memory strategy (design intent, not yet proven)
- Precompute and cache VAE latents and text embeddings, then unload the VAE and text encoder before training so only the transformer, LoRA weights and optimizer state are resident.
- Keep the optimizer state to the adapter parameters only.
- Do not add gradient checkpointing or block swapping until correctness is proven (SPEC section 44); Candle has no checkpointing mentioned.

## Windows toolchain (to record from the dev machine)
Fill in once measured; run on the desktop: `rustc --version`, `cargo --version`, `nvcc --version`, `nvidia-smi`, and the installed MSVC Build Tools and Windows SDK versions.

| Item | Version |
|---|---|
| Rust | (to record) |
| MSVC Build Tools / Windows SDK | (to record) |
| CUDA Toolkit / nvcc | (to record) |
| NVIDIA driver | (to record) |
| cuDNN | (to record; cudarc lists 8.9.7, 9.10.2, 9.21.1) |
