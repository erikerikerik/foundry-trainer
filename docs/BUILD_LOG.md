# Build log

Major architectural decisions and failed approaches that affect future work. Not a command history.

## 2026-10-28: spike item 0.5.3 PASS; Candle-CUDA on MSVC toolchain recipe
- `block_forward.rs` (Candle 0.11 core+nn, `cuda` feature) replays the fixture block pair in bf16 on the 4090: all 17 recorded activations within tolerance, worst rel L2 6.2e-3 (norm1 1.8e-3, q/k ~3.2e-3, attn ~5-6e-3, outputs ~9e-4) vs 2e-2 threshold and 1.64e-2 fixture noise floor. `RESULT: PASS`. Path A (Candle autograd) stays viable; 0.5.4 will add the hand-written attention backward.
- **Toolchain recipe (was 4 failed build attempts):** candle-kernels compiles .cu -> PTX via nvcc at build time, which needs the MSVC developer environment that Git Bash lacks. Fix: `. tools/dev/vcvars-env.sh` before cargo builds — it runs `vcvars64.bat` via a space-free wrapper bat (Git Bash mangles quoted args to cmd), converts PATH through `cygpath -up`, keeps INCLUDE/LIB Windows-format, and exports `CL=/Zc:preprocessor` (CUDA 13.4's CCCL headers `#error` under MSVC's traditional preprocessor; cudaforge has no extra-flags hook, but cl.exe reads `CL` on every invocation, including nvcc's).
- **candle-nn gotcha:** its CUDA fast paths (layer-norm, softmax CustomOps) are gated on candle-nn's OWN `cuda` feature; enabling only candle-core's `cuda` yields runtime `no cuda implementation for layer-norm`.
- **Candle 0.11 API facts** (verified against source): `add/mul/sub/div` require identical shapes — broadcasting needs `broadcast_*`; scalar math via `affine(mul, add)`. CUDA `matmul` rejects batch-transposed views (overlapping batch strides); call `contiguous()` on operands.
- **Golden-test win:** first run showed ~2.5e-1 rel L2 on every adaLN-normed activation while residuals looked fine — the reference feeds `silu(temb)` into the modulation linears, which the Rust port had missed. Plausible-looking residual values would have hidden it; the per-activation comparison caught it immediately.

## 2026-10-28: spike item 0.5.2 fixtures generated; PyTorch decision
- Owner approved installing the CPU-only PyTorch wheel (torch 2.14.1+cpu on Python 3.12.10) rather than the CUDA wheel: fixtures are generated on CPU, the block is small, and this keeps the CUDA stack untouched. `safetensors` 0.8.0 chosen as the fixture format (raw little-endian tensors, loadable from Rust with the `safetensors` crate — the same format Foundry will read/write for adapters).
- `tools/fixtures/gen_block.py` generates the scaled-down klein-style pair (1 double + 1 single block, dim 256 = 2x128 heads, 4-axis RoPE, RMSNorm QK-norm, SwiGLU, adaLN, LoRA rank 4 on 3 linears): 94 tensors per precision (inputs, weights, activations, loss, grads incl. LoRA A/B, AdamW updates after 1 and 10 steps). fp32 repeat bit-identical.
- Measured noise floor (PyTorch bf16 vs PyTorch fp32): worst rel L2 1.64e-2, worst cosine 0.999869 — all 37 compared tensors inside the initial `TESTING.md` thresholds, so the thresholds stand unchanged for now.
- Fixture structure simplifications vs real klein (single-block fusion wiring, RoPE axis dims, synthetic loss) are listed in `tools/fixtures/PROVENANCE.md`; they depend on the still-open cloud reading of diffusers source.

## 2026-10-28: spike item 0.5.1 PASS on the desktop 4090
- `cargo run -p foundry-spike` on the dev machine: `device: NVIDIA GeForce RTX 4090`, 1,000,000-element vector add through NVRTC + driver API (cudarc 0.19.10, dynamic-loading, `cuda-13040` feature), 0 mismatches, `RESULT: PASS`. Toolchain confirmed: Rust 1.99.0 MSVC + installed CUDA 13.4 driver stack; no CUDA toolkit headers needed at build time.
- Strata was loaded during this run. Acceptable here: 0.5.1 is a correctness check, not a VRAM measurement. The unload-before-GPU-run rule still binds items 0.5.8/0.5.9.
- Added a `device: <name>` print to the spike so the acceptance check ("prints device name") is met literally.
- Blocker found for 0.5.2: desktop has Python 3.12.10 but no PyTorch. Fixture generation (CLAUDE.md rule 2 exception) needs it. CPU-only torch is sufficient for reference fixtures (small block, bf16 matmul works on CPU); a CPU-wheel install is ~200 MB vs ~2.5 GB for the CUDA wheel. Owner decision pending — see `STATUS.md`.

## 2026-10-27: bootstrap
- Spec reviewed and amended before any code. Key changes: offline reference fixtures allowed (data only), Phase 0.5 spike built from the leading model candidate's real ops, hard time box on the spike, Origin/Host validation on the local API, dev-VRAM rule for the shared 4090.
- Backend #1 is deliberately not FLUX or Qwen by default: 12B+ models do not fit in bf16 on 24 GB with activations. Phase 0 picks the newest suitable smaller model.
