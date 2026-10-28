# Build log

Major architectural decisions and failed approaches that affect future work. Not a command history.

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
