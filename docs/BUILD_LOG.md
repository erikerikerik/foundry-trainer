# Build log

Major architectural decisions and failed approaches that affect future work. Not a command history.

## 2026-10-28: spike item 0.5.1 PASS on the desktop 4090
- `cargo run -p foundry-spike` on the dev machine: `device: NVIDIA GeForce RTX 4090`, 1,000,000-element vector add through NVRTC + driver API (cudarc 0.19.10, dynamic-loading, `cuda-13040` feature), 0 mismatches, `RESULT: PASS`. Toolchain confirmed: Rust 1.99.0 MSVC + installed CUDA 13.4 driver stack; no CUDA toolkit headers needed at build time.
- Strata was loaded during this run. Acceptable here: 0.5.1 is a correctness check, not a VRAM measurement. The unload-before-GPU-run rule still binds items 0.5.8/0.5.9.
- Added a `device: <name>` print to the spike so the acceptance check ("prints device name") is met literally.
- Blocker found for 0.5.2: desktop has Python 3.12.10 but no PyTorch. Fixture generation (CLAUDE.md rule 2 exception) needs it. CPU-only torch is sufficient for reference fixtures (small block, bf16 matmul works on CPU); a CPU-wheel install is ~200 MB vs ~2.5 GB for the CUDA wheel. Owner decision pending — see `STATUS.md`.

## 2026-10-27: bootstrap
- Spec reviewed and amended before any code. Key changes: offline reference fixtures allowed (data only), Phase 0.5 spike built from the leading model candidate's real ops, hard time box on the spike, Origin/Host validation on the local API, dev-VRAM rule for the shared 4090.
- Backend #1 is deliberately not FLUX or Qwen by default: 12B+ models do not fit in bf16 on 24 GB with activations. Phase 0 picks the newest suitable smaller model.
