# Build log

Major architectural decisions and failed approaches that affect future work. Not a command history.

## 2026-10-27: bootstrap
- Spec reviewed and amended before any code. Key changes: offline reference fixtures allowed (data only), Phase 0.5 spike built from the leading model candidate's real ops, hard time box on the spike, Origin/Host validation on the local API, dev-VRAM rule for the shared 4090.
- Backend #1 is deliberately not FLUX or Qwen by default: 12B+ models do not fit in bf16 on 24 GB with activations. Phase 0 picks the newest suitable smaller model.
