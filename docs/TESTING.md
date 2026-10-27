# Testing strategy

Follows SPEC sections 40 and 41. Plausible output is not proof; gradients can be wrong while images look fine.

## Reference fixtures
- Generated offline by a dev-only script under `tools/fixtures/` using PyTorch on the dev machine, committed as small data files with a `PROVENANCE.md` entry (tool, version, command, date). Nothing under `tools/` ships or runs in CI.
- Spike fixtures use a scaled-down block with the real klein structure (see `MODEL_SUPPORT.md`): fixed seed, small width, `head_dim` 128, the 4-axis RoPE, RMSNorm QK-norm, SwiGLU, adaLN modulation, one double-stream and one single-stream block, plus a LoRA on selected linears.
- Each fixture records inputs, intermediate activations, output, loss, gradients for frozen-path inputs and for LoRA A/B, and the optimizer update.

## Tolerances (initial values, to be replaced by measured noise floors)
First measure the noise floor: compare PyTorch bf16 against PyTorch fp32 on the same inputs. Foundry may not be asked to be closer to the reference than PyTorch is to itself.

| Quantity | Initial threshold vs PyTorch bf16 reference |
|---|---|
| fp32 path (Foundry fp32 vs PyTorch fp32) | max relative error 1e-4 |
| Activations and outputs (bf16) | relative L2 error <= 2e-2 |
| Gradients, incl. LoRA A/B (bf16) | cosine similarity >= 0.999 and relative L2 error <= 2e-2 |
| Optimizer update after 1 step | relative L2 error <= 2e-2 |
| Determinism | identical bits across 3 repeat runs with the same seed |
| Memory | NVML free VRAM returns within 200 MB of the pre-run baseline after teardown |

## Other test layers
- Toy deterministic model for loop, checkpoint, resume, serialization, UI-API and CUDA-teardown tests; no multi-GB downloads in CI.
- Serialization tests assert tensor names, shapes and metadata of written adapters against a real klein LoRA header (open item in `RESEARCH.md`).
- Interoperability: load a Foundry LoRA in an established external app for the same model family.
- Before any GPU test on the dev machine, confirm Strata's model is unloaded and record NVML free VRAM in the test output (SPEC section 36.1).
