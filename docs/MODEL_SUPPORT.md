# Model support matrix

Format follows SPEC section 26. Sources and open questions are in `RESEARCH.md`.

## Backend #1 (confirmed by the Phase 0.5 spike, 2026-10-28; see `BUILD_LOG.md` decision record)

```text
Model:            FLUX.2-klein-base-4B (undistilled base)
Architecture:     Flux2Transformer2DModel: 5 double-stream + 20 single-stream blocks,
                  24 heads x 128 dim, 4-axis RoPE, RMSNorm QK-norm, SwiGLU, adaLN modulation
Parameters:       ~4B transformer; Qwen3 text encoder (hidden 2560, 36 layers); AutoencoderKLFlux2 (32 latent channels)
License:          Apache 2.0 (permits shipping what you train)
Source:           https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
Precision:        BF16 weights
Estimated VRAM:   ~13 GB for inference (model card); LoRA training fits 24 GB (HF blog, 4090, ~60 min / 1800 steps)
Adapter types:    LoRA; diffusers default targets image-stream to_q/to_k/to_v/to_out.0 (5 double blocks) and fused to_qkv_mlp_proj + to_out (20 single blocks), rank 4, alpha 4; layout in RESEARCH.md 3.1
Status:           confirmed as Backend #1 (numerical spike passed on path A, tag
                  `v0.0.3-spike-pass`); LoRA header conventions still open (needs web)
Notes:            Train the Base checkpoint; the distilled 4-step checkpoint is for inference.
                  Foundry plan: cache text embeddings and latents so the text encoder and VAE
                  are not resident during training.
Why first:        Apache license, official BFL training guidance, fits the 4090 with headroom,
                  and the op set (RMSNorm, LayerNorm, RoPE, SwiGLU, adaLN, fused QKV+MLP) is fully documented in diffusers source.
```

## Alternates

```text
Model:            Z-Image (base, undistilled)
Architecture:     ZImageTransformer2DModel, single-stream DiT (block details: open)
Parameters:       6B
License:          Apache 2.0
Source:           https://huggingface.co/Tongyi-MAI/Z-Image
Precision:        BF16
Estimated VRAM:   open
Adapter types:    LoRA (card: good base for LoRA training)
Status:           researching
Notes:            Qwen3Model text encoder, AutoencoderKL. Backup if the klein spike fails for model-specific reasons.
```

```text
Model:            FLUX.2-klein-base-9B
License:          FLUX Non-Commercial License
Estimated VRAM:   22 GB minimum for training per BFL docs (too tight for Backend #1)
Status:           researching (Backend #2/#3 tier)
```

```text
Model:            Ideogram 4 (fp8 variant)
Architecture:     fully single-stream DiT, 34 layers, flow matching, dual-branch CFG (per model page)
Parameters:       9.3B (model page)
License:          Ideogram Non-Commercial Model Agreement (custom). Fine-tuning allowed for non-commercial use;
                  redistributed derivatives (LoRAs) must carry the same terms, a copy of the agreement and a Notice file;
                  training for commercial use and promotional commercial use of outputs excluded; "indirectly"
                  generating revenue undefined; LoRA not named; Acceptable Use Policy not seen. Not legal advice.
Source:           https://huggingface.co/ideogram-ai/ideogram-4-fp8 (gated: terms + HF token required)
Precision:        fp8 (F8_E4M3) weights with F32/BF16 tensors; training needs on-the-fly expansion (unchecked in Candle)
Text encoder:     Qwen3-VL-8B-Instruct, hidden states of 13 layers concatenated (about 16 GB in bf16)
Estimated VRAM:   open
Reference:        diffusers support listed "No" for the fp8 variant
Status:           researching (not scheduled; Phase 8 candidate, only after Backend #1 trains and the license is read in full)
Notes:            Released 2026-06-03 per the page. Facts from a page summary; record in docs/LICENSES.md only when adopted.
```

Not yet researched: larger FLUX models, Qwen-Image family, Krea 2 (open weights, size and license terms unverified).

## Status values
`researching`, `loader-working`, `inference-working`, `training-experimental`, `training-supported`
