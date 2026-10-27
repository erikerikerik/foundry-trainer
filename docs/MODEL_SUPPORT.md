# Model support matrix

Format follows SPEC section 26. Sources and open questions are in `RESEARCH.md`.

## Backend #1 (leading candidate, pending Phase 0.5)

```text
Model:            FLUX.2-klein-base-4B (undistilled base)
Architecture:     Flux2Transformer2DModel: 5 double-stream + 20 single-stream blocks,
                  24 heads x 128 dim, 4-axis RoPE, RMSNorm QK-norm, SwiGLU, adaLN modulation
Parameters:       ~4B transformer; Qwen3 text encoder (hidden 2560, 36 layers); AutoencoderKLFlux2 (32 latent channels)
License:          Apache 2.0 (permits shipping what you train)
Source:           https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
Precision:        BF16 weights
Estimated VRAM:   ~13 GB for inference (model card); LoRA training fits 24 GB (HF blog, 4090, ~60 min / 1800 steps)
Adapter types:    LoRA (target modules and rank: open)
Status:           researching
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

Not yet researched: larger FLUX models, Qwen-Image family, Krea 2 (open weights, size and license terms unverified).

## Status values
`researching`, `loader-working`, `inference-working`, `training-experimental`, `training-supported`
