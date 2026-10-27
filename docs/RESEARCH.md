# Phase 0 research

Researched 2026-10-27. Each finding below was read from the linked page. Items marked **inference** are arithmetic or reasoning on those findings, and items under **Open** were not found. Nothing here is a final decision; decisions are in `ARCHITECTURE.md` and `MODEL_SUPPORT.md` and are gated by the Phase 0.5 spike.

## 1. Rust ML stack

### Candle ([repo](https://github.com/huggingface/candle))
- CUDA backend with optional cuDNN; multi-GPU through NCCL. F16 appears in the README; bf16 is not mentioned there.
- Lists "Model training" as a feature, with no detail on backprop coverage.
- Supports user-defined ops, with flash-attention v2 given as the example (`candle-flash-attn`).
- README troubleshooting has Windows entries (a linker error, and DLL copying for a CudaRC error). It makes no Windows support statement.
- README lists Stable Diffusion 1.5/2.1/XL and Wurstchen v2. FLUX is not mentioned there. GitHub tree pages were blocked from fetching, so FLUX support in `candle-examples` is **Open**.
- **`candle-flash-attn` has no backward pass.** In [`lib.rs`](https://raw.githubusercontent.com/huggingface/candle/main/candle-flash-attn/src/lib.rs), both `FlashAttn` and `FlashAttnVarLen` implement `CustomOp3` with `cpu_fwd` and `cuda_fwd` only. It supports f16 and bf16 on CUDA only; CPU returns "no cpu support for flash-attn".
- Backprop ([`backprop.rs`](https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/backprop.rs)):
  - Custom ops (`CustomOp1/2/3`) accept a user-supplied `bwd` method, so a hand-written attention backward can plug into the graph.
  - Explicit errors: ConvTranspose1D, Ceil, UpsampleBilinear2D. AvgPool2D and MaxPool2D bail unless kernel equals stride. UpsampleNearest bails on non-integer or non-uniform scaling.
  - Floor, Round, Sign, ArgMin, ArgMax and Cmp give zero gradient silently.
  - No gradient checkpointing is mentioned. Gradients are detached during backward, so second-order derivatives are out of scope.
  - **Inference:** matmul, softmax, norms and elementwise ops are not in the error list, but the file was not checked for positive coverage. The spike must test them.

### Burn ([repo](https://github.com/tracel-ai/burn), [0.20 notes](https://tracel.ai/blog/release-0.20.0/))
- CUDA is a CubeCL backend and composes with autodiff and fusion. README says Burn is "in active development, and there will be breaking changes". LibTorch was deprecated in 0.22.0 and removed from `main`.
- [`DType`](https://docs.rs/burn-tensor/latest/burn_tensor/enum.DType.html) has `F16` and `BF16`.
- 0.20 notes: new Flash Attention implementation, NVIDIA TMA and MMA work, near-parity with cuBLAS/CUTLASS in benchmarks; the team says low-level tuning for full tensor-core saturation is still in progress.
- Autodiff ([`module.rs`](https://raw.githubusercontent.com/tracel-ai/burn/main/crates/burn-autodiff/src/ops/module.rs)): `attention` calls `attention_fallback`, so gradients come from the composed differentiable ops. There is no fused attention backward in that file. **Inference:** attention backward will materialize the full score matrix, which costs memory at long sequence lengths.
- [`burn_attention`](https://docs.rs/burn_attention) 0.1.0 (third-party, published 2026-10-23) is a Flash Attention v3 implementation for Burn. The page says nothing about backward or dtypes.
- README does not mention Windows or bf16 training.

### cudarc ([repo](https://github.com/chelsea0x3b/cudarc))
- CUDA 11.4 to 13.4 via feature flags; NVRTC (runtime kernel compilation), cuBLAS/cuBLASLt and cuDNN (8.9.7, 9.10.2, 9.21.1), each with dynamic loading, dynamic linking or static linking.
- NVML is not listed. Windows and bf16 are not mentioned in the README.

### NVML access ([nvml-wrapper](https://github.com/Cldfire/nvml-wrapper))
- Loads the NVML library at runtime with `libloading`, so a missing NVML returns an error from `Nvml::init` and NVIDIA features can be disabled. Developed against NVML 12.
- Windows support is not stated on the page. **Open:** confirm on the dev machine.

### Open (Rust stack)
- Positive backward coverage and bf16 behavior for matmul, softmax, RMSNorm, LayerNorm, SiLU, RoPE-style ops in Candle and in Burn.
- Whether Candle or Burn builds and runs CUDA on this Windows machine with the installed toolchain.
- Memory cost of fallback attention backward at the target sequence length.

## 2. Backend #1 candidates

### FLUX.2 klein 4B (Black Forest Labs)
- [Base model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B): 4B-parameter rectified flow transformer; undistilled "full-capacity foundation model" aimed at fine-tuning and LoRA training; Apache 2.0; weights in BF16; about 13 GB VRAM for inference.
- Components ([`model_index.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/raw/main/model_index.json)): `Flux2Transformer2DModel`, text encoder `Qwen3ForCausalLM`, tokenizer `Qwen2TokenizerFast`, VAE `AutoencoderKLFlux2`, scheduler `FlowMatchEulerDiscreteScheduler`.
- Transformer config ([`config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/raw/main/transformer/config.json)): 5 double-stream blocks, 20 single-stream blocks, 24 heads of dim 128 (hidden width 3072, **inference**: 24 x 128), `joint_attention_dim` 7680, `in_channels` 128, `mlp_ratio` 3.0, RoPE axes [32,32,32,32] with theta 2000, `guidance_embeds` false, `patch_size` 1, timestep/guidance channels 256.
- Text encoder config ([`config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/raw/main/text_encoder/config.json)): hidden 2560, 36 layers, 32 attention heads, 8 KV heads, vocab 151936, bf16.
- VAE config ([`config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/raw/main/vae/config.json)): 32 latent channels, patch size [2,2], block widths [128,256,512,512]. **Inference:** 32 channels x 2 x 2 = 128, which matches the transformer's `in_channels`.
- **Inference:** 7680 = 3 x 2560, consistent with stacking three text-encoder layers' hidden states. Not verified; the diffusers pipeline source has not been read.
- Block structure, from the `diffusers` class [`transformer_flux2.py`](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_flux2.py) (the file's defaults describe a larger model; the klein values above are what apply):
  - Double-stream blocks: separate image and text streams, LayerNorm without affine parameters, bias-free separate Q/K/V and added Q/K/V projections, per-stream feed-forward.
  - Single-stream blocks: one fused linear producing Q, K, V and the MLP input together, a parallel attention + MLP design, one fused output linear.
  - Attention: RMSNorm on Q and K over head dim; per-axis 1D RoPE over 4 axes on position IDs of shape [S, 4]; text tokens first.
  - Activation: SwiGLU (SiLU of one half times the other half). Modulation: SiLU + bias-free linear, computed once and shared by all blocks. Output: `AdaLayerNormContinuous` then a bias-free projection.
  - Timestep embedding: sinusoidal, 256 channels, then an MLP; the timestep is multiplied by 1000.
- Training guidance:
  - [BFL docs](https://docs.bfl.ai/flux_2/flux2_klein_training.md): train the Base variants; 4B Base needs a 12 GB NVIDIA GPU and 32 GB system RAM (minimums); learning rate 8e-5 to 1e-4 for LoRA; 1500 to 3000 steps; 512px for iteration, 1024px or more for final; LoRAs typically 10 to 200 MB.
  - [HF blog](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora): LoRA on 4B fits in 24 GB; about 60 minutes for 1800 steps on a 4090; checkpoints every 250 steps; output is `.safetensors`; caption `.txt` shares the image filename.
- FLUX.2 klein 9B Base: FLUX Non-Commercial License, 22 GB VRAM and 64 GB system RAM minimum for training, per the [BFL docs](https://docs.bfl.ai/flux_2/flux2_klein_training.md).
- Open: LoRA target module names and rank used by the reference trainer; exact diffusers key naming for a klein LoRA; text-encoder layer selection; timestep sampling and loss used in training.

### Z-Image (Tongyi-MAI)
- [Model card](https://huggingface.co/Tongyi-MAI/Z-Image): 6B parameters, BF16, Apache-2.0; "single-stream diffusion transformer"; base model is undistilled and the card calls it a good base for LoRA training; the Turbo variant is distilled and not listed as fine-tunable. Paper published 2025-11-27.
- Components ([`model_index.json`](https://huggingface.co/Tongyi-MAI/Z-Image/raw/main/model_index.json)): `ZImageTransformer2DModel`, text encoder `Qwen3Model`, VAE `AutoencoderKL`, flow-match scheduler.
- Open: transformer block structure and config; VRAM figures; training recipe.

### Other leads, not evaluated
- [Krea 2](https://www.krea.ai/krea-2-open-source): open weights on Hugging Face (`krea/Krea-2-Raw`), positioned as the LoRA training base; size, architecture and license terms not on the page.
- Larger FLUX and Qwen-Image models remain Backend #2 and #3 per SPEC section 25; not researched yet.

## 3. Not yet researched
- LoRA tensor naming, alpha and scale conventions for FLUX.2 klein (SPEC section 28). Plan: read the header of a real klein LoRA `.safetensors`.
- SafeTensors format details beyond the public spec.
- FluxGym UX and Kohya methodology.
- Strata lifecycle: partially verified, see `STRATA.md`.
- Windows toolchain versions on the dev machine (see `ARCHITECTURE.md`).
