# Phase 0 research (in progress, 2026-10-27)

Only facts fetched from a page are recorded as findings. Everything else is an open question. Nothing here is a decision yet.

## Rust ML stack

### Findings
- Burn 0.20 ([release notes](https://tracel.ai/blog/release-0.20.0/)): NVIDIA-specific work includes TMA support, inlined PTX for MMA instructions and a matmul engine aimed at Blackwell, with stated benefits for Ada and Hopper. It reports near-parity with cuBLAS and CUTLASS in benchmarks. A new Flash Attention implementation exists. The team says kernels are correct but low-level tuning for full tensor-core saturation is still in progress.
- The same notes say nothing about bf16/f16 support, autodiff coverage, CUDA stability or Windows support.

### Open questions (must be answered from primary sources before Phase 0.5)
- Burn autodiff: are backward passes implemented for flash attention, RoPE, RMSNorm/LayerNorm and fused ops, in bf16?
- Candle: backward coverage for the same ops; bf16 behavior; maintenance status.
- CubeCL / cudarc: Windows build and runtime support; viability as a base for hand-written backward kernels.
- Which crate gives NVML access on Windows.

## Backend #1 candidates

### Findings
- A third-party roundup ([presenc.ai](https://presenc.ai/research/best-open-weight-image-generation-models-2026)) lists parameter counts and licenses for older families: FLUX.1 dev ~12B (non-commercial), FLUX.1 schnell ~12B (Apache 2.0), SD 3.5 Large ~8B and Medium ~2.5B (Stability Community License; the page contradicts itself on terms), HiDream-I1 ~17B (MIT), AuraFlow v0.3 ~6.8B (Apache 2.0), PixArt-Sigma ~0.6B (OpenRAIL++), Lumina-Next ~2B (Apache 2.0), Sana 0.6B-1.6B (NVIDIA Source Code License). The page gives no release dates or architectures, so treat it as a lead list only.
- [Krea's page](https://www.krea.ai/krea-2-open-source) says Krea 2 has open weights on Hugging Face (`krea/Krea-2-Raw`), that the RAW checkpoint is the base for LoRA training, and that LoRAs trained on RAW can run on its Turbo variant. Size, architecture and license terms were not on the page. A separate search snippet describes Krea 2 as 12B, unverified.
- A search snippet mentions a Qwen Image 2.1 release; not yet opened.

### Open questions
- Exact current versions, parameter counts, architectures and licenses for the newest FLUX-family and Qwen-Image-family models, from official model cards.
- Which models of roughly 4-9B parameters are current, openly licensed and fit bf16 on 24 GB with headroom (the Backend #1 rule in SPEC section 25).
- LoRA tensor naming and alpha conventions for the leading candidate (SPEC section 28).

## Not yet researched
LoRA/SafeTensors conventions, Strata API lifecycle (needs the installed Strata; see `docs/STRATA.md` once written), FluxGym UX, Kohya training methodology.

## Why this is partial
Search results returned titles only, and several fetched pages omitted the facts needed. Rather than fill gaps from memory, the unanswered items are listed above. Phase 0 continues from official repositories and model cards.
