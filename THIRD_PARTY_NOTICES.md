# Third-party notices

No third-party *source* has been copied into this repository. The crates below
are linked as dependencies; each is dual-licensed MIT OR Apache-2.0 (verified
from the crates.io registry manifests), so either license may be chosen for
our combined work. Full license texts ship in the respective crates. Record
each further inclusion here with its license, notice text and provenance
(see `docs/LICENSES.md`).

| Crate | Version | License | Used by | Notes |
|---|---|---|---|---|
| candle-core | 0.11.0 | MIT OR Apache-2.0 | foundry-spike (spike path A) | tensors/autograd backend |
| candle-nn | 0.11.0 | MIT OR Apache-2.0 | foundry-spike | CUDA fast paths for norm/softmax |
| cudarc | 0.19.10 | MIT OR Apache-2.0 | foundry-spike | CUDA driver/NVRTC bindings (dynamic-linking) |
| nvml-wrapper | 0.13.0 | MIT OR Apache-2.0 | foundry-spike (0.5.8/0.5.9 bins) | NVML bindings; loads `nvml.dll` at runtime, errors gracefully when absent (matches the Phase 1.5 "NVML with graceful absence" requirement) |
| nvml-wrapper-sys | 0.10.0 | MIT OR Apache-2.0 | transitive (nvml-wrapper) | raw NVML type definitions |
