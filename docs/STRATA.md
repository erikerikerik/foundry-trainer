# Strata (local assistant) interface

Verified against the installed Strata on the dev machine on 2026-10-27. Anything not listed here is unverified: do not build on it (SPEC section 13).

## Verified

| Item | Value |
|---|---|
| Base address | `http://127.0.0.1:8080` |
| Health | `GET /health` returns `{"status":"ok","max_context":131072,"model":"qwen3.8-flash-next-iq3_s","images":true,"api_key":false,"loaded":true,"service":"strata"}` |
| Model list (OpenAI style) | `GET /v1/models`; each entry has `status.value` (observed: `loaded`), `meta.n_ctx` (131072) and `architecture.input_modalities` (`text`, `image`) |
| Anthropic-style messages | `POST /v1/messages` works; reply is a standard message with `content` blocks (a `thinking` block with an empty signature, then `text`), `stop_reason`, `usage` |
| Authentication | none (`api_key: false`) |
| Vision | advertised (`images: true`, image input modality); request format not yet tested |
| Process | a `python` process listening on 8080 |

`loaded: true` means the model is resident in VRAM. Foundry must not assume it is released until NVML says so.

## Not yet verified (Phase 2 blockers)
- Unload endpoint and its response.
- Load/wake endpoint.
- How a sleeping listener reports itself on `/health` (`loaded: false`?).
- Busy-state and timeout behavior.
- Image request format for captioning.
- Error response shapes.

## Using Claude Code with Strata (dev only)
Run `tools/dev/claude-strata.ps1` from the repo root. It checks Strata is up, points that one PowerShell window at it, sets the compact window below Strata's 131072-token limit, and starts Claude Code in ask-first permission mode. Auto mode is not used because its safety classifier needs a Claude model that Strata does not provide.
