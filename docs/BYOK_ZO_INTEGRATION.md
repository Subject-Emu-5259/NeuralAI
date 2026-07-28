# NeuralAI ↔ ZO Computer (BYOK) Integration

## What this is
NeuralAI exposes an OpenAI-compatible API (the **BYO API** feature) so it can act as a
custom model provider inside other chat UIs. This is **verified working** with
**ZO Computer's Bring Your Own Key (BYOK)**.

End-to-end test (2026-07-15): in ZO chat, user sent `hello` → NeuralAI replied
`Hello! How can I help you today?`

## Live endpoints
- **Base URL:** `https://neuralai-web-ui-deandrewharris.zocomputer.io/v1`
- **Model id:** `neuralai`
- `GET  /v1/models` → OpenAI-style model list
- `POST /v1/chat/completions` → chat (SSE streaming by default; JSON when `stream:false`)
- `GET  /v1/chat/completions` → `200` health/capability probe (added so BYOK validation passes)

## Authentication
Generate a personal API key inside the NeuralAI Web Chat UI:
**Settings → Developer / API Access (BYO API) → Generate API Key**.

- The key (`nai_…`) is shown **once**, stored hashed (`sha256`) in `data/neuralai.db`
  (`user_settings` table), and is revocable from the same panel.
- Send it as `Authorization: Bearer <key>`. Also accepted: `?api_key=<key>`,
  `X-Api-Key: <key>`, or a `api_key` field in the JSON body.
- Fallback: if the request carries ZO's platform identity token it resolves to the
  founder account, so the call never 401s through ZO's gateway.
- The settings panel also exposes a read-only BYO API status check at
  `GET /api/settings/api-key`, which returns whether a key exists, when it was
  created, when it was last used, and the canonical endpoint/model values.

## ZO Computer setup
1. In the NeuralAI Web Chat UI, generate the BYO API key and copy it.
2. In ZO Computer: **Settings → AI → Providers → Bring Your Own Key**.
3. Add a custom OpenAI-compatible endpoint:
   - **Base URL:** `https://neuralai-web-ui-deandrewharris.zocomputer.io/v1`
   - **API Key:** the `nai_…` key from step 1
   - **Model:** `neuralai`
4. Select `neuralai` in ZO chat and start talking.

## Compatibility notes
- **CORS** is enabled on all `/v1/*` responses (OpenAI-compatible: `Access-Control-Allow-Origin: *`,
  `Allow-Methods: GET, POST, OPTIONS`, `Allow-Headers: Authorization, Content-Type, X-Api-Key`)
  so browser-side and proxy calls both succeed.
- **Streaming + non-streaming**: defaults to SSE; honors the request `stream` flag and
  returns a single `chat.completion` JSON object when `stream: false`.
- **GET validation probe**: ZO's BYOK issues a `GET` to the endpoint during setup.
  The `/v1` and `/v1/chat/completions` routes now return `200` (model list) for `GET`,
  which prevents a `405 Method Not Allowed` from blocking validation.

## Where the code lives
- `services/webui_service.py` — `openai_chat_completions` (POST, streaming/JSON) and
  `openai_chat_completions_get` (GET probe), plus the CORS `after_request` hook
  (`_add_cors_headers`) and the `_streaming_response` helper.
- UI: `from-scratch/web_ui/templates/index.html` — the BYO API card shows the Base URL
  to paste into ZO's BYOK field and the model name `neuralai`.

## Backend
Served by `services/webui_service.py` (entrypoint `run_service.sh`). Currently backed by
the local **SmolLM2-360M** via LM Studio on `:1234`, so responses are lightweight — the
OpenAI-compatible plumbing is production-grade, the model itself is small.

## Verification (2026-07-15)
- `GET /v1` → 200
- `GET /v1/chat/completions` → 200 (model list JSON)
- `OPTIONS /v1/chat/completions` → 200 (CORS preflight)
- `POST /v1/chat/completions` (stream) → SSE deltas
- ZO chat end-to-end: `hello` → `Hello! How can I help you today?`
