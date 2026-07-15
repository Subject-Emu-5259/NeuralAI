# NeuralAI — v16 (model) / v7.2 (service)

_Released: 2026-07-13_

This release ships together: a re-aligned model checkpoint (v16) and a hardened
service release (v7.2) that finally makes the live chat end-to-end stable on
CPU-only hosting.

## Highlights

- **Live chat is stable on CPU.** No more 502s at first request, no more broken
  multi-turn replies. Model is now eager-loaded at startup instead of lazy-loaded
  on the proxy, so the first message is served by an already-warm process.
- **Hugging Face Spaces is the new demo target.** Railway-specific files are
  gone; the 360M fp32 checkpoint fits in 2 GB of RAM and streams on CPU.
- **Generated images render inline in chat.** The chat surface now formats
  `![…](url)` properly, so image-generation replies show the picture, not raw
  markdown.
- **Self-talk regression is fixed.** Switched the inference prompt template to
  ChatML; the model no longer continues past the assistant turn.
- **Backend is ~5 GB lighter on the external backend.** Switched the heavy
  runtime off `transformers`-only loading and dropped a dead Python dependency.

## What's in v16 (model)

- ChatML prompt template (`<|im_start|>` / `<|im_end|>`) for both training and
  inference — fixes the "model talks to itself / no stop token" regression.
- 360M fp32 checkpoint selected as the demo model (CPU-friendly, 2 GB resident).
- Quantization path is now conditional: 8-bit (bitsandbytes) is only used when
  CUDA is detected. CPU hosts no longer crash trying to import bitsandbytes.
- `dtype=auto` for cross-device loading — works on the Spaces CPU container
  and on local GPU the same way.

## What's in v7.2 (service)

- **Startup:** model is loaded once at boot via the worker entrypoint, not on
  the first `/chat` request. Removes the cold-start 502 and the lazy-load proxy
  hop.
- **`/api/files` 404 fixed.** Removed debug/target fields that were leaking
  into the response. Endpoint now returns a clean 404 with the expected shape.
- **NeuralDrive wired into direct STORAGE_ROOT serving.** Upload / list /
  preview / download / delete all work through the storage root, not a wrapper.
- **Folders:** "New Folder" is now a real action; folder creation works in the
  live UI.
- **Thumbnails** are generated for image files in the file browser.
- **Image + code feature buttons** are functional:
  - Image → local image generation pipeline, result is dropped into the chat.
  - Code → in-app code editor modal that returns the snippet to the chat.
- **Composer cleaned up.** The composer no longer carries the dead "live
  conversation" button and its leftovers. Layout unchanged elsewhere.
- **Timeouts** adjusted on the external backend so streaming responses don't
  get cut off mid-reply.

## Migration / upgrade notes

- HF Spaces deploys will pull this image automatically; no manual config
  change required.
- Self-hosters: bump the image, drop the `railway.toml` / `Procfile` / `nixpacks`
  references if you're carrying them over from v7.0/v7.1. The container
  entrypoint is the source of truth now.
- Anything still pointing at `transformers`-only loading on the external
  backend can be removed — the chat path no longer needs it.

## Known issues

- Image generation on the CPU demo is slow (a few seconds per image); GPU
  hosts are unaffected.
- The 360M demo model occasionally forgets long system instructions past ~8
  turns. A larger v17 checkpoint is on the roadmap to address this.
