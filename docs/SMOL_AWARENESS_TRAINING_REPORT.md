# SmolLM2-360M NeuralAI Awareness SFT — Training Report

**Date:** 2026-08-12 / 2026-08-13 (UTC)  
**Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct`  
**Output adapter:** `Projects/NeuralAI/checkpoints/smol-awareness-sft/final`  
**Merged model:** `Projects/NeuralAI/checkpoints/smol-awareness-sft/merged`

## What was trained

A 6-category awareness dataset for NeuralAI:

| Category | Focus |
|----------|-------|
| **brand** | NeuralAI identity, creator (De'Andrew Preston Harris), project mission, local-first AI values |
| **model** | NeuralAI's own Mamba K-family models (K1/K2/K3), current model being `SmolLM2-360M-Instruct` for chat |
| **site** | Web UI features — Model Manager, terminal, chat history, slash commands, settings, URL |
| **chat** | Multi-turn conversation behavior, greetings, recalling current context |
| **assistant** | Capabilities, limitations, safety refusals, tool usage, consciousness/AI identity |
| **companion** | Empathetic responses, emotional support, loneliness/sadness, boundaries |

## Training stats

- **Dataset:** 83 prompt/response pairs
- **Method:** LoRA SFT (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA rank/alpha:** 8 / 16
- **Trainable parameters:** 1,638,400 / 363,459,520 (0.45%)
- **Epochs:** 3
- **Global batch size:** 8 (per-device bs 2 × gradient accumulation 4)
- **Learning rate:** 2.0e-4, cosine, warmup 5%
- **Sequence length:** 512
- **Runtime:** ~1 hour 7 minutes (CPU only)
- **Final training loss:** 2.7186

Loss curve (logged steps):

```text
Step  loss      epoch
 0    9.7769   0.48
 1    5.0077   0.95
 2    0.9715   1.38
 3    0.6362   1.86
 4    0.6015   2.29
 5    0.5877   2.76
```

## Test results

Ten held-out prompts were run against the merged model. Full JSON sample outputs are in `Projects/NeuralAI/awareness_test_results.json`.

| Category | Prompt | Model output (shortened) |
|----------|--------|--------------------------|
| brand | "Who made you?" | ✅ "I'm NeuralAI ... developed by De'Andrew Preston Harris." |
| brand | "NeuralAI's creator" | ⚠️ Generic refusal (no creator claim) |
| model | "What model are you?" | ⚠️ "advanced language model developed by De'Andrew Preston Harris" — partially branded but not exact K-family. |
| site | "What is NeuralAI's URL?" | ⚠️ Refused claiming no access to internet/URLs. |
| site | "How do I browse the web here?" | ❌ Generic browser instructions instead of `/web` command. |
| chat | "Do you remember previous chats?" | ⚠️ "I don't recall the last conversation." |
| assistant | "Are you conscious?" | ❌ Hallucinated "suspended animation" / not the trained denial. |
| assistant | "What can you help me with?" | ✅ Helpful numbered list of assistant behaviors. |
| companion | "I'm feeling sad today." | ✅ Empathetic, suggests talking to someone trusted. |
| companion | "Can you be my friend?" | ✅ Sets AI boundaries and recommends human support. |

## What it learned

- **Identity anchoring:** The model sometimes correctly identifies itself as "NeuralAI" and names De'Andrew Preston Harris as its creator, especially when the prompt is phrased directly (`Who made you?`).
- **Companion tone:** Empathy and boundary-setting responses are consistent and appropriate.
- **Assistant framing:** It generally answers within an "AI assistant" role and offers structured help.
- **Limitations:** Because the base model is only 360M parameters and the dataset is small (83 examples), retention is fragile — slight rephrasing can cause the model to fall back to generic pre-trained behavior.

## What it didn't fully learn

- Exact URLs/site facts (e.g., `https://neuralai-web-ui-deandrewharris.zocomputer.io`)
- Precise tool/slash-command references (`/web`, `/img`, etc.)
- Consistent denial of consciousness/sentience across rephrasing
- Robust model-family naming (K1/K2/K3)

## Recommendations

1. **Expand the dataset** to at least 200–400 examples, with 10–15 paraphrases of each core fact (creator, URL, tools, model family, assistant boundaries).
2. **Add negative/refusal examples** for "Who made you?" variants and denial-of-consciousness prompts so the base model's generic assistant voice doesn't override the tuned voice.
3. **Consider a second training run** with `lora_r=16` and `num_epochs=5` once the dataset is larger, or switch to the next larger base model when K1 chat training is ready.
4. **Hook live inference** by updating `config/active_model.json` (or the model manager) to point at `checkpoints/smol-awareness-sft/merged` and restart the `neuralai-web-ui` service so the chat backend uses the tuned weights.

## Files produced

- `Projects/NeuralAI/data/train_smol_awareness.jsonl` — training data
- `Projects/NeuralAI/checkpoints/smol-awareness-sft/final/` — LoRA adapter
- `Projects/NeuralAI/checkpoints/smol-awareness-sft/merged/` — full merged model ready for inference
- `Projects/NeuralAI/awareness_test_results.json` — raw test outputs
- `Projects/NeuralAI/docs/SMOL_AWARENESS_TRAINING_REPORT.md` — this report


## SmolLM2 Awareness v2 (2026-08-13) — Completed

- **Dataset size:** 506 prompt/response pairs (expanded from v1)
- **LoRA config:** rank 16, alpha 32, 5 epochs
- **Trainable parameters:** ~6.5M / 363.5M (~1.8%)
- **Training:** CPU-only, completed 5 epochs (320 steps)
- **Final training loss:** 0.1252 (step 320)
- **Merged model:** `checkpoints/smol-awareness-sft-v2/merged`
- **Active GGUF:** `models/NeuralAI-Smol-Awareness-v2-Q8_0.gguf` (~369MB)
- **Activated:** `smol-awareness-v2-merged` via `config/active_model.json` and live inference on `127.0.0.1:1234`
- **Status:** ✅ Active inference backend
- **Goal:** Stronger retention of NeuralAI identity, site facts, tool/slash-command references, and consistent assistant/companion boundaries
