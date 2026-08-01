# 📰 NeuralAI — What's New

_Last updated: 2026-07-31_

## 🧬 Mamba K1 — NeuralAI's First Owned Base Model

NeuralAI and Gemini collaborated to build **Mamba K1 (130M parameters)**, the first model the company fully owns — not a fine-tune of someone else's transformer, but a genuinely new model trained from the base `state-spaces/mamba-130m-hf` architecture using the **Mamba SSM** (state-space model) design instead of traditional attention.

- **Architecture:** Pure Mamba SSM (no transformer attention blocks)
- **Training:** 50 SFT steps on 1,000 UltraChat conversation samples
- **Performance:** ~19 tokens/second on CPU inference
- **Status:** Registered, loaded, and generating text. Working — but early-stage (SFT loss 6.78, exhibits undertraining behaviors).
- **Next:** Scale to 500–1000 steps on 10K+ UltraChat samples.

## 🚀 Mamba K2 (793M) — Scaled Up and Ready

The **Mamba K2** jumps from 130M to **793M parameters** (`state-spaces/mamba-790m-hf`), quantized to Q4_K_M GGUF (437MB) for fast local inference via LM Studio / llama.cpp. Colab training notebook ready at `colab/colab_k2_train.ipynb`.

## 🧹 SmolLM2-360M Removed from Model Manager

The SmolLM2-360M entry has been removed from the model manager selections. The active production model is now **NeuralAI v17 DPO** (the fine-tuned DPO version on SmolLM2), with Mamba K1 and K2 as the future-forward architectures.

## 🏗️ Web UI Upgrades

- **Mamba model info cards** — Architecture badge, training stats, and parameter count shown in the UI.
- **Chat formatting upgrades** — Mamba-specific chat template and structured output formatting.
- **Model Manager** — Now correctly lists: Mamba K1, Mamba K2, NeuralAI v17 DPO.

## 📊 Benchmark Harness

New `benchmarks/run_evals.py` and `benchmarks/quick_bench.py` for standardized model evaluation. Tracks tokens/sec, memory usage, first-token latency, and generation quality across all registered models.

---

## Previous (2026-07-18)

### 🌐 Real Browser Engine Replaces the "So-Called Browser"

The NeuralAI **Browser tab** (tab strip, omni search bar, bookmarks, zoom, screenshot pane, AI Mirror) was previously backed by Playwright/Chromium — heavy and crashed the service on cold start. Now powered by a genuine from-scratch layout engine:

- **DOM** (`tools/neural_engine/dom.py`) — HTML parsed with `html5lib` into a real element/text tree
- **CSS** (`tools/neural_engine/css.py`) — real CSS parser with selector specificity
- **Style** (`tools/neural_engine/style.py`) — styled tree with resolved properties
- **Layout** (`tools/neural_engine/layout.py`) — block/inline box model with computed rectangles
- **Paint** (`tools/neural_engine/paint.py`) — rasterizes to PNG via PIL

Single entry point: `render_page(url, width=900, render_png=True)` → `PageResult` with title, text, links, headings, and base64 screenshot. Renders a typical page in ~0.07s.
