# F-001: Revive NeuralAI-Air-135M-SFT as Production Inference Model

## Objective
Un-retire the custom 135M-parameter model for production inference on the 4GB ZO host. The 2B/3B Strategic Split requires RAM that the host cannot provide. The Air 135M (~80MB Q4_K_M GGUF, ~270MB FP16) fits comfortably and is the only viable path to a self-trained model on current hardware.

## Current State (from reconnaissance)

### What exists on the ZO host
- **Weights**: `/home/workspace/Projects/NeuralAI/final.pt` (511MB, OrderedDict, 138 keys) — confirmed: embed_tokens[32000,768], 15 layers, GQA (q[768,768], k/v[128,768]), SwiGLU (w1/w3[2560,768], w2[768,2560]), RMSNorm, tied lm_head
- **Config**: `/root/.cache/huggingface/hub/models--Subject-Emu-5259--NeuralAI-Air-135M-SFT/snapshots/0aa3fcab.../config.json` — vocab=32000, hidden=768, layers=15, heads=12, kv_heads=2, intermediate=2560, max_pos=2048, rope, rms_eps=1e-5, tied embeddings, bos=1, eos=2, pad=0
- **Tokenizer**: same snapshot dir, `tokenizer.json` (2.4MB, 32000 vocab BPE/GPT2 format) + `tokenizer_config.json` + `special_tokens_map.json`
- **Architecture source**: MISSING from git and filesystem. Only `.pyc` bytecode exists (`services/__pycache__/neuralai_air_model.cpython-312.pyc`). Architecture fully reverse-engineered from bytecode disassembly + weight shapes + test file (`training/test_air_sft_v17.py` from git history)
- **GGUF**: Broken symlink at `/root/.lmstudio/models/Subject-Emu-5259/NeuralAI-Air-135M-SFT/NeuralAI-Air-135M-SFT.Q4_K_M.gguf` → target directory was removed during "clean deployment" commit
- **LM Studio**: `lms` v0.3.34 installed on ZO, `llama-cpp-python` v0.3.34, `gguf` v0.19.0

### What exists locally
- `models/NeuralAI_Air_135M.py` — reconstructed architecture (written by AI Engineer phase)
- `models/NeuralAI-Air-135M-SFT/config.json` — fetched from ZO
- `models/NeuralAI-Air-135M-SFT/tokenizer_config.json` — fetched from ZO
- `models/NeuralAI-Air-135M-SFT/special_tokens_map.json` — fetched from ZO
- `scripts/convert_air_to_gguf.py` — conversion script (written but not deployed to ZO)
- `services/neuralai_llama_server.py` + `.sh` — existing launcher targeting Air GGUF on :1234 (needs path fix)
- `run_service.sh` — auto-detects :1234 and routes to it

### Architecture confirmation
The Air 135M is structurally identical to Llama:
- RMSNorm (pre-norm) ✓
- RoPE position embeddings ✓
- GQA (12 query heads, 2 KV heads, head_dim=64) ✓
- SwiGLU MLP (gate=w1, up=w3, down=w2) ✓
- Tied embeddings ✓

**Key decision**: Write GGUF with `arch="llama"` so llama.cpp loads it natively. The tensor layout maps 1:1 to Llama's expected tensor names.

## Slices

```
Slice 1: GGUF Conversion (on ZO host)
  AC: NeuralAI-Air-135M-SFT.F16.gguf exists on ZO and loads in llama_cpp
  Depends on: nothing

Slice 2: Inference Verification (on ZO host)
  AC: llama_cpp generates coherent text from a ChatML prompt using the GGUF
  Depends on: Slice 1

Slice 3: Service Wiring (local repo)
  AC: run_service.sh + neuralai_llama_server.sh target the new GGUF path and boot correctly
  Depends on: Slice 2

Slice 4: Live Service Test
  AC: https://neuralai-web-ui-deandrewharris.zocomputer.io chat returns responses from the Air model
  Depends on: Slice 3
```

## Tasks

### Slice 1: GGUF Conversion
  AC: F16 GGUF loads in llama_cpp on ZO

  [ ] Task 1.1: Write converter script to ZO `/tmp/conv_air.py`
      Files: ZO:/tmp/conv_air.py (written via Zo API heredoc in 3-4 chunks)
      Change: Python script that loads final.pt, writes GGUF using gguf.GGUFWriter with arch="llama", F16 tensors, Llama-compatible tensor names, GPT2 tokenizer metadata

  [ ] Task 1.2: Run conversion on ZO
      Files: ZO:/home/workspace/Projects/NeuralAI/models/NeuralAI-Air-135M-SFT.F16.gguf
      Change: Execute `python3 /tmp/conv_air.py`, verify output file size (~260MB expected for F16)

  [ ] Task 1.3: Verify GGUF loads in llama_cpp
      Files: none (verification only)
      Change: `python3 -c "from llama_cpp import Llama; llm = Llama(model_path='...F16.gguf', n_ctx=512); print(llm('Hello', max_tokens=10))"`

### Slice 2: Inference Verification
  AC: Coherent ChatML response from Air GGUF

  [ ] Task 2.1: Test ChatML prompt inference
      Files: none (verification only)
      Change: Load GGUF, send `<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n`, verify response contains NeuralAI identity, not gibberish

  [ ] Task 2.2: Attempt Q4_K_M quantization (optional)
      Files: ZO:NeuralAI-Air-135M-SFT.Q4_K_M.gguf
      Change: Use `llama_cpp` quantize or `lms` to produce Q4_K_M (~80MB). If it fails, F16 is acceptable for 4GB host.

### Slice 3: Service Wiring
  AC: run_service.sh boots and routes to Air GGUF

  [ ] Task 3.1: Fix neuralai_llama_server.sh GGUF path
      Files: services/neuralai_llama_server.sh, services/neuralai_llama_server.py
      Change: Update GGUF path to `/home/workspace/Projects/NeuralAI/models/NeuralAI-Air-135M-SFT.F16.gguf` (or Q4_K_M if quantized)

  [ ] Task 3.2: Verify run_service.sh detection logic
      Files: run_service.sh
      Change: Confirm :1234 detection picks up the Air model alias and sets LLM_MODEL correctly

### Slice 4: Live Service Test
  AC: Live chat returns Air model responses

  [ ] Task 4.1: Restart neuralai-web-ui service
      Files: none (ZO service management)
      Change: Restart the service so it picks up the new GGUF via :1234

  [ ] Task 4.2: Verify live chat
      Files: none (curl test)
      Change: `curl -X POST https://neuralai-web-ui-deandrewharris.zocomputer.io/api/chat -d '{"message":"Who are you?"}'` returns Air model response

## Risks

```
RISK: GGUF arch="llama" may not load due to tensor shape mismatch (GQA with 2 KV heads)
IMPACT: llama_cpp refuses to load the model
LIKELIHOOD: low — Llama arch supports GQA natively, head_count_kv=2 is standard
MITIGATION: If it fails, fall back to Python PyTorch loading (Path B from AI Engineer analysis)

RISK: Tokenizer incompatibility — custom 32k BPE vs SmolLM2's 49k
IMPACT: Model produces gibberish because token IDs don't match
LIKELIHOOD: low — we're using the exact tokenizer from the SFT snapshot (32000 vocab)
MITIGATION: Verify token IDs for <|im_start|>=1, <|im_end|>=2 before inference test

RISK: Zo API payload size limits prevent writing large scripts
IMPACT: Cannot deploy converter to ZO host
LIKELIHOOD: high — already encountered timeout on >5KB payloads
MITIGATION: Write script in 3-4 small heredoc chunks via Zo API, each <2KB

RISK: F16 GGUF at ~260MB may be too large for 4GB host alongside Flask
IMPACT: OOM during inference
LIKELIHOOD: low — 260MB + Flask ~100MB = ~360MB, well under 4GB
MITIGATION: Q4_K_M quantization to ~80MB if F16 causes issues

RISK: Reconstructed architecture (NeuralAI_Air_135M.py) has subtle differences from original
IMPACT: Weights load with strict=False but produce wrong outputs
LIKELIHOOD: low — weight shapes match exactly, architecture is standard Llama-like
MITIGATION: GGUF conversion doesn't use the Python architecture at all — it directly maps state_dict tensors to GGUF tensors. The Python model is only for PyTorch fallback.
```

## Checkpoint Gates

- **Gate 0**: Reconnaissance complete — all integration points identified (DONE)
- **Gate 1**: Slice 1 — F16 GGUF loads in llama_cpp (proves the conversion approach)
- **Gate 2**: Slice 2 — Inference produces coherent text (proves the model works)
- **Gate 3**: Slice 3 — Service boots and detects :1234 (proves integration)
- **Gate Final**: Slice 4 — Live chat returns Air model responses (proves end-to-end)

## Decisions

1. **GGUF arch = "llama"** (not "neuralai-air-135m") — llama.cpp only recognizes known architectures. The Air model is structurally Llama-compatible (RMSNorm + RoPE + GQA + SwiGLU), so we write it as Llama. The Python architecture file is only needed for PyTorch fallback, not for GGUF inference.

2. **F16 first, Q4_K_M optional** — F16 is simpler to produce with the gguf Python library. Q4_K_M requires llama.cpp's quantization tooling which may or may not work via the CLI. F16 at ~260MB fits the 4GB host. Q4_K_M at ~80MB is a nice-to-have optimization.

3. **All conversion runs on ZO** — the weights (511MB) are on ZO, not local. Transferring them would be slow and unnecessary. The converter runs directly on ZO where everything lives.

4. **No PyTorch fallback needed for now** — the GGUF path via llama_cpp is the primary target. Path B (PyTorch direct load) remains available if GGUF fails, using the reconstructed `NeuralAI_Air_135M.py`.
