# F-001 Implementation Log: Revive NeuralAI-Air-135M-SFT

## Status: planning → building (approved)

## Reconnaissance Summary
- Weights: `final.pt` (511MB) on ZO host, 138 tensors, 135M params
- Config: 32000 vocab, 768 hidden, 15 layers, 12 heads, 2 KV heads, 2560 intermediate, 2048 max_pos
- Tokenizer: 32000-vocab GPT2 BPE with ChatML special tokens, in HF cache on ZO
- Architecture: Llama-compatible (RMSNorm + RoPE + GQA + SwiGLU + tied embeddings)
- GGUF path: use arch="llama" for llama.cpp compatibility
- Blockers: Architecture source was deleted, but GGUF conversion bypasses this (maps state_dict directly)

## Pre-plan work completed (AI Engineer phase)
- [x] Reconstructed `models/NeuralAI_Air_135M.py` from bytecode analysis
- [x] Fetched `config.json`, `tokenizer_config.json`, `special_tokens_map.json` to local repo
- [x] Wrote `scripts/convert_air_to_gguf.py` (full converter, needs deployment to ZO)
- [x] Verified GGUFWriter API on ZO (add_tensor, add_uint32, add_array, add_string all available)
- [x] Confirmed Llama tensor name mapping (token_embd, blk.{i}.attn_q, etc.)
- [x] Identified Zo API payload size limitation (need heredoc chunk approach)
