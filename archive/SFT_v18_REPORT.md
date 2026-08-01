# Project Documentation: NeuralAI-Air-135M Fine-Tuning

## 1. Executive Summary
This session focused on resolving architectural instabilities and training data processing errors for the **NeuralAI-Air-135M** model. We successfully stabilized the model's internal activations and implemented a 'Precision Masking' strategy to ensure the model learns strictly from assistant responses.

## 2. Technical Fixes & Improvements

### A. Architectural Stabilization (Exploding Gradients)
- **Problem**: Activation norms were jumping from **0.9** at the embedding layer to over **1200** at Layer 0, causing the model to produce gibberish.
- **Fix**: Re-initialized the model weights using a scaled normal distribution: `std / sqrt(2 * num_layers)`.
- **Result**: Post-fix activation norms stabilized at **0.23**, providing a healthy foundation for training.

### B. Precision Masking (Tokenization)
- **Problem**: The model was incorrectly learning from prompt tokens and padding, leading to 'newline collapse' where it would generate endless empty lines or repetitive special tokens.
- **Fix**: Updated the `tokenize` function to apply `IGNORE = -100` to all user prompts and padding tokens. The model is now penalized only for errors in the assistant's predicted text.

### C. Manual Weights Handling (Tied Tensors)
- **Problem**: Hugging Face's default `save_pretrained` failed due to tied weights (`lm_head` and `embed_tokens`).
- **Fix**: Implemented a manual save/load routine using `torch.save(model.state_dict())` and `HfApi` to bypass the automated check and ensure weights are preserved.

### D. Advanced Sampling (Top-p/Nucleus)
- **Improvement**: Manually injected a `top-p` (Nucleus) sampling mechanism into the model's `generate` method to allow for more diverse and coherent text generation compared to simple temperature scaling.

## 3. Current Model Status
- **Loss Performance**: The final training pass achieved a significantly lower loss, indicating convergence on the provided ChatML dataset.
- **Known Limitations**: At 135M parameters, the model is highly sensitive to sampling parameters. While structured responses (like 'Who are you?') are represented in the weights, the model still requires high-quality data and specific temperature settings to maintain coherence.

## 4. Final Artifacts
- **Model Repo**: `Subject-Emu-5259/NeuralAI-Air-135M-SFT-v18` on Hugging Face.
- **Local Archive**: `checkpoints_v18-sft.zip` containing the final state dictionary.
- **Workspace Snapshot**: `Projects/NeuralAI/checkpoints/v18-sft/`
  - `final/` — `pytorch_model_final_push.bin` renamed to `pytorch_model.bin`
  - `stabilized/` — `pytorch_model_stabilized.bin`
  - `hyper_converged/` — `pytorch_model_hyper_converged.bin`
  - `checkpoint-32/` — intermediate config snapshot
  - `tokenizer.json`, `tokenizer_config.json`, `NeuralAI_Air_135M.py`
