# Model Card for NeuralAI
## NeuralAI-Air-135M-SFT

A separate, lighter model in the NeuralAI ecosystem: the first supervised fine-tune of the from-scratch **NeuralAI-Air-135M** causal language model.

| Property | Value |
| --- | --- |
| **Architecture** | Custom decoder-only Transformer (`neuralai-air`) |
| **Parameters** | 133.72M |
| **Base model** | [`Subject-Emu-5259/NeuralAI-Air-135M`](https://huggingface.co/Subject-Emu-5259/NeuralAI-Air-135M) |
| **Training data** | 37 instruction/response pairs (`data/train_sft_v17.jsonl`) |
| **Format** | ChatML with assistant-only loss masking |
| **Epochs** | 3 |
| **Training setup** | batch 4 × accumulation 4, LR 5e-5, FP16 autocast + FP32 master weights |
| **Hardware** | NVIDIA GPU (Google Colab) |
| **Completed** | 2026-07-26 |
| **HF repo** | [`Subject-Emu-5259/NeuralAI-Air-135M-SFT`](https://huggingface.co/Subject-Emu-5259/NeuralAI-Air-135M-SFT) |

### Intended use
- Lightweight instruction-following assistant for the NeuralAI stack
- Local/cloud inference target for the NeuralAI-Air architecture
- Foundation for future DPO alignment and quantization/GGUF conversion

### Limitations
- Small training set (37 samples); general knowledge comes almost entirely from the base model
- The base model was trained from scratch; verify factual outputs independently
- 2,048-token context window; long-context tasks may need future RoPE scaling work

---


## Model Details

### Model Description

- **Developed by:** [De'Andrew Preston Harris](https://github.com/Subject-Emu-5259) (NeuralAI Core Team)
- **Funded by:** Self-funded / Independent Research
- **Shared by:** [Subject-Emu-5259](https://huggingface.co/Subject-Emu-5259)
- **Model type:** Causal Language Model (LoRA adapter on top of SmolLM2-360M-Instruct)
- **Language(s) (NLP):** English
- **License:** Apache 2.0 (base model: SmolLM2)
- **Finetuned from model:** [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)

### Model Sources

- **Repository:** [github.com/Subject-Emu-5259/NeuralAI](https://github.com/Subject-Emu-5259/NeuralAI)
- **Base Model:** [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- **Demo:** [NeuralAI Web UI](https://neuralai-web-ui-deandrewharris.zocomputer.io)

## Uses

### Direct Use

NeuralAI is a general-purpose conversational AI engine designed for:
- Code generation, debugging, and multi-step reasoning
- Technical Q&A with structured Markdown responses
- Agentic task planning and execution
- Speech-to-speech interaction (via NeuralVoice service)

### Downstream Use

- Integration into custom applications via the OpenAI-compatible API (`/v1/chat/completions`)
- Fine-tuning base for domain-specific instruction following
- Educational tool for AI/ML students

### Out-of-Scope Use

- Real-time safety-critical decision making without human oversight
- Generating harmful, hateful, racist, sexist, lewd, or violent content
- Medical, legal, or financial advice without professional consultation

## Bias, Risks, and Limitations

- The base model (SmolLM2-360M) is a small model and may lack depth on highly specialized topics
- DPO training was conducted on a dataset of 597 preference pairs — coverage is broad but not exhaustive
- The model may occasionally hallucinate or provide outdated information
- Voice interaction requires a valid Gemini API key for STT/TTS processing

### Recommendations

Users should validate critical outputs independently. The model performs best on coding, reasoning, and general knowledge tasks within its training distribution.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")

# Apply NeuralAI LoRA adapter
model = PeftModel.from_pretrained(base_model, "Subject-Emu-5259/NeuralAI")

# Generate
inputs = tokenizer("Hello, how can I help you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Or use with [llmster](https://lmstudio.ai) (LM Studio headless) for optimized inference:

```bash
# Install llmster
curl -fsSL https://lmstudio.ai/install.sh | bash

# Download and run
lms get smollm2-360m-instruct
lms load smollm2-360m-instruct
lms server start --port 1234

# Query via OpenAI-compatible API
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}]}'
```

## Training Details

### Training Data

- **Dataset**: `data/train_dpo_v15.jsonl` — 597 manually curated preference pairs
- **Domains covered**: Code correctness, logic, reasoning, debugging, multi-step tasks, creative writing
- **Method**: Direct Preference Optimization (DPO) with chosen/rejected response pairs

### Training Procedure

#### Preprocessing

- Raw conversation data filtered and deduplicated
- Preference pairs constructed with expert-curated chosen and rejected responses
- Tokenization using SmolLM2's native tokenizer with max sequence length of 1024

#### Training Hyperparameters

- **Training regime:** Mixed precision (FP16/BF16) on Apple Silicon MPS
- **Optimizer:** AdamW
- **Learning rate:** 5e-5
- **Beta (DPO):** 0.1
- **Batch size:** 4 (per device) × 4 (gradient accumulation) = effective batch size 16
- **Max prompt length:** 512 tokens
- **Max sequence length:** 1024 tokens
- **Epochs:** 3
- **Training steps:** 450

#### Speeds, Sizes, Times

- **Total training time:** ~12 minutes (730.5 seconds)
- **Hardware:** Apple Silicon MPS (MacBook Air M4)
- **Adapter size:** ~2.5 MB (LoRA rank 16)
- **Base model size:** ~720 MB (FP16)

## Evaluation

### Testing Data, Factors & Metrics

- Manual evaluation on coding, reasoning, and conversational benchmarks
- Response quality measured by: accuracy, helpfulness, format compliance, and safety

### Results

- **Final training loss:** 0.305
- **Reward margin:** ~3.5 (strong preference for chosen responses)
- **Convergence:** Stable across 3 epochs with no overfitting observed

## Environmental Impact

- **Hardware Type:** Apple Silicon (MacBook Air M4) — energy-efficient ARM architecture
- **Hours used:** ~0.2 hours (training) + ongoing inference on ZO Computer
- **Cloud Provider:** ZO Computer (always-on hosting)
- **Compute Region:** United States
- **Carbon Emitted:** Minimal — trained on local hardware, inference via optimized llama.cpp (258MB RAM footprint)

## Technical Specifications

### Model Architecture and Objective

- **Architecture:** Transformer decoder (causal language model)
- **Base parameters:** 360M
- **Adapter:** LoRA (Low-Rank Adaptation) — rank 16, applied to attention layers
- **Objective:** Next-token prediction with DPO alignment for preferred response selection
- **Context window:** 2048 tokens

### Compute Infrastructure

#### Hardware
- **Training:** MacBook Air M4 (Apple Silicon)
- **Inference:** ZO Computer (Linux x86_64) with llmster/llama.cpp

#### Software
- **Framework:** PyTorch 2.x, Transformers, PEFT, TRL
- **Inference runtime:** llmster 0.0.19 (LM Studio headless, llama.cpp backend)
- **Backend:** Python 3.x, Flask
- **Frontend:** Vanilla JS, HTML5, CSS3

## Citation

**BibTeX:**
```bibtex
@software{neuralai2026,
  author       = {Harris, De'Andrew Preston},
  title        = {NeuralAI: The Generative AI Engine},
  year         = {2026},
  url          = {https://github.com/Subject-Emu-5259/NeuralAI},
  version      = {7.2.0},
  description  = {A DPO-aligned multimodal AI engine with pluggable inference backends}
}
```

**APA:**
> Harris, D. A. P. (2026). NeuralAI: The Generative AI Engine (Version 7.2.0) [Software]. GitHub. https://github.com/Subject-Emu-5259/NeuralAI

## More Information

- **Project Repository:** [github.com/Subject-Emu-5259/NeuralAI](https://github.com/Subject-Emu-5259/NeuralAI)
- **Live Deployment:** [neuralai-web-ui-deandrewharris.zocomputer.io](https://neuralai-web-ui-deandrewharris.zocomputer.io)
- **Training Plan:** See `neural-brain/TRAINING_PLAN.md`
- **Architecture Docs:** See `docs/ARCHITECTURE.md` and `docs/SERVICE_ARCHITECTURE.md`

## Model Card Authors

- **De'Andrew Preston Harris** — Founder, Architect, and Primary Developer

## Model Card Contact

- **GitHub:** [Subject-Emu-5259](https://github.com/Subject-Emu-5259)
- **LinkedIn:** [De'Andrew Harris](https://www.linkedin.com/in/deandrewharris94/)

## Framework Versions

- PEFT 0.19.0
- Transformers 4.x
- TRL 0.12+
- PyTorch 2.x
- Datasets 3.x
