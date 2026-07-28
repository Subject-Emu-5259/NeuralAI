from pathlib import Path
from datasets import Dataset
import json, torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, AutoConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_PATH = '/home/.z/workspaces/con_Be6MM5KUzfA88RWI/neuralair-135m/neuralair-135m/final.pt'
MODEL_NAME = '/home/workspace/Projects/NeuralAI/NeuralAI-v2-merged'
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_PATH  = '/home/workspace/Projects/NeuralAI/data/train_sft_v18.jsonl'
OUT_DIR    = '/home/workspace/Projects/NeuralAI/checkpoints/v18-sft'

# Load tokenizer from local folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model configuration from local folder
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Instantiate model architecture and load weights from BASE_PATH if available
model = AutoModelForCausalLM.from_config(
    config,
    trust_remote_code=True,
    dtype=torch.float16
).to(DEVICE)

# Load base weights if uploaded
base_pt = Path(BASE_PATH)
if base_pt.exists():
    model.load_state_dict(torch.load(str(base_pt), map_location=DEVICE))
    print('Loaded base checkpoint:', base_pt)
else:
    print('WARNING: base checkpoint not found. Continuing from HF init weights (randomly initialized before loading).')

def run_sft():
    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"SFT complete. Model saved to {OUT_DIR}")

if __name__ == "__main__":
    run_sft()
