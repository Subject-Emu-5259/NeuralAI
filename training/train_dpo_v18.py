import torch
from pathlib import Path
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig
)
from trl import DPOConfig, DPOTrainer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_PATH = '/home/.z/workspaces/con_Be6MM5KUzfA88RWI/neuralair-135m/neuralair-135m/final.pt'
MODEL_NAME = '/home/workspace/Projects/NeuralAI/NeuralAI-v2-merged'
HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_PATH  = '/home/workspace/Projects/NeuralAI/data/train_dpo_v18.jsonl'
OUT_DIR    = '/home/workspace/Projects/NeuralAI/checkpoints/v18-dpo'

def run_dpo():
    print(f"Starting DPO on {DEVICE}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(DEVICE)

    base_pt = Path(BASE_PATH)
    if base_pt.exists():
        model.load_state_dict(torch.load(str(base_pt), map_location=DEVICE))
        print('Loaded base checkpoint:', base_pt)
    else:
        print('WARNING: base checkpoint not found.')

    # Load reference model (identical to base for DPO)
    ref_model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=True,
        dtype=torch.float16
    ).to(DEVICE)
    if base_pt.exists():
        ref_model.load_state_dict(torch.load(str(base_pt), map_location=DEVICE))

    with open(DATA_PATH, 'r') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)

    training_args = DPOConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=1,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"DPO complete. Model saved to {OUT_DIR}")

if __name__ == "__main__":
    run_dpo()
