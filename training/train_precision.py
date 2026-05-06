import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import os

def train_precision():
    base_model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # We start from the already aligned DPO model to further refine it
    current_adapter_path = "/home/workspace/Projects/NeuralAI/training/checkpoints/dpo_tpu_model"
    dataset_path = "/home/workspace/Projects/NeuralAI/data/dpo_cli_precision.jsonl"
    output_dir = "/home/workspace/Projects/NeuralAI/training/checkpoints/precision_model"

    print(f"Loading model and adapters from {current_adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the DPO aligned model as the starting point
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(model, current_adapter_path, is_trainable=True)

    # Load precision dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # DPO Config
    config = DPOConfig(
        output_dir=output_dir,
        beta=0.1,
        max_prompt_length=128,
        max_length=512,
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="no",
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting precision refinement training...")
    trainer.train()
    
    # Save the refined model
    trainer.save_model(output_dir)
    print(f"Precision model saved to {output_dir}")

if __name__ == "__main__":
    train_precision()
