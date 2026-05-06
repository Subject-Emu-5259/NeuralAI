import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def run_test():
    base_model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    adapter_path = "/home/workspace/Projects/NeuralAI/training/checkpoints/dpo_tpu_model"
    
    print(f"Loading base model: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )
    
    print(f"Loading DPO adapters from: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    test_prompts = [
        "Write a function to reverse a string",
        "Explain the difference between a list and a tuple in Python",
        "List all files in the current directory including hidden ones"
    ]

    print("\n" + "="*50)
    print("DPO ALIGNMENT TEST RUN")
    print("="*50 + "\n")

    for i, prompt in enumerate(test_prompts):
        print(f"Test {i+1}: {prompt}")
        
        # Format for SmolLM2 Instruct
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print(f"Response:\n{response}")
        print("-" * 30)

if __name__ == "__main__":
    run_test()
