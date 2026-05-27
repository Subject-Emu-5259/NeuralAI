import json
from datetime import datetime
import shutil
import os

pairs = [
    # Tool Use: Breaking the Logo Loop
    {
        "prompt": "generate a moon image",
        "chosen": "<tool>image: a beautiful realistic moon in the night sky</tool>",
        "rejected": "I can help you create a professional logo using your own images. Can you please provide the images you want to use?",
        "category": "tool_use"
    },
    {
        "prompt": "make a picture of a cat",
        "chosen": "<tool>image: a cute fluffy cat sitting on a sofa</tool>",
        "rejected": "I can help you create a logo using your own images. Can you please provide the images you want to use?",
        "category": "tool_use"
    },
    {
        "prompt": "generate a professional tech logo for NeuralAI",
        "chosen": "<tool>image: a professional tech logo for an AI company named NeuralAI, minimalist and futuristic</tool>",
        "rejected": "I can help you create a professional logo using your own images. Can you please provide the images you want to use?",
        "category": "tool_use"
    },
    # Reasoning & Logic
    {
        "prompt": "If you are running a race and you pass the person in second place, what place are you in?",
        "chosen": "Second place. You took the position of the person you passed, who was in second.",
        "rejected": "First place.",
        "category": "reasoning_logic"
    },
    {
        "prompt": "A man is looking at a photograph. Someone asks him, 'Who is in the photo?' The man replies, 'Brothers and sisters, I have none. But that man's father is my father's son.' Who is in the photograph?",
        "chosen": "His son. Since he has no brothers or sisters, 'my father's son' must be himself. Therefore, the man in the photo's father is himself.",
        "rejected": "Himself.",
        "category": "reasoning_logic"
    },
    # Mathematics
    {
        "prompt": "What is the square root of 144 plus 10?",
        "chosen": "The square root of 144 is 12. Adding 10 gives 12 + 10 = 22.",
        "rejected": "24.",
        "category": "mathematics"
    },
    # Debugging
    {
        "prompt": "How do I fix 'Permission denied' when trying to run a script in Linux?",
        "chosen": "You need to make the script executable using the `chmod` command. Run: `chmod +x script.sh`. If it's a system-protected file, you may need to use `sudo` (e.g., `sudo chmod +x script.sh`).",
        "rejected": "You should delete the file and recreate it.",
        "category": "debugging"
    },
    # Memphis Culture
    {
        "prompt": "What is the best place for BBQ in Memphis according to locals?",
        "chosen": "Memphis is famous for its BBQ, and while opinions vary, Central BBQ, The Rendezvous, and Payne's Bar-B-Que are often cited as top choices for their dry-rub ribs and pulled pork.",
        "rejected": "Texas Roadhouse is a great place for BBQ in Memphis.",
        "category": "memphis_founder"
    }
]

base_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v8.jsonl"
output_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v9.jsonl"

# Ensure v8 exists or fallback
if not os.path.exists(base_file):
    base_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v7.jsonl"

shutil.copyfile(base_file, output_file)

with open(output_file, "a") as f:
    for p in pairs:
        p["created"] = datetime.now().isoformat()
        f.write(json.dumps(p) + "\n")
        
print(f"Appended {len(pairs)} pairs to {output_file}")
