import json
from datetime import datetime

data = []

# Reasoning
data.append({
    "prompt": "If Sally has 3 brothers, and each of those brothers has 2 sisters, how many sisters does Sally have?",
    "chosen": "Let's think step by step:\n1. Sally is a person.\n2. She has 3 brothers.\n3. Each of those brothers has 2 sisters.\n4. Since all brothers share the same siblings, the sisters of each brother must be the same group of girls.\n5. The two sisters are Sally herself and one other girl.\n6. Therefore, Sally has 1 sister.",
    "rejected": "Sally has 2 sisters because each brother has 2 sisters.",
    "category": "reasoning"
})

data.append({
    "prompt": "Which is heavier: 1 pound of feathers or 1 pound of lead?",
    "chosen": "They weigh the same: exactly one pound. While lead is much denser than feathers, the weight (one pound) is the same for both.",
    "rejected": "Lead is heavier because it is a metal and denser than feathers.",
    "category": "reasoning"
})

# Math
data.append({
    "prompt": "Solve for x: 2x + 5 = 15",
    "chosen": "To solve 2x + 5 = 15:\n1. Subtract 5 from both sides: 2x = 10\n2. Divide both sides by 2: x = 5\nThe solution is x = 5.",
    "rejected": "x = 10 because 15 - 5 = 10.",
    "category": "math"
})

data.append({
    "prompt": "What is the square root of 144?",
    "chosen": "The square root of 144 is 12, because 12 * 12 = 144.",
    "rejected": "The square root of 144 is 14.",
    "category": "math"
})

# Logic
data.append({
    "prompt": "All humans are mortal. Socrates is a human. What follows logically?",
    "chosen": "By the rule of syllogism, if all humans are mortal and Socrates is a human, it follows that Socrates is mortal.",
    "rejected": "Socrates is a famous philosopher.",
    "category": "logic"
})

data.append({
    "prompt": "If it rains, the ground gets wet. It is raining. What can we conclude?",
    "chosen": "We can conclude that the ground is wet, following the logical rule of Modus Ponens (If P then Q; P; therefore Q).",
    "rejected": "It might stop raining soon.",
    "category": "logic"
})

# Add more robust examples
data.append({
    "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "chosen": "Let the price of the ball be x. Then the price of the bat is x + $1.00.\nThe total is x + (x + 1.00) = 1.10.\n2x + 1.00 = 1.10\n2x = 0.10\nx = 0.05\nThe ball costs 5 cents.",
    "rejected": "The ball costs 10 cents.",
    "category": "reasoning"
})

timestamp = datetime.now().isoformat()
for item in data:
    item["created"] = timestamp

existing_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v8.jsonl"
new_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v11.jsonl"

try:
    with open(existing_file, 'r') as f:
        existing_data = [json.loads(line) for line in f]
    
    combined_data = existing_data + data
    
    with open(new_file, 'w') as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully created {new_file} with {len(combined_data)} total pairs.")
except Exception as e:
    print(f"Error: {e}")
