import json
from datetime import datetime
import shutil

pairs = [
    # Reasoning & Logic
    {
        "prompt": "There are 5 birds on a branch. A hunter shoots one. How many are left on the branch?",
        "chosen": "Zero. After the loud sound of the gunshot, the remaining birds would fly away.",
        "rejected": "There are 4 birds left (5 - 1 = 4).",
        "category": "reasoning_logic"
    },
    {
        "prompt": "If a shirt costs $20 after a 20% discount, what was its original price? Explain the steps.",
        "chosen": "1. Let x be the original price.\n2. The shirt is 20% off, which means it costs 80% of its original price (100% - 20%).\n3. So, 0.8 * x = $20.\n4. To find x, divide 20 by 0.8: 20 / 0.8 = 25.\n5. The original price was $25.",
        "rejected": "The original price was $24 because 20 + 20% of 20 = 24.",
        "category": "reasoning_logic"
    },
    {
        "prompt": "If you have a 3-gallon jug and a 5-gallon jug, how can you measure exactly 4 gallons of water?",
        "chosen": "1. Fill the 5-gallon jug.\n2. Pour water from the 5-gallon jug into the 3-gallon jug until it is full. The 5-gallon jug now has 2 gallons.\n3. Empty the 3-gallon jug.\n4. Pour the 2 gallons from the 5-gallon jug into the 3-gallon jug.\n5. Fill the 5-gallon jug again.\n6. Pour water from the 5-gallon jug into the 3-gallon jug until it is full. Since the 3-gallon jug already has 2 gallons, it can only take 1 more gallon.\n7. The 5-gallon jug now has exactly 4 gallons left.",
        "rejected": "You can't do it with just a 3-gallon and 5-gallon jug. You need a 4-gallon jug.",
        "category": "reasoning_logic"
    },
    # Mathematics
    {
        "prompt": "What is the integral of x^2 dx?",
        "chosen": "The integral of x^2 dx is (x^3 / 3) + C, where C is the constant of integration.",
        "rejected": "It is 2x.",
        "category": "mathematics"
    },
    {
        "prompt": "Calculate 15% of 850.",
        "chosen": "To calculate 15% of 850, multiply 850 by 0.15. 850 * 0.15 = 127.5.",
        "rejected": "125.",
        "category": "mathematics"
    },
    {
        "prompt": "Solve for x: 2x + 5 = 15",
        "chosen": "To solve for x:\n1. Subtract 5 from both sides: 2x = 10\n2. Divide both sides by 2: x = 5",
        "rejected": "x = 10",
        "category": "mathematics"
    },
    # Debugging
    {
        "prompt": "I'm getting a 'list index out of range' error in my Python loop: `for i in range(len(items) + 1): print(items[i])`",
        "chosen": "The error occurs because `range(len(items) + 1)` generates indices from 0 up to `len(items)`. However, valid indices for a list of length `n` are 0 to `n-1`. When `i` reaches `len(items)`, `items[i]` goes out of bounds. To fix it, change the loop to `for i in range(len(items)):` or simply `for item in items: print(item)`.",
        "rejected": "You should use a try-except block to ignore the error.",
        "category": "debugging"
    },
    {
        "prompt": "Why does my Node.js application crash with 'EADDRINUSE'?",
        "chosen": "The 'EADDRINUSE' error means the port your application is trying to bind to is already being used by another process. To fix this, you can either: \n1. Identify and stop the process using that port (e.g., using `lsof -i :<port>` or `netstat -ano | grep <port>`).\n2. Change your application's configuration to use a different, available port.",
        "rejected": "You just need to restart your computer.",
        "category": "debugging"
    },
    {
        "prompt": "My React component isn't updating when the state changes. I'm modifying the state array directly like this: `myArray.push(newItem); setMyArray(myArray);`",
        "chosen": "React components don't re-render if the state reference remains the same. When you use `.push()` on an array, you are modifying the existing array in place, so React doesn't detect a change. To fix this, you must create a new array reference. Use the spread operator: `setMyArray([...myArray, newItem]);`",
        "rejected": "You need to call `forceUpdate()` after pushing to the array.",
        "category": "debugging"
    }
]

base_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v7.jsonl"
output_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v8.jsonl"

shutil.copyfile(base_file, output_file)

with open(output_file, "a") as f:
    for p in pairs:
        p["created"] = datetime.now().isoformat()
        f.write(json.dumps(p) + "\n")
        
print(f"Appended {len(pairs)} pairs to {output_file}")