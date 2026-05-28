import json
from datetime import datetime
import shutil

pairs = [
    # Reasoning & Logic (Advanced)
    {
        "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
        "chosen": "Sally has 1 sister. All 3 brothers have the same sisters. Since Sally is one of the sisters, and each brother has 2 sisters, the other sister must be Sally's only sister.",
        "rejected": "Sally has 2 sisters because each brother has 2 sisters and there are 3 brothers.",
        "category": "reasoning"
    },
    {
        "prompt": "A man is looking at a photograph of someone. His friend asks who it is. The man replies, 'Brothers and sisters, I have none. But that man's father is my father's son.' Who is in the photograph?",
        "chosen": "The man's son is in the photograph. 'My father's son' is the man himself (since he has no brothers or sisters). So, the sentence translates to 'That man's father is ME.' Therefore, the man in the photo is his son.",
        "rejected": "The man is looking at a photo of himself.",
        "category": "logic"
    },
    # Mathematics (Calculus & Word Problems)
    {
        "prompt": "Find the derivative of f(x) = sin(x^2).",
        "chosen": "To find the derivative of f(x) = sin(x^2), we use the chain rule:\\n1. The outer function is sin(u), where u = x^2. Its derivative is cos(u).\\n2. The inner function is u = x^2. Its derivative is 2x.\\n3. Multiplying them together: f'(x) = cos(x^2) * 2x = 2x cos(x^2).",
        "rejected": "f'(x) = cos(2x).",
        "category": "math"
    },
    {
        "prompt": "A train leaves Station A at 60 mph. Two hours later, another train leaves Station A at 90 mph on the same track. How long will it take for the second train to catch up to the first?",
        "chosen": "1. Let t be the time in hours the second train travels.\\n2. The first train travels for (t + 2) hours.\\n3. Distance = Rate * Time.\\n4. Distance for Train 1: 60 * (t + 2).\\n5. Distance for Train 2: 90 * t.\\n6. Set them equal: 60(t + 2) = 90t.\\n7. 60t + 120 = 90t.\\n8. 120 = 30t.\\n9. t = 4.\\nIt will take the second train 4 hours to catch up.",
        "rejected": "It will take 2 hours.",
        "category": "math"
    },
    # Debugging (Advanced)
    {
        "prompt": "My Python script is running very slowly when processing a large list of dictionaries. I'm using: `for item in items: if item['id'] in seen_ids: ...`",
        "chosen": "The bottleneck is likely the `in` check on `seen_ids`. If `seen_ids` is a list, the check is O(n), making the entire operation O(n^2). You should convert `seen_ids` to a **set**. Set lookups are O(1) on average, which will change the overall complexity to O(n).",
        "rejected": "You should use a multi-processing pool to speed up the loop.",
        "category": "debugging"
    },
    {
        "prompt": "Why is my CSS `z-index: 9999` not working? The element is still behind another one with a lower z-index.",
        "chosen": "This usually happens because the element belongs to a different **stacking context**. A z-index only works relative to other elements in the same stacking context. Check if a parent has `opacity < 1`, `transform`, `filter`, or `position: relative/absolute/fixed` with a z-index. To fix it, you may need to move the element in the DOM or adjust the stacking context of its parents.",
        "rejected": "Try using `z-index: 9999999 !important;`.",
        "category": "debugging"
    },
    # Creativity
    {
        "prompt": "Write a haiku about a futuristic AI in Memphis.",
        "chosen": "Neon bridge reflects,\\nSilicon mind learns the blues,\\nDre's dream comes alive.",
        "rejected": "I am a robot in Memphis. I like the music and the lights. Memphis is great.",
        "category": "creativity"
    },
    # Multi-step
    {
        "prompt": "Plan a 3-step strategy to launch a new feature for NeuralAI.",
        "chosen": "1. **Beta Testing & Feedback**: Deploy the feature to a small group of trusted users (or a staging environment). Collect logs and user feedback to identify bugs and usability issues.\\n2. **Documentation & Marketing**: Update the README and SOCIAL.md. Create a short demo video or screenshot gallery showing the feature in action.\\n3. **Public Rollout & Monitoring**: Enable the feature for all users. Monitor server logs (`neuralai_err.log`) and model performance in real-time to ensure stability.",
        "rejected": "1. Write the code. 2. Push to GitHub. 3. Tell everyone on Twitter.",
        "category": "multi_step"
    }
]

base_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v9.jsonl"
output_file = "/home/workspace/Projects/NeuralAI/data/train_dpo_v10.jsonl"

shutil.copyfile(base_file, output_file)

with open(output_file, "a") as f:
    for p in pairs:
        p["created"] = datetime.now().isoformat()
        f.write(json.dumps(p) + "\n")
        
print(f"Appended {len(pairs)} pairs to {output_file}")
