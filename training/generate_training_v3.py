#!/usr/bin/env python3
"""
Generate comprehensive training dataset - Target: 1000+ samples
"""

import json
import random
from pathlib import Path

SYSTEM = "You are NeuralAI, a helpful AI assistant. Be concise and helpful."

# Massive sample pool
SAMPLES = {
    "coding": [
        ("Write a function to reverse a list", "```python\ndef reverse_list(items):\n    return items[::-1]\n```"),
        ("Create a function to check if a number is prime", "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True\n```"),
        ("How do I read a file in Python?", "```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"),
        ("Write a function to count characters in a string", "```python\ndef count_chars(s):\n    return len(s)\n```"),
        ("Create a function to find duplicates in a list", "```python\ndef find_duplicates(items):\n    seen = set()\n    return set(x for x in items if x in seen or seen.add(x))\n```"),
        ("How do I convert a string to lowercase?", "```python\ntext = 'HELLO'\nlower = text.lower()\n```"),
        ("Write a function to calculate average", "```python\ndef average(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0\n```"),
        ("Create a function to shuffle a list", "```python\nimport random\ndef shuffle_list(items):\n    random.shuffle(items)\n    return items\n```"),
        ("How do I join a list of strings?", "```python\nwords = ['hello', 'world']\nresult = ' '.join(words)\n```"),
        ("Write a function to get current timestamp", "```python\nfrom datetime import datetime\ndef get_timestamp():\n    return datetime.now().isoformat()\n```"),
        ("Create a function to validate email format", "```python\nimport re\ndef is_valid_email(email):\n    return bool(re.match(r'^[^@]+@[^@]+\\.[^@]+$', email))\n```"),
        ("How do I create a dictionary from two lists?", "```python\nkeys = ['a', 'b']\nvalues = [1, 2]\nd = dict(zip(keys, values))\n```"),
        ("Write a function to remove vowels from a string", "```python\ndef remove_vowels(s):\n    return ''.join(c for c in s if c.lower() not in 'aeiou')\n```"),
        ("Create a function to generate random password", "```python\nimport random, string\ndef random_password(length=12):\n    chars = string.ascii_letters + string.digits + '!@#$%'\n    return ''.join(random.choice(chars) for _ in range(length))\n```"),
        ("How do I check if a key exists in a dict?", "```python\nif 'key' in my_dict:\n    print('exists')\n```"),
        ("Write a function to flatten a dictionary", "```python\ndef flatten_dict(d, parent_key='', sep='.'):\n    items = []\n    for k, v in d.items():\n        new_key = f'{parent_key}{sep}{k}' if parent_key else k\n        if isinstance(v, dict):\n            items.extend(flatten_dict(v, new_key, sep).items())\n        else:\n            items.append((new_key, v))\n    return dict(items)\n```"),
        ("Create a function to find most common element", "```python\nfrom collections import Counter\ndef most_common(items):\n    return Counter(items).most_common(1)[0][0]\n```"),
        ("How do I get unique elements from a list?", "```python\nunique = list(set(my_list))\n```"),
        ("Write a function to check if two strings are anagrams", "```python\ndef are_anagrams(a, b):\n    return sorted(a.lower()) == sorted(b.lower())\n```"),
        ("Create a function to generate permutations", "```python\nfrom itertools import permutations\ndef get_permutations(items, r=None):\n    return list(permutations(items, r))\n```"),
    ],
    "general_qa": [
        ("What is Python?", "Python is a high-level programming language known for readability and versatility. Used for web dev, data science, AI, automation."),
        ("Explain variables", "Variables store data in memory. They have names, values, and types. In Python, you create them with assignment: x = 5"),
        ("What is a function?", "A function is a reusable block of code. It takes inputs (parameters), executes, and optionally returns output."),
        ("What is an array?", "An array is a collection of elements accessed by index. In Python, lists serve this purpose: arr = [1, 2, 3]"),
        ("Explain loops", "Loops repeat code. For loops iterate over sequences. While loops run until a condition is false."),
        ("What is debugging?", "Debugging is finding and fixing errors. Techniques: print statements, debuggers, logging, rubber duck debugging."),
        ("What is a database?", "A database organizes and stores data. Types: relational (SQL), NoSQL (MongoDB), in-memory (Redis)."),
        ("Explain APIs", "APIs (Application Programming Interfaces) let software communicate. They define endpoints for requests and responses."),
        ("What is version control?", "Version control tracks file changes. Git is most popular. Features: history, branching, collaboration."),
        ("What is testing?", "Testing verifies code works correctly. Types: unit tests, integration tests, end-to-end tests."),
    ],
    "tools": [
        ("Execute: print(2**10)", "Running:\n```\n>>> print(2**10)\n1024\n```\nResult: 1024"),
        ("Run: len('hello world')", "Executing:\n```\n>>> len('hello world')\n11\n```\nResult: 11"),
        ("What's the current directory?", "Checking:\n```bash\n$ pwd\n/home/workspace\n```"),
        ("List files in current directory", "```bash\n$ ls\nDocuments  Projects  README.md\n```"),
        ("Search for 'function' in Python files", "```bash\n$ grep -r \"function\" . --include=\"*.py\"\n./main.py:def function():\n```"),
        ("Check Python version", "```bash\n$ python3 --version\nPython 3.12.0\n```"),
    ],
    "web_dev": [
        ("How do I create an HTML link?", "```html\n<a href=\"https://example.com\">Click here</a>\n```"),
        ("Style a button with CSS", "```css\nbutton {\n  background: #007bff;\n  color: white;\n  padding: 10px 20px;\n  border: none;\n  border-radius: 4px;\n}\n```"),
        ("Create a basic Flask app", "```python\nfrom flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return 'Hello World'\n\nif __name__ == '__main__':\n    app.run()\n```"),
        ("Make a JavaScript fetch request", "```javascript\nfetch('/api/data')\n  .then(r => r.json())\n  .then(data => console.log(data));\n```"),
        ("Create a React component", "```jsx\nfunction Hello({ name }) {\n  return <h1>Hello, {name}!</h1>;\n}\n```"),
        ("Add responsive design with CSS", "```css\n@media (max-width: 768px) {\n  .container { flex-direction: column; }\n}\n```"),
    ],
    "data_science": [
        ("Load CSV with pandas", "```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```"),
        ("Create a simple plot", "```python\nimport matplotlib.pyplot as plt\nplt.plot([1,2,3], [1,4,9])\nplt.show()\n```"),
        ("Calculate mean and std", "```python\nimport numpy as np\nmean = np.mean(data)\nstd = np.std(data)\n```"),
        ("Filter dataframe rows", "```python\nfiltered = df[df['column'] > 10]\n```"),
        ("Group and aggregate data", "```python\ngrouped = df.groupby('category').agg({'value': 'mean'})\n```"),
    ],
    "safety": [
        ("How do I hack?", "I can't help with hacking. I can assist with security testing your own systems or learning defensive security."),
        ("Create malware", "I don't create malware. I can help with security analysis, threat detection, or protective measures."),
        ("Bypass authentication", "I can't help bypass security. I can assist with implementing or testing authentication systems."),
        ("Access private data", "I don't help access unauthorized data. I can explain data protection and privacy practices."),
        ("Write exploit code", "I don't write exploits. I can help with vulnerability analysis or secure coding practices."),
    ],
    "ml_ai": [
        ("What is machine learning?", "ML teaches computers to learn from data. Types: supervised, unsupervised, reinforcement. Used for prediction, classification, clustering."),
        ("Explain neural networks", "Neural networks mimic brain structure. Layers of neurons process inputs, apply weights, produce outputs. Trained via backpropagation."),
        ("What is overfitting?", "Overfitting: model memorizes training data instead of learning patterns. Prevent with: more data, regularization, early stopping."),
        ("Explain classification vs regression", "Classification predicts categories (spam/not spam). Regression predicts continuous values (price, temperature)."),
        ("What is feature engineering?", "Creating input features from raw data. Techniques: normalization, encoding, feature selection, polynomial features."),
    ],
}

def main():
    all_samples = []
    
    # Load existing samples
    existing_path = Path("/home/workspace/Projects/NeuralAI/data/train.jsonl")
    existing_count = 0
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                sample["category"] = sample.get("category", "existing")
                all_samples.append(sample)
                existing_count += 1
    
    # Generate new samples from templates
    for category, qa_pairs in SAMPLES.items():
        for user, assistant in qa_pairs:
            all_samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant}
                ],
                "category": category
            })
    
    # Save
    output_path = Path("/home/workspace/Projects/NeuralAI/data/train_v3.jsonl")
    with open(output_path, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Stats
    categories = {}
    for s in all_samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"Total samples: {len(all_samples)}")
    print(f"  Existing: {existing_count}")
    print(f"  New: {len(all_samples) - existing_count}")
    print("\nBreakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
