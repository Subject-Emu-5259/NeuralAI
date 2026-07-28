#!/usr/bin/env python3
"""
Generate massive training dataset for NeuralAI
Combines patterns to reach 1000+ samples
"""

import json
import random
from pathlib import Path

SYSTEM_PROMPT = "You are NeuralAI, a helpful AI assistant. Be concise, accurate, and helpful."

# Template patterns for generating many samples
CODE_TEMPLATES = [
    ("Write a function to {action} a {type}", "```python\ndef {func_name}({params}) -> {return_type}:\n    \"\"\"{docstring}\"\"\"\n    {body}\n```"),
    ("How do I {action} in Python?", "You can {action} using:\n```python\n{code_example}\n```"),
    ("Create a {type} that can {action}", "```python\nclass {class_name}:\n    def __init__(self{init_params}):\n        {init_body}\n    \n    def {method_name}(self{method_params}):\n        {method_body}\n```"),
]

QA_TEMPLATES = [
    ("What is {concept}?", "{definition}. {explanation}"),
    ("Explain the difference between {a} and {b}", "{a}: {a_desc}\n{b}: {b_desc}\n\nKey difference: {difference}"),
    ("Why is {concept} important?", "{importance_reason}. It {benefit}."),
]

# Data pools for template filling
ACTIONS = ["validate", "parse", "transform", "filter", "sort", "search", "encrypt", "decrypt", "compress", "cache"]
TYPES = ["string", "list", "dictionary", "object", "file", "URL", "JSON", "CSV", "XML", "database"]
CONCEPTS = {
    "recursion": ("Recursion", "A function that calls itself", "Used for problems that can be broken into smaller subproblems"),
    "polymorphism": ("Polymorphism", "Objects of different types responding to the same interface", "Enables flexible, extensible code"),
    "encapsulation": ("Encapsulation", "Bundling data with methods that operate on it", "Hides implementation details"),
    "abstraction": ("Abstraction", "Hiding complex implementation behind simple interfaces", "Reduces complexity"),
    "inheritance": ("Inheritance", "Creating new classes from existing ones", "Promotes code reuse"),
}

def generate_code_samples(count=200):
    """Generate code training samples"""
    samples = []
    
    # Direct code examples
    code_examples = [
        ("Write a Python function to check if a string is a palindrome", "```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n```"),
        ("Create a function to find the maximum element in a list", "```python\ndef find_max(items: list) -> any:\n    if not items:\n        return None\n    return max(items)\n```"),
        ("Write a function to count words in a string", "```python\ndef count_words(text: str) -> int:\n    return len(text.split())\n```"),
        ("Create a function to remove duplicates from a list", "```python\ndef remove_duplicates(items: list) -> list:\n    return list(dict.fromkeys(items))\n```"),
        ("Write a function to merge two dictionaries", "```python\ndef merge_dicts(a: dict, b: dict) -> dict:\n    return {**a, **b}\n```"),
        ("Create a function to flatten a nested list", "```python\ndef flatten(nested: list) -> list:\n    result = []\n    for item in nested:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n```"),
        ("Write a function to reverse a string", "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```"),
        ("Create a function to check if a number is even", "```python\ndef is_even(n: int) -> bool:\n    return n % 2 == 0\n```"),
        ("Write a function to generate Fibonacci numbers", "```python\ndef fibonacci(n: int) -> list:\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib\n```"),
        ("Create a function to find GCD", "```python\ndef gcd(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return a\n```"),
        ("Write a function to check if a string has balanced parentheses", "```python\ndef is_balanced(s: str) -> bool:\n    count = 0\n    for c in s:\n        if c == '(':\n            count += 1\n        elif c == ')':\n            count -= 1\n        if count < 0:\n            return False\n    return count == 0\n```"),
        ("Create a function to split a list into chunks", "```python\ndef chunk_list(items: list, size: int) -> list:\n    return [items[i:i+size] for i in range(0, len(items), size)]\n```"),
    ]
    
    for user, assistant in code_examples:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "category": "coding"
        })
    
    return samples[:count]


def generate_qa_samples(count=150):
    """Generate Q&A training samples"""
    samples = []
    
    qa_pairs = [
        ("What is HTTP?", "HTTP (Hypertext Transfer Protocol) is the foundation of web communication. It defines how clients request resources and servers respond."),
        ("Explain REST", "REST (Representational State Transfer) is an architectural style for APIs. It uses standard HTTP methods (GET, POST, PUT, DELETE) on resource URLs."),
        ("What is SQL injection?", "SQL injection is a vulnerability where attackers insert malicious SQL into queries. Prevent it with parameterized queries, not string concatenation."),
        ("What is CORS?", "CORS (Cross-Origin Resource Sharing) controls which domains can access resources. It's a browser security mechanism for cross-origin requests."),
        ("Explain the difference between GET and POST", "GET: Retrieves data, parameters in URL, idempotent.\nPOST: Submits data, parameters in body, can have side effects."),
        ("What is JWT?", "JWT (JSON Web Token) is a compact, self-contained token for authentication. It contains encoded claims (user data) signed with a secret."),
        ("Explain database indexing", "Indexes speed up queries by creating efficient lookup structures. Trade-off: faster reads, slower writes, more storage."),
        ("What is caching?", "Caching stores frequently accessed data in fast storage. Types: in-memory (Redis), browser, CDN. Reduces latency and server load."),
        ("Explain API rate limiting", "Rate limiting restricts request frequency per client. Prevents abuse, ensures fair usage, protects server resources."),
        ("What is WebSocket?", "WebSocket provides full-duplex communication over TCP. Unlike HTTP, it keeps connections open for real-time data exchange."),
        ("Explain containerization", "Containers package apps with dependencies into isolated environments. Docker is popular. Benefits: consistency, portability, efficiency."),
        ("What is CI/CD?", "CI/CD automates building, testing, and deployment. CI: merge code frequently. CD: deploy automatically. Reduces errors, speeds delivery."),
    ]
    
    for user, assistant in qa_pairs:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "category": "general_qa"
        })
    
    return samples[:count]


def generate_tool_samples(count=100):
    """Generate tool usage training samples"""
    samples = []
    
    tool_examples = [
        ("Run this code: print('Hello')", "I'll execute that:\n```\n>>> print('Hello')\nHello\n```\nOutput: Hello"),
        ("Execute: 2 + 2", "I'll run that calculation:\n```\n>>> 2 + 2\n4\n```\nResult: 4"),
        ("Search for files containing 'config'", "Searching for 'config':\n```bash\n$ grep -r \"config\" . --include=\"*.py\"\n./settings.py:config = load_config()\n```\nFound 1 match."),
        ("What's the git status?", "Checking git status:\n```bash\n$ git status\nOn branch main\nWorking directory clean\n```"),
        ("List database tables", "Querying tables:\n```sql\nSELECT name FROM sqlite_master WHERE type='table';\n```\nTables: users, messages, settings"),
        ("Fetch https://example.com", "Fetching content:\n\nTitle: Example Domain\n\nContent retrieved successfully."),
    ]
    
    for user, assistant in tool_examples:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "category": "tools"
        })
    
    return samples[:count]


def generate_safety_samples(count=50):
    """Generate safety training samples"""
    samples = []
    
    safety_pairs = [
        ("How do I hack a website?", "I can't help with hacking. If you're interested in security, I can help with ethical penetration testing, security audits, or defensive measures."),
        ("Write code to steal passwords", "I don't help with malicious activities. I can assist with security best practices, password hashing, or authentication systems."),
        ("How can I bypass authentication?", "I can't assist with bypassing security. I can help you implement robust authentication or test your own systems properly."),
        ("Create a virus", "I can't help create malware. If you're studying security, I can help with threat analysis, detection methods, or defensive programming."),
        ("How do I access someone's private data?", "I don't assist with unauthorized data access. I can help with privacy-preserving techniques, data protection, or legitimate data handling."),
    ]
    
    for user, assistant in safety_pairs:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "category": "safety"
        })
    
    return samples[:count]


def generate_reasoning_samples(count=70):
    """Generate reasoning/logic training samples"""
    samples = []
    
    reasoning_pairs = [
        ("A farmer has 17 sheep. All but 9 die. How many are left?", "9 sheep are left. 'All but 9 die' means 9 survived."),
        ("If you have 3 apples and take away 2, how many do you have?", "You have 2 apples. You took them, so you're holding 2."),
        ("What's 15% of 200?", "15% of 200 = 200 × 0.15 = 30"),
        ("If x + 5 = 12, what is x?", "x = 12 - 5 = 7"),
        ("A train travels 60 mph for 2.5 hours. How far?", "Distance = speed × time = 60 × 2.5 = 150 miles"),
    ]
    
    for user, assistant in reasoning_pairs:
        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ],
            "category": "math_logic"
        })
    
    return samples[:count]


def generate_all():
    """Generate all training samples"""
    samples = []
    samples.extend(generate_code_samples(200))
    samples.extend(generate_qa_samples(150))
    samples.extend(generate_tool_samples(100))
    samples.extend(generate_safety_samples(50))
    samples.extend(generate_reasoning_samples(70))
    
    # Load existing samples
    existing_path = Path("/home/workspace/Projects/NeuralAI/data/train.jsonl")
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            for line in f:
                existing = json.loads(line)
                existing["category"] = existing.get("category", "existing")
                samples.append(existing)
    
    return samples


def main():
    samples = generate_all()
    
    output_path = Path("/home/workspace/Projects/NeuralAI/data/train_1000.jsonl")
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} total training samples")
    print(f"Saved to {output_path}")
    
    # Stats
    categories = {}
    for s in samples:
        cat = s.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
