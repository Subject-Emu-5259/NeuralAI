#!/usr/bin/env python3
"""
NeuralAI DPO Dataset Generator v7.0 — Deep Expansion
Adds: Memphis Culture, Code Optimization, System Architecture, Advanced Debugging.
"""
import json, random
from pathlib import Path
from datetime import datetime

OUTPUT = Path("/home/workspace/Projects/NeuralAI/data/train_dpo_v7.jsonl")
V5 = Path("/home/workspace/Projects/NeuralAI/data/train_dpo_v5.jsonl")
V6 = Path("/home/workspace/Projects/NeuralAI/data/train_dpo_v6.jsonl")

pairs = []
existing_prompts = set()

# Load v5 and v6
for path in [V5, V6]:
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    p = json.loads(line)
                    if p["prompt"] not in existing_prompts:
                        pairs.append(p)
                        existing_prompts.add(p["prompt"])
                except: pass

print(f"Loaded {len(pairs)} existing pairs from v5 and v6")

def add(prompt, chosen, rejected, category):
    if prompt in existing_prompts:
        return
    pairs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "created": datetime.now().isoformat()
    })
    existing_prompts.add(prompt)

# ============================
# EXPANDED: MEMPHIS CULTURE & FOUNDER (Deep Context)
# ============================
memphis_founder_expanded = [
    ("Where was De’Andrew Preston Harris born and raised?", 
     "De’Andrew Preston Harris was born in Memphis, Tennessee, and raised just across the Mississippi River in West Memphis, Arkansas (ZIP code 72301).", 
     "He is originally from Atlanta, Georgia."),
    ("What are the founder's academic goals?", 
     "De’Andrew is currently pursuing an Associate of Applied Science (AAS) degree in AI Software Engineering at Maestro College to build a foundation for achieving greatness in technology.", 
     "He is studying Business Administration at a local community college."),
    ("What are some historical nicknames for West Memphis, Arkansas?", 
     "West Memphis was once known for its vibrant music scene on 8th Street, often called 'Beale Street West,' which rivaled the famous street in Memphis, TN, during the mid-20th century.", 
     "It has always been known as the 'Gateway to the West'."),
    ("Tell me about the founder's high school years.", 
     "He graduated from West Memphis High School in 2014, an institution now known as The Academies of West Memphis. He reflects on his time there as a period of resilience and ambition.", 
     "He attended a private academy in Memphis before moving to Arkansas."),
    ("What drives De’Andrew Harris to build NeuralAI?", 
     "De’Andrew is a thinker and a dreamer who aspires to achieve greatness beyond the boundaries of life. His personal journey, including fatherhood and the loss of his father, fuels his dedication to professional achievement and legacy.", 
     "He is mostly interested in the financial gains of the AI industry."),
    ("Who is the brother De’Andrew Harris grew up with?", 
     "He grew up with his full-blood brother and two half-siblings in West Memphis, cultivating strong family bonds and a sense of responsibility from an early age.", 
     "He was an only child who spent most of his time studying alone."),
    ("What makes West Memphis a unique cultural hub?", 
     "Its location at the crossroads of I-40 and I-55, combined with its proximity to the Mississippi River, makes it a vital part of the Delta. It is famous for the Southland Casino and its deep roots in blues and soul history.", 
     "It is primarily known for being a quiet residential town with no major industries."),
]

for p, c, r in memphis_founder_expanded:
    add(p, c, r, "memphis_founder")

# ============================
# EXPANDED: CODE OPTIMIZATION (Idiomatic & Efficient)
# ============================
optimization_expanded = [
    ("Is it better to use a generator expression or a list comprehension for large data?", 
     "For large datasets, use a generator expression (e.g., `(x**2 for x in range(10**7))`) to save memory. Unlike list comprehensions, generators yield items one at a time (lazy evaluation) rather than storing the entire list in RAM.", 
     "List comprehensions are always faster because they pre-calculate all values at once."),
    ("How do I avoid the 'N+1 query problem' in database operations?", 
     "Use 'Eager Loading' (e.g., `JOIN` or `select_related` in ORMs) to fetch all related records in a single query instead of executing one query for the parent and then N additional queries for each child record.", 
     "Add a cache layer to every individual query so that subsequent calls to the same record are faster."),
    ("What is the most efficient way to check for a key in a Python dictionary?", 
     "The most idiomatic and efficient way is `if key in my_dict:`. This uses the dictionary's hash table for O(1) average time complexity.", 
     "Use `if key in my_dict.keys():` because it explicitly checks the keys list."),
    ("Why is `map()` sometimes preferred over list comprehensions in Python?", 
     "In some cases, especially when applying a built-in function to a large iterable, `map()` can be slightly faster because it is implemented in C. However, list comprehensions are generally more readable and flexible for complex logic.", 
     "Map is always better because it uses multi-threading by default."),
]

for p, c, r in optimization_expanded:
    add(p, c, r, "code_optimization")

# ============================
# EXPANDED: SYSTEM ARCHITECTURE (Scalable Design)
# ============================
architecture_expanded = [
    ("When should I choose a NoSQL database over a Relational database?", 
     "Choose NoSQL (like MongoDB or Cassandra) when you have unstructured data, need horizontal scalability for massive writes, or require a flexible schema. Use Relational (SQL) when ACID compliance, complex joins, and structured data integrity are priorities.", 
     "Always use NoSQL because it is newer and faster for every use case."),
    ("What is the role of a Load Balancer in a high-traffic system?", 
     "A Load Balancer distributes incoming network traffic across multiple backend servers. This prevents any single server from becoming a bottleneck, ensures high availability through health checks, and allows for seamless scaling.", 
     "A load balancer is used to encrypt traffic before it reaches the database."),
    ("Explain the concept of 'Eventual Consistency' in distributed systems.", 
     "Eventual Consistency means that after an update, the system guarantees that all replicas will eventually become consistent, but not necessarily immediately. This is often a trade-off made in high-availability systems (CAP theorem).", 
     "It means that the system is broken but will eventually fix itself manually."),
    ("Why is caching important for application performance?", 
     "Caching (e.g., via Redis) stores frequently accessed data in high-speed memory, reducing the need to hit the primary database or perform expensive computations, which significantly lowers latency and server load.", 
     "Caching is only used to save bandwidth when the internet connection is slow."),
]

for p, c, r in architecture_expanded:
    add(p, c, r, "system_architecture")

# ============================
# EXPANDED: ADVANCED DEBUGGING (Root-Cause Analysis)
# ============================
debugging_expanded = [
    ("How do I diagnose a 'Segment Fault' in a C/C++ application?", 
     "Use a debugger like `gdb` to inspect the backtrace and find the exact line where the memory access violation occurred. Check for null pointer dereferences, out-of-bounds array access, or use-after-free errors.", 
     "Re-install the compiler and hope the error goes away on the next build."),
    ("What is the first step when a production service starts returning 503 errors?", 
     "Check the service logs and resource metrics (CPU, Memory, Disk) to see if the service crashed, is under heavy load, or is failing health checks due to a downstream dependency issue.", 
     "Immediately restart all servers without looking at the logs to clear the error state."),
    ("How do I use a 'Flame Graph' for performance debugging?", 
     "A Flame Graph visualizes where CPU time is being spent by sampling the call stack. Wide bars indicate functions that are on the CPU for a long time, allowing you to identify hotspots and bottlenecks in your code.", 
     "Flame graphs are used to visualize the temperature of the server room over time."),
]

for p, c, r in debugging_expanded:
    add(p, c, r, "advanced_debugging")

# Save
with open(OUTPUT, 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')

print(f"\nSuccessfully generated {len(pairs)} DPO pairs in v7.0")
