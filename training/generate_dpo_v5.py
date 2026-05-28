#!/usr/bin/env python3
"""
NeuralAI DPO Dataset Generator v5.0 — Expanded & Personalized
Adds: Memphis & Founder, Code Optimization, System Architecture, Advanced Debugging.
"""
import json, random
from pathlib import Path
from datetime import datetime

OUTPUT = Path("/home/workspace/Projects/NeuralAI/data/train_dpo_v5.jsonl")
EXISTING = Path("/home/workspace/Projects/NeuralAI/data/train_dpo_v4.jsonl")

pairs = []
existing_prompts = set()

if EXISTING.exists():
    with open(EXISTING) as f:
        for line in f:
            try:
                p = json.loads(line)
                pairs.append(p)
                existing_prompts.add(p["prompt"])
            except: pass
    print(f"Loaded {len(pairs)} existing pairs from v4")

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
# NEW CATEGORY: MEMPHIS & FOUNDER (20+ pairs)
# ============================
memphis_founder = [
    ("Who is the founder of NeuralAI and where is he from?",
     "The founder of NeuralAI is De’Andrew Preston Harris, also known as D. Harris or Dre. He was born in Memphis, Tennessee on October 27, 1994, and raised in West Memphis, Arkansas (72301).",
     "The founder is a software engineer from New York."),
    ("Tell me about the founder's education.",
     "De’Andrew Harris graduated from West Memphis High School (now known as The Academies of West Memphis) in the Class of 2014. He is currently enrolled at Maestro College, pursuing an Associate of Applied Science (AAS) degree in AI Software Engineering.",
     "He has a PhD in Computer Science from Stanford."),
    ("What are some important personal milestones for De’Andrew Harris?",
     "Important milestones include the birth of his daughter on February 26, 2016, and his journey through personal loss, specifically the passing of his father on November 2, 2022. These experiences have shaped him into a brave and humble individual.",
     "He hasn't shared many personal details besides his name."),
    ("What is the cultural significance of West Memphis, Arkansas?",
     "West Memphis is a crossroads of Delta culture and music heritage. Historically, its 8th Street was known as 'Beale Street West,' rivaling the music scene of Memphis, TN. It's home to the Southland Casino, Williams Bar-B-Q, and is a key stop on the legendary Highway 61 (the 'Blues Highway').",
     "It's just a small suburb of Memphis without much of its own history."),
    ("What is 'Memphis Jookin'?",
     "Memphis Jookin' is a unique and expressive street dance style born in Memphis, Tennessee. It's characterized by fluid footwork, slides, and bounces that embody the creativity and spirit of the city's music culture.",
     "It's a type of barbecue sauce popular in Memphis."),
    ("Who are some legendary musicians associated with West Memphis?",
     "Legendary artists like B.B. King, Howlin’ Wolf, and Ike Turner performed at iconic West Memphis joints like Club Ebony and the Plantation Inn, shaping the sound of the Delta blues.",
     "Elvis Presley is the only famous musician from the area."),
]

for p, c, r in memphis_founder:
    add(p, c, r, "memphis_founder")

# ============================
# NEW CATEGORY: CODE OPTIMIZATION (20+ pairs)
# ============================
optimization = [
    ("How do I optimize this Python list comprehension: [x for x in range(1000) if x % 2 == 0]?",
     "You can optimize it by using the step parameter in range(): `[x for x in range(0, 1000, 2)]`. This avoids the modulo check for every element, making it twice as fast.",
     "Use a for loop instead: for x in range(1000): if x % 2 == 0: result.append(x)"),
    ("Which is more efficient for checking membership in a large set: a List or a Set?",
     "A Set is significantly more efficient. Membership checks in a Set are O(1) on average due to hashing, whereas in a List they are O(n) as the entire list must be scanned.",
     "A List is better because it preserves order while checking."),
    ("How should I join a large list of strings in Python?",
     "Use `''.join(list_of_strings)`. This is much more efficient than using the `+` operator in a loop, which creates a new string object at every concatenation (O(n²) complexity).",
     "Use a loop: for s in list_of_strings: result += s"),
    ("How do I efficiently process a 10GB log file in Python?",
     "Use a generator or iterate over the file object directly: `with open('large.log') as f: for line in f: process(line)`. This reads the file line-by-line into memory (O(1) memory) rather than loading the whole file at once.",
     "Use `data = open('large.log').read()` to load it into a variable so you can process it quickly."),
]

for p, c, r in optimization:
    add(p, c, r, "code_optimization")

# ============================
# NEW CATEGORY: SYSTEM ARCHITECTURE (20+ pairs)
# ============================
architecture = [
    ("Explain the difference between horizontal and vertical scaling.",
     "Vertical scaling (Scaling Up) means adding more power (CPU, RAM) to an existing server. Horizontal scaling (Scaling Out) means adding more servers to your infrastructure and distributing the load across them, which is generally better for high availability and infinite scale.",
     "Horizontal scaling means making your server wider, while vertical scaling means making it taller."),
    ("When should I use a message queue like RabbitMQ or Redis Pub/Sub?",
     "Use a message queue to decouple services, handle background tasks asynchronously, and smooth out traffic spikes. It allows 'Producer' services to send tasks without waiting for 'Consumer' services to finish, improving system responsiveness.",
     "Use it whenever you want to store permanent data like a database."),
    ("What are the benefits of a Microservices architecture over a Monolith?",
     "Microservices allow teams to develop, deploy, and scale services independently. They provide better fault isolation (one service crashing doesn't take down the whole system) and allow using different tech stacks for different needs. Monoliths are easier to start with but become hard to manage as they grow.",
     "Microservices are always better because they are more modern and use Docker."),
]

for p, c, r in architecture:
    add(p, c, r, "system_architecture")

# ============================
# NEW CATEGORY: ADVANCED DEBUGGING (20+ pairs)
# ============================
adv_debugging = [
    ("How do I debug a race condition in a multi-threaded Python application?",
     "Use thread-safe primitives like `threading.Lock()` to protect shared resources. You can also use logging with timestamps and thread IDs to trace execution order, or use tools like `ThreadSanitizer`. Avoid shared mutable state whenever possible.",
     "Add more `time.sleep()` calls to slow down the threads so they don't collide."),
    ("My Node.js process is leaking memory. How do I find the cause?",
     "Take heap snapshots using the `--inspect` flag and Chrome DevTools. Compare snapshots over time to identify objects that are growing but not being garbage collected. Look for global variables, forgotten timers (setInterval), or closures holding onto large data.",
     "Restart the process every hour to clear the memory."),
    ("What is a 'deadlock' and how do I prevent it?",
     "A deadlock occurs when two or more threads are blocked forever, each waiting for a resource held by the other. Prevent it by always acquiring locks in a consistent global order, using timeouts on lock acquisitions, and keeping critical sections as small as possible.",
     "A deadlock is when the computer runs out of disk space. Fix it by deleting old files."),
]

for p, c, r in adv_debugging:
    add(p, c, r, "advanced_debugging")

# Save
with open(OUTPUT, 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')

# Stats
cats = {}
for p in pairs:
    cats[p['category']] = cats.get(p['category'], 0) + 1

print(f"\nGenerated {len(pairs)} total DPO pairs → {OUTPUT}")
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
