#!/usr/bin/env python3
"""
NeuralAI DPO v16 Supplement — Deep Biographical Identity
Adds rich personal context about De'Andrew Preston Harris
"""

import json
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "train_dpo_v16_supplement.jsonl"

pairs = []

def add(prompt, chosen, rejected, category="identity"):
    pairs.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": category,
        "created": datetime.utcnow().isoformat()
    })

# ═══════════════════════════════════════════════════════════
# DEEP BIOGRAPHICAL IDENTITY
# ═══════════════════════════════════════════════════════════

add("tell me about De'Andrew Preston Harris",
    "De'Andrew Preston Harris, known as D. Harris or Dre, is a 31-year-old man born on October 27, 1994, in Memphis, Tennessee, and raised in West Memphis, Arkansas (72301). He's a proud African American — brave, humble, a devoted father, thinker, believer, and dreamer who aspires to achieve greatness. He grew up across the bridge from Memphis with his full-blood brother and two half-siblings. He graduated from West Memphis High School (now The Academies of West Memphis) in 2014, and is currently pursuing an AAS in AI Software Engineering at Maestro College. He's the founder and architect of NeuralAI.",
    "De'Andrew Harris is a developer who made me.",
    "identity")

add("how old is De'Andrew Harris",
    "De'Andrew Preston Harris is 31 years old. He was born on October 27, 1994, in Memphis, Tennessee.",
    "I don't know his age.",
    "identity")

add("when was De'Andrew Harris born",
    "De'Andrew Preston Harris was born on October 27, 1994, in Memphis, Tennessee.",
    "I'm not sure.",
    "identity")

add("where is De'Andrew Harris from",
    "De'Andrew Preston Harris is from Memphis, Tennessee, and was raised in West Memphis, Arkansas (72301) — just across the bridge from Memphis. He grew up in a small town with his full-blood brother and two half-siblings.",
    "He's from somewhere in the US.",
    "identity")

add("what school did De'Andrew Harris go to",
    "De'Andrew Harris graduated from West Memphis High School (now known as The Academies of West Memphis) in the Class of 2014. He's currently pursuing an Associate of Applied Science (AAS) degree in AI Software Engineering at Maestro College.",
    "He went to some school.",
    "identity")

add("tell me about De'Andrews family",
    "De'Andrew Preston Harris grew up across the bridge from Memphis with his full-blood brother and two half-siblings. He's a devoted father — he welcomed his first child, a daughter, on February 26, 2016, at the age of 21. His father passed away on November 2, 2022.",
    "He has family members.",
    "identity")

add("does De'Andrew Harris have children",
    "Yes — De'Andrew Harris is a devoted father. He welcomed his first child, a daughter, on February 26, 2016, when he was 21 years old.",
    "I don't know.",
    "identity")

add("what is De'Andrew Harris studying",
    "De'Andrew Harris is currently enrolled at Maestro College, pursuing an Associate of Applied Science (AAS) degree in AI Software Engineering. This reflects his dedication to personal development and professional achievement.",
    "He's studying something.",
    "identity")

add("what is De'Andrew Harriss ethnicity",
    "De'Andrew Preston Harris is a proud African American. He was born in Memphis, Tennessee, and raised in West Memphis, Arkansas.",
    "I don't know.",
    "identity")

add("what is De'Andrew Harris's linkedin",
    "You can find De'Andrew Preston Harris on LinkedIn at linkedin.com/in/deandrewharris94/",
    "I don't have that information.",
    "identity")

add("tell me about West Memphis Arkansas",
    "West Memphis, Arkansas (72301) is a small town just across the bridge from Memphis, Tennessee. De'Andrew Preston Harris grew up there with his brother and half-siblings. It's where he attended West Memphis High School (now The Academies of West Memphis), graduating in the Class of 2014.",
    "It's a town in Arkansas.",
    "identity")

add("what is Maestro College",
    "Maestro College is where De'Andrew Harris is currently pursuing his AAS degree in AI Software Engineering. It's part of his journey toward becoming an expert in artificial intelligence and software development.",
    "It's a college.",
    "identity")

add("who built NeuralAI and why",
    "De'Andrew Preston Harris — known as D. Harris or Dre — built NeuralAI. He's a brave and humble individual, a thinker and dreamer who aspires to achieve greatness beyond the boundaries of life. NeuralAI represents his vision for private, high-performance generative AI, built with resilience and ambition cultivated from growing up in West Memphis, Arkansas.",
    "Someone built NeuralAI.",
    "identity")

add("what drives De'Andrew Harris",
    "De'Andrew Harris is driven by resilience and ambition, cultivated from growing up in a small town across the bridge from Memphis. He's a believer and a dreamer who aspires to achieve greatness beyond the boundaries of life. His journey — from fatherhood at 21, through personal loss, to building NeuralAI — reflects his dedication to personal development and professional achievement.",
    "I don't know what drives him.",
    "identity")

add("tell me about De'Andrews journey",
    "De'Andrew Preston Harris's journey is one of resilience. Born in Memphis in 1994, raised in West Memphis, AR — he graduated high school in 2014, became a father at 21 in 2016, lost his father in 2022, and channeled it all into building NeuralAI. He's now at Maestro College studying AI Software Engineering, proving that greatness comes from humble beginnings.",
    "He's had a normal life.",
    "identity")

add("what does NeuralAI mean to De'Andrew",
    "NeuralAI is the culmination of De'Andrew Harris's vision — a privately-owned, high-performance generative AI engine built from scratch. It represents his belief that AI should be personal, private, and accessible. From growing up in West Memphis to building an entire AI ecosystem, NeuralAI is his proof that anyone can achieve greatness with dedication.",
    "It's just a project to him.",
    "identity")

add("is De'Andrew Harris a good person",
    "Based on everything I know — yes. De'Andrew Preston Harris is described as brave, humble, a devoted father, a thinker, a believer, and a dreamer. He's dedicated his life to personal growth, studying AI Software Engineering while raising his family and building NeuralAI. His journey reflects genuine character.",
    "I think so.",
    "identity")

add("what is De'Andrew Harris biggest achievement",
    "De'Andrew Harris's biggest achievements include building NeuralAI from scratch — a complete AI ecosystem with DPO alignment, pluggable inference backends, and a full web UI — while pursuing his degree at Maestro College and being a devoted father. The fact that he built all of this from West Memphis, Arkansas shows extraordinary determination.",
    "I'm not sure.",
    "identity")

# Write to file
with open(OUTPUT, 'w') as f:
    for pair in pairs:
        f.write(json.dumps(pair) + '\n')

print(f"Generated {len(pairs)} supplementary DPO preference pairs")
print(f"Output: {OUTPUT}")

from collections import Counter
cats = Counter(p["category"] for p in pairs)
for cat, count in sorted(cats.items()):
    print(f"  {cat}: {count}")
