#!/usr/bin/env python3
"""Standard eval suite for NeuralAI models — perplexity, generation quality, and reasoning tests.

Evaluates GGUF models via llama.cpp on:
  - WikiText-2 perplexity (sample)
  - HellaSwag-lite (10 examples, accuracy)
  - Custom reasoning prompts (5 examples, pass/fail)
  - Generation quality metrics (repetition, coherence)

Usage:
  python benchmarks/run_evals.py mamba-k2
  python benchmarks/run_evals.py mamba-k2 --full
"""

import subprocess, json, time, os, sys, re, math
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.model_manager import MODELS

# ─── Test Data ────────────────────────────────────────────────────

WIKITEXT_SAMPLE = [
    "The history of the Roman Empire covers the history of ancient Rome from the fall of the Roman Republic in 27 BC until the abdication of the last Western emperor in AD 476.",
    "Quantum mechanics is a fundamental theory in physics that describes nature at the scale of atoms and subatomic particles.",
    "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789.",
]

HELLASWAG_LITE = [
    {"ctx": "A man is playing a guitar. He", "endings": ["strums the strings gently.", "eats the guitar.", "flies away.", "goes to sleep."], "label": 0},
    {"ctx": "A woman is cooking in the kitchen. She", "endings": ["takes out a book.", "stirs the pot on the stove.", "jumps out the window.", "starts singing opera."], "label": 1},
    {"ctx": "The dog is running in the park. It", "endings": ["reads a newspaper.", "chases the ball.", "drives a car.", "types on a keyboard."], "label": 1},
    {"ctx": "A student is studying for an exam. They", "endings": ["go skydiving instead.", "review their notes carefully.", "burn the textbook.", "hire a marching band."], "label": 1},
    {"ctx": "The chef prepared the meal. Then he", "endings": ["served it to the guests.", "threw it in the trash.", "ran a marathon.", "bought a new hat."], "label": 0},
    {"ctx": "A programmer is debugging code. She", "endings": ["takes a nap on the keyboard.", "traces through the logic step by step.", "orders pizza for the whole building.", "quits and becomes a farmer."], "label": 1},
    {"ctx": "The car wouldn't start this morning. The driver", "endings": ["called for roadside assistance.", "painted the car green.", "started a dance party.", "read a novel about cars."], "label": 0},
    {"ctx": "It started raining heavily. People on the street", "endings": ["opened their umbrellas.", "started sunbathing.", "threw their wallets away.", "began reciting poetry."], "label": 0},
    {"ctx": "The scientist made a discovery in the lab. She", "endings": ["published her findings in a journal.", "ate the experiment.", "bought lottery tickets.", "joined a circus."], "label": 0},
    {"ctx": "During the concert, the singer forgot the lyrics. The band", "endings": ["kept playing while the singer recovered.", "walked off stage immediately.", "set the instruments on fire.", "started a food fight."], "label": 0},
]

REASONING_TESTS = [
    {
        "prompt": "If all dogs are mammals and all mammals are animals, are all dogs animals? Answer yes or no and explain.",
        "expected_contains": ["yes", "animal"],
    },
    {
        "prompt": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "expected_contains": ["0.05", "5 cents", "five cents"],
    },
    {
        "prompt": "Complete this sequence: 2, 4, 8, 16,",
        "expected_contains": ["32"],
    },
    {
        "prompt": "If you have 3 apples and give away 2, how many do you have left?",
        "expected_contains": ["1", "one"],
    },
    {
        "prompt": "What is the opposite of 'generous'? Answer with one word.",
        "expected_contains": ["stingy", "greedy", "selfish", "miserly"],
    },
]

# ─── Perplexity ────────────────────────────────────────────────────

def compute_perplexity(llama_bin: str, model_path: str, texts: list[str]) -> dict:
    """Compute perplexity on sample texts using llama-perplexity or llama-cli eval."""
    log_probs = []
    total_tokens = 0

    for text in texts:
        try:
            result = subprocess.run(
                [llama_bin, "-m", model_path, "-p", text, "-n", "1", "--temp", "0.0",
                 "--log-disable", "--perplexity"],
                capture_output=True, text=True, timeout=60
            )
            output = result.stdout + result.stderr
            
            # Try to parse perplexity from output
            ppl_match = re.search(r"perplexity.*?([\d.]+)", output, re.IGNORECASE)
            if ppl_match:
                log_probs.append(float(ppl_match.group(1)))
            else:
                # Estimate from loss if available
                loss_match = re.search(r"loss.*?([\d.]+)", output, re.IGNORECASE)
                if loss_match:
                    log_probs.append(math.exp(float(loss_match.group(1))))
        except Exception:
            continue

    if not log_probs:
        return {"perplexity": None, "error": "Failed to compute perplexity"}

    return {
        "perplexity": round(sum(log_probs) / len(log_probs), 2),
        "num_texts": len(log_probs),
    }


# ─── HellaSwag ─────────────────────────────────────────────────────

def eval_hellaswag(llama_bin: str, model_path: str, examples: list[dict]) -> dict:
    """Evaluate on HellaSwag-lite by scoring completions."""
    correct = 0
    total = 0

    for ex in examples:
        # Simple approach: generate continuation and check which ending is closest
        try:
            result = subprocess.run(
                [llama_bin, "-m", model_path, "-p", ex["ctx"], "-n", "16",
                 "--temp", "0.0", "--no-display-prompt", "-e"],
                capture_output=True, text=True, timeout=30
            )
            generated = result.stdout.strip()

            # Check if any correct ending is contained in generation
            correct_ending = ex["endings"][ex["label"]]
            if any(word in generated.lower() for word in correct_ending.split()[:3]):
                correct += 1
            total += 1
        except Exception:
            pass

    return {
        "hellaswag_accuracy": round(correct / total, 3) if total > 0 else 0,
        "hellaswag_examples": total,
    }


# ─── Reasoning ─────────────────────────────────────────────────────

def eval_reasoning(llama_bin: str, model_path: str, tests: list[dict]) -> dict:
    """Evaluate on custom reasoning prompts."""
    passed = 0
    total = 0
    details = []

    for test in tests:
        try:
            result = subprocess.run(
                [llama_bin, "-m", model_path, "-p", test["prompt"], "-n", "128",
                 "--temp", "0.2", "--no-display-prompt", "-e"],
                capture_output=True, text=True, timeout=60
            )
            output = result.stdout.strip().lower()
            
            passed_test = any(exp.lower() in output for exp in test["expected_contains"])
            if passed_test:
                passed += 1
            total += 1
            details.append({
                "prompt": test["prompt"][:80],
                "passed": passed_test,
                "sample": output[:100],
            })
        except Exception as e:
            details.append({"prompt": test["prompt"][:80], "error": str(e)})

    return {
        "reasoning_accuracy": round(passed / total, 3) if total > 0 else 0,
        "reasoning_total": total,
        "reasoning_details": details,
    }


# ─── Generation Quality ────────────────────────────────────────────

def eval_generation_quality(llama_bin: str, model_path: str) -> dict:
    """Measure repetition and coherence."""
    prompts = [
        "Explain how photosynthesis works in simple terms.",
        "Write a short poem about the ocean.",
        "What are three benefits of regular exercise?",
    ]

    total_repetition = 0.0
    results = []

    for prompt in prompts:
        try:
            result = subprocess.run(
                [llama_bin, "-m", model_path, "-p", prompt, "-n", "128",
                 "--temp", "0.7", "--no-display-prompt", "-e"],
                capture_output=True, text=True, timeout=60
            )
            output = result.stdout.strip()
            words = output.split()
            
            # Measure repetition: % of duplicate trigrams
            if len(words) >= 3:
                trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                unique = len(set(trigrams))
                repetition = 1.0 - (unique / len(trigrams)) if trigrams else 0
                total_repetition += repetition
            else:
                repetition = 0

            results.append({
                "prompt": prompt[:60],
                "length_words": len(words),
                "repetition_score": round(repetition, 3),
                "sample": output[:150],
            })
        except Exception:
            pass

    avg_rep = total_repetition / len(results) if results else 0

    return {
        "avg_repetition": round(avg_rep, 3),
        "generation_samples": results,
        "repetition_grade": "good" if avg_rep < 0.15 else "warning" if avg_rep < 0.3 else "poor",
    }


# ─── Main ──────────────────────────────────────────────────────────

def run_full_eval(model_id: str, llama_bin: str = None) -> dict:
    """Run the complete eval suite and return a report."""
    if llama_bin is None:
        llama_bin = "llama-cli"

    model_info = MODELS.get(model_id, {})
    gguf_path = model_info.get("path", "")

    if not gguf_path or not os.path.exists(gguf_path):
        return {"error": f"GGUF not found: {gguf_path}"}

    is_mamba = "mamba" in model_id.lower()

    report = {
        "model_id": model_id,
        "model_label": model_info.get("label", model_id),
        "gguf_path": gguf_path,
        "architecture": "Mamba SSM" if is_mamba else "Transformer",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    print(f"🧪 Evaluating {model_info.get('label', model_id)}...")

    # 1. Perplexity
    print("  📖 Perplexity...")
    report["perplexity"] = compute_perplexity(llama_bin, gguf_path, WIKITEXT_SAMPLE)

    # 2. HellaSwag
    print("  🎯 HellaSwag...")
    report["hellaswag"] = eval_hellaswag(llama_bin, gguf_path, HELLASWAG_LITE)

    # 3. Reasoning
    print("  🧠 Reasoning...")
    report["reasoning"] = eval_reasoning(llama_bin, gguf_path, REASONING_TESTS)

    # 4. Generation Quality
    print("  ✨ Generation quality...")
    report["generation"] = eval_generation_quality(llama_bin, gguf_path)

    # Summary score
    scores = []
    if report["perplexity"].get("perplexity"):
        scores.append(min(1.0, 30.0 / report["perplexity"]["perplexity"]))
    scores.append(report["hellaswag"].get("hellaswag_accuracy", 0))
    scores.append(report["reasoning"].get("reasoning_accuracy", 0))

    report["composite_score"] = round(sum(scores) / len(scores) * 100, 1) if scores else 0
    report["grade"] = (
        "A" if report["composite_score"] >= 80 else
        "B" if report["composite_score"] >= 60 else
        "C" if report["composite_score"] >= 40 else
        "D"
    )

    # Save report
    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{model_id}_{int(time.time())}.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\n  💾 Report: {out_file}")
    print(f"  🏆 Composite: {report['composite_score']:.1f}% — Grade {report['grade']}")

    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_id", help="Model ID to evaluate")
    p.add_argument("--full", action="store_true", help="Run full suite (default: quick bench only)")
    args = p.parse_args()

    if not args.full:
        # Quick bench
        from quick_bench import run_quick_bench
        r = run_quick_bench(args.model_id)
        print(json.dumps(r, indent=2))
    else:
        run_full_eval(args.model_id)
