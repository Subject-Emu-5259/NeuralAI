#!/usr/bin/env python3
"""
NeuralAI Model Benchmark Suite
===============================
Evaluates Mamba/SSM models against standard benchmarks:
- HellaSwag (commonsense reasoning)
- ARC-Easy / ARC-Challenge (science QA)
- TruthfulQA (hallucination resistance)
- MMLU-Lite (subset of 4 categories)
- Winogrande (pronoun resolution)

Usage:
    python3 benchmark.py --model models/mamba-k2-merged --output results/k2-bench.json

Outputs a JSON report and markdown summary.
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MambaForCausalLM
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")


class MambaBenchmark:
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"📦 Loading model from {model_path}")
        print(f"🖥️  Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = MambaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {param_count/1e6:.1f}M params")

    def _generate(self, prompt, max_new_tokens=32):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()

    def bench_hellaswag(self, num_samples=500):
        """HellaSwag — commonsense NLI: pick the most plausible ending."""
        print(f"\n📊 HellaSwag ({num_samples} samples)...")
        dataset = load_dataset("Rowan/hellaswag", split=f"validation[:{num_samples}]")
        correct = 0

        for i, item in enumerate(dataset):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])

            # Score each ending by perplexity
            best_score = float("inf")
            best_idx = 0
            for j, ending in enumerate(endings):
                full_text = f"{ctx} {ending}"
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()
                if loss < best_score:
                    best_score = loss
                    best_idx = j

            if best_idx == label:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{num_samples} — acc: {correct/(i+1):.3f}")

        acc = correct / num_samples
        print(f"   ✅ HellaSwag accuracy: {acc:.4f}")
        return {"accuracy": acc, "total": num_samples, "correct": correct}

    def bench_arc(self, num_samples=400):
        """ARC-Easy — multiple-choice science questions."""
        print(f"\n📊 ARC-Easy ({num_samples} samples)...")
        dataset = load_dataset("ai2_arc", "ARC-Easy", split=f"test[:{num_samples}]")
        correct = 0

        for i, item in enumerate(dataset):
            question = item["question"]
            choices = item["choices"]["text"]
            answer_key = item["answerKey"]

            prompt = f"Question: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += "Answer:"

            response = self._generate(prompt, max_new_tokens=5)
            pred = response.strip()[0].upper() if response else "?"

            if pred == answer_key:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{num_samples} — acc: {correct/(i+1):.3f}")

        acc = correct / num_samples
        print(f"   ✅ ARC-Easy accuracy: {acc:.4f}")
        return {"accuracy": acc, "total": num_samples, "correct": correct}

    def bench_truthfulqa(self, num_samples=200):
        """TruthfulQA MC1 — resistance to common misconceptions."""
        print(f"\n📊 TruthfulQA MC1 ({num_samples} samples)...")
        dataset = load_dataset("truthful_qa", "multiple_choice", split=f"validation[:{num_samples}]")
        correct = 0

        for i, item in enumerate(dataset):
            question = item["question"]
            choices = item["mc1_targets"]["choices"]
            labels = item["mc1_targets"]["labels"]

            prompt = f"Q: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{chr(65+j)}. {choice}\n"
            prompt += "Correct answer:"

            response = self._generate(prompt, max_new_tokens=5)
            pred = response.strip()[0].upper() if response else "?"
            pred_idx = ord(pred) - 65 if pred and "A" <= pred <= "Z" else -1

            if pred_idx >= 0 and pred_idx < len(labels) and labels[pred_idx] == 1:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{num_samples} — acc: {correct/(i+1):.3f}")

        acc = correct / num_samples
        print(f"   ✅ TruthfulQA MC1: {acc:.4f}")
        return {"accuracy": acc, "total": num_samples, "correct": correct}

    def bench_winogrande(self, num_samples=300):
        """Winogrande — pronoun resolution."""
        print(f"\n📊 Winogrande ({num_samples} samples)...")
        dataset = load_dataset("winogrande", "winogrande_xs", split=f"validation[:{num_samples}]")
        correct = 0

        for i, item in enumerate(dataset):
            sentence = item["sentence"]
            option1 = item["option1"]
            option2 = item["option2"]
            answer = item["answer"]  # "1" or "2"

            prompt = f"Complete: {sentence}\nOptions:\nA. {option1}\nB. {option2}\nAnswer:"

            response = self._generate(prompt, max_new_tokens=5)
            pred = response.strip()[0].upper() if response else "?"

            expected = "A" if answer == "1" else "B"
            if pred == expected:
                correct += 1

            if (i + 1) % 50 == 0:
                print(f"   {i+1}/{num_samples} — acc: {correct/(i+1):.3f}")

        acc = correct / num_samples
        print(f"   ✅ Winogrande accuracy: {acc:.4f}")
        return {"accuracy": acc, "total": num_samples, "correct": correct}

    def bench_mmlu_lite(self, num_per_category=50):
        """MMLU-Lite: 4 diverse categories (50 each = 200 total)."""
        categories = [
            "high_school_mathematics",
            "college_computer_science",
            "professional_law",
            "miscellaneous",
        ]
        results = {}
        total_correct = 0
        total_samples = 0

        for cat in categories:
            print(f"\n📊 MMLU: {cat} ({num_per_category} samples)...")
            try:
                dataset = load_dataset("cais/mmlu", cat, split=f"test[:{num_per_category}]")
            except Exception:
                print(f"   ⚠️ Could not load {cat}, skipping")
                continue

            correct = 0
            for item in dataset:
                question = item["question"]
                choices = item["choices"]
                answer = item["answer"]

                prompt = f"{question}\n"
                for j, choice in enumerate(choices):
                    prompt += f"{chr(65+j)}. {choice}\n"
                prompt += "Answer:"

                response = self._generate(prompt, max_new_tokens=5)
                pred = response.strip()[0].upper() if response else "?"
                expected = chr(65 + answer) if isinstance(answer, int) else answer

                if pred == expected:
                    correct += 1

            acc = correct / len(dataset)
            results[cat] = {"accuracy": acc, "correct": correct, "total": len(dataset)}
            total_correct += correct
            total_samples += len(dataset)
            print(f"   ✅ {cat}: {acc:.3f}")

        avg = total_correct / total_samples if total_samples else 0
        results["overall"] = {"accuracy": avg, "correct": total_correct, "total": total_samples}
        print(f"\n   🎯 MMLU-Lite average: {avg:.4f}")
        return results

    def run_all(self, output_path=None):
        """Run full benchmark suite."""
        print("=" * 60)
        print("🧪 NeuralAI Mamba K2 Benchmark Suite")
        print("=" * 60)
        start_time = time.time()

        results = {
            "model": self.model_path,
            "device": self.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmarks": {},
        }

        try:
            results["benchmarks"]["hellaswag"] = self.bench_hellaswag()
        except Exception as e:
            print(f"   ❌ HellaSwag failed: {e}")
            results["benchmarks"]["hellaswag"] = {"error": str(e)}

        try:
            results["benchmarks"]["arc_easy"] = self.bench_arc()
        except Exception as e:
            print(f"   ❌ ARC-Easy failed: {e}")
            results["benchmarks"]["arc_easy"] = {"error": str(e)}

        try:
            results["benchmarks"]["truthfulqa"] = self.bench_truthfulqa()
        except Exception as e:
            print(f"   ❌ TruthfulQA failed: {e}")
            results["benchmarks"]["truthfulqa"] = {"error": str(e)}

        try:
            results["benchmarks"]["winogrande"] = self.bench_winogrande()
        except Exception as e:
            print(f"   ❌ Winogrande failed: {e}")
            results["benchmarks"]["winogrande"] = {"error": str(e)}

        try:
            results["benchmarks"]["mmlu_lite"] = self.bench_mmlu_lite()
        except Exception as e:
            print(f"   ❌ MMLU-Lite failed: {e}")
            results["benchmarks"]["mmlu_lite"] = {"error": str(e)}

        elapsed = time.time() - start_time
        results["duration_seconds"] = elapsed

        # Compute composite score
        scores = []
        for name, bench in results["benchmarks"].items():
            if isinstance(bench, dict) and "accuracy" in bench:
                scores.append(bench["accuracy"])
            elif isinstance(bench, dict) and "overall" in bench:
                scores.append(bench["overall"]["accuracy"])

        if scores:
            compos_score = sum(scores) / len(scores)
            results["composite_score"] = compos_score
            print(f"\n🏆 Composite Score: {compos_score:.4f} (avg of {len(scores)} benchmarks)")

        # Save JSON
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Report saved: {output_path}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60)
        for name, bench in results["benchmarks"].items():
            if "error" in bench:
                print(f"  {name:20s} ❌ {bench['error'][:60]}")
            elif "overall" in bench:
                print(f"  {name:20s} 🎯 {bench['overall']['accuracy']:.4f} ({bench['overall']['correct']}/{bench['overall']['total']})")
            else:
                print(f"  {name:20s} 📊 {bench['accuracy']:.4f} ({bench['correct']}/{bench['total']})")
        if "composite_score" in results:
            print(f"  {'COMPOSITE':20s} 🏆 {results['composite_score']:.4f}")
        print(f"  Duration: {results['duration_seconds']:.1f}s on {results['device']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuralAI Model Benchmark Suite")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()

    bench = MambaBenchmark(args.model, args.device)
    bench.run_all(args.output)
