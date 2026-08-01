#!/usr/bin/env python3
"""Lightning-fast benchmark for Mamba models using llama.cpp via subprocess."""

import subprocess, json, time, os, sys, resource
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.model_manager import MODELS


def find_llama_cli() -> str:
    """Locate llama-cli binary."""
    paths = [
        "/usr/local/bin/llama-cli",
        "/usr/bin/llama-cli",
        str(Path.home() / ".local/bin/llama-cli"),
    ]
    # Also search common llmster paths
    for p in Path("/tmp").glob("lmstudio-*/llama-cli"):
        paths.insert(0, str(p))
    for p in paths:
        if os.path.exists(p):
            return p
    return "llama-cli"


def run_quick_bench(model_id: str, llama_bin: str = None) -> dict:
    """Run a fast benchmark on a GGUF model and return metrics."""
    if llama_bin is None:
        llama_bin = find_llama_cli()

    model_info = MODELS.get(model_id, {})
    gguf_path = model_info.get("path", "")
    
    if not gguf_path or not os.path.exists(gguf_path):
        return {"error": f"GGUF not found: {gguf_path}"}

    results = {
        "model_id": model_id,
        "model_label": model_info.get("label", model_id),
        "gguf_path": gguf_path,
        "tokens_per_sec": 0,
        "memory_mb": 0,
        "first_token_ms": 0,
    }

    prompt = "The capital of France is"

    # Measure memory
    try:
        mem_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except Exception:
        mem_before = 0

    # Run generation with timing
    t0 = time.time()
    try:
        result = subprocess.run(
            [llama_bin, "-m", gguf_path, "-p", prompt, "-n", "64", "--temp", "0.0", "--no-display-prompt", "-e"],
            capture_output=True, text=True, timeout=120
        )
        elapsed = time.time() - t0

        try:
            mem_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        except Exception:
            mem_after = 0

        # Parse token count from output
        output = result.stdout + result.stderr
        tokens = 0
        for line in output.splitlines():
            if "eval time" in line.lower():
                # Extract timing
                parts = line.split()
                for i, p in enumerate(parts):
                    if "tokens" in p.lower() and i > 0:
                        try:
                            tokens = float(parts[i - 1])
                        except ValueError:
                            pass

        if tokens > 0 and elapsed > 0:
            results["tokens_per_sec"] = round(tokens / elapsed, 1)
        else:
            # Fallback: estimate from output length
            gen_text = result.stdout.strip()
            est_tokens = len(gen_text.split()) * 1.3  # rough token estimate
            if elapsed > 0:
                results["tokens_per_sec"] = round(est_tokens / elapsed, 1)

        results["memory_mb"] = round((mem_after - mem_before) / 1024, 1)
        results["output_sample"] = result.stdout.strip()[:200]
        results["status"] = "ok"

    except subprocess.TimeoutExpired:
        results["error"] = "Timeout (120s)"
        results["status"] = "timeout"
    except FileNotFoundError:
        results["error"] = f"llama-cli not found at {llama_bin}"
        results["status"] = "no_binary"
    except Exception as e:
        results["error"] = str(e)
        results["status"] = "error"

    return results


def bench_all():
    """Benchmark all GGUF models in the registry."""
    llama_bin = find_llama_cli()
    if not os.path.exists(llama_bin):
        print(f"llama-cli not found at {llama_bin}")
        print("Install llama.cpp or LM Studio to run benchmarks")
        return

    print(f"Using: {llama_bin}\n")

    for mid, info in MODELS.items():
        if not info.get("path", "").endswith(".gguf"):
            print(f"  Skipping {mid} — no GGUF path")
            continue
        if not os.path.exists(info["path"]):
            print(f"  Skipping {mid} — GGUF not found at {info['path']}")
            continue

        print(f"  Benchmarking {info['label']}...")
        r = run_quick_bench(mid, llama_bin)
        
        if r.get("status") == "ok":
            print(f"    → {r['tokens_per_sec']:.1f} tok/s, {r['memory_mb']:.0f} MB RAM")
        else:
            print(f"    → {r.get('error', 'unknown error')}")

        # Save
        out_dir = Path(__file__).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"bench_{mid}_{int(time.time())}.json"
        out_file.write_text(json.dumps(r, indent=2))
        print(f"    Saved: {out_file}")
        print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_id", nargs="?", help="Model ID to benchmark")
    p.add_argument("--all", action="store_true", help="Benchmark all GGUF models")
    args = p.parse_args()

    if args.all:
        bench_all()
    elif args.model_id:
        r = run_quick_bench(args.model_id)
        print(json.dumps(r, indent=2))
    else:
        # Default: benchmark active model
        from scripts.model_manager import active_model_id
        r = run_quick_bench(active_model_id())
        print(json.dumps(r, indent=2))
