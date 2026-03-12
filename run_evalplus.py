#!/usr/bin/env python3
"""
Evaluate LLaDA on HumanEval/MBPP coding benchmarks via EvalPlus.

Usage:
    # Base model, HumanEval, pass@1 (greedy)
    python run_evalplus.py --benchmark humaneval --no-adapter --num-samples 1

    # Fine-tuned model, MBPP, pass@10
    python run_evalplus.py --benchmark mbpp --adapter-path runs/llada_lora/lora_adapter \
        --num-samples 10 --temperature 0.6

    # Instruct-wrapped prompting
    python run_evalplus.py --benchmark humaneval --no-adapter --instruct

    # Evaluate after generation (if samples already exist)
    python run_evalplus.py --eval-only --samples-file runs/benchmark_results/samples.jsonl \
        --benchmark humaneval
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from benchmark_generate import (
    generate_completion,
    generate_completion_instruct,
    resolve_mask_token_id,
)
from inference import MODEL_NAME, load_model_and_tokenizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "benchmark_results"


# ============================================================
# Dataset Loading
# ============================================================
def load_benchmark_problems(benchmark: str) -> Dict[str, Dict]:
    """
    Load HumanEval or MBPP problems from the evalplus package.

    Returns dict mapping task_id -> {"prompt": str, "entry_point": str, ...}
    """
    if benchmark in ("humaneval", "humaneval+"):
        from evalplus.data import get_human_eval_plus

        return get_human_eval_plus()
    elif benchmark in ("mbpp", "mbpp+"):
        from evalplus.data import get_mbpp_plus

        return get_mbpp_plus()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Use 'humaneval' or 'mbpp'.")


# ============================================================
# Sample Generation
# ============================================================
def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    problems: Dict[str, Dict],
    num_samples: int = 1,
    max_new_tokens: int = 512,
    steps: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    instruct: bool = False,
    device: str = "cuda",
) -> List[Dict]:
    """
    Generate code completions for all benchmark problems.

    Returns list of {"task_id": str, "completion": str} dicts.
    """
    mask_token_id = resolve_mask_token_id(tokenizer)
    gen_fn = generate_completion_instruct if instruct else generate_completion

    all_samples = []
    total = len(problems)

    for idx, (task_id, problem) in enumerate(problems.items()):
        prompt = problem["prompt"]
        t0 = time.time()

        completions = gen_fn(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            mask_token_id=mask_token_id,
            steps=steps,
            temperature=temperature,
            remasking=remasking,
            num_samples=num_samples,
            device=device,
        )

        elapsed = time.time() - t0

        for comp in completions:
            all_samples.append({
                "task_id": task_id,
                "completion": comp,
            })

        # Progress
        print(
            f"  [{idx + 1}/{total}] {task_id} "
            f"samples={len(completions)} "
            f"avg_len={sum(len(c) for c in completions) / len(completions):.0f}chars "
            f"t={elapsed:.1f}s"
        )

    return all_samples


# ============================================================
# EvalPlus Evaluation
# ============================================================
def run_evalplus_evaluation(
    samples_file: Path,
    benchmark: str,
) -> Optional[Dict]:
    """
    Run EvalPlus evaluation on generated samples.

    Returns parsed results dict or None if evaluation fails.
    """
    dataset = "humaneval" if "humaneval" in benchmark else "mbpp"

    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", dataset,
        "--samples", str(samples_file),
    ]

    print(f"\n[EVAL] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] EvalPlus failed:\n{result.stderr}")
        return None

    # Try to find the results file that EvalPlus generates
    results_dir = samples_file.parent
    eval_results = list(results_dir.glob("*eval_results*"))
    if eval_results:
        with open(eval_results[-1]) as f:
            return json.load(f)

    return None


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLaDA on HumanEval/MBPP via EvalPlus"
    )

    # Model
    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (omit for base model)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run base model without LoRA")

    # Benchmark
    parser.add_argument("--benchmark", type=str, default="humaneval",
                        choices=["humaneval", "mbpp"],
                        help="Which benchmark to run")

    # Generation
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per problem (k for pass@k)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per completion")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--instruct", action="store_true",
                        help="Use instruct chat template wrapping")

    # Output
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # Eval-only mode (skip generation, just evaluate existing samples)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, evaluate existing samples file")
    parser.add_argument("--samples-file", type=str, default=None,
                        help="Path to existing samples JSONL (for --eval-only)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"])

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a descriptive filename
    model_tag = "base" if args.no_adapter or not args.adapter_path else "lora"
    prompt_tag = "instruct" if args.instruct else "raw"
    samples_filename = (
        f"samples_{args.benchmark}_{model_tag}_{prompt_tag}"
        f"_n{args.num_samples}_s{args.steps}_t{args.temperature}.jsonl"
    )
    samples_path = output_dir / samples_filename

    if not args.eval_only:
        # Load model
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        adapter = None if args.no_adapter else args.adapter_path
        model, tokenizer = load_model_and_tokenizer(
            base_model_name=args.base_model,
            adapter_path=adapter,
            device=args.device,
            dtype=dtype,
        )

        # Load problems
        print(f"\n[INFO] Loading {args.benchmark} problems...")
        problems = load_benchmark_problems(args.benchmark)
        print(f"[INFO] {len(problems)} problems loaded")

        # Generate
        print(
            f"\n[GEN] benchmark={args.benchmark} model={model_tag} prompt={prompt_tag}"
            f"\n[GEN] samples={args.num_samples} steps={args.steps} "
            f"temp={args.temperature} max_tokens={args.max_new_tokens}"
        )

        samples = generate_samples(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=args.temperature,
            remasking=args.remasking,
            instruct=args.instruct,
            device=args.device,
        )

        # Write samples JSONL
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"\n[INFO] Samples written to: {samples_path}")
        print(f"[INFO] Total samples: {len(samples)}")

    else:
        # Eval-only: use provided samples file
        if args.samples_file:
            samples_path = Path(args.samples_file)
        if not samples_path.exists():
            print(f"[ERROR] Samples file not found: {samples_path}")
            sys.exit(1)
        print(f"[INFO] Using existing samples: {samples_path}")

    # Run EvalPlus evaluation
    print("\n" + "=" * 60)
    print("RUNNING EVALPLUS EVALUATION")
    print("=" * 60)
    results = run_evalplus_evaluation(samples_path, args.benchmark)

    if results:
        results_path = output_dir / samples_filename.replace(".jsonl", "_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
