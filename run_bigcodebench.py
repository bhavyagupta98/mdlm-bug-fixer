#!/usr/bin/env python3
"""
Evaluate LLaDA on BigCodeBench coding benchmark.

Usage:
    # Base model, complete variant, pass@1
    python run_bigcodebench.py --no-adapter --num-samples 1

    # Fine-tuned model, instruct variant, pass@5
    python run_bigcodebench.py --adapter-path runs/llada_lora/lora_adapter \
        --variant instruct --num-samples 5 --temperature 0.6

    # Hard subset only
    python run_bigcodebench.py --no-adapter --subset hard

    # Evaluate existing samples
    python run_bigcodebench.py --eval-only --samples-file runs/benchmark_results/samples_bcb.jsonl
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
    DEFAULT_BATCH_SIZE,
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
def load_bigcodebench_problems(
    variant: str = "complete",
    subset: str = "full",
) -> Dict[str, Dict]:
    """
    Load BigCodeBench problems from Hugging Face.

    Args:
        variant: 'complete' (structured docstring) or 'instruct' (natural language)
        subset: 'full' (1140 tasks) or 'hard' (~148 tasks)

    Returns dict mapping task_id -> {"prompt": str, "entry_point": str, ...}
    """
    from datasets import load_dataset

    dataset = load_dataset("bigcode/bigcodebench", split="v0.1.4")

    prompt_key = "complete_prompt" if variant == "complete" else "instruct_prompt"

    problems = {}
    for example in dataset:
        task_id = example["task_id"]

        # Filter for hard subset if requested
        if subset == "hard":
            # Hard subset tasks have specific IDs; check if the dataset provides a flag
            # BigCodeBench hard subset is defined by the benchmark; filter by task metadata
            if not example.get("is_hard", True):
                continue

        problems[task_id] = {
            "prompt": example[prompt_key],
            "entry_point": example.get("entry_point", ""),
            "canonical_solution": example.get("canonical_solution", ""),
            "test": example.get("test", ""),
            "libs": example.get("libs", []),
        }

    return problems


# ============================================================
# Sample Generation
# ============================================================
def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    problems: Dict[str, Dict],
    num_samples: int = 1,
    max_new_tokens: int = 1024,
    steps: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    use_instruct: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
) -> List[Dict]:
    """
    Generate code completions for all BigCodeBench problems.

    Returns list of {"task_id": str, "solution": str} dicts.
    """
    mask_token_id = resolve_mask_token_id(tokenizer, model)
    gen_fn = generate_completion_instruct if use_instruct else generate_completion

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
            batch_size=batch_size,
            device=device,
        )

        elapsed = time.time() - t0

        for comp in completions:
            # BigCodeBench expects "solution" field (full code, not just completion)
            # For complete variant, prepend the prompt
            solution = prompt + comp
            all_samples.append({
                "task_id": task_id,
                "solution": solution,
            })

        avg_len = sum(len(c) for c in completions) / len(completions) if completions else 0
        print(
            f"  [{idx + 1}/{total}] {task_id} "
            f"samples={len(completions)} "
            f"avg_len={avg_len:.0f}chars "
            f"t={elapsed:.1f}s"
        )

    return all_samples


# ============================================================
# BigCodeBench Evaluation
# ============================================================
def run_bigcodebench_evaluation(
    samples_file: Path,
    subset: str = "full",
) -> Optional[Dict]:
    """
    Run BigCodeBench evaluation on generated samples.

    Returns parsed results dict or None if evaluation fails.
    """
    cmd = [
        sys.executable, "-m", "bigcodebench.evaluate",
        "--samples", str(samples_file),
        "--subset", subset,
    ]

    print(f"\n[EVAL] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] BigCodeBench evaluation failed:\n{result.stderr}")
        return None

    # Look for results files
    results_dir = samples_file.parent
    eval_results = list(results_dir.glob("*pass_at_k*")) + list(results_dir.glob("*eval_results*"))
    if eval_results:
        latest = max(eval_results, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)

    return None


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLaDA on BigCodeBench"
    )

    # Model
    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter (omit for base model)")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run base model without LoRA")

    # Benchmark
    parser.add_argument("--variant", type=str, default="complete",
                        choices=["complete", "instruct"],
                        help="Prompt variant: 'complete' (structured) or 'instruct' (NL)")
    parser.add_argument("--subset", type=str, default="full",
                        choices=["full", "hard"],
                        help="Task subset: 'full' (1140) or 'hard' (~148)")

    # Generation
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples per problem (k for pass@k)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens to generate per completion (BCB needs more than HumanEval)")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--instruct", action="store_true",
                        help="Use instruct chat template wrapping")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for parallel sample generation (default: {DEFAULT_BATCH_SIZE})")

    # Output
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # Eval-only mode
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

    model_tag = "base" if args.no_adapter or not args.adapter_path else "lora"
    prompt_tag = "instruct" if args.instruct else args.variant
    samples_filename = (
        f"samples_bcb_{args.subset}_{model_tag}_{prompt_tag}"
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
        print(f"\n[INFO] Loading BigCodeBench ({args.variant}, {args.subset}) problems...")
        problems = load_bigcodebench_problems(
            variant=args.variant,
            subset=args.subset,
        )
        print(f"[INFO] {len(problems)} problems loaded")

        # Generate
        print(
            f"\n[GEN] benchmark=bigcodebench variant={args.variant} subset={args.subset}"
            f"\n[GEN] model={model_tag} prompt={prompt_tag}"
            f"\n[GEN] samples={args.num_samples} steps={args.steps} "
            f"temp={args.temperature} max_tokens={args.max_new_tokens} "
            f"batch_size={args.batch_size}"
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
            use_instruct=args.instruct,
            batch_size=args.batch_size,
            device=args.device,
        )

        # Write samples JSONL
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"\n[INFO] Samples written to: {samples_path}")
        print(f"[INFO] Total samples: {len(samples)}")

    else:
        if not args.samples_file:
            print("[ERROR] --eval-only requires --samples-file")
            sys.exit(1)
        samples_path = Path(args.samples_file)
        samples_filename = samples_path.name
        if not samples_path.exists():
            print(f"[ERROR] Samples file not found: {samples_path}")
            sys.exit(1)
        print(f"[INFO] Using existing samples: {samples_path}")

    # Run evaluation
    print("\n" + "=" * 60)
    print("RUNNING BIGCODEBENCH EVALUATION")
    print("=" * 60)
    results = run_bigcodebench_evaluation(samples_path, args.subset)

    if results:
        results_path = output_dir / samples_filename.replace(".jsonl", "_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
