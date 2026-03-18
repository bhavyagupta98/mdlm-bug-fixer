#!/usr/bin/env python3
"""
Separate inference/evaluation entrypoint for sub-7B diffusion experiments.

This script reuses evaluation logic from `inference.py` but adds:
- sub-7B model presets
- AutoModelForCausalLM -> AutoModel fallback
- separate default output paths
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from inference import evaluate


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PRESET = "coda-base"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed_train_coda_v0_base.jsonl.gz"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "sub7b" / "eval_results"

MODEL_PRESETS = {
    "coda-base": "Salesforce/CoDA-v0-Base",
    "coda-instruct": "Salesforce/CoDA-v0-Instruct",
    "fast-dllm-1.5b": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    "sdlm-3b-d4": "OpenGVLab/SDLM-3B-D4",
    "sdlm-3b-d8": "OpenGVLab/SDLM-3B-D8",
}


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "__")


def resolve_model_name(model_arg: str) -> str:
    return MODEL_PRESETS.get(model_arg, model_arg)


def default_adapter_path(model_arg: str) -> Path:
    if model_arg in MODEL_PRESETS:
        model_short = model_arg
    else:
        model_short = sanitize_model_name(model_arg)
    return BASE_DIR / "runs" / "sub7b" / f"{model_short}_lora" / "lora_adapter"


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Optional[Path],
    device: str,
    dtype: torch.dtype,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    print("[INFO] Loading tokenizer:", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=True)

    print(f"[INFO] Loading model: {base_model_name} (dtype={dtype})")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        print("[INFO] Loaded via AutoModelForCausalLM")
    except ValueError:
        print("[INFO] AutoModelForCausalLM failed, falling back to AutoModel")
        model = AutoModel.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    if adapter_path:
        if adapter_path.exists():
            print(f"[INFO] Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            print("[INFO] Merging LoRA weights into base model...")
            model = model.merge_and_unload()
        else:
            print(f"[WARN] Adapter path not found: {adapter_path}")
            print("[INFO] Continuing with base model only")
    else:
        print("[INFO] Running base model only (no LoRA adapter)")

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference/evaluation for sub-7B diffusion model experiments (separate from inference.py)."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "Model preset or full HF model id. "
            f"Presets: {', '.join(MODEL_PRESETS.keys())}"
        ),
    )
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--no-adapter", action="store_true")

    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)

    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--codebleu", action="store_true")

    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = resolve_model_name(args.base_model)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    if args.no_adapter:
        adapter_path = None
    elif args.adapter_path:
        adapter_path = Path(args.adapter_path)
    else:
        adapter_path = default_adapter_path(args.base_model)

    print("[INFO] Resolved model:", model_name)
    if adapter_path:
        print("[INFO] Adapter path:", adapter_path)

    model, tokenizer = load_model_and_tokenizer(
        base_model_name=model_name,
        adapter_path=adapter_path,
        device=args.device,
        dtype=dtype,
    )

    summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        split=args.split,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        max_records=args.max_records,
        max_seq_len=args.max_seq_len,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        top_k=args.top_k,
        compute_codebleu_flag=args.codebleu,
        output_dir=Path(args.output_dir),
        device=args.device,
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
