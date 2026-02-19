#!/usr/bin/env python3
"""
Inference and evaluation for LLaDA LoRA fine-tuned on multi-hunk Java bug repair.

Implements custom iterative denoising for infilling known masked regions.
"""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from eval_metrics import (
    token_exact_match_rate,
    per_hunk_exact_match,
    all_hunks_correct,
    masked_cross_entropy,
    token_edit_distance,
    top_k_accuracy,
    compute_codebleu,
    MetricsAggregator,
)


# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "processed_train.jsonl.gz"
DEFAULT_ADAPTER_PATH = BASE_DIR / "runs" / "llada_lora" / "lora_adapter"
DEFAULT_OUTPUT_PATH = BASE_DIR / "runs" / "eval_results"
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


# ============================================================
# HELPERS (reused from model.py)
# ============================================================
def _filter_spans(spans: List[List[int]], L: int) -> List[Tuple[int, int]]:
    """Clamp spans to [0, L]."""
    filtered: List[Tuple[int, int]] = []
    for s, e in spans:
        s = int(s); e = int(e)
        if s < 0:
            s = 0
        if e > L:
            e = L
        if e > s:
            filtered.append((s, e))
    return filtered


def record_split(record_id: str, test_fraction: float = 0.1, seed: int = 42) -> str:
    """Deterministic hash-based train/test assignment."""
    h = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()
    if int(h[:8], 16) / 0xFFFFFFFF < test_fraction:
        return "test"
    return "train"


# ============================================================
# DENOISING UTILITIES
# ============================================================
def compute_transfer_schedule(num_masked: int, steps: int) -> List[int]:
    """
    Distribute num_masked tokens across `steps` denoising steps.
    Linear schedule: floor(num_masked/steps) per step, +1 for first remainder steps.
    """
    if steps <= 0 or num_masked <= 0:
        return []
    steps = min(steps, num_masked)
    base = num_masked // steps
    remainder = num_masked % steps
    return [base + (1 if i < remainder else 0) for i in range(steps)]


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Gumbel-max trick for sampling. Uses float64 for numerical stability.
    When temperature=0, returns logits unchanged (greedy).
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-10)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def infill_denoise(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask_positions: torch.Tensor,
    mask_token_id: int,
    steps: int = 64,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    return_logits: bool = False,
) -> Dict[str, Any]:
    """
    Iterative denoising for infilling known masked regions.

    Args:
        model: LLaDA model (or LoRA-merged model)
        input_ids: (1, seq_len) with mask_token_id at hunk positions
        mask_positions: (seq_len,) boolean, True where masked
        mask_token_id: token ID used for masking
        steps: number of denoising iterations
        temperature: sampling temperature (0=greedy)
        remasking: 'low_confidence' or 'random'
        return_logits: whether to return final-step logits

    Returns dict with predicted_ids, final_logits, steps_taken, confidence_history.
    """
    x = input_ids.clone()
    num_masked = mask_positions.sum().item()

    if num_masked == 0:
        return {
            "predicted_ids": x,
            "final_logits": None,
            "steps_taken": 0,
            "confidence_history": [],
        }

    effective_steps = min(steps, num_masked)
    schedule = compute_transfer_schedule(num_masked, effective_steps)

    still_masked = mask_positions.clone()
    confidence_history = []
    logits = None

    for t, k_this_step in enumerate(schedule):
        # Forward pass
        logits = model(x, attention_mask=torch.ones_like(x)).logits  # (1, seq_len, V)

        # Sample predictions
        if temperature == 0.0:
            predicted_tokens = logits.argmax(dim=-1)  # (1, seq_len)
        else:
            noisy_logits = add_gumbel_noise(logits[0], temperature)
            predicted_tokens = noisy_logits.argmax(dim=-1).unsqueeze(0)

        # Compute confidence for still-masked positions
        if remasking == "low_confidence":
            probs = F.softmax(logits.float(), dim=-1)
            conf = probs.gather(-1, predicted_tokens.unsqueeze(-1)).squeeze(-1).squeeze(0)
        else:  # random
            conf = torch.rand(x.size(1), device=x.device)

        # Only consider still-masked positions
        conf[~still_masked] = -float("inf")

        valid_conf = conf[still_masked]
        if valid_conf.numel() > 0:
            confidence_history.append(valid_conf.mean().item())

        # Last step: unmask everything remaining
        if t == len(schedule) - 1:
            transfer_indices = still_masked.nonzero(as_tuple=True)[0]
        else:
            _, transfer_indices = torch.topk(conf, k=k_this_step)

        # Commit predictions at selected positions
        x[0, transfer_indices] = predicted_tokens[0, transfer_indices]
        still_masked[transfer_indices] = False

        # Re-mask remaining positions
        x[0, still_masked] = mask_token_id

    return {
        "predicted_ids": x,
        "final_logits": logits if return_logits else None,
        "steps_taken": len(schedule),
        "confidence_history": confidence_history,
    }


@torch.no_grad()
def compute_loss_single_pass(
    model: torch.nn.Module,
    masked_input: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    mask_positions: torch.Tensor,
) -> float:
    """
    Single forward pass CE loss on masked positions.
    Directly comparable to training loss from MDLMTrainer.
    """
    logits = model(masked_input, attention_mask=torch.ones_like(masked_input)).logits
    labels = ground_truth_ids.clone()
    labels[0, ~mask_positions] = -100
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss.item()


# ============================================================
# MODEL LOADING
# ============================================================
def load_model_and_tokenizer(
    base_model_name: str = MODEL_NAME,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Load base LLaDA model, optionally with LoRA adapter merged."""
    print("[INFO] Loading tokenizer:", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True, use_fast=True
    )

    print(f"[INFO] Loading model: {base_model_name} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"[INFO] Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("[INFO] Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return model, tokenizer


# ============================================================
# DATA LOADING
# ============================================================
def stream_records(
    data_path: Path,
    split: str = "test",
    test_fraction: float = 0.1,
    split_seed: int = 42,
    max_records: Optional[int] = None,
    max_seq_len: int = 2048,
) -> Iterator[Dict[str, Any]]:
    """Stream records from processed JSONL gz, filtering by split."""
    count = 0
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue

            record_id = ex.get("id", "NA")

            # Filter by split
            if split != "all":
                assigned = record_split(record_id, test_fraction, split_seed)
                if assigned != split:
                    continue

            input_ids = ex.get("input_ids", [])
            spans = ex.get("hunk_token_spans", [])
            mask_token_id = ex.get("mask_token_id")

            if not input_ids or not spans or mask_token_id is None:
                continue

            # Cap to max_seq_len
            input_ids = input_ids[:max_seq_len]
            L = len(input_ids)
            filtered = _filter_spans(spans, L)

            if not filtered:
                continue

            yield {
                "id": record_id,
                "input_ids": input_ids,
                "hunk_token_spans": filtered,
                "mask_token_id": int(mask_token_id),
            }

            count += 1
            if max_records is not None and count >= max_records:
                return


# ============================================================
# MASKING
# ============================================================
def prepare_masked_input(
    record: Dict[str, Any],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]], int]:
    """
    Create masked input tensor from a processed record.

    Returns:
        masked_input_ids: (1, seq_len)
        ground_truth_ids: (1, seq_len)
        mask_positions: (seq_len,) boolean
        filtered_spans: List of (start, end)
        mask_token_id: int
    """
    input_ids = record["input_ids"]
    spans = record["hunk_token_spans"]
    mask_token_id = record["mask_token_id"]

    gt = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    masked = gt.clone()
    mask_pos = torch.zeros(len(input_ids), dtype=torch.bool, device=device)

    for s, e in spans:
        masked[0, s:e] = mask_token_id
        mask_pos[s:e] = True

    return masked, gt, mask_pos, spans, mask_token_id


# ============================================================
# EVALUATION LOOP
# ============================================================
def evaluate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    data_path: Path,
    split: str = "test",
    steps: int = 64,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    max_records: Optional[int] = None,
    max_seq_len: int = 2048,
    test_fraction: float = 0.1,
    split_seed: int = 42,
    top_k: int = 5,
    compute_codebleu_flag: bool = False,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Main evaluation loop. Returns summary metrics dict."""
    agg = MetricsAggregator()

    results_path = None
    results_file = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"results_{split}_{steps}steps.jsonl"
        results_file = open(results_path, "w")

    print(f"\n[EVAL] split={split} steps={steps} temp={temperature} remasking={remasking}")
    print(f"[EVAL] data={data_path}")

    for i, record in enumerate(stream_records(
        data_path, split, test_fraction, split_seed, max_records, max_seq_len
    )):
        t0 = time.time()

        masked, gt, mask_pos, spans, mask_tid = prepare_masked_input(record, device)
        num_masked = mask_pos.sum().item()

        # Single-pass CE loss (training-parity metric)
        sp_ce = compute_loss_single_pass(model, masked, gt, mask_pos)

        # Iterative denoising
        result = infill_denoise(
            model, masked, mask_pos, mask_tid,
            steps=steps, temperature=temperature,
            remasking=remasking, return_logits=True,
        )

        pred_ids = result["predicted_ids"]
        pred_list = pred_ids[0].tolist()
        gt_list = gt[0].tolist()

        # Extract masked span tokens for metrics
        pred_masked = []
        gt_masked = []
        for s, e in spans:
            pred_masked.extend(pred_list[s:e])
            gt_masked.extend(gt_list[s:e])

        # Metrics
        em_rate = token_exact_match_rate(pred_masked, gt_masked)
        hunk_results = per_hunk_exact_match(pred_list, gt_list, spans)
        ahc = all_hunks_correct(hunk_results)
        ed = token_edit_distance(pred_masked, gt_masked)

        # Top-k and CE from final logits
        tk_acc = None
        denoise_ce = None
        if result["final_logits"] is not None:
            final_logits = result["final_logits"][0]  # (seq_len, V)
            masked_logits = final_logits[mask_pos]
            gt_masked_tensor = gt[0][mask_pos]
            tk_acc = top_k_accuracy(masked_logits, gt_masked_tensor, k=top_k)
            denoise_ce = masked_cross_entropy(final_logits, gt[0], mask_pos)

        # CodeBLEU (optional)
        cb = None
        if compute_codebleu_flag:
            pred_tokens = []
            gt_tokens = []
            for s, e in spans:
                pred_tokens.extend(pred_list[s:e])
                gt_tokens.extend(gt_list[s:e])
            pred_code = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            gt_code = tokenizer.decode(gt_tokens, skip_special_tokens=True)
            cb_result = compute_codebleu(pred_code, gt_code, language="java")
            if cb_result:
                cb = cb_result.get("codebleu")

        wall_time = time.time() - t0

        record_metrics = {
            "id": record["id"],
            "seq_len": len(record["input_ids"]),
            "num_hunks": len(spans),
            "total_masked_tokens": int(num_masked),
            "token_exact_match": em_rate,
            "per_hunk_exact": hunk_results,
            "all_hunks_correct": ahc,
            "single_pass_ce": sp_ce,
            "cross_entropy_loss": denoise_ce,
            "edit_distance": ed,
            "top_k_accuracy": tk_acc,
            "codebleu": cb,
            "denoising_steps": result["steps_taken"],
            "wall_time_seconds": round(wall_time, 2),
        }

        agg.add(record_metrics)

        if results_file:
            results_file.write(json.dumps(record_metrics) + "\n")
            results_file.flush()

        # Progress
        status = "ALL_CORRECT" if ahc else f"EM={em_rate:.2f}"
        print(
            f"  [{i+1}] id={record['id'][:12]}.. "
            f"hunks={len(spans)} masked={int(num_masked)} "
            f"{status} CE={sp_ce:.3f} t={wall_time:.1f}s"
        )

    if results_file:
        results_file.close()
        print(f"\n[EVAL] Per-record results: {results_path}")

    summary = agg.summary()

    if output_dir:
        summary_path = output_dir / f"summary_{split}_{steps}steps.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[EVAL] Summary saved: {summary_path}")

    return summary


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference and evaluation for LLaDA multi-hunk bug repair"
    )

    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run base model without LoRA (baseline)")

    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "all"])
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)

    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--codebleu", action="store_true")

    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"])

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    output_dir = Path(args.output_dir)

    adapter = None if args.no_adapter else args.adapter_path
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        adapter_path=adapter,
        device=device,
        dtype=dtype,
    )

    summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_path=Path(args.data_path),
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
        output_dir=output_dir,
        device=device,
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
