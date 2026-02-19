#!/usr/bin/env python3
"""
Metric computation for masked diffusion hunk infilling evaluation.
All functions are stateless and operate on plain Python lists/tensors.
"""

from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F


# ============================================================
# Token-level Metrics
# ============================================================

def token_exact_match_rate(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
) -> float:
    """Fraction of predicted tokens matching ground truth (same-length lists)."""
    if not predicted_ids:
        return 1.0
    matches = sum(p == g for p, g in zip(predicted_ids, ground_truth_ids))
    return matches / len(ground_truth_ids)


def per_hunk_exact_match(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
    hunk_spans: List[Tuple[int, int]],
) -> List[bool]:
    """For each hunk span, check if ALL tokens in that span match exactly."""
    results = []
    for s, e in hunk_spans:
        match = predicted_ids[s:e] == ground_truth_ids[s:e]
        results.append(match)
    return results


def all_hunks_correct(hunk_results: List[bool]) -> bool:
    """True if every hunk was exactly reconstructed."""
    return all(hunk_results) if hunk_results else True


# ============================================================
# Loss Metric
# ============================================================

def masked_cross_entropy(
    logits: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    mask_positions: torch.Tensor,
) -> float:
    """
    Cross-entropy loss on masked positions only.
    logits: (seq_len, vocab_size)
    ground_truth_ids: (seq_len,)
    mask_positions: (seq_len,) boolean
    """
    if mask_positions.sum() == 0:
        return 0.0
    masked_logits = logits[mask_positions]
    masked_labels = ground_truth_ids[mask_positions]
    loss = F.cross_entropy(masked_logits, masked_labels)
    return loss.item()


# ============================================================
# Edit Distance
# ============================================================

def token_edit_distance(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
) -> int:
    """Levenshtein edit distance between two token sequences."""
    n, m = len(predicted_ids), len(ground_truth_ids)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if predicted_ids[i - 1] == ground_truth_ids[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


# ============================================================
# Top-k Accuracy
# ============================================================

def top_k_accuracy(
    logits: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Fraction of positions where ground truth is in the top-k predictions.
    logits: (num_positions, vocab_size)
    ground_truth_ids: (num_positions,)
    """
    if logits.size(0) == 0:
        return 1.0
    _, topk_ids = logits.topk(k, dim=-1)
    matches = (topk_ids == ground_truth_ids.unsqueeze(-1)).any(dim=-1)
    return matches.float().mean().item()


# ============================================================
# CodeBLEU (optional)
# ============================================================

def compute_codebleu(
    predicted_code: str,
    reference_code: str,
    language: str = "java",
) -> Optional[Dict[str, float]]:
    """CodeBLEU between predicted and reference code. Returns None if not installed."""
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=[[reference_code]],
            predictions=[predicted_code],
            lang=language,
        )
        return result
    except ImportError:
        return None


# ============================================================
# Aggregator
# ============================================================

class MetricsAggregator:
    """Accumulates per-record metrics and computes summary statistics."""

    def __init__(self):
        self.records: List[Dict] = []

    def add(self, metrics: Dict) -> None:
        self.records.append(metrics)

    def summary(self) -> Dict:
        if not self.records:
            return {"num_records": 0}

        n = len(self.records)

        def _mean(key):
            vals = [r[key] for r in self.records if key in r and r[key] is not None]
            return sum(vals) / len(vals) if vals else None

        # Per-hunk accuracy: flatten all hunk results across all records
        all_hunk_results = []
        for r in self.records:
            if "per_hunk_exact" in r:
                all_hunk_results.extend(r["per_hunk_exact"])

        return {
            "num_records": n,
            "num_hunks_total": len(all_hunk_results),
            "token_exact_match": _mean("token_exact_match"),
            "per_hunk_accuracy": (
                sum(all_hunk_results) / len(all_hunk_results)
                if all_hunk_results else None
            ),
            "all_hunks_correct_rate": (
                sum(1 for r in self.records if r.get("all_hunks_correct")) / n
            ),
            "mean_cross_entropy": _mean("cross_entropy_loss"),
            "mean_single_pass_ce": _mean("single_pass_ce"),
            "mean_edit_distance": _mean("edit_distance"),
            "mean_top_k_accuracy": _mean("top_k_accuracy"),
            "mean_codebleu": _mean("codebleu"),
        }
