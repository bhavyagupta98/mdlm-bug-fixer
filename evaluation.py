#!/usr/bin/env python3
"""
Unified evaluation metrics for multi-hunk infilling comparison.

Used by both:
  - inference.py               (MDLM iterative denoising)
  - baseline_reconstruction_eval.py  (FIM / instruct zero-shot baselines)

All functions are stateless and operate on plain Python lists / strings.
Tensor-based functions (masked_cross_entropy, top_k_accuracy) are MDLM-only
and are safe to skip in AR/FIM baselines.
"""

import math
import unicodedata
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# Token-level exact match  (variable-length safe)
# ============================================================

def token_exact_match_rate(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
) -> float:
    """
    Fraction of ground-truth token positions that are correctly predicted.
    Positions beyond the shorter sequence count as mismatches, so the
    function is safe when predicted and gold have different lengths.
    """
    if not ground_truth_ids:
        return 1.0
    n = len(ground_truth_ids)
    matches = sum(
        predicted_ids[i] == ground_truth_ids[i]
        for i in range(min(len(predicted_ids), n))
    )
    return matches / n


# ============================================================
# Per-hunk span exact match  (for MDLM token-ID output)
# ============================================================

def per_hunk_exact_match(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
    hunk_spans: List[Tuple[int, int]],
) -> List[bool]:
    """For each hunk span [s, e), check if ALL tokens match exactly."""
    results = []
    for s, e in hunk_spans:
        match = predicted_ids[s:e] == ground_truth_ids[s:e]
        results.append(match)
    return results


def all_hunks_correct(hunk_results: List[bool]) -> bool:
    return all(hunk_results) if hunk_results else True


# ============================================================
# String-level exact match  (for AR / FIM text output)
# ============================================================

def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def patch_string_exact_match(predicted_text: str, gold_text: str) -> bool:
    """True if predicted and gold patch strings match after normalization."""
    return _normalize(predicted_text) == _normalize(gold_text)


# ============================================================
# Edit distance
# ============================================================

def token_edit_distance(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
) -> int:
    """Levenshtein edit distance between two token-ID sequences."""
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


def normalized_edit_distance(
    predicted_ids: List[int],
    ground_truth_ids: List[int],
) -> float:
    """
    token_edit_distance normalised by max(|pred|, |gold|).
    Range [0, 1] — 0 is identical, 1 is completely different.
    """
    denom = max(len(predicted_ids), len(ground_truth_ids), 1)
    return token_edit_distance(predicted_ids, ground_truth_ids) / denom


# ============================================================
# Patch BLEU  (sentence-BLEU-4, no external deps)
# ============================================================

def _ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def patch_bleu(predicted_text: str, gold_text: str, max_n: int = 4) -> float:
    """
    Sentence BLEU-4 on whitespace-tokenised patch text.
    No external nltk/sacrebleu dependency required.
    Returns 0.0 for empty predictions.
    """
    hyp = predicted_text.split()
    ref = gold_text.split()
    if not hyp:
        return 0.0
    bp = 1.0 if len(hyp) >= len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp, n)
        ref_ng = _ngrams(ref, n)
        if not hyp_ng:
            return 0.0
        clipped = sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
        total = sum(hyp_ng.values())
        if clipped == 0:
            return 0.0
        log_avg += math.log(clipped / total)
    return bp * math.exp(log_avg / max_n)


# ============================================================
# Pass@k  (unbiased estimator — Chen et al. 2021)
# ============================================================

def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from the Codex paper.

    Args:
        n: total generated samples for a problem
        c: number of correct samples among n
        k: k in pass@k

    Returns the probability that at least one of k randomly drawn
    samples is correct.
    """
    if n < k:
        return float(c > 0)
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k), in log space for stability
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)


# ============================================================
# Loss / logit metrics  (MDLM-only; safe to skip in AR/FIM)
# ============================================================

def masked_cross_entropy(
    logits: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    mask_positions: torch.Tensor,
) -> float:
    """
    Cross-entropy on masked positions only.
    logits: (seq_len, vocab_size)
    ground_truth_ids: (seq_len,)
    mask_positions: (seq_len,) boolean
    """
    if mask_positions.sum() == 0:
        return 0.0
    loss = F.cross_entropy(logits[mask_positions], ground_truth_ids[mask_positions])
    return loss.item()


def top_k_accuracy(
    logits: torch.Tensor,
    ground_truth_ids: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Fraction of positions where the ground-truth token is in the top-k
    predictions.
    logits: (num_positions, vocab_size)
    ground_truth_ids: (num_positions,)
    """
    if logits.size(0) == 0:
        return 1.0
    _, topk_ids = logits.topk(k, dim=-1)
    matches = (topk_ids == ground_truth_ids.unsqueeze(-1)).any(dim=-1)
    return matches.float().mean().item()


# ============================================================
# CodeBLEU  (optional external dependency)
# ============================================================

def compute_codebleu(
    predicted_code: str,
    reference_code: str,
    language: str = "java",
) -> Optional[Dict[str, float]]:
    """
    CodeBLEU between predicted and reference code strings.
    Returns None if the `codebleu` package is not installed.
    """
    try:
        from codebleu import calc_codebleu
        return calc_codebleu(
            references=[[reference_code]],
            predictions=[predicted_code],
            lang=language,
        )
    except ImportError:
        return None


# ============================================================
# Aggregator  (shared between MDLM and baselines)
# ============================================================

class MetricsAggregator:
    """
    Accumulates per-record metric dicts and computes summary statistics.

    Expected keys per record (all optional — missing keys are skipped):

    Shared (MDLM + baselines):
      per_hunk_exact      list[bool]   one entry per hunk
      all_hunks_correct   bool
      patch_string_em     float        fraction of hunks with string EM
      patch_bleu          float        mean sentence-BLEU across hunks
      patch_ned           float        mean normalised edit distance
      codebleu            float | None

    MDLM-only (silently absent for AR/FIM):
      token_exact_match   float
      cross_entropy_loss  float
      single_pass_ce      float
      top_k_accuracy      float
    """

    def __init__(self) -> None:
        self.records: List[Dict] = []

    def add(self, metrics: Dict) -> None:
        self.records.append(metrics)

    def summary(self) -> Dict:
        if not self.records:
            return {"num_records": 0}

        n = len(self.records)

        def _mean(key: str) -> Optional[float]:
            vals = [r[key] for r in self.records if key in r and r[key] is not None]
            return sum(vals) / len(vals) if vals else None

        # Flatten per-hunk boolean results across all records
        all_hunk_results: List[bool] = []
        for r in self.records:
            if "per_hunk_exact" in r:
                all_hunk_results.extend(r["per_hunk_exact"])

        result: Dict = {
            "num_records": n,
            "num_hunks_total": len(all_hunk_results),
            # --- primary shared metrics ---
            "per_hunk_accuracy": (
                sum(all_hunk_results) / len(all_hunk_results)
                if all_hunk_results else None
            ),
            "all_hunks_correct_rate": (
                sum(1 for r in self.records if r.get("all_hunks_correct")) / n
            ),
            "patch_string_em": _mean("patch_string_em"),
            "mean_patch_bleu": _mean("patch_bleu"),
            "mean_patch_ned": _mean("patch_ned"),
            # --- secondary shared metrics ---
            "mean_codebleu": _mean("codebleu"),
        }

        # Token-level EM — populated by MDLM; may also be set by baselines
        v = _mean("token_exact_match")
        if v is not None:
            result["token_exact_match"] = v

        # MDLM-only (emitted only when present so baseline summaries stay clean)
        for src_key, out_key in [
            ("cross_entropy_loss", "mean_cross_entropy"),
            ("single_pass_ce",     "mean_single_pass_ce"),
            ("top_k_accuracy",     "mean_top_k_accuracy"),
        ]:
            v = _mean(src_key)
            if v is not None:
                result[out_key] = v

        return result
