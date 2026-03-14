#!/usr/bin/env python3
"""
Ablation study runner for LLaDA multi-hunk bug repair.

Runs three ablation axes and prints LaTeX-ready tables + saves JSON results.

All hyperparameters are cross-referenced against the rest of the codebase:
  model.py         — TEST_FRACTION=0.1, SPLIT_SEED=42, LoRA r=16/alpha=32
  data_preprocess.py — MAX_LENGTH=2048, HUNK_CAP_PER_FILE=5
  inference.py     — default steps=64, remasking=low_confidence, greedy (T=0)
  run_all_benchmarks.sh — evaluation uses --max-records 200 (matched here)

Ablation 1 — Denoising Steps
  Fixed:   LoRA model, remasking=low_confidence, greedy
  Sweep:   T in {1, 16, 32, 64, 128}
  Note:    infill_denoise caps effective steps at min(T, num_masked_tokens).
           With HUNK_CAP_PER_FILE=5 hunks × ~10-40 tokens/hunk, most records
           have 20-100 masked tokens. T=1 is single-pass (no refinement);
           T=64 is the training default; T=128 approaches one-step-per-token.

Ablation 2 — Re-masking Strategy
  Fixed:   LoRA model, T=64 (inference default), greedy
  Sweep:   remasking in {low_confidence, random}

Ablation 3 — Fine-tuning Contribution
  Fixed:   T=64, remasking=low_confidence, greedy
  Sweep:   {base LLaDA-8B-Instruct (no LoRA), LLaDA-8B + LoRA}

Model is loaded once per checkpoint and reused across runs that share it,
minimising redundant HuggingFace downloads and VRAM pressure.

Usage:
    # Full ablation (200 records — matches run_all_benchmarks.sh evaluation)
    python run_ablations.py --codebleu

    # Quick sanity check (5 records, skips CodeBLEU)
    python run_ablations.py --dry-run

    # Only one axis
    python run_ablations.py --ablation steps

    # Custom adapter / output directory
    python run_ablations.py \\
        --adapter-path runs/llada_lora/lora_adapter \\
        --output-dir runs/ablations \\
        --codebleu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# ── Local imports ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from inference import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_DATA_PATH,
    MODEL_NAME,
    evaluate,
    load_model_and_tokenizer,
)

# ── Constants pinned to the rest of the codebase ─────────────────────────────
# model.py: TEST_FRACTION = 0.1, SPLIT_SEED = 42
TEST_FRACTION = 0.1
SPLIT_SEED    = 42
# data_preprocess.py: MAX_LENGTH = 2048, HUNK_CAP_PER_FILE = 5
MAX_SEQ_LEN   = 2048
HUNK_CAP      = 5   # max hunks per record → max 2^5 - 5 - 1 = 26 masked regions
# inference.py: steps=64, remasking="low_confidence", temperature=0.0
DEFAULT_T        = 64
DEFAULT_REMASKING = "low_confidence"
# run_all_benchmarks.sh: evaluation uses --max-records 200
DEFAULT_MAX_RECORDS = 200

# ── Sweep definitions ─────────────────────────────────────────────────────────
# T=1   : single forward pass, no iterative refinement (AR-equivalent)
# T=16  : light refinement
# T=32  : moderate refinement
# T=64  : training default (inference.py)
# T=128 : approaches one-step-per-token ceiling
#         (infill_denoise clips to min(T, num_masked), so T=128 is effectively
#          T=num_masked for most records given HUNK_CAP=5 hunks)
STEPS_SWEEP     = [1, 16, 32, 64, 128]
REMASKING_SWEEP = ["low_confidence", "random"]

DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "ablations"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v: Optional[float], scale: float = 1.0, decimals: int = 2) -> str:
    """Format a float for table display; '--' for None."""
    if v is None:
        return "--"
    return f"{v * scale:.{decimals}f}"


def _pct(v: Optional[float]) -> str:
    return _fmt(v, scale=100.0, decimals=1)


def _f2(v: Optional[float]) -> str:
    return _fmt(v, scale=1.0, decimals=4)


def _latency(v: Optional[float]) -> str:
    if v is None:
        return "--"
    return f"{v:.2f}s"


def run_single(
    model,
    tokenizer,
    data_path: Path,
    output_dir: Path,
    run_id: str,
    steps: int,
    remasking: str,
    max_records: int,
    max_seq_len: int,
    compute_codebleu: bool,
    device: str,
) -> Dict[str, Any]:
    """
    Run a single evaluation configuration. Returns the summary dict from
    inference.evaluate(), augmented with configuration metadata.

    Split constants (test_fraction=0.1, split_seed=42) are pinned to their
    values in model.py to guarantee the same deterministic test split.
    """
    run_out = output_dir / run_id
    run_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary = evaluate(
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        split="test",
        steps=steps,
        temperature=0.0,
        remasking=remasking,
        max_records=max_records,
        max_seq_len=max_seq_len,
        test_fraction=TEST_FRACTION,   # pinned: model.py TEST_FRACTION = 0.1
        split_seed=SPLIT_SEED,         # pinned: model.py SPLIT_SEED    = 42
        compute_codebleu_flag=compute_codebleu,
        output_dir=run_out,
        device=device,
    )
    wall = time.time() - t0

    summary["_run_id"] = run_id
    summary["_steps"] = steps
    summary["_remasking"] = remasking
    summary["_wall_seconds"] = round(wall, 1)

    # Persist full summary
    with open(run_out / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX table printers
# ══════════════════════════════════════════════════════════════════════════════

def _latex_steps_table(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Effect of denoising steps $T$ on reconstruction quality "
        r"(LLaDA-8B + LoRA, greedy, \texttt{low\_confidence} remasking).}",
        r"\label{tab:ablation_steps}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Steps $T$} & \textbf{All-Hunks (\%)} & \textbf{Token EM (\%)} "
        r"& \textbf{BLEU-4} & \textbf{CodeBLEU} & \textbf{Latency (s/sample)} \\",
        r"\midrule",
    ]
    for r in rows:
        ahc  = _pct(r.get("all_hunks_correct_rate"))
        em   = _pct(r.get("token_exact_match_mean"))
        bleu = _f2(r.get("patch_bleu_mean"))
        cb   = _f2(r.get("codebleu_mean"))
        lat  = _latency(r.get("wall_time_seconds_mean"))
        lines.append(
            f"  {r['_steps']} & {ahc} & {em} & {bleu} & {cb} & {lat} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _latex_remasking_table(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Re-masking strategy comparison ($T=64$, LLaDA-8B + LoRA, greedy).}",
        r"\label{tab:ablation_remask}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Re-masking} & \textbf{All-Hunks (\%)} & \textbf{Hunk EM (\%)} "
        r"& \textbf{Token EM (\%)} & \textbf{BLEU-4} & \textbf{CodeBLEU} \\",
        r"\midrule",
    ]
    for r in rows:
        tag  = r["_remasking"].replace("_", r"\_")
        ahc  = _pct(r.get("all_hunks_correct_rate"))
        hem  = _pct(r.get("per_hunk_exact_rate"))
        em   = _pct(r.get("token_exact_match_mean"))
        bleu = _f2(r.get("patch_bleu_mean"))
        cb   = _f2(r.get("codebleu_mean"))
        lines.append(
            f"  \\texttt{{{tag}}} & {ahc} & {hem} & {em} & {bleu} & {cb} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _latex_finetuning_table(rows: List[Dict]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Effect of LoRA fine-tuning vs.\ base LLaDA-8B-Instruct "
        r"($T=64$, greedy, \texttt{low\_confidence} remasking).}",
        r"\label{tab:ablation_finetune}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{All-Hunks} & \textbf{Hunk EM} "
        r"& \textbf{Token EM} & \textbf{Patch EM} "
        r"& \textbf{BLEU-4} & \textbf{CodeBLEU} & \textbf{CE}$\downarrow$ \\",
        r" & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{(\%)} & & & \\",
        r"\midrule",
    ]
    for r in rows:
        label = r["_label"]
        ahc  = _pct(r.get("all_hunks_correct_rate"))
        hem  = _pct(r.get("per_hunk_exact_rate"))
        em   = _pct(r.get("token_exact_match_mean"))
        pem  = _pct(r.get("patch_string_em_mean"))
        bleu = _f2(r.get("patch_bleu_mean"))
        cb   = _f2(r.get("codebleu_mean"))
        ce   = _fmt(r.get("cross_entropy_loss_mean"), decimals=4)
        lines.append(
            f"  {label} & {ahc} & {hem} & {em} & {pem} & {bleu} & {cb} & {ce} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _print_divider(title: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study runner for MDLM Bug Fixer"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=["all", "steps", "remasking", "finetuning"],
        help="Which ablation axis to run (default: all)",
    )
    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(DEFAULT_ADAPTER_PATH),
        help="Path to LoRA adapter for the fine-tuned model",
    )
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,   # 200 — matches run_all_benchmarks.sh evaluation
        help="Maximum test records per run (default: 200, matching run_all_benchmarks.sh)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,           # 2048 — matches data_preprocess.py MAX_LENGTH
        help="Sequence length cap (default: 2048, matching data_preprocess.py MAX_LENGTH)",
    )
    parser.add_argument(
        "--codebleu",
        action="store_true",
        help="Compute CodeBLEU (slower, requires codebleu package)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias for --max-records 5; quick smoke-test",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        args.max_records = 5  # always override to 5 on dry-run regardless of --max-records

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    adapter_path = args.adapter_path
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    run_steps      = args.ablation in ("all", "steps")
    run_remasking  = args.ablation in ("all", "remasking")
    run_finetuning = args.ablation in ("all", "finetuning")

    all_summaries: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Load LoRA model (used by steps ablation + remasking ablation +      #
    #                   fine-tuning ablation "lora" arm)                  #
    # ------------------------------------------------------------------ #
    lora_needed = run_steps or run_remasking or run_finetuning
    if lora_needed:
        _print_divider("Loading LoRA-adapted model")
        model_lora, tokenizer = load_model_and_tokenizer(
            base_model_name=args.base_model,
            adapter_path=adapter_path,
            device=args.device,
            dtype=dtype,
        )

    # ═══════════════════════════════════════════════════════════════════ #
    #  ABLATION 1 — Denoising Steps                                       #
    # ═══════════════════════════════════════════════════════════════════ #
    steps_results: List[Dict] = []
    if run_steps:
        _print_divider("Ablation 1: Denoising Steps")
        for T in STEPS_SWEEP:
            print(f"\n── T={T} ──")
            s = run_single(
                model=model_lora,
                tokenizer=tokenizer,
                data_path=data_path,
                output_dir=output_dir / "steps",
                run_id=f"steps_T{T}",
                steps=T,
                remasking=DEFAULT_REMASKING,
                max_records=args.max_records,
                max_seq_len=args.max_seq_len,
                compute_codebleu=args.codebleu,
                device=args.device,
            )
            steps_results.append(s)
            all_summaries[f"steps_T{T}"] = s

    # ═══════════════════════════════════════════════════════════════════ #
    #  ABLATION 2 — Re-masking Strategy                                   #
    # ═══════════════════════════════════════════════════════════════════ #
    remasking_results: List[Dict] = []
    if run_remasking:
        _print_divider("Ablation 2: Re-masking Strategy")
        for strategy in REMASKING_SWEEP:
            print(f"\n── remasking={strategy} ──")
            s = run_single(
                model=model_lora,
                tokenizer=tokenizer,
                data_path=data_path,
                output_dir=output_dir / "remasking",
                run_id=f"remasking_{strategy}",
                steps=DEFAULT_T,
                remasking=strategy,
                max_records=args.max_records,
                max_seq_len=args.max_seq_len,
                compute_codebleu=args.codebleu,
                device=args.device,
            )
            remasking_results.append(s)
            all_summaries[f"remasking_{strategy}"] = s

    # ═══════════════════════════════════════════════════════════════════ #
    #  ABLATION 3 — Fine-tuning Contribution                              #
    # ═══════════════════════════════════════════════════════════════════ #
    finetuning_results: List[Dict] = []
    if run_finetuning:
        _print_divider("Ablation 3: Fine-tuning (LoRA arm)")

        # LoRA arm (model already loaded above)
        print("\n── LLaDA-8B + LoRA ──")
        s_lora = run_single(
            model=model_lora,
            tokenizer=tokenizer,
            data_path=data_path,
            output_dir=output_dir / "finetuning",
            run_id="finetuning_lora",
            steps=DEFAULT_T,
            remasking=DEFAULT_REMASKING,
            max_records=args.max_records,
            max_seq_len=args.max_seq_len,
            compute_codebleu=args.codebleu,
            device=args.device,
        )
        s_lora["_label"] = r"LLaDA-8B + LoRA"
        finetuning_results.append(s_lora)
        all_summaries["finetuning_lora"] = s_lora

        # Free LoRA model before loading base to keep peak VRAM low
        del model_lora
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _print_divider("Ablation 3: Fine-tuning (base arm)")
        print("\n── LLaDA-8B base (no LoRA) ──")
        model_base, _ = load_model_and_tokenizer(
            base_model_name=args.base_model,
            adapter_path=None,  # no LoRA
            device=args.device,
            dtype=dtype,
        )
        s_base = run_single(
            model=model_base,
            tokenizer=tokenizer,
            data_path=data_path,
            output_dir=output_dir / "finetuning",
            run_id="finetuning_base",
            steps=DEFAULT_T,
            remasking=DEFAULT_REMASKING,
            max_records=args.max_records,
            max_seq_len=args.max_seq_len,
            compute_codebleu=args.codebleu,
            device=args.device,
        )
        s_base["_label"] = r"LLaDA-8B (base)"
        # Insert base before LoRA for a logical table ordering
        finetuning_results.insert(0, s_base)
        all_summaries["finetuning_base"] = s_base

        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================================== #
    #  Save all summaries                                                 #
    # ================================================================== #
    combined_path = output_dir / "ablation_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[✓] All summaries saved → {combined_path}")

    # ================================================================== #
    #  Print LaTeX tables                                                 #
    # ================================================================== #
    latex_lines: List[str] = [
        "%% ============================================================",
        "%%  AUTO-GENERATED LaTeX TABLES — run_ablations.py",
        "%%  Copy-paste the tables you need into report/report.tex",
        "%% ============================================================",
        "",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        "",
    ]

    if run_steps and steps_results:
        _print_divider("LaTeX — Ablation 1: Denoising Steps")
        tbl = _latex_steps_table(steps_results)
        print(tbl)
        latex_lines += [tbl, ""]

    if run_remasking and remasking_results:
        _print_divider("LaTeX — Ablation 2: Re-masking Strategy")
        tbl = _latex_remasking_table(remasking_results)
        print(tbl)
        latex_lines += [tbl, ""]

    if run_finetuning and finetuning_results:
        _print_divider("LaTeX — Ablation 3: Fine-tuning Contribution")
        tbl = _latex_finetuning_table(finetuning_results)
        print(tbl)
        latex_lines += [tbl, ""]

    latex_out = output_dir / "ablation_tables.tex"
    with open(latex_out, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"\n[✓] LaTeX tables saved → {latex_out}")

    # ================================================================== #
    #  Plain-text summary to stdout (easy copy-paste for report)         #
    # ================================================================== #
    _print_divider("Summary")
    if steps_results:
        print("\nDenoising Steps")
        print(f"{'T':>6}  {'All-Hunks%':>11}  {'TokenEM%':>9}  {'BLEU-4':>8}  {'CodeBLEU':>9}  {'s/sample':>9}")
        print("─" * 70)
        for r in steps_results:
            print(
                f"{r['_steps']:>6}  "
                f"{_pct(r.get('all_hunks_correct_rate')):>11}  "
                f"{_pct(r.get('token_exact_match_mean')):>9}  "
                f"{_f2(r.get('patch_bleu_mean')):>8}  "
                f"{_f2(r.get('codebleu_mean')):>9}  "
                f"{_latency(r.get('wall_time_seconds_mean')):>9}"
            )

    if remasking_results:
        print("\nRe-masking Strategy (T=64)")
        print(f"{'Strategy':<20}  {'All-Hunks%':>11}  {'HunkEM%':>8}  {'TokenEM%':>9}  {'BLEU-4':>8}  {'CodeBLEU':>9}")
        print("─" * 80)
        for r in remasking_results:
            print(
                f"{r['_remasking']:<20}  "
                f"{_pct(r.get('all_hunks_correct_rate')):>11}  "
                f"{_pct(r.get('per_hunk_exact_rate')):>8}  "
                f"{_pct(r.get('token_exact_match_mean')):>9}  "
                f"{_f2(r.get('patch_bleu_mean')):>8}  "
                f"{_f2(r.get('codebleu_mean')):>9}"
            )

    if finetuning_results:
        print("\nFine-tuning Contribution (T=64, low_confidence)")
        print(f"{'Model':<30}  {'All-Hunks%':>11}  {'HunkEM%':>8}  {'TokenEM%':>9}  {'BLEU-4':>8}  {'CodeBLEU':>9}")
        print("─" * 90)
        for r in finetuning_results:
            print(
                f"{r['_label']:<30}  "
                f"{_pct(r.get('all_hunks_correct_rate')):>11}  "
                f"{_pct(r.get('per_hunk_exact_rate')):>8}  "
                f"{_pct(r.get('token_exact_match_mean')):>9}  "
                f"{_f2(r.get('patch_bleu_mean')):>8}  "
                f"{_f2(r.get('codebleu_mean')):>9}"
            )

    print(f"\n[✓] Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
