#!/usr/bin/env python3
"""
Run a Java SWE-style benchmark for LLaDA/LoRA using Defects4J.

This is intentionally not presented as canonical SWE-bench. The current model
is a localized multi-hunk infiller, not a repository-level issue-solving agent.
Accordingly, this runner evaluates the model in an oracle-localized setting:

1. Use Defects4J modified source files for file localization.
2. Diff buggy vs fixed Java file to identify changed hunks.
3. Mask those hunks in the fixed file using the LLaDA tokenizer.
4. Reconstruct the hunks with the current denoising inference pipeline.
5. Optionally write predicted files into the buggy checkout and validate with
   `defects4j compile`, trigger tests, and the full test suite.

Outputs:
  - file_results.jsonl: one record per modified Java file
  - bug_results.jsonl: one record per Defects4J bug
  - bug_results.csv: compact bug-level table
  - summary.json: report-ready aggregate metrics
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from benchmark_generate import resolve_mask_token_id
from data_preprocess import compute_hunks, new_line_spans, char_span_to_token_span
from evaluation import (
    all_hunks_correct,
    compute_codebleu,
    masked_cross_entropy,
    normalized_edit_distance,
    patch_bleu,
    patch_string_exact_match,
    per_hunk_exact_match,
    token_edit_distance,
    token_exact_match_rate,
    top_k_accuracy,
)
from inference import (
    MODEL_NAME,
    compute_holistic_patch_metrics,
    compute_loss_single_pass,
    infill_denoise,
    load_model_and_tokenizer,
    prepare_masked_input,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = BASE_DIR / "runs" / "llada_lora" / "lora_adapter"
DEFAULT_OUTPUT_DIR = BASE_DIR / "runs" / "java_swebench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Defects4J-backed Java SWE-style benchmark for LLaDA"
    )

    parser.add_argument("--projects", type=str, required=True,
                        help="Comma-separated Defects4J project ids, e.g. Lang,Math")
    parser.add_argument("--bug-ids", type=str, default=None,
                        help="Optional bug ids/ranges applied to each project, e.g. 1,2,5-10")
    parser.add_argument("--bug-limit", type=int, default=None,
                        help="Optional cap on bugs per project after filtering")

    parser.add_argument("--defects4j-home", type=str, default=None,
                        help="Path to Defects4J root. If omitted, use --defects4j-cmd or PATH.")
    parser.add_argument("--defects4j-cmd", type=str, default="defects4j",
                        help="Defects4J executable name/path if not using --defects4j-home")

    parser.add_argument("--base-model", type=str, default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH),
                        help="Path to LoRA adapter. Defaults to runs/llada_lora/lora_adapter if it exists.")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run base model without LoRA")

    parser.add_argument(
        "--mode", type=str, default="localized",
        choices=["localized", "generative"],
        help=(
            "localized: oracle-localized diff masking (original mode); "
            "generative: feed prompt + full buggy code, model generates the fixed file "
            "without being told where the bug is"
        ),
    )

    # Generative mode generation budget
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="(generative mode) number of mask tokens to append for generation")
    parser.add_argument("--oracle-length", action="store_true",
                        help=(
                            "(generative mode) use the fixed-file token count as the generation "
                            "budget instead of --max-new-tokens"
                        ))
    parser.add_argument("--oracle-slack", type=int, default=64,
                        help="(generative mode) extra tokens added on top of oracle length")

    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--compute-codebleu", action="store_true")

    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip defects4j compile/test validation and report only patch metrics")
    parser.add_argument("--run-full-test-suite", action="store_true",
                        help="Also run `defects4j test` for the full suite (slower)")
    parser.add_argument("--keep-checkouts", action="store_true",
                        help="Keep temporary Defects4J checkouts under the output dir for debugging")

    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float32"])

    return parser.parse_args()


def resolve_defects4j_command(defects4j_home: Optional[str], defects4j_cmd: str) -> str:
    if defects4j_home:
        candidate = Path(defects4j_home) / "framework" / "bin" / "defects4j"
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Defects4J executable not found under {candidate}")
    return defects4j_cmd


def parse_bug_id_filter(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None

    values = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range: {item}")
            for bug_id in range(start, end + 1):
                values.add(bug_id)
        else:
            values.add(int(item))

    return sorted(values)


def run_command(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
) -> Tuple[int, str, str]:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def defects4j_export(defects4j_cmd: str, work_dir: Path, prop: str) -> str:
    code, stdout, stderr = run_command([defects4j_cmd, "export", "-p", prop], cwd=work_dir)
    if code != 0:
        raise RuntimeError(f"defects4j export failed for {prop}: {stderr or stdout}")
    return stdout.strip()


def list_bug_ids(defects4j_cmd: str, project_id: str) -> List[int]:
    code, stdout, stderr = run_command([defects4j_cmd, "bids", "-p", project_id])
    if code != 0:
        raise RuntimeError(f"defects4j bids failed for {project_id}: {stderr or stdout}")
    ids = []
    for line in stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            ids.append(int(text))
    if not ids:
        raise RuntimeError(f"No bug ids returned for project {project_id}")
    return ids


def checkout_bug(defects4j_cmd: str, project_id: str, bug_id: int, version: str, work_dir: Path) -> None:
    code, stdout, stderr = run_command(
        [
            defects4j_cmd,
            "checkout",
            "-p", project_id,
            "-v", f"{bug_id}{version}",
            "-w", str(work_dir),
        ]
    )
    if code != 0:
        raise RuntimeError(
            f"defects4j checkout failed for {project_id}-{bug_id}{version}: {stderr or stdout}"
        )


def parse_failing_tests_output(stdout: str, stderr: str) -> Optional[int]:
    merged = "\n".join([stdout, stderr])
    match = re.search(r"Failing tests:\s*(\d+)", merged)
    if match:
        return int(match.group(1))
    return None


def run_defects4j_compile(defects4j_cmd: str, work_dir: Path) -> Dict[str, Any]:
    code, stdout, stderr = run_command([defects4j_cmd, "compile"], cwd=work_dir)
    return {
        "ok": code == 0,
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
    }


def run_defects4j_test(defects4j_cmd: str, work_dir: Path, test_name: Optional[str] = None) -> Dict[str, Any]:
    cmd = [defects4j_cmd, "test"]
    if test_name:
        cmd.extend(["-t", test_name])
    code, stdout, stderr = run_command(cmd, cwd=work_dir)
    failing_tests = parse_failing_tests_output(stdout, stderr)
    ok = (failing_tests == 0) if failing_tests is not None else (code == 0)
    return {
        "ok": ok,
        "exit_code": code,
        "failing_tests": failing_tests,
        "stdout": stdout,
        "stderr": stderr,
        "test_name": test_name,
    }


def class_name_to_java_path(class_name: str, src_dir: str) -> Path:
    base = class_name.split("$", 1)[0]
    return (Path(src_dir) / Path(*base.split("."))).with_suffix(".java")


# ============================================================
# Generative Mode Helpers
# ============================================================
JAVA_REPAIR_SYSTEM = (
    "You are an expert Java developer. "
    "Fix the bug in the Java code below. "
    "Return the complete corrected file with no explanation or commentary."
)


def build_repair_prompt(buggy_code: str, tokenizer) -> str:
    """Build a chat-formatted prompt for generative Java bug repair."""
    messages = [
        {"role": "system", "content": JAVA_REPAIR_SYSTEM},
        {"role": "user", "content": f"Buggy Java code:\n```java\n{buggy_code}\n```"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: plain-text format for tokenizers without a chat template
        return (
            f"{JAVA_REPAIR_SYSTEM}\n\n"
            f"Buggy Java code:\n```java\n{buggy_code}\n```\n\n"
            "Fixed code:\n```java\n"
        )


def _extract_java_code(text: str) -> str:
    """
    Strip markdown fences and leading prose from model output.
    Returns the raw Java source text.
    """
    # Prefer fenced java block; fall back to any fenced block
    blocks = re.findall(r"```(?:java)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()

    # If output starts with a Java token already, return as-is
    stripped = text.lstrip()
    for kw in ("package ", "import ", "public ", "class ", "//", "/*", "@"):
        if stripped.startswith(kw):
            return stripped

    # Find the first Java-looking line
    for pattern in [r"\npackage ", r"\nimport ", r"\npublic ", r"\nclass "]:
        match = re.search(pattern, text)
        if match:
            return text[match.start() + 1:]

    return text


def build_localized_record(
    tokenizer,
    mask_token_id: int,
    buggy_code: str,
    fixed_code: str,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    raw_hunks = compute_hunks(buggy_code, fixed_code)
    char_spans = new_line_spans(fixed_code, raw_hunks)
    if not char_spans:
        return None

    encoded = tokenizer(
        fixed_code,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    token_spans: List[Tuple[int, int]] = []
    for span in char_spans:
        mapped = char_span_to_token_span(encoded["offset_mapping"], span)
        if mapped is not None:
            token_spans.append(mapped)

    if not token_spans:
        return None

    return {
        "input_ids": encoded["input_ids"],
        "hunk_token_spans": token_spans,
        "mask_token_id": int(mask_token_id),
    }


def evaluate_instance(
    model: torch.nn.Module,
    tokenizer,
    record: Dict[str, Any],
    fixed_code: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    masked, gt, mask_pos, spans, mask_tid = prepare_masked_input(record, args.device)
    t0 = time.time()

    single_pass_ce = compute_loss_single_pass(model, masked, gt, mask_pos)
    result = infill_denoise(
        model=model,
        input_ids=masked,
        mask_positions=mask_pos,
        mask_token_id=mask_tid,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        return_logits=True,
    )

    pred_ids = result["predicted_ids"]
    pred_list = pred_ids[0].tolist()
    gt_list = gt[0].tolist()

    pred_masked: List[int] = []
    gt_masked: List[int] = []
    for s, e in spans:
        pred_masked.extend(pred_list[s:e])
        gt_masked.extend(gt_list[s:e])

    token_em = token_exact_match_rate(pred_masked, gt_masked)
    hunk_exact = per_hunk_exact_match(pred_list, gt_list, spans)
    holistic = compute_holistic_patch_metrics(tokenizer, pred_list, gt_list, spans)

    denoise_ce = None
    tk_acc = None
    if result["final_logits"] is not None:
        final_logits = result["final_logits"][0]
        denoise_ce = masked_cross_entropy(final_logits, gt[0], mask_pos)
        tk_acc = top_k_accuracy(final_logits[mask_pos], gt[0][mask_pos], k=args.top_k)

    pred_code = tokenizer.decode(
        pred_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    gt_code = tokenizer.decode(
        gt_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    codebleu = None
    if args.compute_codebleu:
        cb = compute_codebleu(pred_code, gt_code, language="java")
        if cb:
            codebleu = cb.get("codebleu")

    return {
        "predicted_code": pred_code,
        "reference_code": gt_code,
        "full_file_exact_match": patch_string_exact_match(pred_code, gt_code),
        "reference_matches_raw_fixed": patch_string_exact_match(gt_code, fixed_code),
        "num_hunks": len(spans),
        "total_masked_tokens": int(mask_pos.sum().item()),
        "token_exact_match": token_em,
        "per_hunk_exact": hunk_exact,
        "all_hunks_correct": all_hunks_correct(hunk_exact),
        "patch_string_em": holistic["patch_string_em"],
        "patch_bleu": holistic["patch_bleu"],
        "patch_ned": holistic["patch_ned"],
        "edit_distance": token_edit_distance(pred_masked, gt_masked),
        "single_pass_ce": single_pass_ce,
        "cross_entropy_loss": denoise_ce,
        "top_k_accuracy": tk_acc,
        "codebleu": codebleu,
        "wall_time_sec": time.time() - t0,
    }


# ============================================================
# Generative Mode Evaluation (prompt + buggy code, no masking hints)
# ============================================================
def _is_causal_lm(model: torch.nn.Module) -> bool:
    """
    Return True if the model is a standard causal LM (e.g. Llama, Mistral)
    rather than a masked-diffusion model like LLaDA.
    LLaDA exposes mask_token_id in its config; causal LMs do not.
    """
    config = getattr(model, "config", None)
    return getattr(config, "mask_token_id", None) is None


def evaluate_instance_generative_causal(
    model: torch.nn.Module,
    tokenizer,
    buggy_code: str,
    fixed_code: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Generative evaluation for standard causal/autoregressive LMs (e.g. Llama-2).
    Uses model.generate() instead of infill_denoise.
    """
    prompt = build_repair_prompt(buggy_code, tokenizer)
    fixed_ids = tokenizer.encode(fixed_code, add_special_tokens=False)

    if args.oracle_length:
        max_new_tokens = len(fixed_ids) + args.oracle_slack
    else:
        max_new_tokens = args.max_new_tokens

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len - max_new_tokens - 2,
    ).to(args.device)
    P = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    gen_ids = output_ids[0, P:].tolist()
    pred_code_raw = tokenizer.decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
    pred_code = _extract_java_code(pred_code_raw)
    pred_code_ids = tokenizer.encode(pred_code, add_special_tokens=False)

    file_em = patch_string_exact_match(pred_code, fixed_code)
    bleu = patch_bleu(pred_code, fixed_code)
    ned = normalized_edit_distance(pred_code_ids, fixed_ids)
    ed = token_edit_distance(pred_code_ids, fixed_ids)

    codebleu = None
    if args.compute_codebleu:
        cb = compute_codebleu(pred_code, fixed_code, language="java")
        if cb:
            codebleu = cb.get("codebleu")

    return {
        "predicted_code": pred_code,
        "reference_code": fixed_code,
        "full_file_exact_match": file_em,
        "num_hunks": None,
        "total_masked_tokens": max_new_tokens,
        "token_exact_match": None,
        "per_hunk_exact": None,
        "all_hunks_correct": None,
        "patch_string_em": float(file_em),
        "patch_bleu": bleu,
        "patch_ned": ned,
        "edit_distance": ed,
        "single_pass_ce": None,
        "cross_entropy_loss": None,
        "top_k_accuracy": None,
        "codebleu": codebleu,
        "wall_time_sec": elapsed,
        "prompt_tokens": P,
        "generated_tokens": len(gen_ids),
    }


def evaluate_instance_generative(
    model: torch.nn.Module,
    tokenizer,
    buggy_code: str,
    fixed_code: str,
    mask_token_id: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Generative evaluation (LLaDA masked-diffusion): no diff-based masking.
    The model receives a prompt containing the full buggy file and must
    produce the complete fixed file via append-and-denoise.

    Generation budget:
      - If --oracle-length: use len(fixed_file_tokens) + --oracle-slack
      - Otherwise: --max-new-tokens (fixed cap)
    """
    prompt = build_repair_prompt(buggy_code, tokenizer)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    fixed_ids = tokenizer.encode(fixed_code, add_special_tokens=False)

    if args.oracle_length:
        max_new_tokens = len(fixed_ids) + args.oracle_slack
    else:
        max_new_tokens = args.max_new_tokens

    # Ensure prompt + generation fits within max_seq_len
    max_prompt_len = args.max_seq_len - max_new_tokens - 2
    if len(prompt_ids) > max_prompt_len:
        print(f"  [WARN] Prompt truncated: {len(prompt_ids)} → {max_prompt_len} tokens")
        prompt_ids = prompt_ids[:max_prompt_len]

    max_new_tokens = max(
        min(max_new_tokens, args.max_seq_len - len(prompt_ids) - 2), 1
    )

    P = len(prompt_ids)
    M = max_new_tokens
    total_len = P + M

    device = args.device
    input_ids = torch.full((1, total_len), mask_token_id, dtype=torch.long, device=device)
    input_ids[0, :P] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    mask_positions = torch.zeros(total_len, dtype=torch.bool, device=device)
    mask_positions[P:] = True

    t0 = time.time()
    result = infill_denoise(
        model=model,
        input_ids=input_ids,
        mask_positions=mask_positions,
        mask_token_id=mask_token_id,
        steps=args.steps,
        temperature=args.temperature,
        remasking=args.remasking,
        return_logits=False,
    )
    elapsed = time.time() - t0

    pred_ids_full = result["predicted_ids"][0].tolist()
    gen_ids = pred_ids_full[P:]  # only the generated part
    pred_code_raw = tokenizer.decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    pred_code = _extract_java_code(pred_code_raw)

    # Re-tokenize the cleaned prediction for token-level metrics
    pred_code_ids = tokenizer.encode(pred_code, add_special_tokens=False)

    file_em = patch_string_exact_match(pred_code, fixed_code)
    bleu = patch_bleu(pred_code, fixed_code)
    ned = normalized_edit_distance(pred_code_ids, fixed_ids)
    ed = token_edit_distance(pred_code_ids, fixed_ids)

    codebleu = None
    if args.compute_codebleu:
        cb = compute_codebleu(pred_code, fixed_code, language="java")
        if cb:
            codebleu = cb.get("codebleu")

    return {
        "predicted_code": pred_code,
        "reference_code": fixed_code,
        "full_file_exact_match": file_em,
        "num_hunks": None,           # not applicable in generative mode
        "total_masked_tokens": M,
        "token_exact_match": None,
        "per_hunk_exact": None,
        "all_hunks_correct": None,
        "patch_string_em": float(file_em),
        "patch_bleu": bleu,
        "patch_ned": ned,
        "edit_distance": ed,
        "single_pass_ce": None,
        "cross_entropy_loss": None,
        "top_k_accuracy": None,
        "codebleu": codebleu,
        "wall_time_sec": elapsed,
        "prompt_tokens": P,
        "generated_tokens": M,
    }


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return (sum(cleaned) / len(cleaned)) if cleaned else None


def aggregate_summary(
    file_records: List[Dict[str, Any]],
    bug_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    def rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
        if not records:
            return None
        return sum(1.0 if r.get(key) else 0.0 for r in records) / len(records)

    is_generative = args.mode == "generative"
    benchmark_name = (
        "java_swe_generative_defects4j" if is_generative
        else "java_swe_localized_defects4j"
    )
    benchmark_mode_desc = (
        "generative: prompt + full buggy file, model produces complete fixed file"
        if is_generative
        else "oracle-localized file repair with optional repository validation"
    )

    return {
        "benchmark": benchmark_name,
        "benchmark_mode": benchmark_mode_desc,
        "language": "java",
        "projects": [p.strip() for p in args.projects.split(",") if p.strip()],
        "model": args.base_model,
        "adapter_path": None if args.no_adapter else args.adapter_path,
        "num_file_instances": len(file_records),
        "num_bug_instances": len(bug_records),
        "file_exact_match_rate": rate(file_records, "full_file_exact_match"),
        # Only meaningful in localized mode, None in generative mode
        "file_all_hunks_correct_rate": None if is_generative else rate(file_records, "all_hunks_correct"),
        "bug_exact_match_rate": rate(bug_records, "all_files_exact"),
        "compile_success_rate": rate(bug_records, "compile_ok"),
        "trigger_test_pass_rate": rate(bug_records, "trigger_tests_ok"),
        "full_test_suite_pass_rate": rate(bug_records, "full_test_suite_ok") if args.run_full_test_suite else None,
        "mean_token_exact_match": None if is_generative else mean_or_none([r.get("token_exact_match") for r in file_records]),
        "mean_patch_string_em": mean_or_none([r.get("patch_string_em") for r in file_records]),
        "mean_patch_bleu": mean_or_none([r.get("patch_bleu") for r in file_records]),
        "mean_patch_ned": mean_or_none([r.get("patch_ned") for r in file_records]),
        "mean_codebleu": mean_or_none([r.get("codebleu") for r in file_records]),
        "mean_wall_time_sec_per_file": mean_or_none([r.get("wall_time_sec") for r in file_records]),
        # Generative mode budget info
        "mean_generated_tokens": mean_or_none([r.get("generated_tokens") for r in file_records]) if is_generative else None,
        "oracle_length_budget": args.oracle_length if is_generative else None,
        "validation_enabled": not args.skip_validation,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_bug_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "project_id",
        "bug_id",
        "num_modified_files",
        "all_files_exact",
        "compile_ok",
        "trigger_tests_ok",
        "full_test_suite_ok",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    defects4j_cmd = resolve_defects4j_command(args.defects4j_home, args.defects4j_cmd)
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    adapter = None if args.no_adapter else args.adapter_path
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model,
        adapter_path=adapter,
        device=args.device,
        dtype=dtype,
    )
    causal = _is_causal_lm(model)
    mask_token_id = None if causal else resolve_mask_token_id(tokenizer, model)

    print("\n" + "=" * 60)
    print("JAVA SWE-STYLE BENCHMARK")
    print("=" * 60)
    print(f"[INFO] Model type: {'causal LM (autoregressive)' if causal else 'masked diffusion (LLaDA)'}")
    if args.mode == "generative":
        print("[INFO] Mode: generative — prompt + full buggy file, no masking hints")
        budget = (
            f"oracle length + {args.oracle_slack} slack"
            if args.oracle_length
            else f"{args.max_new_tokens} fixed tokens"
        )
        print(f"[INFO] Generation budget: {budget}")
    else:
        if causal:
            print("[WARN] Mode: localized — not applicable for causal LMs; skipping to generative")
            args.mode = "generative"
        else:
            print("[INFO] Mode: oracle-localized Defects4J file repair")
    print(f"[INFO] Model: {args.base_model}")
    print(f"[INFO] Adapter: {'base-only' if args.no_adapter else (args.adapter_path or 'base-only')}")
    print(f"[INFO] Defects4J cmd: {defects4j_cmd}")

    requested_bug_ids = parse_bug_id_filter(args.bug_ids)
    projects = [item.strip() for item in args.projects.split(",") if item.strip()]

    file_records: List[Dict[str, Any]] = []
    bug_records: List[Dict[str, Any]] = []

    work_root: Optional[Path]
    if args.keep_checkouts:
        work_root = output_dir / "checkouts"
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        work_root = None

    for project_id in projects:
        project_bug_ids = list_bug_ids(defects4j_cmd, project_id)
        if requested_bug_ids is not None:
            project_bug_ids = [bug_id for bug_id in project_bug_ids if bug_id in requested_bug_ids]
        if args.bug_limit is not None:
            project_bug_ids = project_bug_ids[:args.bug_limit]

        print(f"\n[PROJECT] {project_id}: {len(project_bug_ids)} bugs selected")

        for bug_id in project_bug_ids:
            print(f"\n[BUG] {project_id}-{bug_id}")

            if work_root is None:
                temp_root = Path(tempfile.mkdtemp(prefix=f"d4j_{project_id}_{bug_id}_"))
            else:
                temp_root = work_root / f"{project_id}_{bug_id}"
                temp_root.mkdir(parents=True, exist_ok=True)

            buggy_dir = temp_root / "buggy"
            fixed_dir = temp_root / "fixed"

            try:
                checkout_bug(defects4j_cmd, project_id, bug_id, "b", buggy_dir)
                checkout_bug(defects4j_cmd, project_id, bug_id, "f", fixed_dir)

                src_dir = defects4j_export(defects4j_cmd, fixed_dir, "dir.src.classes")
                modified_classes_raw = defects4j_export(defects4j_cmd, fixed_dir, "classes.modified")
                trigger_tests_raw = defects4j_export(defects4j_cmd, fixed_dir, "tests.trigger")
                modified_classes = [line.strip() for line in modified_classes_raw.splitlines() if line.strip()]
                trigger_tests = [line.strip() for line in trigger_tests_raw.splitlines() if line.strip()]

                bug_file_records: List[Dict[str, Any]] = []

                for class_name in modified_classes:
                    rel_path = class_name_to_java_path(class_name, src_dir)
                    buggy_file = buggy_dir / rel_path
                    fixed_file = fixed_dir / rel_path

                    if not buggy_file.exists() or not fixed_file.exists():
                        print(f"  [SKIP] {rel_path} missing in buggy/fixed checkout")
                        continue

                    buggy_code = buggy_file.read_text(encoding="utf-8", errors="ignore")
                    fixed_code = fixed_file.read_text(encoding="utf-8", errors="ignore")

                    if args.mode == "generative":
                        # No diff-based masking: model receives full buggy file and must
                        # generate the complete fixed file from scratch.
                        if causal:
                            # Standard autoregressive model (e.g. Llama-2)
                            metrics = evaluate_instance_generative_causal(
                                model, tokenizer, buggy_code, fixed_code, args
                            )
                        else:
                            # LLaDA masked-diffusion append-and-denoise
                            metrics = evaluate_instance_generative(
                                model, tokenizer, buggy_code, fixed_code, mask_token_id, args
                            )
                    else:
                        # Oracle-localized mode: diff hunks are masked, model fills them in.
                        # Only supported for masked-diffusion models (LLaDA).
                        record = build_localized_record(
                            tokenizer=tokenizer,
                            mask_token_id=mask_token_id,
                            buggy_code=buggy_code,
                            fixed_code=fixed_code,
                            max_seq_len=args.max_seq_len,
                        )
                        if record is None:
                            print(f"  [SKIP] {rel_path} has no usable multi-hunk diff after tokenization")
                            continue
                        metrics = evaluate_instance(model, tokenizer, record, fixed_code, args)

                    metrics.update({
                        "mode": args.mode,
                        "project_id": project_id,
                        "bug_id": bug_id,
                        "class_name": class_name,
                        "file_path": str(rel_path),
                    })

                    predicted_out = predictions_dir / project_id / str(bug_id) / rel_path
                    ensure_parent(predicted_out)
                    predicted_out.write_text(metrics["predicted_code"], encoding="utf-8")
                    metrics["predicted_file"] = str(predicted_out)

                    # Apply predicted file to buggy checkout for optional validation.
                    buggy_file.write_text(metrics["predicted_code"], encoding="utf-8")

                    file_records.append(metrics)
                    bug_file_records.append(metrics)

                    hunks_info = (
                        f"gen_tokens={metrics['generated_tokens']}"
                        if args.mode == "generative"
                        else f"hunks={metrics['num_hunks']}"
                    )
                    print(
                        f"  [FILE] {rel_path} "
                        f"{hunks_info} "
                        f"exact={metrics['full_file_exact_match']} "
                        f"patch_em={metrics['patch_string_em']:.4f} "
                        f"bleu={metrics['patch_bleu']:.4f}"
                    )

                bug_result: Dict[str, Any] = {
                    "project_id": project_id,
                    "bug_id": bug_id,
                    "num_modified_files": len(bug_file_records),
                    "all_files_exact": bool(bug_file_records) and all(
                        record["full_file_exact_match"] for record in bug_file_records
                    ),
                    "compile_ok": None,
                    "trigger_tests_ok": None,
                    "full_test_suite_ok": None,
                    "trigger_test_count": len(trigger_tests),
                }

                if not args.skip_validation and bug_file_records:
                    compile_result = run_defects4j_compile(defects4j_cmd, buggy_dir)
                    bug_result["compile_ok"] = compile_result["ok"]
                    bug_result["compile_stdout"] = compile_result["stdout"]
                    bug_result["compile_stderr"] = compile_result["stderr"]

                    if compile_result["ok"]:
                        if trigger_tests:
                            trigger_ok = True
                            failing_trigger_tests: List[str] = []
                            for test_name in trigger_tests:
                                test_result = run_defects4j_test(defects4j_cmd, buggy_dir, test_name=test_name)
                                if not test_result["ok"]:
                                    trigger_ok = False
                                    failing_trigger_tests.append(test_name)
                            bug_result["trigger_tests_ok"] = trigger_ok
                            bug_result["failing_trigger_tests"] = failing_trigger_tests
                        else:
                            bug_result["trigger_tests_ok"] = None

                        if args.run_full_test_suite:
                            full_suite_result = run_defects4j_test(defects4j_cmd, buggy_dir)
                            bug_result["full_test_suite_ok"] = full_suite_result["ok"]
                            bug_result["full_suite_failing_tests"] = full_suite_result["failing_tests"]
                    else:
                        bug_result["trigger_tests_ok"] = False if trigger_tests else None
                        if args.run_full_test_suite:
                            bug_result["full_test_suite_ok"] = False

                bug_records.append(bug_result)

                print(
                    f"  [BUG-SUMMARY] files={bug_result['num_modified_files']} "
                    f"all_exact={bug_result['all_files_exact']} "
                    f"compile={bug_result['compile_ok']} "
                    f"trigger_tests={bug_result['trigger_tests_ok']}"
                )

            finally:
                if work_root is None and temp_root.exists():
                    shutil.rmtree(temp_root, ignore_errors=True)

    summary = aggregate_summary(file_records, bug_records, args)

    file_results_path = output_dir / "file_results.jsonl"
    bug_results_path = output_dir / "bug_results.jsonl"
    bug_csv_path = output_dir / "bug_results.csv"
    summary_path = output_dir / "summary.json"

    write_jsonl(file_results_path, file_records)
    write_jsonl(bug_results_path, bug_records)
    write_bug_csv(bug_csv_path, bug_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("JAVA SWE-STYLE SUMMARY")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\n[INFO] File results: {file_results_path}")
    print(f"[INFO] Bug results: {bug_results_path}")
    print(f"[INFO] Bug CSV: {bug_csv_path}")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()