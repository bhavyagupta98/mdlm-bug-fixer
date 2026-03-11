#!/usr/bin/env python3
"""
Zero-shot baseline evaluation for multi-hunk infilling.

Compares against MDLM (model.py / inference.py) on an apples-to-apples basis:

  Strategy A - FIM chaining  (fill-in-the-middle native models)
    Models:
      deepseek-coder  : deepseek-ai/deepseek-coder-6.7b-base
      starcoder2      : bigcode/starcoder2-7b
      codellama       : codellama/CodeLlama-7b-hf
    Each hunk is filled sequentially using the model's native FIM tokens.
    After each hunk is filled the working code is updated, so downstream
    hunks see already-repaired prefix/suffix context -- mirroring iterative
    denoising in MDLM.

  Strategy B - instruct zero-shot  (chat / instruct models)
    Models:
      qwen              : Qwen/Qwen2.5-Coder-7B-Instruct
      deepseek-instruct : deepseek-ai/deepseek-coder-7b-instruct-v1.5
    A single prompt names all masked regions; the model returns PATCH_i lines.
    No fine-tuning.  This is the minimal AR reference point.

Both strategies use the same evaluation.py metrics, producing results directly
comparable to inference.py (MDLM).

Usage:
  # FIM chaining with deepseek-coder (default)
  python baseline_reconstruction_eval.py --strategy fim

  # FIM with starcoder2
  python baseline_reconstruction_eval.py --strategy fim --baseline starcoder2

  # Instruct zero-shot with Qwen
  python baseline_reconstruction_eval.py --strategy instruct --baseline qwen

  # pass@10 sampling
  python baseline_reconstruction_eval.py --strategy instruct --num-samples 10 --k 10

  # Quick sanity check
  python baseline_reconstruction_eval.py --strategy fim --dry-run --max-records 5
"""

import argparse
import gzip
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation import (
    MetricsAggregator,
    all_hunks_correct,
    compute_codebleu,
    normalized_edit_distance,
    pass_at_k_unbiased,
    patch_bleu,
    patch_string_exact_match,
    token_edit_distance,
)

# ============================================================
# CONSTANTS
# ============================================================
BASE_DIR              = Path(__file__).resolve().parent
DEFAULT_DATA_GZ       = BASE_DIR / "processed_train.jsonl.gz"
DEFAULT_OUT_DIR       = BASE_DIR / "runs" / "baselines"
SOURCE_TOKENIZER_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_MAX_LEN       = 4096
RANDOM_SEED           = 0

# ---- FIM model presets ----
# Each entry has the HF model ID and the three FIM special-token strings.
FIM_PRESETS: Dict[str, Dict[str, Any]] = {
    "deepseek-coder": {
        "model":  "deepseek-ai/deepseek-coder-6.7b-base",
        "prefix": "<\uff5cfim\u25a0begin\uff5c>",
        "suffix": "<\uff5cfim\u25a0suffix\uff5c>",
        "middle": "<\uff5cfim\u25a0hole\uff5c>",
    },
    "starcoder2": {
        "model":  "bigcode/starcoder2-7b",
        "prefix": "<fim_prefix>",
        "suffix": "<fim_suffix>",
        "middle": "<fim_middle>",
    },
    "codellama": {
        "model":  "codellama/CodeLlama-7b-hf",
        "prefix": "\u2581<PRE>",
        "suffix": "\u2581<SUF>",
        "middle": "\u2581<MID>",
    },
}

# ---- Instruct model presets ----
INSTRUCT_PRESETS: Dict[str, str] = {
    "qwen":              "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
}

PATCH_LINE_RE = re.compile(r"^\s*PATCH_(\d+)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)


# ============================================================
# HELPERS
# ============================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _filter_spans(spans: List[Any], L: int) -> List[Tuple[int, int]]:
    out = []
    for s, e in spans:
        s, e = max(0, int(s)), min(L, int(e))
        if e > s:
            out.append((s, e))
    return out


def decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


# ============================================================
# DATA LOADING
# ============================================================

@dataclass
class Record:
    idx: int
    input_ids: List[int]
    hunk_spans: List[Tuple[int, int]]


def load_records(
    data_gz: Path,
    source_tokenizer,
    max_records: Optional[int],
    max_seq_len: int,
    min_hunks: int = 2,
) -> List[Record]:
    """
    Load records with at least `min_hunks` valid spans.
    Mirrors filtering in model.py / inference.py.
    """
    records: List[Record] = []
    with gzip.open(data_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue

            ids   = ex.get("input_ids", [])
            spans = ex.get("hunk_token_spans", [])
            if not ids or not spans:
                continue

            ids      = ids[:max_seq_len]
            filtered = _filter_spans(spans, len(ids))
            if len(filtered) < min_hunks:
                continue

            records.append(Record(idx=len(records), input_ids=ids, hunk_spans=filtered))
            if max_records is not None and len(records) >= max_records:
                break

    if not records:
        raise RuntimeError(f"No records with >= {min_hunks} hunks found in {data_gz}")
    return records


# ============================================================
# STRATEGY A: FIM CHAINING
# ============================================================

class FIMBaseline:
    """
    Fill-in-the-Middle chaining baseline.

    For each hunk i in left-to-right order:
      1. Build FIM prompt:
             <prefix_tok> code_up_to_hunk <suffix_tok> code_after_hunk <middle_tok>
         where 'code_up_to_hunk' already includes predictions for hunks 0..i-1.
      2. Greedily decode the middle.
      3. Update the working token sequence before moving to hunk i+1.

    This setup is architecturally equivalent to MDLM infilling: the model
    sees the full prefix AND suffix context for every hunk.  The only
    difference is AR generation vs masked diffusion.
    """

    def __init__(
        self,
        model,
        tokenizer,
        preset: Dict[str, Any],
        device: torch.device,
        max_new_tokens: int = 256,
        source_tokenizer=None,
    ):
        self.model            = model
        self.tokenizer        = tokenizer
        self.preset           = preset
        self.device           = device
        self.max_new_tokens   = max_new_tokens
        self.source_tokenizer = source_tokenizer

        for role in ("prefix", "suffix", "middle"):
            tok_str = preset[role]
            tok_id  = tokenizer.convert_tokens_to_ids(tok_str)
            if tok_id == tokenizer.unk_token_id:
                print(f"[WARN] FIM token '{tok_str}' not in vocab -- FIM quality may degrade.")

    @torch.no_grad()
    def _fill_one(self, prefix_text: str, suffix_text: str) -> str:
        """Single greedy FIM call; returns generated middle text."""
        fim_input = (
            self.preset["prefix"] + prefix_text
            + self.preset["suffix"] + suffix_text
            + self.preset["middle"]
        )
        enc = self.tokenizer(
            fim_input, return_tensors="pt",
            truncation=True, max_length=DEFAULT_MAX_LEN,
        ).to(self.device)

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][enc["input_ids"].size(1):]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def run(self, record: Record, num_samples: int = 1) -> List[List[str]]:
        """
        Returns `num_samples` candidate patch lists.
        Sample 0 is always greedy; samples 1+ use temperature=0.8.
        """
        src_tok     = self.source_tokenizer
        all_samples: List[List[str]] = []

        for sample_i in range(num_samples):
            current_ids = list(record.input_ids)
            spans       = list(record.hunk_spans)
            patches: List[str] = []

            for hi, (s, e) in enumerate(spans):
                prefix_text = decode_ids(src_tok, current_ids[:s])
                suffix_text = decode_ids(src_tok, current_ids[e:])

                if sample_i == 0:
                    filled = self._fill_one(prefix_text, suffix_text)
                else:
                    fim_input = (
                        self.preset["prefix"] + prefix_text
                        + self.preset["suffix"] + suffix_text
                        + self.preset["middle"]
                    )
                    enc = self.tokenizer(
                        fim_input, return_tensors="pt",
                        truncation=True, max_length=DEFAULT_MAX_LEN,
                    ).to(self.device)
                    with torch.no_grad():
                        out = self.model.generate(
                            **enc,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=True, temperature=0.8, top_p=0.95,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    filled = self.tokenizer.decode(
                        out[0][enc["input_ids"].size(1):], skip_special_tokens=True
                    )

                patches.append(filled)

                # Update current_ids and re-map downstream span offsets
                filled_ids  = src_tok(filled, add_special_tokens=False)["input_ids"]
                delta       = len(filled_ids) - (e - s)
                current_ids = current_ids[:s] + filled_ids + current_ids[e:]
                spans = [
                    (a, b) if j <= hi else (a + delta, b + delta)
                    for j, (a, b) in enumerate(spans)
                ]

            all_samples.append(patches)

        return all_samples


# ============================================================
# STRATEGY B: INSTRUCT ZERO-SHOT
# ============================================================

class InstructBaseline:
    """
    Zero-shot instruct baseline.

    One prompt presents all masked regions simultaneously; the model returns
    PATCH_i: <text> lines.  No fine-tuning.  This is the minimal AR baseline:
    it shows what a capable code-repair LLM can do without any task-specific
    training, and without the benefit of suffix context per hunk.
    """

    SYSTEM_PROMPT = (
        "You are a code repair assistant. "
        "Return ONLY the replacement lines in the exact format requested. "
        "Do not explain anything. Do not output the full file."
    )

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        device: torch.device,
        max_new_tokens: int = 256,
        source_tokenizer=None,
    ):
        self.model            = model
        self.tokenizer        = tokenizer
        self.model_name       = model_name
        self.device           = device
        self.max_new_tokens   = max_new_tokens
        self.source_tokenizer = source_tokenizer

    def _build_prompt(self, masked_code: str, n_hunks: int, language: str) -> str:
        patch_spec = "\n".join(f"PATCH_{i}: <replacement_{i}>" for i in range(n_hunks))
        return (
            "The following code contains masked regions.\n"
            "Each masked region is marked with <MASK_i>.\n"
            "Return ONLY replacements in this exact format:\n\n"
            f"{patch_spec}\n\n"
            "Rules:\n"
            "1. Do not output the full file.\n"
            "2. One PATCH_i line per mask, in order.\n\n"
            f"Language: {language}\n"
            f"Code:\n```{language}\n{masked_code}\n```\n\n"
            "Replacements:\n"
        )

    def _apply_chat_template(self, raw_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": raw_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return raw_prompt

    def mask_code(self, record: Record) -> Tuple[str, List[str]]:
        """Returns (masked_code_str, gold_patch_texts)."""
        ids, spans = record.input_ids, record.hunk_spans
        parts: List[str] = []
        gold:  List[str] = []
        cursor = 0
        for i, (s, e) in enumerate(spans):
            parts.append(decode_ids(self.source_tokenizer, ids[cursor:s]))
            parts.append(f"<MASK_{i}>")
            gold.append(decode_ids(self.source_tokenizer, ids[s:e]))
            cursor = e
        parts.append(decode_ids(self.source_tokenizer, ids[cursor:]))
        return "".join(parts), gold

    @staticmethod
    def _parse_patches(text: str, n_hunks: int) -> List[str]:
        found: Dict[int, str] = {}
        for line in text.splitlines():
            m = PATCH_LINE_RE.match(line)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < n_hunks and idx not in found:
                    found[idx] = m.group(2)
        return [found.get(i, "") for i in range(n_hunks)]

    @torch.no_grad()
    def run(
        self,
        record: Record,
        num_samples: int = 1,
        temperature: float = 0.8,
        language: str = "java",
    ) -> Tuple[List[List[str]], List[str]]:
        """Returns (list_of_patch_lists, gold_patch_texts)."""
        masked_code, gold = self.mask_code(record)
        raw_prompt        = self._build_prompt(masked_code, len(record.hunk_spans), language)
        prompt            = self._apply_chat_template(raw_prompt)

        enc = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=DEFAULT_MAX_LEN,
        ).to(self.device)

        do_sample = (num_samples > 1)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.95 if do_sample else None,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        input_len   = enc["input_ids"].size(1)
        all_samples = []
        for seq in out:
            gen_text = self.tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            all_samples.append(self._parse_patches(gen_text, len(record.hunk_spans)))

        return all_samples, gold


# ============================================================
# SHARED EVALUATION LOOP
# ============================================================

def _evaluate(
    strategy: str,
    baseline,
    records: List[Record],
    source_tokenizer,
    num_samples: int,
    k: int,
    temperature: float,
    codebleu_lang: str,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Unified evaluation loop for both FIM and instruct strategies.

    For each record, `num_samples` patch candidates are generated.
    The best candidate (by patch_string_em) contributes to the aggregator.
    pass@k is computed with the unbiased Codex estimator from evaluation.py.
    """
    agg = MetricsAggregator()
    pass_at_k_per_record: List[float] = []
    gen_tokens = 0
    gen_time   = 0.0

    for i, record in enumerate(records):
        gold_full = decode_ids(source_tokenizer, record.input_ids)

        t0 = time.perf_counter()
        if strategy == "fim":
            candidate_lists = baseline.run(record, num_samples=num_samples)
            gold_patches    = [
                decode_ids(source_tokenizer, record.input_ids[s:e])
                for s, e in record.hunk_spans
            ]
        else:
            candidate_lists, gold_patches = baseline.run(
                record,
                num_samples=num_samples,
                temperature=temperature,
                language=codebleu_lang,
            )
        gen_time += time.perf_counter() - t0

        gold_patch_ids = [
            source_tokenizer(g, add_special_tokens=False)["input_ids"]
            for g in gold_patches
        ]

        correct_count = 0
        best_metrics  = None
        best_score    = -1.0

        for patches in candidate_lists:
            # Reconstruct full file from patch predictions
            if strategy == "fim":
                current_ids = list(record.input_ids)
                spans       = list(record.hunk_spans)
                for hi, (s, e) in enumerate(spans):
                    filled_ids  = source_tokenizer(patches[hi], add_special_tokens=False)["input_ids"]
                    delta       = len(filled_ids) - (e - s)
                    current_ids = current_ids[:s] + filled_ids + current_ids[e:]
                    spans = [
                        (a, b) if j <= hi else (a + delta, b + delta)
                        for j, (a, b) in enumerate(spans)
                    ]
                pred_full = decode_ids(source_tokenizer, current_ids)
            else:
                masked_code, _ = baseline.mask_code(record)
                pred_full = masked_code
                for hi, patch in enumerate(patches):
                    pred_full = pred_full.replace(f"<MASK_{hi}>", patch)

            if pred_full.strip() == gold_full.strip():
                correct_count += 1

            # Per-hunk metrics
            per_hunk_em: List[bool]  = []
            hunk_bleu:   List[float] = []
            hunk_ned:    List[float] = []
            hunk_str_em: List[float] = []

            for pred_patch, gold_patch, gold_ids in zip(patches, gold_patches, gold_patch_ids):
                pred_ids    = source_tokenizer(pred_patch, add_special_tokens=False)["input_ids"]
                gen_tokens += len(pred_ids)
                is_em       = patch_string_exact_match(pred_patch, gold_patch)
                per_hunk_em.append(is_em)
                hunk_str_em.append(float(is_em))
                hunk_bleu.append(patch_bleu(pred_patch, gold_patch))
                hunk_ned.append(normalized_edit_distance(pred_ids, gold_ids))

            cb    = compute_codebleu(pred_full, gold_full, language=codebleu_lang)
            score = sum(hunk_str_em) / max(len(hunk_str_em), 1)

            rec_metrics = {
                "per_hunk_exact":    per_hunk_em,
                "all_hunks_correct": all_hunks_correct(per_hunk_em),
                "patch_string_em":   score,
                "patch_bleu":        sum(hunk_bleu) / max(len(hunk_bleu), 1),
                "patch_ned":         sum(hunk_ned)  / max(len(hunk_ned),  1),
                "codebleu":          cb.get("codebleu") if cb else None,
            }
            if score > best_score:
                best_score   = score
                best_metrics = rec_metrics

        pass_at_k_per_record.append(pass_at_k_unbiased(num_samples, correct_count, k))
        if best_metrics is not None:
            agg.add(best_metrics)

        if (i + 1) % verbose_every == 0:
            print(f"[EVAL] {i + 1}/{len(records)} records done")

    summary = agg.summary()
    summary.update({
        "pass_at_k":                 sum(pass_at_k_per_record) / max(len(pass_at_k_per_record), 1),
        "k":                         k,
        "num_samples":               num_samples,
        "avg_gen_tokens_per_record": gen_tokens / max(len(records), 1),
        "avg_decode_sec_per_record": gen_time   / max(len(records), 1),
        "tokens_per_second":         gen_tokens / max(gen_time, 1e-8),
    })
    return summary


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot FIM / instruct baselines for multi-hunk infilling"
    )
    parser.add_argument(
        "--strategy", choices=["fim", "instruct"], default="fim",
        help=(
            "fim      = FIM-chaining with a native FIM model "
            "(deepseek-coder / starcoder2 / codellama).  "
            "instruct = zero-shot prompting of a chat model "
            "(qwen / deepseek-instruct)."
        ),
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help=(
            "Named preset.  "
            "FIM presets: " + ", ".join(FIM_PRESETS) + ".  "
            "Instruct presets: " + ", ".join(INSTRUCT_PRESETS) + ".  "
            "Default: deepseek-coder (fim), qwen (instruct)."
        ),
    )
    parser.add_argument("--model-name", type=str, default=None,
                        help="Explicit HF model ID; overrides --baseline preset.")
    parser.add_argument("--data-gz",          type=str, default=str(DEFAULT_DATA_GZ))
    parser.add_argument("--source-tokenizer", type=str, default=SOURCE_TOKENIZER_NAME)
    parser.add_argument("--out-dir",          type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-records",      type=int, default=None,
                        help="Cap number of records loaded (for quick tests).")
    parser.add_argument("--max-seq-len",      type=int, default=4096)
    parser.add_argument("--max-new-tokens",   type=int, default=256)
    parser.add_argument("--num-samples",      type=int, default=1,
                        help="Samples per record for unbiased pass@k estimation.")
    parser.add_argument("--k",                type=int, default=1,
                        help="k in pass@k.")
    parser.add_argument("--temperature",      type=float, default=0.8,
                        help="Sampling temperature (used when num-samples > 1).")
    parser.add_argument("--language",         type=str, default="java")
    parser.add_argument("--use-bf16",         action="store_true")
    parser.add_argument("--dry-run",          action="store_true",
                        help="Print first prompt/response then exit.")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype   = torch.bfloat16 if (args.use_bf16 and bf16_ok) else torch.float32

    # ---- Resolve model ----
    if args.strategy == "fim":
        preset_name = args.baseline or "deepseek-coder"
        if preset_name not in FIM_PRESETS:
            raise ValueError(
                f"Unknown FIM baseline '{preset_name}'. Choices: {sorted(FIM_PRESETS)}"
            )
        preset   = FIM_PRESETS[preset_name]
        model_id = args.model_name or preset["model"]
    else:
        preset_name = args.baseline or "qwen"
        if preset_name not in INSTRUCT_PRESETS:
            raise ValueError(
                f"Unknown instruct baseline '{preset_name}'. Choices: {sorted(INSTRUCT_PRESETS)}"
            )
        preset   = None
        model_id = args.model_name or INSTRUCT_PRESETS[preset_name]

    out_dir = Path(args.out_dir) / f"{args.strategy}_{preset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizers ----
    print(f"[INFO] Source tokenizer : {args.source_tokenizer}")
    source_tok = AutoTokenizer.from_pretrained(
        args.source_tokenizer, trust_remote_code=True, use_fast=True
    )

    print(f"[INFO] Model tokenizer  : {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # ---- Model ----
    print(f"[INFO] Loading model    : {model_id}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device).eval()

    # ---- Data ----
    print(f"[INFO] Loading records from {args.data_gz}")
    records = load_records(
        Path(args.data_gz), source_tok,
        max_records=args.max_records,
        max_seq_len=args.max_seq_len,
    )
    print(f"[INFO] {len(records)} records loaded  (>= 2 hunks each)")

    # ---- Baseline object ----
    if args.strategy == "fim":
        baseline = FIMBaseline(
            model=model, tokenizer=tok, preset=preset,
            device=device, max_new_tokens=args.max_new_tokens,
            source_tokenizer=source_tok,
        )
    else:
        baseline = InstructBaseline(
            model=model, tokenizer=tok, model_name=model_id,
            device=device, max_new_tokens=args.max_new_tokens,
            source_tokenizer=source_tok,
        )

    # ---- Dry-run ----
    if args.dry_run:
        rec = records[0]
        print(f"\n[DRY-RUN] record 0: {len(rec.input_ids)} tokens, "
              f"{len(rec.hunk_spans)} hunks: {rec.hunk_spans}")
        if args.strategy == "fim":
            gold_patches = [decode_ids(source_tok, rec.input_ids[s:e]) for s, e in rec.hunk_spans]
            prefix_text  = decode_ids(source_tok, rec.input_ids[:rec.hunk_spans[0][0]])
            suffix_text  = decode_ids(source_tok, rec.input_ids[rec.hunk_spans[0][1]:])
            print(f"\nFIM prompt hunk-0 prefix (first 300 chars):\n{prefix_text[:300]!r}")
            print(f"FIM prompt hunk-0 suffix (last  300 chars):\n{suffix_text[-300:]!r}")
            patches = baseline.run(rec, num_samples=1)[0]
        else:
            masked_code, gold_patches = baseline.mask_code(rec)
            prompt = baseline._build_prompt(masked_code, len(rec.hunk_spans), args.language)
            print(f"\nInstruct prompt (first 2000 chars):\n{prompt[:2000]}")
            samples, gold_patches = baseline.run(rec, num_samples=1, language=args.language)
            patches = samples[0]
        print(f"\nPredicted patches : {patches}")
        print(f"Gold patches      : {gold_patches}")
        return

    # ---- Evaluate ----
    print(f"\n[INFO] strategy={args.strategy}  model={model_id}  "
          f"num_samples={args.num_samples}  k={args.k}")

    summary = _evaluate(
        strategy=args.strategy,
        baseline=baseline,
        records=records,
        source_tokenizer=source_tok,
        num_samples=args.num_samples,
        k=args.k,
        temperature=args.temperature,
        codebleu_lang=args.language,
    )
    summary["model"]    = model_id
    summary["strategy"] = args.strategy

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:45s}: {val:.4f}")
        else:
            print(f"  {key:45s}: {val}")

    results_path = out_dir / "eval_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Results saved to {results_path}")


if __name__ == "__main__":
    main()
