#!/usr/bin/env python3
"""
Train and evaluate strong autoregressive reconstruction baselines on the same
multi-hunk processed dataset used by the masked diffusion pipeline.

What this script assumes
------------------------
- The .jsonl.gz file already contains:
    * input_ids               : tokenized clean/fixed code in SOURCE tokenizer space
    * hunk_token_spans        : spans of repair regions in SOURCE tokenizer space
    * mask_token_id           : present but not needed here
- One base record expands to multiple subset-of-hunk training samples.
- We split at the BASE RECORD level first, then expand, to avoid leakage.

What baseline setting this implements
-------------------------------------
This is a *localized reconstruction* baseline, not a no-localization baseline.
Each example gets oracle masked spans and must reconstruct the gold content in
those spans autoregressively.

Supported baselines
-------------------
- qwen       -> Qwen/Qwen2.5-Coder-7B-Instruct
- starcoder2 -> bigcode/starcoder2-7b
You can also pass a raw Hugging Face model name via --model-name.

Evaluation
----------
This script reuses evaluation.py when available and additionally reports:
- pass@k                     : exact full-code hit among k sampled candidates
- localization_accuracy      : 1.0 here, because spans are oracle
- patch_minimality           : normalized patch edit quality
- generation_efficiency      : decode speed / generated token counts
"""

import argparse
import bisect
import gzip
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Optional reuse of your existing evaluation helpers.
try:
    from eval_metrics import (
        token_exact_match_rate,
        token_edit_distance,
        compute_codebleu,
        MetricsAggregator,
    )
except Exception:
    token_exact_match_rate = None
    token_edit_distance = None
    compute_codebleu = None
    MetricsAggregator = None


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_GZ = BASE_DIR / "processed_train.jsonl.gz"
DEFAULT_OUT_DIR = BASE_DIR / "runs"
SOURCE_TOKENIZER_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_FALLBACK_MAX_LEN = 2048
RANDOM_SEED = 0

BASELINE_PRESETS = {
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "starcoder2": "bigcode/starcoder2-7b",
}

PATCH_RE = re.compile(r"<PATCH_(\d+)>(.*?)</PATCH_\1>", re.DOTALL)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_max_len(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return DEFAULT_FALLBACK_MAX_LEN
    for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    return DEFAULT_FALLBACK_MAX_LEN


def _filter_spans(spans: List[List[int]], L: int) -> List[Tuple[int, int]]:
    filtered: List[Tuple[int, int]] = []
    for s, e in spans:
        s = max(0, int(s))
        e = min(L, int(e))
        if e > s:
            filtered.append((s, e))
    return filtered


def all_valid_subset_masks(k: int, min_hunks: int = 2) -> List[int]:
    out = []
    for mask in range(1, 1 << k):
        if bin(mask).count('1') >= min_hunks:
            out.append(mask)
    return out


def decode_tokens(tok, ids: Sequence[int]) -> str:
    return tok.decode(
        list(ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_masked_code_and_targets(
    source_tokenizer,
    source_input_ids: List[int],
    chosen_spans: List[Tuple[int, int]],
) -> Tuple[str, List[str], str]:
    """
    Returns
    -------
    masked_code_text : full code with <BUG_MASK_i> placeholders
    patch_texts      : gold text for each chosen hunk in order
    full_code_text   : original clean/fixed full code
    """
    full_code_text = decode_tokens(source_tokenizer, source_input_ids)

    chosen_spans = sorted(chosen_spans, key=lambda x: x[0])
    parts: List[str] = []
    patch_texts: List[str] = []
    cursor = 0
    for i, (s, e) in enumerate(chosen_spans):
        parts.append(decode_tokens(source_tokenizer, source_input_ids[cursor:s]))
        parts.append(f"<BUG_MASK_{i}>")
        patch_texts.append(decode_tokens(source_tokenizer, source_input_ids[s:e]))
        cursor = e
    parts.append(decode_tokens(source_tokenizer, source_input_ids[cursor:]))

    return "".join(parts), patch_texts, full_code_text


def build_prompt(masked_code_text: str, n_patches: int, language: str = "java") -> str:
    patch_spec = "\n".join(f"<PATCH_{i}>...replacement...</PATCH_{i}>" for i in range(n_patches))
    return (
        "You are a code repair assistant. The input code contains one or more masked bug regions.\n"
        "Each masked region is marked as <BUG_MASK_i>. Return ONLY the missing replacements, in order,\n"
        "using exactly the following format and nothing else:\n"
        f"{patch_spec}\n\n"
        "Do not return the full file. Do not explain anything.\n\n"
        f"Language: {language}\n"
        f"Masked code:\n```{language}\n{masked_code_text}\n```\n\n"
        "Replacements:\n"
    )


def build_target_patch_block(patch_texts: List[str]) -> str:
    return "\n".join(f"<PATCH_{i}>{txt}</PATCH_{i}>" for i, txt in enumerate(patch_texts))


def parse_patch_block(text: str, expected_count: int) -> List[str]:
    matches = PATCH_RE.findall(text)
    found: Dict[int, str] = {}
    for idx_str, body in matches:
        idx = int(idx_str)
        if idx not in found:
            found[idx] = body
    return [found.get(i, "") for i in range(expected_count)]


def reconstruct_full_code(masked_code_text: str, predicted_patches: List[str]) -> str:
    repaired = masked_code_text
    for i, patch in enumerate(predicted_patches):
        repaired = repaired.replace(f"<BUG_MASK_{i}>", patch)
    return repaired


def patch_minimality_ratio(pred_ids: List[int], gt_ids: List[int]) -> float:
    if token_edit_distance is None:
        return 0.0
    denom = max(1, len(gt_ids))
    return max(0.0, 1.0 - (token_edit_distance(pred_ids, gt_ids) / denom))


class BaseRecordIndex:
    def __init__(self, path: Path, max_records: Optional[int], max_source_seq_len: int):
        self.path = path
        self.offsets: List[int] = []
        self.valid_masks: List[List[int]] = []
        self.k_valid: List[int] = []
        self.max_source_seq_len = int(max_source_seq_len)

        with gzip.open(self.path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue

                try:
                    ex = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                input_ids = ex.get("input_ids", [])
                spans = ex.get("hunk_token_spans", [])
                if not isinstance(input_ids, list) or len(input_ids) == 0:
                    continue

                L = min(len(input_ids), self.max_source_seq_len)
                filtered = _filter_spans(spans, L)
                k = len(filtered)
                if k < 2:
                    continue

                masks = all_valid_subset_masks(k, min_hunks=2)
                if not masks:
                    continue

                self.offsets.append(pos)
                self.valid_masks.append(masks)
                self.k_valid.append(k)

                if max_records is not None and len(self.offsets) >= max_records:
                    break

        if not self.offsets:
            raise RuntimeError(f"No usable records found in {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def read_record(self, idx: int) -> Dict[str, Any]:
        with gzip.open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            line = f.readline().decode("utf-8")
        ex = json.loads(line)
        input_ids = ex["input_ids"][: self.max_source_seq_len]
        spans = _filter_spans(ex["hunk_token_spans"], len(input_ids))
        ex["input_ids"] = input_ids
        ex["hunk_token_spans"] = spans
        return ex


def split_base_records(num_records: int, eval_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(range(num_records))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_eval = max(1, int(round(num_records * eval_ratio)))
    eval_ids = sorted(ids[:n_eval])
    train_ids = sorted(ids[n_eval:])
    return train_ids, eval_ids


class ExpandedRepairDataset(Dataset):
    def __init__(
        self,
        index: BaseRecordIndex,
        source_tokenizer,
        base_record_ids: List[int],
        max_examples: Optional[int] = None,
        language: str = "java",
    ):
        self.index = index
        self.source_tokenizer = source_tokenizer
        self.base_record_ids = list(base_record_ids)
        self.language = language

        self.prefix: List[int] = [0]
        for rid in self.base_record_ids:
            self.prefix.append(self.prefix[-1] + len(self.index.valid_masks[rid]))
        self.total_examples = min(self.prefix[-1], int(max_examples)) if max_examples is not None else self.prefix[-1]

    def __len__(self) -> int:
        return self.total_examples

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(global_idx)
        r = bisect.bisect_right(self.prefix, global_idx) - 1
        local = global_idx - self.prefix[r]
        base_id = self.base_record_ids[r]
        subset_mask = self.index.valid_masks[base_id][local]
        return base_id, subset_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_id, subset_mask = self._locate(idx)
        rec = self.index.read_record(base_id)
        input_ids: List[int] = rec["input_ids"]
        spans: List[Tuple[int, int]] = rec["hunk_token_spans"]

        chosen_spans = [spans[i] for i in range(len(spans)) if (subset_mask >> i) & 1]
        masked_code_text, patch_texts, full_code_text = build_masked_code_and_targets(
            self.source_tokenizer,
            input_ids,
            chosen_spans,
        )
        prompt = build_prompt(masked_code_text, len(patch_texts), language=self.language)
        target = build_target_patch_block(patch_texts)

        return {
            "base_record_id": base_id,
            "subset_mask": subset_mask,
            "prompt": prompt,
            "target": target,
            "masked_code_text": masked_code_text,
            "gold_patch_texts": patch_texts,
            "gold_full_code": full_code_text,
            "num_hunks": len(patch_texts),
        }


@dataclass
class ARRepairCollator:
    tokenizer: Any
    max_seq_len: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_id_tensors = []
        attn_tensors = []
        label_tensors = []
        meta = []

        for ex in batch:
            prompt_ids = self.tokenizer(
                ex["prompt"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"]
            target_ids = self.tokenizer(
                ex["target"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"]

            eos = [] if self.tokenizer.eos_token_id is None else [self.tokenizer.eos_token_id]
            merged = (prompt_ids + target_ids + eos)[: self.max_seq_len]

            labels = [-100] * min(len(prompt_ids), len(merged))
            labels.extend(merged[len(labels):])
            labels = labels[: len(merged)]

            ids = torch.tensor(merged, dtype=torch.long)
            input_id_tensors.append(ids)
            attn_tensors.append(torch.ones_like(ids))
            label_tensors.append(torch.tensor(labels, dtype=torch.long))
            meta.append(ex)

        max_len = max(x.size(0) for x in input_id_tensors)
        pad_id = self.tokenizer.pad_token_id

        def pad_1d(x: torch.Tensor, pad_val: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            pad = torch.full((max_len - x.size(0),), pad_val, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        return {
            "input_ids": torch.stack([pad_1d(x, pad_id) for x in input_id_tensors], dim=0),
            "attention_mask": torch.stack([pad_1d(x, 0) for x in attn_tensors], dim=0),
            "labels": torch.stack([pad_1d(x, -100) for x in label_tensors], dim=0),
            "meta": meta,
        }


class MetaIgnoringTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        inputs.pop("meta", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def evaluate_model(
    model,
    baseline_tokenizer,
    source_tokenizer,
    eval_ds: ExpandedRepairDataset,
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
    num_return_sequences: int,
    codebleu_lang: str = "java",
) -> Dict[str, Any]:
    model.eval()

    agg = MetricsAggregator() if MetricsAggregator is not None else None
    total_samples = 0
    total_hunks = 0
    pass_at_k_hits = 0
    patch_minimality_values = []
    gen_tokens = 0
    gen_time_sec = 0.0

    for i in range(len(eval_ds)):
        ex = eval_ds[i]
        enc = baseline_tokenizer(
            ex["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=getattr(model.config, "max_position_embeddings", DEFAULT_FALLBACK_MAX_LEN),
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=(num_return_sequences > 1),
                temperature=0.8 if num_return_sequences > 1 else None,
                top_p=0.95 if num_return_sequences > 1 else None,
                num_beams=num_beams if num_return_sequences == 1 else 1,
                num_return_sequences=num_return_sequences,
                pad_token_id=baseline_tokenizer.pad_token_id,
                eos_token_id=baseline_tokenizer.eos_token_id,
            )
        gen_time_sec += time.perf_counter() - start

        input_len = enc["input_ids"].size(1)
        ref_full_ids = source_tokenizer(ex["gold_full_code"], add_special_tokens=False)["input_ids"]

        exact_hit = False
        best_metrics = None
        best_exact = -1.0

        for seq in out:
            gen_part = seq[input_len:]
            gen_tokens += int(gen_part.numel())
            text = baseline_tokenizer.decode(gen_part, skip_special_tokens=True)
            patches = parse_patch_block(text, ex["num_hunks"])
            pred_full_code = reconstruct_full_code(ex["masked_code_text"], patches)
            pred_full_ids = source_tokenizer(pred_full_code, add_special_tokens=False)["input_ids"]

            if pred_full_ids == ref_full_ids:
                exact_hit = True

            em = 1.0 if pred_full_ids == ref_full_ids else 0.0
            if token_exact_match_rate is not None and len(pred_full_ids) == len(ref_full_ids):
                em = token_exact_match_rate(pred_full_ids, ref_full_ids)

            per_hunk_exact = []
            for pred_patch_text, gold_patch_text in zip(patches, ex["gold_patch_texts"]):
                pred_patch_ids = source_tokenizer(pred_patch_text, add_special_tokens=False)["input_ids"]
                gold_patch_ids = source_tokenizer(gold_patch_text, add_special_tokens=False)["input_ids"]
                per_hunk_exact.append(pred_patch_ids == gold_patch_ids)
                patch_minimality_values.append(patch_minimality_ratio(pred_patch_ids, gold_patch_ids))

            record = {
                "token_exact_match": em,
                "per_hunk_exact": per_hunk_exact,
                "all_hunks_correct": all(per_hunk_exact) if per_hunk_exact else True,
                "edit_distance": token_edit_distance(pred_full_ids, ref_full_ids) if token_edit_distance is not None else None,
                "codebleu": compute_codebleu(pred_full_code, ex["gold_full_code"], language=codebleu_lang).get("codebleu") if compute_codebleu is not None else None,
                "training_samples_generated": 1,
            }

            if record["token_exact_match"] > best_exact:
                best_exact = record["token_exact_match"]
                best_metrics = record

        pass_at_k_hits += int(exact_hit)
        if agg is not None and best_metrics is not None:
            agg.add(best_metrics)

        total_samples += 1
        total_hunks += ex["num_hunks"]

        if (i + 1) % 50 == 0:
            print(f"[EVAL] done {i + 1}/{len(eval_ds)}")

    summary = agg.summary() if agg is not None else {"num_records": total_samples}
    summary.update({
        "pass@k": pass_at_k_hits / max(1, total_samples),
        "localization_accuracy": 1.0,
        "patch_minimality": sum(patch_minimality_values) / max(1, len(patch_minimality_values)),
        "generation_efficiency": {
            "avg_generated_tokens_per_sample": gen_tokens / max(1, total_samples),
            "avg_decode_seconds_per_sample": gen_time_sec / max(1, total_samples),
            "tokens_per_second": gen_tokens / max(1e-8, gen_time_sec),
        },
        "num_eval_samples": total_samples,
        "num_eval_hunks": total_hunks,
        "num_return_sequences": num_return_sequences,
    })
    return summary


def resolve_model_name(baseline: str, explicit_model_name: Optional[str]) -> str:
    if explicit_model_name:
        return explicit_model_name
    if baseline not in BASELINE_PRESETS:
        raise ValueError(f"Unknown baseline '{baseline}'. Choices: {sorted(BASELINE_PRESETS)}")
    return BASELINE_PRESETS[baseline]


def pick_lora_targets(model_name: str) -> List[str]:
    name = model_name.lower()
    if "starcoder" in name:
        return ["c_attn", "c_proj", "c_fc"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate AR reconstruction baselines on multi-hunk repair data")
    parser.add_argument("--baseline", type=str, default="qwen", choices=sorted(BASELINE_PRESETS))
    parser.add_argument("--model-name", type=str, default=None, help="Optional explicit HF model name; overrides --baseline preset")
    parser.add_argument("--train-gz", type=str, default=str(DEFAULT_TRAIN_GZ))
    parser.add_argument("--source-tokenizer", type=str, default=SOURCE_TOKENIZER_NAME)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--max-source-seq-len", type=int, default=4096)
    parser.add_argument("--max-model-seq-len", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--language", type=str, default="java")
    args, _ = parser.parse_known_args()

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = resolve_model_name(args.baseline, args.model_name)
    out_dir = Path(args.out_dir) if args.out_dir else (DEFAULT_OUT_DIR / f"ar_{args.baseline}")

    print(f"[INFO] baseline={args.baseline} model_name={model_name}")
    print(f"[INFO] Loading source tokenizer: {args.source_tokenizer}")
    source_tok = AutoTokenizer.from_pretrained(args.source_tokenizer, trust_remote_code=True, use_fast=True)

    print(f"[INFO] Loading baseline tokenizer: {model_name}")
    baseline_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    if baseline_tok.pad_token_id is None:
        if baseline_tok.eos_token_id is None:
            raise RuntimeError("Baseline tokenizer has neither pad nor eos token.")
        baseline_tok.pad_token = baseline_tok.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if (args.use_bf16 and bf16_ok) else torch.float32

    print(f"[INFO] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)

    max_model_seq_len = args.max_model_seq_len or get_model_max_len(model)
    print(f"[INFO] max_model_seq_len={max_model_seq_len}")

    index = BaseRecordIndex(Path(args.train_gz), args.max_records, args.max_source_seq_len)
    train_ids, eval_ids = split_base_records(len(index), eval_ratio=args.eval_ratio, seed=RANDOM_SEED)

    train_ds = ExpandedRepairDataset(index, source_tok, train_ids, args.max_train_examples, args.language)
    eval_ds = ExpandedRepairDataset(index, source_tok, eval_ids, args.max_eval_examples, args.language)

    print(f"[INFO] Base train records: {len(train_ids):,}")
    print(f"[INFO] Base eval records: {len(eval_ids):,}")
    print(f"[INFO] Expanded train examples: {len(train_ds):,}")
    print(f"[INFO] Expanded eval examples: {len(eval_ds):,}")

    collator = ARRepairCollator(tokenizer=baseline_tok, max_seq_len=max_model_seq_len)

    if args.dry_run:
        ex = train_ds[0]
        batch = collator([ex])
        print("[DRY-RUN] prompt snippet:\n", ex["prompt"][:800])
        print("[DRY-RUN] target:\n", ex["target"])
        for k, v in batch.items():
            if k != "meta":
                print(f"[DRY-RUN] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        return

    if not args.eval_only:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=pick_lora_targets(model_name),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        try:
            model.gradient_checkpointing_enable()
            print("[INFO] Gradient checkpointing enabled")
        except Exception:
            print("[WARN] Could not enable gradient checkpointing")

        train_args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=bool(args.use_bf16 and bf16_ok),
            fp16=False,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            optim="adamw_torch",
        )

        trainer = MetaIgnoringTrainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            data_collator=collator,
            tokenizer=baseline_tok,
        )

        print("[INFO] Starting AR baseline training...")
        trainer.train()

        adapter_dir = out_dir / "lora_adapter"
        print(f"[INFO] Saving final AR baseline adapter to {adapter_dir}")
        trainer.save_model(str(adapter_dir))

    print("[INFO] Starting evaluation...")
    summary = evaluate_model(
        model=model,
        baseline_tokenizer=baseline_tok,
        source_tokenizer=source_tok,
        eval_ds=eval_ds,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        codebleu_lang=args.language,
    )

    summary.update({
        "baseline": args.baseline,
        "model_name": model_name,
    })

    print("[RESULT] Evaluation summary:")
    print(json.dumps(summary, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote eval summary to {out_path}")


if __name__ == "__main__":
    main()
