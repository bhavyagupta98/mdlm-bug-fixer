#!/usr/bin/env python3
import gzip
import hashlib
import json
import os
import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


# ======================
# CONFIG (EDIT ME)
# ======================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_GZ = BASE_DIR / "processed_train.jsonl.gz"
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"

MODEL_PRESETS = {
    "llada-8b": "GSAI-ML/LLaDA-8B-Instruct",
    "llada2-mini": "inclusionAI/LLaDA2.0-mini",
    "dream-7b": "Dream-org/Dream-v0-Instruct-7B",
}

# Debug: stop after N base records indexed from gzip (set None for full)
DEBUG_MAX_RECORDS: Optional[int] = None

RANDOM_SEED = 0
DEFAULT_FALLBACK_MAX_LEN = 1024
TEST_FRACTION = 0.1
SPLIT_SEED = 42


# ======================
# HELPERS
# ======================
def record_split(record_id: str, test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> str:
    """Deterministic hash-based train/test assignment. Must match inference.py."""
    h = hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()
    if int(h[:8], 16) / 0xFFFFFFFF < test_fraction:
        return "test"
    return "train"


def num_training_samples_from_k(k: int) -> int:
    """
    Compute number of training examples for k hunks.
    Includes all subsets with size >= 2: sum_{r=2..k} C(k,r) = 2^k - k - 1
    """
    if k < 2:
        return 0
    return (1 << k) - k - 1


def _filter_spans(spans: List[List[int]], L: int) -> List[Tuple[int, int]]:
    """
    Clamp spans to [0, L] where L is the current (capped) input length.
    """
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


def get_model_max_len(model) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return DEFAULT_FALLBACK_MAX_LEN
    v = getattr(cfg, "max_position_embeddings", None)
    if v is None:
        return DEFAULT_FALLBACK_MAX_LEN
    return int(v)


# ======================
# DATASET (ALL COMBINATIONS)
# ======================
class StreamingAllCombosDataset(Dataset):
    """
    Each base JSONL record becomes (2^k - k - 1) training examples, where k is the number
    of valid hunk spans. We enumerate ALL subsets of hunks with size >= 2:
      size 2, 3, ..., k.
    This focuses on multi-hunk error infilling, aligning with iterative denoising inference.

    Memory use:
      - offsets: one int per base record
      - k_valid: one small int per base record
      - valid_masks: list of valid subset masks per record
      - prefix: one int per base record (cumulative expanded example counts)
    """
    def __init__(self, path: Path, max_records: Optional[int], max_seq_len: int, split: str = "train"):
        self.path = path
        self.max_seq_len = int(max_seq_len)
        self.split = split

        self.offsets: List[int] = []
        self.k_valid: List[int] = []
        self.valid_masks: List[List[int]] = []  # valid subset masks per record
        self.prefix: List[int] = [0]  # prefix sums of expanded example counts

        n_skipped_split = 0
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

                # Filter by train/test split
                if split != "all":
                    record_id = ex.get("id", ex.get("commit", "NA"))
                    if record_split(record_id) != split:
                        n_skipped_split += 1
                        continue

                input_ids = ex.get("input_ids", [])
                spans = ex.get("hunk_token_spans", [])

                if not isinstance(input_ids, list) or len(input_ids) == 0:
                    continue

                # Cap tokens to model context length
                L = min(len(input_ids), self.max_seq_len)

                filtered = _filter_spans(spans, L)
                k = len(filtered)

                # keep records with at least 2 valid hunks (for multi-hunk masking)
                if k >= 2:
                    # Generate all subsets with 2+ hunks (minimum 2 bits set)
                    valid = []
                    for mask in range(1, (1 << k)):
                        if bin(mask).count('1') >= 2:  # At least 2 bits set
                            valid.append(mask)
                    
                    n_sub = len(valid)
                    if n_sub > 0:
                        self.offsets.append(pos)
                        self.k_valid.append(k)
                        self.valid_masks.append(valid)
                        self.prefix.append(self.prefix[-1] + n_sub)

                        if max_records is not None and len(self.offsets) >= max_records:
                            break

        if not self.offsets:
            raise RuntimeError(f"No usable records found in {self.path} (need k >= 2 hunks)")
        if split != "all":
            print(f"[INFO] Split='{split}': kept {len(self.offsets)} records, skipped {n_skipped_split} ({split} only)")

    def __len__(self) -> int:
        return self.prefix[-1]

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        """
        Map global example idx -> (record_idx, subset_mask)
        subset_mask is a valid mask with 2+ bits set.
        """
        if global_idx < 0 or global_idx >= len(self):
            raise IndexError(global_idx)

        r = bisect.bisect_right(self.prefix, global_idx) - 1
        base = self.prefix[r]
        local = global_idx - base  # Index within this record's valid masks
        subset_mask = self.valid_masks[r][local]
        return r, subset_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_i, subset_mask = self._locate(idx)

        with gzip.open(self.path, "rb") as f:
            f.seek(self.offsets[rec_i])
            line = f.readline().decode("utf-8")

        ex = json.loads(line)

        input_ids: List[int] = ex["input_ids"]
        spans: List[List[int]] = ex["hunk_token_spans"]
        mask_token_id: int = int(ex["mask_token_id"])

        # Cap to model context length
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
        L = len(input_ids)

        filtered_spans = _filter_spans(spans, L)
        k = len(filtered_spans)

        # Defensive: if something changed, require k >= 2
        if k < 2:
            chosen_spans: List[Tuple[int, int]] = []
        else:
            chosen_spans = [filtered_spans[i] for i in range(k) if (subset_mask >> i) & 1]

        return {
            "input_ids": input_ids,
            "hunk_spans": chosen_spans,
            "mask_token_id": mask_token_id,
        }


# ======================
# COLLATOR
# ======================
@dataclass
class HunkCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_tensors: List[torch.Tensor] = []
        label_tensors: List[torch.Tensor] = []
        attn_tensors: List[torch.Tensor] = []

        for ex in batch:
            ids = torch.tensor(ex["input_ids"], dtype=torch.long)
            labels = ids.clone()

            noisy = ids.clone()
            mask_token_id = int(ex["mask_token_id"])
            spans: List[Tuple[int, int]] = ex["hunk_spans"]

            supervise = torch.zeros_like(ids, dtype=torch.bool)

            for (s, e) in spans:
                s = int(s); e = int(e)
                if e <= s:
                    continue
                noisy[s:e] = mask_token_id
                supervise[s:e] = True

            # Loss only on supervised tokens
            labels[~supervise] = -100

            input_tensors.append(noisy)
            label_tensors.append(labels)
            attn_tensors.append(torch.ones_like(noisy, dtype=torch.long))

        max_len = max(x.size(0) for x in input_tensors)

        def pad_1d(x: torch.Tensor, pad_val: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            pad = torch.full((max_len - x.size(0),), pad_val, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x, self.pad_token_id) for x in input_tensors], dim=0)
        labels = torch.stack([pad_1d(x, -100) for x in label_tensors], dim=0)
        attention_mask = torch.stack([pad_1d(x, 0) for x in attn_tensors], dim=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ======================
# TRAIN
# ======================
def main():
    parser = argparse.ArgumentParser(description="Train LoRA on ALL hunk subsets (2^k-1 per record), bf16 only (no fp16)")
    parser.add_argument("--dry-run", action="store_true", help="Dataset+collator dry run and exit")
    parser.add_argument("--max-records", type=int, default=DEBUG_MAX_RECORDS, help="Limit number of base gzip records to index")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"],
                        help="Which data split to train on (default: train)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or preset. Presets: {', '.join(MODEL_PRESETS.keys())}. Or pass a full HuggingFace model ID.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--use-bf16", action="store_true", help="Use bf16 if supported by GPU (otherwise float32)")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Optional override. If not set, uses model.config.max_position_embeddings.",
    )
    args, _ = parser.parse_known_args()  # Colab/Jupyter safe

    # Resolve model name: preset shorthand or full HuggingFace ID
    model_name = MODEL_PRESETS.get(args.model, args.model)

    # Derive output dir from model short name (e.g. "runs/llada2-mini_lora")
    model_short = args.model if args.model in MODEL_PRESETS else model_name.split("/")[-1]
    out_dir = BASE_DIR / f"runs/{model_short}_lora"

    torch.manual_seed(RANDOM_SEED)

    print("[INFO] Loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    # Ensure pad token for batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer missing pad_token_id and eos_token_id; cannot pad.")
        tokenizer.pad_token = tokenizer.eos_token

    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()
    use_bf16 = bool(args.use_bf16) and bf16_supported

    # Per your requirement: NO fp16. If bf16 isn't supported, we fall back to float32.
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # DRY-RUN can avoid loading model if user gives max-seq-len
    if args.dry_run and args.max_seq_len is not None:
        max_seq_len = int(args.max_seq_len)
        print(f"[INFO] DRY-RUN: using --max-seq-len={max_seq_len} (model not loaded)")
        print("[INFO] Indexing dataset:", TRAIN_GZ)
        train_ds = StreamingAllCombosDataset(TRAIN_GZ, max_records=args.max_records, max_seq_len=max_seq_len, split=args.split)
        print(f"[INFO] Base records (k >= 2 hunks): {len(train_ds.offsets):,}")
        print(f"[INFO] Total expanded examples: {len(train_ds):,}")
        print(f"[INFO] Example per-record counts (k -> training samples):")
        for k in range(2, 8):
            samples = num_training_samples_from_k(k)
            print(f"       k={k} hunks -> {samples} training samples")
        collator = HunkCollator(pad_token_id=tokenizer.pad_token_id)

        print("[DRY-RUN] Building one batch...")
        n = min(4, len(train_ds))
        batch = collator([train_ds[i] for i in range(n)])
        for k, v in batch.items():
            print(f"[DRY-RUN] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        labels = batch["labels"][0].tolist()
        supervised = [i for i, t in enumerate(labels) if t != -100]
        print("[DRY-RUN] First row supervised indices (first 50):", supervised[:50])
        return

    print(f"[INFO] CUDA={cuda} bf16_supported={bf16_supported} use_bf16={use_bf16} dtype={load_dtype}")
    print("[INFO] Loading model:", model_name)
    is_causal_lm = True
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=load_dtype,
        )
    except ValueError:
        print("[INFO] AutoModelForCausalLM failed, falling back to AutoModel")
        is_causal_lm = False
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=load_dtype,
        )
    if cuda:
        model = model.cuda()

    max_seq_len = int(args.max_seq_len) if args.max_seq_len is not None else get_model_max_len(model)
    print(f"[INFO] Using max_seq_len = model.config.max_position_embeddings = {max_seq_len}")

    print("[INFO] Indexing dataset:", TRAIN_GZ)
    train_ds = StreamingAllCombosDataset(TRAIN_GZ, max_records=args.max_records, max_seq_len=max_seq_len, split=args.split)
    print(f"[INFO] Base records (k >= 2 hunks): {len(train_ds.offsets):,}")
    print(f"[INFO] Total expanded examples: {len(train_ds):,}")
    print(f"[INFO] Per-record sample formula: 2^k - k - 1 for k hunks")

    collator = HunkCollator(pad_token_id=tokenizer.pad_token_id)

    if args.dry_run:
        print("[DRY-RUN] Building one batch...")
        n = min(4, len(train_ds))
        batch = collator([train_ds[i] for i in range(n)])
        for k, v in batch.items():
            print(f"[DRY-RUN] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        return

    # LoRA config — use CAUSAL_LM for standard models, None for custom (e.g. Dream)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" if is_causal_lm else None,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    # Print a few target matches for sanity
    matched = []
    for name, _ in model.named_modules():
        for cand in lora_cfg.target_modules:
            if cand in name:
                matched.append(name)
    if not matched:
        print("[WARN] No LoRA target modules matched. You may need to adjust target_modules for this architecture.")
    else:
        uniq = list(dict.fromkeys(matched))
        print("[INFO] LoRA module examples (up to 10):", uniq[:10])

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Ensure input embeddings require grad (needed for gradient checkpointing)
    model.enable_input_require_grads()

    # Memory saver
    try:
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled.")
    except ValueError:
        print("[WARN] Model does not support gradient checkpointing, skipping.")

    args_tf = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,          # bf16 only
        fp16=False,             # explicitly disabled
        report_to="none",
        dataloader_num_workers=4,  # Colab stability
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    # LLaDA's forward() returns logits only, no loss — compute it ourselves
    class MDLMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            # Drop attention_mask for models that don't handle 2D masks (e.g. Dream)
            if not is_causal_lm and "attention_mask" in inputs:
                del inputs["attention_mask"]
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, seq_len, vocab)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    trainer = MDLMTrainer(
        model=model,
        args=args_tf,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving final LoRA adapter...")
    trainer.save_model(str(out_dir / "lora_adapter"))
    print("[DONE]")


if __name__ == "__main__":
    main()
