#!/usr/bin/env python3
"""
Separate LoRA training entrypoint for sub-7B diffusion model experiments.

This file intentionally does not modify the existing `model.py` flow.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from model import HunkCollator, StreamingAllCombosDataset, get_model_max_len, num_training_samples_from_k


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed_train_coda_v0_base.jsonl.gz"
DEFAULT_MODEL_PRESET = "coda-base"
RANDOM_SEED = 0

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LoRA for sub-7B diffusion model experiments (separate from model.py)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PRESET,
        help=(
            "Model preset or full HF model id. "
            f"Presets: {', '.join(MODEL_PRESETS.keys())}"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to processed JSONL.GZ generated for the same tokenizer family.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to runs/sub7b/<model>_lora",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model_name = resolve_model_name(args.model)
    model_short = args.model if args.model in MODEL_PRESETS else sanitize_model_name(model_name)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = BASE_DIR / "runs" / "sub7b" / f"{model_short}_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)

    print("[INFO] Loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer missing pad_token_id and eos_token_id; cannot pad.")
        tokenizer.pad_token = tokenizer.eos_token

    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()
    use_bf16 = bool(args.use_bf16) and bf16_supported
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32

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
    print(f"[INFO] Using max_seq_len={max_seq_len}")

    print("[INFO] Indexing dataset:", data_path)
    train_ds = StreamingAllCombosDataset(
        path=data_path,
        max_records=args.max_records,
        max_seq_len=max_seq_len,
        split=args.split,
    )
    print(f"[INFO] Base records (k >= 2 hunks): {len(train_ds.offsets):,}")
    print(f"[INFO] Total expanded examples: {len(train_ds):,}")
    print("[INFO] Per-record sample formula: 2^k - k - 1 for k hunks")

    collator = HunkCollator(pad_token_id=tokenizer.pad_token_id)

    if args.dry_run:
        print("[DRY-RUN] Building one batch...")
        n = min(4, len(train_ds))
        batch = collator([train_ds[i] for i in range(n)])
        for k, v in batch.items():
            print(f"[DRY-RUN] {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        print("[DRY-RUN] Sample training-samples-per-record:")
        for k in range(2, 8):
            print(f"  k={k} -> {num_training_samples_from_k(k)}")
        return

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" if is_causal_lm else None,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    matched = []
    for name, _ in model.named_modules():
        for cand in lora_cfg.target_modules:
            if cand in name:
                matched.append(name)

    if not matched:
        print("[WARN] No LoRA target modules matched. Verify architecture-specific module names.")
    else:
        uniq = list(dict.fromkeys(matched))
        print("[INFO] LoRA module examples (up to 10):", uniq[:10])

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    try:
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled.")
    except ValueError:
        print("[WARN] Model does not support gradient checkpointing, skipping.")

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        weight_decay=0.0,
        warmup_ratio=0.03,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    class Sub7BTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            if not is_causal_lm and "attention_mask" in inputs:
                # Some custom diffusion models reject 2D attention masks.
                del inputs["attention_mask"]
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    trainer = Sub7BTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    print("[INFO] Saving final LoRA adapter:", adapter_dir)
    trainer.save_model(str(adapter_dir))
    print("[DONE]")


if __name__ == "__main__":
    main()
