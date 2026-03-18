#!/usr/bin/env python3
"""
Separate preprocessing entrypoint for sub-7B diffusion model experiments.

Produces tokenizer-specific processed JSONL.GZ files so experiments do not reuse
incompatible token IDs from the existing LLaDA pipeline.
"""

import argparse
import difflib
import gzip
import json
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import AutoConfig, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "raw" / "bugs_java.jsonl.gz"
DEFAULT_TOKENIZER_PRESET = "coda-base"

TOKENIZER_PRESETS = {
    "coda-base": "Salesforce/CoDA-v0-Base",
    "coda-instruct": "Salesforce/CoDA-v0-Instruct",
    "fast-dllm-1.5b": "Efficient-Large-Model/Fast_dLLM_v2_1.5B",
    "sdlm-3b-d4": "OpenGVLab/SDLM-3B-D4",
    "sdlm-3b-d8": "OpenGVLab/SDLM-3B-D8",
}

RANDOM_SEED = 0


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def resolve_tokenizer_name(tokenizer_arg: str) -> str:
    return TOKENIZER_PRESETS.get(tokenizer_arg, tokenizer_arg)


def resolve_default_output(tokenizer_name: str) -> Path:
    slug = slugify(tokenizer_name)
    return BASE_DIR / "data" / f"processed_train_{slug}.jsonl.gz"


def compute_hunks(old: str, new: str) -> List[Tuple[int, int, int, int]]:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines)
    hunks: List[Tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            hunks.append((i1, i2, j1, j2))
    return hunks


def new_line_spans(new: str, hunks: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    lines = new.splitlines(keepends=True)
    if not lines:
        return []

    offsets: List[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    spans: List[Tuple[int, int]] = []
    for (_, _, j1, j2) in hunks:
        if j2 <= j1:
            continue
        if j1 < 0 or j2 > len(lines):
            continue
        start = offsets[j1]
        end = offsets[j2 - 1] + len(lines[j2 - 1])
        spans.append((start, end))
    return spans


def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def char_span_to_token_span(
    offset_mapping: List[Tuple[int, int]],
    span: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    cs, ce = span
    idxs: List[int] = []
    for i, (s, e) in enumerate(offset_mapping):
        if s == 0 and e == 0:
            continue
        if spans_overlap((s, e), (cs, ce)):
            idxs.append(i)
    if not idxs:
        return None
    return min(idxs), max(idxs) + 1


def num_training_samples_from_k(k: int) -> int:
    if k < 2:
        return 0
    return (1 << k) - k - 1


def resolve_mask_token_id(tokenizer, tokenizer_name: str, mask_token_id_override: Optional[int]) -> int:
    if mask_token_id_override is not None:
        return int(mask_token_id_override)

    try:
        cfg = AutoConfig.from_pretrained(tokenizer_name, trust_remote_code=True)
        cfg_mask_token_id = getattr(cfg, "mask_token_id", None)
        if cfg_mask_token_id is not None:
            return int(cfg_mask_token_id)
    except Exception:
        pass

    vocab = tokenizer.get_vocab()
    for token in ("<|mask|>", "<|mdm_mask|>"):
        if token in vocab:
            return int(vocab[token])

    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)

    raise RuntimeError(
        "Could not resolve mask token id. Pass --mask-token-id explicitly for this tokenizer/model family."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenizer-specific preprocessing for sub-7B diffusion experiments."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER_PRESET,
        help=(
            "Tokenizer preset or full HF id. "
            f"Presets: {', '.join(TOKENIZER_PRESETS.keys())}"
        ),
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--hunk-cap-per-file", type=int, default=5)
    parser.add_argument("--debug-write-limit", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    tokenizer_name = resolve_tokenizer_name(args.tokenizer)
    output_path = Path(args.output) if args.output else resolve_default_output(tokenizer_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading tokenizer:", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)

    mask_token_id = resolve_mask_token_id(tokenizer, tokenizer_name, args.mask_token_id)
    print("[INFO] mask_token_id:", mask_token_id)

    rng = random.Random(args.seed)

    n_in = 0
    n_out = 0
    total_training_samples = 0

    print("[INFO] Streaming read:", input_path)
    print("[INFO] Streaming write:", output_path)
    print("[INFO] MAX_LENGTH:", args.max_length)
    print("[INFO] HUNK_CAP_PER_FILE:", args.hunk_cap_per_file)
    print("[INFO] DEBUG_WRITE_LIMIT:", args.debug_write_limit)

    with gzip.open(input_path, "rt", encoding="utf-8") as fin, gzip.open(output_path, "wt", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1

            try:
                ex = json.loads(line)
                old = ex.get("old_contents", "")
                new = ex.get("new_contents", "")
                if not old or not new:
                    continue

                raw_hunks = compute_hunks(old, new)
                spans = new_line_spans(new, raw_hunks)

                enc = tokenizer(
                    new,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=True,
                )

                token_spans: List[Tuple[int, int]] = []
                for span in spans:
                    token_span = char_span_to_token_span(enc["offset_mapping"], span)
                    if token_span is not None:
                        token_spans.append(token_span)

                if not token_spans:
                    continue

                if len(token_spans) > args.hunk_cap_per_file:
                    idxs = list(range(len(token_spans)))
                    rng.shuffle(idxs)
                    idxs = sorted(idxs[: args.hunk_cap_per_file])
                    token_spans = [token_spans[i] for i in idxs]

                k = len(token_spans)
                if k < 2:
                    continue

                samples_here = num_training_samples_from_k(k)
                total_training_samples += samples_here

                record = {
                    "id": ex.get("commit", "NA"),
                    "input_ids": enc["input_ids"],
                    "hunk_token_spans": token_spans,
                    "mask_token_id": int(mask_token_id),
                }
                fout.write(json.dumps(record) + "\n")

                n_out += 1
                if n_out % 1000 == 0:
                    print(
                        f"[INFO] wrote={n_out} processed={n_in} "
                        f"hunks={k} samples_here={samples_here} total_samples={total_training_samples}"
                    )

                if args.debug_write_limit is not None and n_out >= args.debug_write_limit:
                    print("[INFO] DEBUG break after writing", args.debug_write_limit, "records")
                    break

            except Exception as exc:
                if n_in <= 5:
                    print("[WARN] Skipping a line due to error:", repr(exc))
                continue

    print("\n[DONE]")
    print("tokenizer:", tokenizer_name)
    print("output_file:", output_path)
    print("input_records_seen:", n_in)
    print("output_records_written:", n_out)
    print("total_training_samples (with cap):", total_training_samples)


if __name__ == "__main__":
    main()
