#!/usr/bin/env python3

import gzip
import json
import difflib
import random
from pathlib import Path
from typing import List, Tuple, Optional

from transformers import AutoTokenizer


# ======================
# CONFIG
# ======================
BASE_DIR = Path(__file__).resolve().parent

INPUT_JSONL_GZ = BASE_DIR / "data/raw/bugs_java.jsonl.gz"
OUTPUT_JSONL_GZ = BASE_DIR / "data/processed_train.jsonl.gz"

TOKENIZER_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MAX_LENGTH = 2048

HUNK_CAP_PER_FILE = 5
DEBUG_WRITE_LIMIT = 5   # set to None to run full file
RANDOM_SEED = 0


# ======================
# HELPERS
# ======================
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
    for ln in lines:
        offsets.append(pos)
        pos += len(ln)

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
    span: Tuple[int, int]
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
    return (min(idxs), max(idxs) + 1)


def num_training_samples_from_k(k: int) -> int:
    # sum_{r=2..k} C(k,r) = 2^k - k - 1
    if k < 2:
        return 0
    return (1 << k) - k - 1


def main():
    if not INPUT_JSONL_GZ.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_JSONL_GZ}")

    OUTPUT_JSONL_GZ.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading tokenizer:", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, use_fast=True
    )

    mask_token_id = None
    # Resolve mask token id (LLaDA does not define one by default)
    if tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        MASK_TOKEN = "<|mask|>"
        if MASK_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [MASK_TOKEN]}
            )
        mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)


    print("[INFO] mask_token_id:", mask_token_id)
    if mask_token_id is None:
        raise RuntimeError(
            "Tokenizer has no mask_token_id. Pick a tokenizer/config that defines a mask token."
        )

    rng = random.Random(RANDOM_SEED)

    n_in = 0
    n_out = 0
    total_training_samples = 0

    print("[INFO] Streaming read:", INPUT_JSONL_GZ)
    print("[INFO] Streaming write:", OUTPUT_JSONL_GZ)
    print("[INFO] HUNK_CAP_PER_FILE:", HUNK_CAP_PER_FILE)
    print("[INFO] DEBUG_WRITE_LIMIT:", DEBUG_WRITE_LIMIT)

    with gzip.open(INPUT_JSONL_GZ, "rt", encoding="utf-8") as fin, gzip.open(
        OUTPUT_JSONL_GZ, "wt", encoding="utf-8"
    ) as fout:
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
                    max_length=MAX_LENGTH,
                    add_special_tokens=True,
                )

                token_spans: List[Tuple[int, int]] = []
                for sp in spans:
                    ts = char_span_to_token_span(enc["offset_mapping"], sp)
                    if ts is not None:
                        token_spans.append(ts)

                if not token_spans:
                    continue

                if len(token_spans) > HUNK_CAP_PER_FILE:
                    idxs = list(range(len(token_spans)))
                    rng.shuffle(idxs)
                    idxs = sorted(idxs[:HUNK_CAP_PER_FILE])
                    token_spans = [token_spans[i] for i in idxs]

                k = len(token_spans)

                # omit single-hunk (and zero-hunk) records
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

                # if DEBUG_WRITE_LIMIT is not None and n_out >= DEBUG_WRITE_LIMIT:
                #     print("[INFO] DEBUG break after writing", DEBUG_WRITE_LIMIT, "records.")
                #     break

            except Exception as e:
                if n_in <= 5:
                    print("[WARN] Skipping a line due to error:", repr(e))
                continue

    print("\n[DONE]")
    print("input_records_seen:", n_in)
    print("output_records_written:", n_out)
    print("total_training_samples (with cap):", total_training_samples)
    print("[NOTE] Set DEBUG_WRITE_LIMIT=None to process the full file.")


if __name__ == "__main__":
    main()
