#!/usr/bin/env python3

import gzip
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
GZ_PATH = (BASE_DIR / "data/one_example_train.jsonl.gz").resolve()


def main():
    print("\n=== Reading gz file ===")
    print("Path:", GZ_PATH)

    if not GZ_PATH.exists():
        raise FileNotFoundError(f"GZ file not found: {GZ_PATH}")

    with gzip.open(GZ_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"\n--- RECORD {i} ---")
            # obj = json.loads(line)

            # Pretty-print JSON
            # print(json.dumps(obj, indent=2))

            # Stop after first record

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
