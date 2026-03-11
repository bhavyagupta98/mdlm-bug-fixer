#!/usr/bin/env bash
set -euo pipefail

# End-to-end benchmark runner:
# 1) Train MDLM (model.py) with 1000 base records
# 2) Evaluate MDLM on 200 records
# 3) Evaluate 3 FIM baselines on 200 records each
# 4) Write combined results JSON

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="$ROOT_DIR/runs/benchmark_${TS}"
MDLM_EVAL_DIR="$RUN_DIR/mdlm_eval"
BASELINES_OUT_DIR="$RUN_DIR/baselines"
COMBINED_JSON="$RUN_DIR/combined_results.json"
COMPARISON_JSON="$RUN_DIR/comparison_table.json"
COMPARISON_CSV="$RUN_DIR/comparison_table.csv"

mkdir -p "$MDLM_EVAL_DIR" "$BASELINES_OUT_DIR"

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Run directory: $RUN_DIR"

echo "[STEP 1/4] Training MDLM with max-records=1000"
python model.py \
  --use-bf16 \
  --max-records 1000

echo "[STEP 2/4] Evaluating MDLM on 200 records"
python inference.py \
  --adapter-path "$ROOT_DIR/runs/llada_lora/lora_adapter" \
  --split test \
  --max-records 200 \
  --codebleu \
  --output-dir "$MDLM_EVAL_DIR"

echo "[STEP 3/4] Evaluating FIM baselines on 200 records each"
FIM_MODELS=(
  deepseek-coder
  starcoder2
  codellama
)

for model_name in "${FIM_MODELS[@]}"; do
  echo "  [BASELINE] $model_name"
  python baseline_reconstruction_eval.py \
    --strategy fim \
    --baseline "$model_name" \
    --max-records 200 \
    --num-samples 1 \
    --k 1 \
    --language java \
    --out-dir "$BASELINES_OUT_DIR"
done

echo "[STEP 4/4] Aggregating results into one JSON"
python - <<PYEOF
import json
import csv
from pathlib import Path

run_dir = Path(r"$RUN_DIR")
mdlm_dir = Path(r"$MDLM_EVAL_DIR")
baselines_dir = Path(r"$BASELINES_OUT_DIR")
out_file = Path(r"$COMBINED_JSON")
comparison_json = Path(r"$COMPARISON_JSON")
comparison_csv = Path(r"$COMPARISON_CSV")

mdlm_summary = None
mdlm_candidates = sorted(mdlm_dir.glob("summary_*.json"))
if mdlm_candidates:
    mdlm_summary = json.loads(mdlm_candidates[-1].read_text())

baseline_summaries = {}
for p in sorted(baselines_dir.glob("fim_*/eval_summary.json")):
    key = p.parent.name
    baseline_summaries[key] = json.loads(p.read_text())

combined = {
    "run_dir": str(run_dir),
    "mdlm": mdlm_summary,
    "baselines": baseline_summaries,
}

out_file.write_text(json.dumps(combined, indent=2))
print(f"[INFO] Wrote combined results: {out_file}")

# Side-by-side comparable metrics (holistic aligned set)
metrics = [
  "per_hunk_accuracy",
  "all_hunks_correct_rate",
  "token_exact_match",
  "patch_string_em",
  "mean_patch_bleu",
  "mean_patch_ned",
  "mean_codebleu",
]

rows = []
if mdlm_summary:
  rows.append({
    "model": "mdlm",
    **{m: mdlm_summary.get(m) for m in metrics},
  })

for name, summary in baseline_summaries.items():
  rows.append({
    "model": name,
    **{m: summary.get(m) for m in metrics},
  })

comparison_json.write_text(json.dumps(rows, indent=2))
print(f"[INFO] Wrote comparison table JSON: {comparison_json}")

with comparison_csv.open("w", newline="") as f:
  writer = csv.DictWriter(f, fieldnames=["model", *metrics])
  writer.writeheader()
  writer.writerows(rows)
print(f"[INFO] Wrote comparison table CSV: {comparison_csv}")
PYEOF

echo "[DONE]"
echo "  Combined results: $COMBINED_JSON"
echo "  Comparison JSON:  $COMPARISON_JSON"
echo "  Comparison CSV:   $COMPARISON_CSV"
echo "  MDLM eval dir:    $MDLM_EVAL_DIR"
echo "  Baseline eval dir:$BASELINES_OUT_DIR"
