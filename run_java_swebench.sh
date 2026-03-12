#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFECTS4J_HOME="${DEFECTS4J_HOME:-}"
PROJECTS="${PROJECTS:-Lang}"
BUG_IDS="${BUG_IDS:-}"
BUG_LIMIT="${BUG_LIMIT:-10}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/java_swebench}"

# Modes to run: localized, generative, or both (default)
MODES="${MODES:-localized generative}"

# Generative mode generation budget
ORACLE_LENGTH="${ORACLE_LENGTH:-1}"   # 1=use oracle length, 0=use --max-new-tokens
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ----------------------------------------------------------------
# Shared base args (common to both modes)
# ----------------------------------------------------------------
BASE_ARGS=(
  python run_java_swebench.py
  --projects "$PROJECTS"
  --bug-limit "$BUG_LIMIT"
)

if [[ -n "$DEFECTS4J_HOME" ]]; then
  BASE_ARGS+=(--defects4j-home "$DEFECTS4J_HOME")
fi

if [[ -n "$BUG_IDS" ]]; then
  BASE_ARGS+=(--bug-ids "$BUG_IDS")
fi

ADAPTER_PATH="${ADAPTER_PATH:-$ROOT_DIR/runs/llada_lora/lora_adapter}"

echo "[INFO] Projects:   $PROJECTS"
echo "[INFO] Bug ids:    ${BUG_IDS:-<all>}"
echo "[INFO] Bug limit:  $BUG_LIMIT"
echo "[INFO] Modes:      $MODES"

# ----------------------------------------------------------------
# Run each (mode, adapter) combination
# ----------------------------------------------------------------
for MODE in $MODES; do
  for ADAPTER_LABEL in base finetuned; do
    if [[ "$ADAPTER_LABEL" == "finetuned" ]]; then
      if [[ ! -d "$ADAPTER_PATH" ]]; then
        echo "[WARN] Adapter not found at $ADAPTER_PATH — skipping finetuned run"
        continue
      fi
      ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")
    else
      ADAPTER_ARGS=(--no-adapter)
    fi

    OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODE}_${ADAPTER_LABEL}"
    CMD=("${BASE_ARGS[@]}" --mode "$MODE" --output-dir "$OUTPUT_DIR" "${ADAPTER_ARGS[@]}")

    if [[ "$MODE" == "generative" ]]; then
      if [[ "$ORACLE_LENGTH" == "1" ]]; then
        CMD+=(--oracle-length)
      else
        CMD+=(--max-new-tokens "$MAX_NEW_TOKENS")
      fi
    fi

    echo ""
    echo "========================================================"
    echo "[INFO] Mode: $MODE  |  Adapter: $ADAPTER_LABEL  →  $OUTPUT_DIR"
    echo "========================================================"
    printf '[INFO] Command: %q ' "${CMD[@]}"
    printf '\n'

    "${CMD[@]}"
  done
done

echo ""
echo "[INFO] All modes complete. Results under: $BASE_OUTPUT_DIR"