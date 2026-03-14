#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DEFECTS4J_HOME="${DEFECTS4J_HOME:-}"
PROJECTS="${PROJECTS:-Lang}"
BUG_IDS="${BUG_IDS:-}"
BUG_LIMIT="${BUG_LIMIT:-10}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/runs/java_swebench}"

# Models to benchmark (default: both)
# Format: "llada llama" or override as MODELS="llada"
MODELS="${MODELS:-llada llama}"

# Modes to run (localized, generative, or both)
MODES="${MODES:-localized generative}"

# Generative mode generation budget
ORACLE_LENGTH="${ORACLE_LENGTH:-1}"   # 1=use oracle length, 0=use --max-new-tokens
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# LoRA adapter for LLaDA
ADAPTER_PATH="${ADAPTER_PATH:-$ROOT_DIR/runs/llada_lora/lora_adapter}"

# ================================================================
# Model configuration (name → HF model ID)
# ================================================================
declare -A MODEL_IDS=(
  ["llada"]="GSAI-ML/LLaDA-8B-Instruct"
  ["llama"]="meta-llama/Llama-2-7b-chat-hf"
)

declare -A HAS_ADAPTER=(
  ["llada"]=1        # LLaDA has LoRA adapter variants
  ["llama"]=0        # Llama is used as-is
)

# ================================================================
# Shared base args (common to all runs)
# ================================================================
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
echo "[INFO] Models:     $MODELS"
echo "[INFO] Modes:      $MODES"

# ================================================================
# Main benchmark loop: model × mode × (adapter for llada)
# ================================================================
for MODEL in $MODELS; do
  MODEL_ID="${MODEL_IDS[$MODEL]}"
  SUPPORTS_ADAPTER=${HAS_ADAPTER[$MODEL]}

  echo ""
  echo "╔═════════════════════════════════════════════════════════════╗"
  echo "║  MODEL: $MODEL ($MODEL_ID)"
  echo "╚═════════════════════════════════════════════════════════════╝"

  for MODE in $MODES; do
    if [[ $SUPPORTS_ADAPTER -eq 1 ]]; then
      # LLaDA: run both base and finetuned
      for ADAPTER_LABEL in base finetuned; do
        if [[ "$ADAPTER_LABEL" == "finetuned" ]]; then
          if [[ ! -d "$ADAPTER_PATH" ]]; then
            echo "[WARN] Adapter not found at $ADAPTER_PATH — skipping $MODEL finetuned run"
            continue
          fi
          ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")
        else
          ADAPTER_ARGS=(--no-adapter)
        fi

        OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL}_${ADAPTER_LABEL}_${MODE}"
        CMD=("${BASE_ARGS[@]}" --base-model "$MODEL_ID" --mode "$MODE" \
             --output-dir "$OUTPUT_DIR" "${ADAPTER_ARGS[@]}")

        if [[ "$MODE" == "generative" ]]; then
          if [[ "$ORACLE_LENGTH" == "1" ]]; then
            CMD+=(--oracle-length)
          else
            CMD+=(--max-new-tokens "$MAX_NEW_TOKENS")
          fi
        fi

        echo ""
        echo "──────────────────────────────────────────────────────────"
        echo "[RUN] $MODEL (${ADAPTER_LABEL}) + $MODE mode"
        echo "      Output: $OUTPUT_DIR"
        echo "──────────────────────────────────────────────────────────"

        "${CMD[@]}"
      done
    else
      # Llama: just run without adapters
      OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL}_${MODE}"
      CMD=("${BASE_ARGS[@]}" --base-model "$MODEL_ID" --mode "$MODE" \
           --output-dir "$OUTPUT_DIR" --no-adapter)

      if [[ "$MODE" == "generative" ]]; then
        if [[ "$ORACLE_LENGTH" == "1" ]]; then
          CMD+=(--oracle-length)
        else
          CMD+=(--max-new-tokens "$MAX_NEW_TOKENS")
        fi
      fi

      echo ""
      echo "──────────────────────────────────────────────────────────"
      echo "[RUN] $MODEL + $MODE mode"
      echo "      Output: $OUTPUT_DIR"
      echo "──────────────────────────────────────────────────────────"

      "${CMD[@]}"
    fi
  done
done

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE"
echo "║  Results:  $BASE_OUTPUT_DIR"
echo "╚═════════════════════════════════════════════════════════════╝"