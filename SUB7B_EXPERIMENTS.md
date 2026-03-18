# Sub-7B Diffusion Experiments (Separate Pipeline)

This workflow is fully separate from the existing `model.py` / `inference.py` path.

## Files

- `data_preprocess_sub7b.py`: tokenizer-specific preprocessing
- `model_sub7b.py`: LoRA training for sub-7B models
- `inference_sub7b.py`: evaluation/inference with LoRA merge support

## 1) Preprocess (CoDA Base)

```bash
python3 data_preprocess_sub7b.py \
  --tokenizer coda-base \
  --input data/raw/bugs_java.jsonl.gz \
  --output data/processed_train_coda_v0_base.jsonl.gz \
  --max-length 2048 \
  --hunk-cap-per-file 5
```

## 2) Dry-run Training Sanity Check

```bash
python3 model_sub7b.py \
  --model coda-base \
  --data-path data/processed_train_coda_v0_base.jsonl.gz \
  --split train \
  --dry-run \
  --max-records 64
```

## 3) Train LoRA (CoDA Base)

```bash
python3 model_sub7b.py \
  --model coda-base \
  --data-path data/processed_train_coda_v0_base.jsonl.gz \
  --split train \
  --batch-size 1 \
  --grad-accum 4 \
  --use-bf16
```

Default adapter output:

```text
runs/sub7b/coda-base_lora/lora_adapter
```

## 4) Evaluate / Inference (CoDA Base + LoRA)

```bash
python3 inference_sub7b.py \
  --base-model coda-base \
  --data-path data/processed_train_coda_v0_base.jsonl.gz \
  --split test \
  --steps 64 \
  --temperature 0.0 \
  --dtype bfloat16
```

## 5) Base-Only Evaluation (No Adapter)

```bash
python3 inference_sub7b.py \
  --base-model coda-base \
  --data-path data/processed_train_coda_v0_base.jsonl.gz \
  --split test \
  --no-adapter
```

## Other model presets

```text
coda-instruct
fast-dllm-1.5b
sdlm-3b-d4
sdlm-3b-d8
```

Example (Fast-dLLM):

```bash
python3 data_preprocess_sub7b.py --tokenizer fast-dllm-1.5b --output data/processed_train_fast_dllm_v2_1_5b.jsonl.gz
python3 model_sub7b.py --model fast-dllm-1.5b --data-path data/processed_train_fast_dllm_v2_1_5b.jsonl.gz --use-bf16
python3 inference_sub7b.py --base-model fast-dllm-1.5b --data-path data/processed_train_fast_dllm_v2_1_5b.jsonl.gz
```

## Notes

- Keep preprocessing tokenizer aligned with training/inference model family.
- Do not reuse existing LLaDA-tokenized datasets for CoDA/Fast-dLLM/SDLM experiments.
