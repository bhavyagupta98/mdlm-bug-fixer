# MDLM Bug Fixer
Multi-Hunk Bug Fix Using LLaDA-8B with LoRA Fine-tuning

## Overview
This project trains and evaluates a multi-hop debug language model (MDLM) for automated multi-hunk Java bug repair. It uses LLaDA-8B-Instruct with LoRA fine-tuning to generate patches for code bugs across multiple file hunks.

## Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU recommended (for training/inference)
- ~50GB disk space for model, data, and outputs

### Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
.
├── model.py                          # Training script (LoRA fine-tuning)
├── inference.py                      # Inference & evaluation on test set
├── evaluation.py                     # Evaluation metrics (exact match, BLEU, etc.)
├── data_preprocess.py                # Data preprocessing utilities
├── baseline_reconstruction_eval.py   # Baseline evaluation (zero-shot LLaDA-8B)
├── run_all_benchmarks.sh             # End-to-end benchmark pipeline
├── benchmarks/                       # Benchmark scripts
│   ├── run_bigcodebench.py
│   ├── run_evalplus.py
│   ├── run_java_swebench.py
│   └── run_ablations.py
├── k8s/                              # Kubernetes job manifests
├── runs/                             # Output directory (ignored in git)
├── processed_train.jsonl.gz          # training data (large, ignored in git)
└── report/                           # Evaluation results & reports
```

## Quick Start

### 1. Train the MDLM Model
```bash
python model.py \
  --use-bf16 \
  --max-records 1000 \
  --output-dir runs/llada_lora
```
**Options:**
- `--use-bf16`: Use bfloat16 precision (requires A100/H100 GPUs)
- `--max-records`: Limit training records (None = full dataset)
- `--output-dir`: Output directory for fine-tuned model

### 2. Run Inference & Evaluation
```bash
python inference.py \
  --model GSAI-ML/LLaDA-8B-Instruct \
  --adapter runs/llada_lora/checkpoint-500 \
  --max-records 200
```
**Output:** Evaluation metrics (exact match %, BLEU, edit distance, etc.)

### 3. Evaluate Baselines (Zero-shot)
```bash
python baseline_reconstruction_eval.py \
  --model GSAI-ML/LLaDA-8B-Instruct \
  --num-records 200
```

### 4. Run Full Benchmark Suite
```bash
bash run_all_benchmarks.sh
```
This runs:
1. MDLM training (1000 records)
2. MDLM evaluation (200 records)
3. Baseline models (200 records each)
4. Generates comparison JSON/CSV

## Running on Kubernetes
Submit jobs using manifests in `k8s/`:
```bash
kubectl apply -f k8s/train-job.yaml
kubectl apply -f k8s/benchmark-job.yaml
kubectl apply -f k8s/swe_benchmark.yaml
```

## Evaluation Metrics
- **Token Exact Match Rate**: Exact token-level match
- **Per-Hunk Exact Match**: Match per individual code hunk
- **All Hunks Correct**: All hunks in a patch must be correct
- **Patch BLEU**: BLEU score on generated patches
- **Token Edit Distance**: Normalized Levenshtein distance
- **Masked Cross-Entropy**: Loss on masked regions

See `evaluation.py` for implementation details.

## Data Format
Training data: `processed_train.jsonl.gz` (gzipped JSONL)
```json
{
  "record_id": "swe-bench-123",
  "buggy_code": "...",
  "fixed_code": "...",
  "hunks": [{"start": 10, "end": 20, "fixed": "..."}, ...],
  "repo": "django/django",
  "commit": "abc123..."
}
```

## Configuration
Edit constants in `model.py`:
```python
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
TRAIN_GZ = Path(__file__).parent / "processed_train.jsonl.gz"
DEFAULT_FALLBACK_MAX_LEN = 1024
TEST_FRACTION = 0.1  # 10% test split (deterministic hash-based)
```

## Notes
- Train/test split is **deterministic** based on record ID hash
- Both `model.py` and `inference.py` use same split logic for consistency
- LoRA adapter is saved separately; can be combined with base model
- All results saved to `runs/` directory with timestamp

## License & Attribution
Built with [LLaDA-8B](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and inspired by SWE-Bench evaluation framework.
