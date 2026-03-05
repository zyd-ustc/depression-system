#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

echo "[1/4] Distill depression scores (DeepSeek + student model)"
python -m training.score_distill.run_distill \
  --input-dialogues data/generate/DataSynCards/dialogues.jsonl \
  --output-labels data/generate/processed/distilled_scores.jsonl \
  --sample-ratio 0.1 \
  --epochs 30 \
  --batch-size 32

echo "[2/4] Convert dialogues to SFT data with distill metadata"
python data/generate/data_convert.py

echo "[3/4] Build benchmark dataset (prefer distilled labels)"
python -m eval.benchmark_v2.dataset_builder \
  --source auto \
  --distilled-path data/generate/processed/distilled_scores.jsonl \
  --output-samples eval/benchmark_v2/data/samples.v1.json \
  --output-counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json

echo "[4/4] Run benchmark"
python -m eval.benchmark_v2.run_benchmark_v2 \
  --samples eval/benchmark_v2/data/samples.v1.json \
  --counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json \
  --config eval/benchmark_v2/configs/closed_source.v1.json \
  --repeats 3 \
  --output-dir eval/benchmark_v2/outputs

echo "Unified workflow finished."

