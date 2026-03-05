# Benchmark v2 (Depression Scoring)

This module provides a structured benchmark framework for closed-source model evaluation with four layers:

1. Accuracy: `MAE`, `RMSE`, `Pearson`, `Spearman`, `CCC`, `bucket_mae`
2. Stability: `test_retest_std`, `prompt_paraphrase_std`, `ICC(2,1)`
3. Logical consistency: `counterfactual_order_accuracy`
4. Safety and rationale consistency (optional judge): `judge_alignment_mean`, `unsafe_advice_rate`

## Data schema

Sample file (`json` or `jsonl`) fields:

- `id`: unique sample id
- `dialogue`: raw dialogue text
- `human_score`: float in `[0, 100]`
- `symptom_labels`: list of symptom tags
- `risk_label`: risk level (`low|medium|high|...`)

Counterfactual triples file fields:

- `id`: unique triple id
- `light_id`: milder sample id
- `heavy_id`: severer sample id
- `expected_order`: default `heavy_gt_light`

See:

- `eval/benchmark_v2/samples.example.json`
- `eval/benchmark_v2/counterfactuals.example.json`

## Quick start

From repository root:

```bash
python -m eval.benchmark_v2.run_benchmark_v2 \
  --samples eval/benchmark_v2/samples.example.json \
  --counterfactuals eval/benchmark_v2/counterfactuals.example.json \
  --mode mock \
  --repeats 3 \
  --output-dir eval/benchmark_v2/outputs
```

Outputs:

- `eval/benchmark_v2/outputs/results_raw.jsonl`
- `eval/benchmark_v2/outputs/summary.json`

## Build dataset from current repo data

If `data/generate/processed/distilled_scores.jsonl` exists, `--source auto` will use distilled labels first. Otherwise it falls back to weak-supervision labels from symptom/risk keywords and diagnosis text.

```bash
python -m eval.benchmark_v2.dataset_builder \
  --source auto \
  --distilled-path data/generate/processed/distilled_scores.jsonl \
  --cards-path data/generate/cards.jsonl \
  --pa20-path data/generate/raw_data/pa20.json \
  --output-samples eval/benchmark_v2/data/samples.v1.json \
  --output-counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json
```

Then run benchmark:

```bash
python -m eval.benchmark_v2.run_benchmark_v2 \
  --samples eval/benchmark_v2/data/samples.v1.json \
  --counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json \
  --mode mock \
  --repeats 3 \
  --output-dir eval/benchmark_v2/outputs
```

## OpenAI-compatible mode

```bash
export OPENAI_API_KEY="..."
python -m eval.benchmark_v2.run_benchmark_v2 \
  --samples /path/to/your_samples.json \
  --mode openai-compatible \
  --repeats 3 \
  --use-judge
```

Optional judge envs:

- `JUDGE_API_KEY`
- `JUDGE_BASE_URL`
- `JUDGE_MODEL`

## Config-driven closed-source matrix

Use one config file to define all model adapters and judge adapter:

```bash
python -m eval.benchmark_v2.run_benchmark_v2 \
  --config eval/benchmark_v2/configs/closed_source.v1.json \
  --samples eval/benchmark_v2/data/samples.v1.json \
  --counterfactuals eval/benchmark_v2/data/counterfactuals.v1.json \
  --repeats 3 \
  --temperature 0.1 \
  --output-dir eval/benchmark_v2/outputs
```

## Notes

- The framework is model-adapter based. Add new vendors by implementing `BaseScoreClient`.
- For a stable leaderboard, use a fixed test split and fixed random seed in your data preparation stage.
