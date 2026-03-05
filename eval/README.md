# Evaluation (MindEval)

[MindEval](https://github.com/SWORDHealth/mind-eval) benchmarks multi-turn mental health support. Only usage and wiring are documented here.

## Setup

Clone and install mind-eval (see their README; Poetry, Python 3.10, gcloud credentials).

## Run benchmark

From the mind-eval repo root, run their pipeline with your clinician model API params:

```bash
bash run_benchmark.sh '<YOUR_CLINICIAN_MODEL_API_PARAMS>'
```

Or step by step:

1. `generate_interactions.py` — profiles + clinician + member models → `interactions.jsonl`
2. `generate_judgments.py` — judge model scores interactions → `judgments.jsonl`
3. Aggregate scores (e.g. `df.mean()` on parsed judgments)

To evaluate this project’s trained model, expose it via an API compatible with mind-eval’s `InferenceEngine` and pass the corresponding params as the clinician model.

## Reference

- Paper: [MindEval: Benchmarking Language Models on Multi-turn Mental Health Support](https://arxiv.org/abs/2511.18491)

## Benchmark v2 (this repo)

This repository also includes a local depression-scoring benchmark framework with:

- accuracy metrics (MAE/RMSE/correlation/CCC),
- stability metrics (test-retest variance, prompt paraphrase variance, ICC),
- counterfactual order consistency,
- optional judge-based rationale/safety checks.

See `eval/benchmark_v2/README.md`.

When `data/generate/processed/distilled_scores.jsonl` is available (from `training/score_distill`), benchmark v2 will use distilled scores as the primary label source to keep training/evaluation label logic consistent.
