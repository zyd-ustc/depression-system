# Depression Dialogue System

A full-stack pipeline for building and evaluating an AI-assisted depression support dialogue system. The project covers synthetic dialogue generation, depression-score distillation, SFT data preparation, model training (via LLaMA-Factory), local inference with a Gradio UI, and evaluation with a local benchmark. An optional RAG module augments responses with psychology literature and methodology.

## Overview

The system is designed to support research and development of conversational agents for mental health: it generates realistic doctor–patient multi-turn dialogues, assigns depression-related scores (and clinical-scale proxies) through a teacher–student distillation process, and uses those labels both for supervised fine-tuning and for evaluation. This keeps training and evaluation on a consistent scoring scheme while reducing reliance on expensive human annotations.

**Main capabilities:**

- **Synthetic dialogue generation** — LLM-based doctor and patient simulators produce multi-turn dialogues from patient profiles (e.g. card-based or PA-20), with configurable turn limits and model backends.
- **Score distillation** — A teacher (e.g. DeepSeek API) plus a local student model produce depression scores, emotion dimensions, and scale-style outputs (PHQ-9, HAM-D, MADRS, QIDS) for each dialogue. Labels are written for downstream SFT and evaluation.
- **SFT data pipeline** — Raw dialogues are converted into SFT-ready samples with embedded metadata (distilled scores, emotions, scales) so the model is trained with the same conceptual labels used at evaluation time.
- **Training** — Configuration and dataset wiring for [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); no framework code is bundled. You provide the base model and run training in your own LLaMA-Factory clone.
- **Inference** — A Gradio-based UI for loading a fine-tuned model (or Hugging Face model ID) and conducting local multi-turn conversations.
- **Evaluation** — A local benchmark (benchmark v2) with accuracy (e.g. MAE, RMSE, correlation, CCC), stability (test–retest, prompt paraphrase variance, ICC), and counterfactual consistency. Integration points for [MindEval](https://github.com/SWORDHealth/mind-eval) are documented for API-based evaluation.
- **RAG (optional)** — Retrieval over psychology-related documents (theory, methodology, cases) to enrich model answers. Implemented in the `rag/` subtree with vector search and configurable data ingestion.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Data synthesis (data/generate)                                         │
│  Doctor + Patient simulators → multi-turn dialogues (e.g. dialogues.jsonl)│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Score distillation (training/score_distill)                             │
│  Teacher (e.g. DeepSeek) + student → distilled_scores.jsonl              │
│  (depression_score, level, emotion_scores, scales: PHQ-9/HAM-D/…)        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SFT data (data/generate/data_convert.py)                                │
│  Dialogues + distilled metadata → processed_dialogues.json              │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Training (LLaMA-Factory, config in training/)                          │
│  SFT on base LLM → fine-tuned model / LoRA adapter                       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐
│  Inference UI    │   │  Benchmark v2    │   │  RAG (optional)          │
│  (Gradio)        │   │  (local eval)    │   │  Psychology retrieval    │
└─────────────────┘   └─────────────────┘   └─────────────────────────┘
```

## Project Structure

| Path | Description |
|------|-------------|
| `data/generate/` | Dialogue synthesis: `main.py` (doctor–patient generation), `data_convert.py` (to SFT + metadata), card/PA-20 inputs, prompts, and `processed/` outputs (e.g. `distilled_scores.jsonl`, `processed_dialogues.json`). |
| `training/` | `score_distill/` (distillation script and student model), LLaMA-Factory config (`sft_config.yaml`, `dataset_info.json`). |
| `src/inference_pipeline/` | Gradio app and launcher script for local inference. |
| `eval/benchmark_v2/` | Local benchmark: dataset builder, sample/counterfactual generation, metrics (accuracy, stability, counterfactual), configs, and output schema. |
| `eval/` | Top-level eval docs and pointers to MindEval. |
| `rag/` | RAG pipeline: PDF/text ingestion, vector store, and retrieval for psychology content; see `rag/README.md`. |
| `scripts/` | Unified workflow script that chains distillation → SFT conversion → benchmark dataset build → benchmark run. |

## Dependencies and Integration

- **Python**: 3.10+ recommended.
- **Key libraries**: See `src/inference_pipeline/requirements_ui.txt` and `training/score_distill/requirements.txt`. RAG has its own `rag/requirements.txt`.
- **External services**: Optional DeepSeek API key for the distillation teacher; without it, a heuristic teacher is used (lower quality). LLaMA-Factory and MindEval are used externally; this repo only provides configs and wiring.

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — training framework used for SFT.
- [MindEval](https://github.com/SWORDHealth/mind-eval) — multi-turn mental health support benchmark; see `eval/README.md` for how to plug in this project’s model.

For step-by-step setup, environment variables, and one-shot script usage, see the docs in `training/README.md`, `eval/README.md`, `eval/benchmark_v2/README.md`, and `scripts/run_unified_workflow.sh`.
