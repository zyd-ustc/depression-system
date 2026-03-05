# Training (LLaMA-Factory)

SFT uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Only config and dataset wiring live here.

## Setup

1. Clone and install LLaMA-Factory (or install `llamafactory`).
2. Copy `training/sft_config.yaml` and `training/dataset_info.json` into LLaMA-Factory (or point their CLI at this repo).
3. Export SFT data from this project into LLaMA-Factory’s data dir:
   - From repo root: run `data/generate/main.py` (cards or pa20), then `data/generate/data_convert.py`.
   - Copy `data/generate/processed/processed_dialogues.json` to `LLaMA-Factory/data/depression.json`.
4. Merge `training/dataset_info.json` into LLaMA-Factory’s `data/dataset_info.json` (add the `depression_dialogue` entry).

## Train

Edit `sft_config.yaml`: set `model_name_or_path` (base model) and `output_dir`. Then:

```bash
cd /path/to/LLaMA-Factory
llamafactory-cli train /path/to/depression_system/training/sft_config.yaml
```

Checkpoints are written to `output_dir`. Use the merged model (or LoRA adapter) as `MODEL_ID` for the inference UI.

## Score distillation workflow (DeepSeek + Student model)

To reduce labeling cost while keeping score consistency across training and evaluation, this repo includes a distillation module:

- `training/score_distill/run_distill.py`
- Input: `data/generate/DataSynCards/dialogues.jsonl`
- Output labels: `data/generate/processed/distilled_scores.jsonl`

Run:

```bash
pip install -r training/score_distill/requirements.txt
python -m training.score_distill.run_distill \
  --input-dialogues data/generate/DataSynCards/dialogues.jsonl \
  --output-labels data/generate/processed/distilled_scores.jsonl \
  --sample-ratio 0.1 \
  --epochs 30
```

Then run `data/generate/data_convert.py`; converted SFT samples will include distill metadata (`depression_score`, `teacher_score`, `score_label_source`).
The distill output `distilled_scores.jsonl` also includes:

- `depression_level`
- `emotion_scores` (sadness/fear/aversion/anger/anticipation/surprise)
- `scales` (`PHQ-9`, `HAM-D`, `MADRS`, `QIDS`)

## End-to-end unified pipeline

For one-command orchestration from distillation to benchmark:

```bash
bash scripts/run_unified_workflow.sh
```
