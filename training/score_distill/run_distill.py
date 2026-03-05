from __future__ import annotations

import argparse

from training.score_distill.config import DistillConfig
from training.score_distill.pipeline import run_distillation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSeek-assisted score distillation")
    parser.add_argument("--input-dialogues", default="data/generate/DataSynCards/dialogues.jsonl")
    parser.add_argument("--output-labels", default="data/generate/processed/distilled_scores.jsonl")
    parser.add_argument("--cache-path", default="training/score_distill/artifacts/teacher_cache.json")
    parser.add_argument("--model-ckpt", default="training/score_distill/artifacts/student_model.pt")
    parser.add_argument("--roberta-model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--teacher-model-name", default="deepseek-chat")
    parser.add_argument("--teacher-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = DistillConfig(
        input_dialogues=args.input_dialogues,
        output_labels=args.output_labels,
        cache_path=args.cache_path,
        model_ckpt=args.model_ckpt,
        roberta_model_name=args.roberta_model_name,
        teacher_model_name=args.teacher_model_name,
        teacher_base_url=args.teacher_base_url,
        sample_ratio=args.sample_ratio,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
        max_turns=args.max_turns,
        device=args.device,
    )
    summary = run_distillation(cfg)
    print("Distillation completed.")
    print(summary)


if __name__ == "__main__":
    main()

