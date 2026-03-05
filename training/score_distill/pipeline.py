from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

from training.score_distill.config import DistillConfig
from training.score_distill.features import RobertaFeatureExtractor
from training.score_distill.io_utils import (
    TurnSample,
    build_turn_samples,
    load_dialogues,
    write_distilled_labels,
)
from training.score_distill.student import StudentRegressor, TrainConfig
from training.score_distill.teacher import DeepSeekTeacher
from training.score_distill.postprocess import enrich_output


NEGATIVE_HINTS = [
    "情绪低落",
    "睡不好",
    "失眠",
    "绝望",
    "焦虑",
    "不想活",
    "自杀",
    "自残",
    "无意义",
]


def _heuristic_teacher_score(sample: TurnSample) -> float:
    text = f"{sample.dialogue_window}\n{sample.instruction}"
    score = 20.0
    for hint in NEGATIVE_HINTS:
        if hint in text:
            score += 10.0
    return max(0.0, min(100.0, score))


def _select_teacher_indices(total: int, sample_ratio: float, seed: int) -> list[int]:
    count = max(1, int(total * sample_ratio))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:count])


def run_distillation(cfg: DistillConfig) -> dict:
    dialogues = load_dialogues(cfg.input_dialogues)
    turn_samples = build_turn_samples(dialogues, max_turns=cfg.max_turns)
    if not turn_samples:
        raise ValueError("No turn samples built from input dialogues.")

    features = RobertaFeatureExtractor(cfg.roberta_model_name, device=cfg.device).encode(
        [f"{sample.dialogue_window}\n患者当前陈述：{sample.instruction}" for sample in turn_samples],
        batch_size=cfg.batch_size,
    )

    teacher_indices = _select_teacher_indices(len(turn_samples), cfg.sample_ratio, cfg.random_seed)
    teacher_scores: dict[int, float] = {}

    api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    teacher = None
    if api_key:
        teacher = DeepSeekTeacher(
            model_name=cfg.teacher_model_name,
            base_url=cfg.teacher_base_url,
            api_key=api_key,
            cache_path=cfg.cache_path,
        )

    for index in teacher_indices:
        sample = turn_samples[index]
        if teacher is None:
            score = _heuristic_teacher_score(sample)
        else:
            score = teacher.score(sample)
        teacher_scores[index] = score

    if teacher is not None:
        teacher.flush()

    x_train = [features[index] for index in teacher_indices]
    y_train = [teacher_scores[index] for index in teacher_indices]

    student = StudentRegressor(input_dim=len(features[0]), device=cfg.device)
    student.fit(
        x_train=x_train,
        y_train=y_train,
        cfg=TrainConfig(
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            random_seed=cfg.random_seed,
        ),
    )
    student.save(cfg.model_ckpt)

    preds = student.predict(features, batch_size=max(cfg.batch_size, 128))
    rows = []
    for index, sample in enumerate(turn_samples):
        pred_score = round(float(preds[index]), 3)
        enriched = enrich_output(
            dialogue_text=f"{sample.dialogue_window}\n患者当前陈述：{sample.instruction}",
            score_0_100=pred_score,
        )
        row = {
            "key": sample.key,
            "source_idx": sample.source_idx,
            "turn_index": sample.turn_index,
            "instruction": sample.instruction,
            "doctor_reply": sample.doctor_reply,
            "dialogue_window": sample.dialogue_window,
            "depression_score": pred_score,
            "depression_level": enriched["depression_level"],
            "emotion_scores": enriched["emotion_scores"],
            "scales": enriched["scales"],
            "label_source": "teacher" if index in teacher_scores else "student",
            "teacher_score": (
                round(float(teacher_scores[index]), 3) if index in teacher_scores else None
            ),
        }
        rows.append(row)

    write_distilled_labels(cfg.output_labels, rows)
    summary = {
        "config": asdict(cfg),
        "input_dialogues": len(dialogues),
        "turn_samples": len(turn_samples),
        "teacher_labeled": len(teacher_indices),
        "output_labels": cfg.output_labels,
        "model_ckpt": cfg.model_ckpt,
    }
    summary_path = cfg.artifacts_dir / "distill_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
