from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    "depressed_mood": ["情绪低落", "悲观", "绝望", "高兴不起来", "心情差", "情绪很低"],
    "anhedonia": ["兴趣减退", "无愉悦感", "提不起兴趣", "什么都不想做"],
    "sleep_problem": ["失眠", "入睡困难", "早醒", "睡眠差", "睡不好", "多梦"],
    "fatigue": ["疲劳", "精力不足", "没力气", "乏力"],
    "appetite_change": ["食欲减退", "食欲下降", "暴食", "厌食", "体重减轻", "体重下降"],
    "attention_problem": ["注意力不集中", "反应变慢", "记忆减退", "思维迟缓"],
    "guilt_worthlessness": ["无价值感", "自责", "内疚", "没用"],
    "psychomotor_change": ["言语减少", "活动减少", "迟缓"],
    "anxiety": ["焦虑", "恐惧", "紧张", "心慌"],
    "self_harm_ideation": ["自残", "自杀", "轻生", "不想活", "活着没意思", "消极意念"],
}

HIGH_RISK_KEYWORDS = ["自杀", "自残", "轻生", "结束生命", "不想活", "活着没意思"]
MEDIUM_RISK_KEYWORDS = ["绝望", "消极", "无望", "崩溃", "抑郁", "严重焦虑"]

DIAGNOSIS_BASE_SCORE = {
    "重度抑郁": 85.0,
    "中度抑郁": 68.0,
    "轻度抑郁": 45.0,
    "抑郁障碍": 62.0,
    "抑郁状态": 58.0,
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_distilled_jsonl(path: Path) -> list[dict]:
    return _read_jsonl(path)


def _extract_symptoms(text: str) -> list[str]:
    labels: list[str] = []
    for symptom, words in SYMPTOM_KEYWORDS.items():
        if any(word in text for word in words):
            labels.append(symptom)
    return labels


def _infer_risk_label(text: str) -> str:
    if any(word in text for word in HIGH_RISK_KEYWORDS):
        return "high"
    if any(word in text for word in MEDIUM_RISK_KEYWORDS):
        return "medium"
    return "low"


def _infer_score(text: str, diagnosis_text: str = "") -> float:
    score = 20.0

    # Diagnosis prior (for pa20 this is useful and stronger than pure keyword counting).
    for key, value in DIAGNOSIS_BASE_SCORE.items():
        if key in diagnosis_text or key in text:
            score = max(score, value)
            break

    symptom_labels = _extract_symptoms(text)
    score += len(symptom_labels) * 6.0

    # Frequency-sensitive distress load: repeated symptom mentions increase severity estimate.
    repeat_hits = 0
    for words in SYMPTOM_KEYWORDS.values():
        for word in words:
            repeat_hits += len(re.findall(re.escape(word), text))
    score += min(20.0, repeat_hits * 1.2)
    if "self_harm_ideation" in symptom_labels:
        score += 20.0
    if "anhedonia" in symptom_labels and "sleep_problem" in symptom_labels:
        score += 5.0
    if _infer_risk_label(text) == "high":
        score += 18.0
    elif _infer_risk_label(text) == "medium":
        score += 8.0
    return max(0.0, min(100.0, round(score, 1)))


def _build_counterfactuals(samples: list[dict], seed: int, max_pairs: int) -> list[dict]:
    sorted_samples = sorted(samples, key=lambda item: item["human_score"])
    if len(sorted_samples) < 4:
        return []

    rng = random.Random(seed)
    lower_bucket = sorted_samples[: max(1, len(sorted_samples) // 3)]
    upper_bucket = sorted_samples[-max(1, len(sorted_samples) // 3) :]

    pair_set: set[tuple[str, str]] = set()
    for light in lower_bucket:
        for heavy in upper_bucket:
            if heavy["human_score"] - light["human_score"] >= 12:
                pair_set.add((light["id"], heavy["id"]))

    if not pair_set:
        return []

    pair_list = list(pair_set)
    rng.shuffle(pair_list)
    selected = pair_list[:max_pairs]
    return [
        {
            "id": f"cf_{idx + 1:04d}",
            "light_id": light_id,
            "heavy_id": heavy_id,
            "expected_order": "heavy_gt_light",
        }
        for idx, (light_id, heavy_id) in enumerate(selected)
    ]


def build_from_cards(cards_path: Path, max_samples: int | None = None) -> list[dict]:
    rows = _read_jsonl(cards_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    output: list[dict] = []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("inner_monologue", "")).strip()
        if not text:
            continue
        symptom_labels = _extract_symptoms(text)
        risk_label = _infer_risk_label(text)
        score = _infer_score(text)
        output.append(
            {
                "id": f"cards_{index:04d}",
                "dialogue": text,
                "human_score": score,
                "symptom_labels": symptom_labels,
                "risk_label": risk_label,
                "source": "cards",
                "label_quality": "weak_supervision",
                "meta": {
                    "cognitive_bias": row.get("cognitive_bias", ""),
                    "seed_scenario": (row.get("metadata") or {}).get("seed_scenario", ""),
                },
            }
        )
    return output


def _flatten_pa20_dialogue(row: dict) -> str:
    complaint = str(row.get("主诉", ""))
    history = str(row.get("现病史", ""))
    mental = row.get("精神检查", {}) or {}
    mental_text = "；".join(f"{key}:{value}" for key, value in mental.items())
    return f"主诉：{complaint}\n现病史：{history}\n精神检查：{mental_text}".strip()


def build_from_pa20(pa20_path: Path, max_samples: int | None = None) -> list[dict]:
    rows = _read_json(pa20_path)
    if not isinstance(rows, list):
        raise ValueError("pa20 source must be a JSON array")
    if max_samples is not None:
        rows = rows[:max_samples]

    output: list[dict] = []
    for index, row in enumerate(rows, start=1):
        text = _flatten_pa20_dialogue(row)
        diagnosis_text = str(row.get("诊断结果", ""))
        symptom_labels = _extract_symptoms(text)
        risk_label = _infer_risk_label(text)
        score = _infer_score(text, diagnosis_text=diagnosis_text)
        output.append(
            {
                "id": f"pa20_{index:04d}",
                "dialogue": text,
                "human_score": score,
                "symptom_labels": symptom_labels,
                "risk_label": risk_label,
                "source": "pa20",
                "label_quality": "weak_supervision",
                "meta": {
                    "diagnosis_result": diagnosis_text,
                    "icd_code": row.get("ICD编码", ""),
                },
            }
        )
    return output


def build_from_distilled(distilled_path: Path, max_samples: int | None = None) -> list[dict]:
    rows = _read_distilled_jsonl(distilled_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    output: list[dict] = []
    for row in rows:
        dialogue_text = str(row.get("dialogue_window", "")).strip()
        instruction = str(row.get("instruction", "")).strip()
        if instruction:
            dialogue_text = f"{dialogue_text}\n患者当前陈述：{instruction}".strip()
        if not dialogue_text:
            continue
        symptom_labels = _extract_symptoms(dialogue_text)
        risk_label = _infer_risk_label(dialogue_text)
        score = float(row.get("depression_score", _infer_score(dialogue_text)))
        output.append(
            {
                "id": f"distill_{row.get('key')}",
                "dialogue": dialogue_text,
                "human_score": max(0.0, min(100.0, score)),
                "symptom_labels": symptom_labels,
                "risk_label": risk_label,
                "source": "distilled",
                "label_quality": "distilled",
                "meta": {
                    "distill_key": row.get("key"),
                    "teacher_score": row.get("teacher_score"),
                    "label_source": row.get("label_source"),
                    "depression_level": row.get("depression_level"),
                    "emotion_scores": row.get("emotion_scores"),
                    "scales": row.get("scales"),
                },
            }
        )
    return output


def build_dataset(
    source: str,
    cards_path: Path,
    pa20_path: Path,
    distilled_path: Path,
    max_samples: int | None,
    seed: int,
    max_counterfactuals: int,
) -> tuple[list[dict], list[dict]]:
    if source == "cards":
        samples = build_from_cards(cards_path, max_samples=max_samples)
    elif source == "pa20":
        samples = build_from_pa20(pa20_path, max_samples=max_samples)
    elif source == "auto":
        if distilled_path.exists():
            samples = build_from_distilled(distilled_path, max_samples=max_samples)
        else:
            cards_samples = build_from_cards(cards_path, max_samples=max_samples)
            pa20_samples = build_from_pa20(pa20_path, max_samples=max_samples)
            samples = cards_samples + pa20_samples
    elif source == "distilled":
        samples = build_from_distilled(distilled_path, max_samples=max_samples)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if not samples:
        raise ValueError("No samples built from source data")

    counterfactuals = _build_counterfactuals(
        samples=samples,
        seed=seed,
        max_pairs=min(max_counterfactuals, max(1, len(samples) // 3)),
    )
    return samples, counterfactuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark v2 dataset from local sources")
    parser.add_argument("--source", choices=["cards", "pa20", "distilled", "auto"], default="auto")
    parser.add_argument("--cards-path", default="data/generate/cards.jsonl")
    parser.add_argument("--pa20-path", default="data/generate/raw_data/pa20.json")
    parser.add_argument("--distilled-path", default="data/generate/processed/distilled_scores.jsonl")
    parser.add_argument("--output-samples", default="eval/benchmark_v2/data/samples.v1.json")
    parser.add_argument(
        "--output-counterfactuals",
        default="eval/benchmark_v2/data/counterfactuals.v1.json",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-counterfactuals", type=int, default=60)
    args = parser.parse_args()

    samples, counterfactuals = build_dataset(
        source=args.source,
        cards_path=Path(args.cards_path),
        pa20_path=Path(args.pa20_path),
        distilled_path=Path(args.distilled_path),
        max_samples=args.max_samples,
        seed=args.seed,
        max_counterfactuals=args.max_counterfactuals,
    )

    output_samples = Path(args.output_samples)
    output_counterfactuals = Path(args.output_counterfactuals)
    output_samples.parent.mkdir(parents=True, exist_ok=True)
    output_counterfactuals.parent.mkdir(parents=True, exist_ok=True)

    with output_samples.open("w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
    with output_counterfactuals.open("w", encoding="utf-8") as file:
        json.dump(counterfactuals, file, ensure_ascii=False, indent=2)

    print(f"Built samples: {len(samples)} -> {output_samples}")
    print(f"Built counterfactuals: {len(counterfactuals)} -> {output_counterfactuals}")
    weak_count = sum(1 for sample in samples if sample.get("label_quality") == "weak_supervision")
    print(f"Weak supervision labels: {weak_count}/{len(samples)}")


if __name__ == "__main__":
    main()
