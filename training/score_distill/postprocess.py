from __future__ import annotations

from dataclasses import dataclass


EMOTION_KEYWORDS: dict[str, list[str]] = {
    "sadness": ["难过", "悲伤", "情绪低落", "绝望", "无助", "空虚"],
    "fear": ["害怕", "恐惧", "紧张", "担心", "焦虑", "不安"],
    "aversion": ["厌恶", "反感", "恶心", "排斥", "烦躁"],
    "anger": ["生气", "愤怒", "火大", "发脾气", "恼火"],
    "anticipation": ["希望", "期待", "打算", "计划", "明天会好"],
    "surprise": ["突然", "震惊", "意外", "没想到"],
}


@dataclass
class ScaleResult:
    score: int
    level: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def depression_level(score_0_100: float) -> str:
    if score_0_100 < 20:
        return "无明显抑郁"
    if score_0_100 < 40:
        return "轻度"
    if score_0_100 < 60:
        return "中度"
    if score_0_100 < 80:
        return "中重度"
    return "重度"


def _phq9_level(score: int) -> str:
    if score <= 4:
        return "无/最轻"
    if score <= 9:
        return "轻度"
    if score <= 14:
        return "中度"
    if score <= 19:
        return "中重度"
    return "重度"


def _hamd_level(score: int) -> str:
    if score <= 7:
        return "无/最轻"
    if score <= 16:
        return "轻度"
    if score <= 23:
        return "中度"
    return "重度"


def _madrs_level(score: int) -> str:
    if score <= 6:
        return "无/最轻"
    if score <= 19:
        return "轻度"
    if score <= 34:
        return "中度"
    return "重度"


def _qids_level(score: int) -> str:
    if score <= 5:
        return "无/最轻"
    if score <= 10:
        return "轻度"
    if score <= 15:
        return "中度"
    if score <= 20:
        return "重度"
    return "极重度"


def map_scales(score_0_100: float) -> dict[str, dict]:
    score = _clamp(score_0_100, 0.0, 100.0)
    phq9 = int(round(score / 100.0 * 27))
    hamd = int(round(score / 100.0 * 52))
    madrs = int(round(score / 100.0 * 60))
    qids = int(round(score / 100.0 * 27))
    return {
        "PHQ-9": {"score": phq9, "level": _phq9_level(phq9)},
        "HAM-D": {"score": hamd, "level": _hamd_level(hamd)},
        "MADRS": {"score": madrs, "level": _madrs_level(madrs)},
        "QIDS": {"score": qids, "level": _qids_level(qids)},
    }


def emotion_scores(text: str, score_0_100: float) -> dict[str, float]:
    text = text or ""
    base_depression = _clamp(score_0_100 / 100.0, 0.0, 1.0)
    values = {
        "sadness": 0.25 + 0.6 * base_depression,
        "fear": 0.2 + 0.4 * base_depression,
        "aversion": 0.1 + 0.3 * base_depression,
        "anger": 0.1 + 0.2 * base_depression,
        "anticipation": 0.35 - 0.25 * base_depression,
        "surprise": 0.2,
    }
    for emotion, keywords in EMOTION_KEYWORDS.items():
        hit_count = sum(text.count(keyword) for keyword in keywords)
        values[emotion] += min(0.35, hit_count * 0.08)

    output: dict[str, float] = {}
    for key, value in values.items():
        output[key] = round(_clamp(value, 0.0, 1.0), 3)
    return output


def enrich_output(dialogue_text: str, score_0_100: float) -> dict:
    return {
        "depression_level": depression_level(score_0_100),
        "emotion_scores": emotion_scores(dialogue_text, score_0_100),
        "scales": map_scales(score_0_100),
    }

