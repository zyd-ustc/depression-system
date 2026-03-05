from __future__ import annotations

import hashlib

from eval.benchmark_v2.adapters.base import BaseJudgeClient, BaseScoreClient
from eval.benchmark_v2.types import ScoreResult


class MockScoreClient(BaseScoreClient):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def score_dialogue(self, dialogue: str, prompt_text: str, temperature: float) -> ScoreResult:
        seed_text = f"{self.model_name}|{prompt_text}|{dialogue}|{temperature}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) % 101
        rationale = f"Mock rationale for {self.model_name}, synthetic score={value}."
        return ScoreResult(
            score=float(value),
            rationale=rationale,
            raw_text=rationale,
            parse_ok=True,
            error="",
        )


class MockJudgeClient(BaseJudgeClient):
    def judge_alignment(
        self,
        dialogue: str,
        rationale: str,
        score: float,
        symptom_labels: list[str],
        risk_label: str,
    ) -> dict:
        alignment = 0.8 if 25 <= score <= 85 else 0.6
        unsafe_advice = False
        return {
            "alignment_score": alignment,
            "unsafe_advice": unsafe_advice,
            "comments": "mock judge",
        }

