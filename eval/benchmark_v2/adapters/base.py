from __future__ import annotations

from abc import ABC, abstractmethod

from eval.benchmark_v2.types import ScoreResult


class BaseScoreClient(ABC):
    @abstractmethod
    def score_dialogue(self, dialogue: str, prompt_text: str, temperature: float) -> ScoreResult:
        raise NotImplementedError


class BaseJudgeClient(ABC):
    @abstractmethod
    def judge_alignment(
        self,
        dialogue: str,
        rationale: str,
        score: float,
        symptom_labels: list[str],
        risk_label: str,
    ) -> dict:
        raise NotImplementedError

