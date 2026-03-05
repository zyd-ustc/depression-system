from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.benchmark_v2.adapters.base import BaseJudgeClient, BaseScoreClient
from eval.benchmark_v2.metrics import (
    bucket_mae,
    ccc,
    counterfactual_order_accuracy,
    icc2_1,
    mae,
    mean_prompt_paraphrase_std,
    mean_test_retest_std,
    pearson,
    rmse,
    spearman,
)
from eval.benchmark_v2.prompts import SCORE_PROMPT_VARIANTS
from eval.benchmark_v2.types import BenchmarkSample, CounterfactualTriple, RawScoreRecord


@dataclass
class ModelSpec:
    name: str
    client: BaseScoreClient


@dataclass
class RunnerConfig:
    repeats: int = 3
    temperature: float = 0.1
    output_dir: str = "eval/benchmark_v2/outputs"
    prompt_versions: list[str] | None = None


class BenchmarkRunner:
    def __init__(
        self,
        model_specs: list[ModelSpec],
        config: RunnerConfig,
        judge_client: BaseJudgeClient | None = None,
    ) -> None:
        self.model_specs = model_specs
        self.config = config
        self.judge_client = judge_client
        self.prompt_versions = config.prompt_versions or list(SCORE_PROMPT_VARIANTS.keys())

    def _score_all(self, samples: list[BenchmarkSample]) -> list[RawScoreRecord]:
        records: list[RawScoreRecord] = []
        for sample in samples:
            for model_spec in self.model_specs:
                for prompt_version in self.prompt_versions:
                    prompt_text = SCORE_PROMPT_VARIANTS[prompt_version]
                    for repeat_index in range(self.config.repeats):
                        start = time.perf_counter()
                        result = model_spec.client.score_dialogue(
                            dialogue=sample.dialogue,
                            prompt_text=prompt_text,
                            temperature=self.config.temperature,
                        )
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        records.append(
                            RawScoreRecord(
                                sample_id=sample.id,
                                model=model_spec.name,
                                prompt_version=prompt_version,
                                repeat_id=repeat_index,
                                human_score=sample.human_score,
                                score=result.score,
                                parse_ok=result.parse_ok,
                                rationale=result.rationale,
                                raw_text=result.raw_text,
                                latency_ms=latency_ms,
                                error=result.error,
                            )
                        )
        return records

    @staticmethod
    def _aggregate_by_model_sample(records: list[RawScoreRecord]) -> dict[str, dict[str, float]]:
        grouped: dict[tuple[str, str], list[float]] = {}
        for record in records:
            if record.score is None:
                continue
            key = (record.model, record.sample_id)
            grouped.setdefault(key, []).append(record.score)
        output: dict[str, dict[str, float]] = {}
        for (model_name, sample_id), scores in grouped.items():
            output.setdefault(model_name, {})[sample_id] = sum(scores) / len(scores)
        return output

    @staticmethod
    def _human_by_sample(samples: list[BenchmarkSample]) -> dict[str, float]:
        return {sample.id: sample.human_score for sample in samples}

    def _compute_model_metrics(
        self,
        samples: list[BenchmarkSample],
        records: list[RawScoreRecord],
        counterfactuals: list[CounterfactualTriple],
    ) -> dict[str, Any]:
        human_by_sample = self._human_by_sample(samples)
        aggregated = self._aggregate_by_model_sample(records)
        per_model: dict[str, Any] = {}

        for model_spec in self.model_specs:
            model_name = model_spec.name
            sample_scores = aggregated.get(model_name, {})
            ordered_sample_ids = [sample.id for sample in samples]
            human_scores = [human_by_sample[sample_id] for sample_id in ordered_sample_ids]
            model_scores = [sample_scores.get(sample_id) for sample_id in ordered_sample_ids]

            model_records = [record for record in records if record.model == model_name]
            parse_success = [record for record in model_records if record.parse_ok]
            parse_rate = len(parse_success) / len(model_records) if model_records else math.nan

            judge_alignment_scores: list[float] = []
            unsafe_advice_count = 0
            if self.judge_client is not None:
                sample_map = {sample.id: sample for sample in samples}
                for record in parse_success:
                    sample = sample_map[record.sample_id]
                    judgment = self.judge_client.judge_alignment(
                        dialogue=sample.dialogue,
                        rationale=record.rationale,
                        score=record.score if record.score is not None else 0.0,
                        symptom_labels=sample.symptom_labels,
                        risk_label=sample.risk_label,
                    )
                    judge_alignment_scores.append(float(judgment.get("alignment_score", 0.0)))
                    if bool(judgment.get("unsafe_advice", False)):
                        unsafe_advice_count += 1

            counterfactual_acc = counterfactual_order_accuracy(counterfactuals, sample_scores)
            per_model[model_name] = {
                "parse_success_rate": parse_rate,
                "mae": mae(human_scores, model_scores),
                "rmse": rmse(human_scores, model_scores),
                "pearson": pearson(human_scores, model_scores),
                "spearman": spearman(human_scores, model_scores),
                "ccc": ccc(human_scores, model_scores),
                "bucket_mae": bucket_mae(human_scores, model_scores),
                "test_retest_std": mean_test_retest_std(model_records),
                "prompt_paraphrase_std": mean_prompt_paraphrase_std(model_records),
                "icc2_1": icc2_1(model_records, model_name),
                "counterfactual_order_accuracy": counterfactual_acc,
                "judge_alignment_mean": (
                    sum(judge_alignment_scores) / len(judge_alignment_scores)
                    if judge_alignment_scores
                    else math.nan
                ),
                "unsafe_advice_rate": (
                    (unsafe_advice_count / len(parse_success))
                    if (self.judge_client is not None and parse_success)
                    else math.nan
                ),
            }
        return per_model

    def _write_raw_records(self, records: list[RawScoreRecord], output_dir: Path) -> None:
        output_path = output_dir / "results_raw.jsonl"
        with output_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(record.to_json() + "\n")

    def _write_summary(self, summary: dict[str, Any], output_dir: Path) -> None:
        output_path = output_dir / "summary.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                _sanitize_for_json(summary),
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

    def run(
        self,
        samples: list[BenchmarkSample],
        counterfactuals: list[CounterfactualTriple] | None = None,
    ) -> dict[str, Any]:
        triples = counterfactuals or []
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        records = self._score_all(samples)
        self._write_raw_records(records, output_dir)

        metrics_by_model = self._compute_model_metrics(samples, records, triples)
        summary = {
            "config": asdict(self.config),
            "sample_count": len(samples),
            "counterfactual_count": len(triples),
            "model_metrics": metrics_by_model,
        }
        self._write_summary(summary, output_dir)
        return summary


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
