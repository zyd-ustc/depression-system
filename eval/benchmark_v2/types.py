from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkSample:
    id: str
    dialogue: str
    human_score: float
    symptom_labels: list[str]
    risk_label: str


@dataclass
class CounterfactualTriple:
    id: str
    light_id: str
    heavy_id: str
    expected_order: str = "heavy_gt_light"


@dataclass
class ScoreResult:
    score: float | None
    rationale: str
    raw_text: str
    parse_ok: bool
    error: str = ""


@dataclass
class RawScoreRecord:
    sample_id: str
    model: str
    prompt_version: str
    repeat_id: int
    human_score: float
    score: float | None
    parse_ok: bool
    rationale: str
    raw_text: str
    latency_ms: int
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"JSON root must be a list: {path}")
    raise ValueError(f"Unsupported format: {path}")


def load_samples(path: str | Path) -> list[BenchmarkSample]:
    data_path = Path(path)
    rows = _read_json_or_jsonl(data_path)
    samples: list[BenchmarkSample] = []
    for row in rows:
        sample = BenchmarkSample(
            id=str(row["id"]),
            dialogue=str(row["dialogue"]),
            human_score=float(row["human_score"]),
            symptom_labels=list(row.get("symptom_labels", [])),
            risk_label=str(row.get("risk_label", "unknown")),
        )
        if sample.human_score < 0 or sample.human_score > 100:
            raise ValueError(f"human_score out of range [0, 100] for sample {sample.id}")
        samples.append(sample)
    return samples


def load_counterfactuals(path: str | Path | None) -> list[CounterfactualTriple]:
    if path is None:
        return []
    triples_path = Path(path)
    rows = _read_json_or_jsonl(triples_path)
    triples: list[CounterfactualTriple] = []
    for row in rows:
        triples.append(
            CounterfactualTriple(
                id=str(row["id"]),
                light_id=str(row["light_id"]),
                heavy_id=str(row["heavy_id"]),
                expected_order=str(row.get("expected_order", "heavy_gt_light")),
            )
        )
    return triples

