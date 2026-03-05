from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DistillConfig:
    input_dialogues: str = "data/generate/DataSynCards/dialogues.jsonl"
    output_labels: str = "data/generate/processed/distilled_scores.jsonl"
    cache_path: str = "training/score_distill/artifacts/teacher_cache.json"
    model_ckpt: str = "training/score_distill/artifacts/student_model.pt"
    roberta_model_name: str = "hfl/chinese-roberta-wwm-ext"
    teacher_model_name: str = "deepseek-chat"
    teacher_base_url: str = "https://api.deepseek.com/v1"
    sample_ratio: float = 0.1
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 32
    random_seed: int = 42
    max_turns: int | None = None
    device: str = "auto"

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.cache_path).parent

