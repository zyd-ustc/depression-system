from __future__ import annotations

import json
import re
from pathlib import Path

from training.score_distill.io_utils import TurnSample


class DeepSeekTeacher:
    def __init__(self, model_name: str, base_url: str, api_key: str, cache_path: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for DeepSeekTeacher") from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, float]:
        if not self.cache_path.exists():
            return {}
        with self.cache_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_cache(self) -> None:
        with self.cache_path.open("w", encoding="utf-8") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_score(text: str) -> float:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            raise ValueError(f"Cannot parse score from teacher output: {text}")
        return max(0.0, min(100.0, float(match.group(1))))

    def _prompt(self, sample: TurnSample) -> str:
        return (
            "你是一位资深临床心理评估专家，请对以下来访者对话片段进行抑郁严重度评分。\n"
            "输出要求：只输出0-100一个数字，不要输出其他内容。\n"
            f"对话片段：\n{sample.dialogue_window}\n"
            f"当前用户陈述：{sample.instruction}\n"
        )

    def score(self, sample: TurnSample) -> float:
        if sample.key in self.cache:
            return float(self.cache[sample.key])
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self._prompt(sample)}],
            temperature=0.1,
        )
        text = response.choices[0].message.content or ""
        score = self._parse_score(text)
        self.cache[sample.key] = score
        if len(self.cache) % 10 == 0:
            self._save_cache()
        return score

    def flush(self) -> None:
        self._save_cache()

