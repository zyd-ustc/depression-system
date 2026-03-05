from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.benchmark_v2.adapters.mock import MockJudgeClient, MockScoreClient
from eval.benchmark_v2.adapters.openai_compatible import (
    OpenAICompatibleClient,
    OpenAICompatibleJudgeClient,
)
from eval.benchmark_v2.pipeline import BenchmarkRunner, ModelSpec, RunnerConfig
from eval.benchmark_v2.types import load_counterfactuals, load_samples


def _build_mock_specs() -> list[ModelSpec]:
    return [
        ModelSpec(name="GPT-4-mock", client=MockScoreClient("GPT-4")),
        ModelSpec(name="Claude-3-mock", client=MockScoreClient("Claude-3")),
        ModelSpec(name="Gemini-Pro-mock", client=MockScoreClient("Gemini-Pro")),
    ]


def _build_openai_compatible_specs() -> list[ModelSpec]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for openai-compatible mode")

    return [
        ModelSpec(
            name="gpt-4-turbo",
            client=OpenAICompatibleClient(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model_name=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            ),
        ),
    ]


def _build_specs_from_config(config_path: str) -> tuple[list[ModelSpec], OpenAICompatibleJudgeClient | MockJudgeClient | None]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    mode = str(config.get("mode", "openai-compatible"))
    model_specs: list[ModelSpec] = []
    for item in config.get("models", []):
        name = str(item["name"])
        if mode == "mock":
            model_specs.append(ModelSpec(name=name, client=MockScoreClient(name)))
            continue
        api_key_env = str(item.get("api_key_env", "OPENAI_API_KEY"))
        base_url = str(item["base_url"])
        model_name = str(item["model_name"])
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing required env var: {api_key_env} for model {name}")
        model_specs.append(
            ModelSpec(
                name=name,
                client=OpenAICompatibleClient(
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                ),
            )
        )

    judge_cfg = config.get("judge", {})
    if not judge_cfg or not bool(judge_cfg.get("enabled", False)):
        judge_client = None
    elif mode == "mock":
        judge_client = MockJudgeClient()
    else:
        judge_api_env = str(judge_cfg.get("api_key_env", "JUDGE_API_KEY"))
        judge_api_key = os.getenv(judge_api_env, "")
        if not judge_api_key:
            raise RuntimeError(f"Missing required env var: {judge_api_env} for judge")
        judge_client = OpenAICompatibleJudgeClient(
            api_key=judge_api_key,
            base_url=str(judge_cfg["base_url"]),
            model_name=str(judge_cfg["model_name"]),
        )
    return model_specs, judge_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark v2")
    parser.add_argument("--samples", required=True, help="Path to samples json/jsonl")
    parser.add_argument("--counterfactuals", default=None, help="Path to counterfactual triples")
    parser.add_argument("--mode", choices=["mock", "openai-compatible"], default="mock")
    parser.add_argument("--config", default=None, help="Path to benchmark config JSON")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-dir", default="eval/benchmark_v2/outputs")
    parser.add_argument("--use-judge", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.samples)
    counterfactuals = load_counterfactuals(args.counterfactuals)

    if args.config:
        model_specs, judge_client = _build_specs_from_config(args.config)
    elif args.mode == "mock":
        model_specs = _build_mock_specs()
        judge_client = MockJudgeClient() if args.use_judge else None
    else:
        model_specs = _build_openai_compatible_specs()
        if args.use_judge:
            judge_client = OpenAICompatibleJudgeClient(
                api_key=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                base_url=os.getenv("JUDGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
                model_name=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
            )
        else:
            judge_client = None

    config = RunnerConfig(
        repeats=args.repeats,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )
    runner = BenchmarkRunner(model_specs=model_specs, config=config, judge_client=judge_client)
    summary = runner.run(samples=samples, counterfactuals=counterfactuals)
    print("Benchmark completed.")
    print(f"Summary written to: {args.output_dir}/summary.json")
    print(summary["model_metrics"])


if __name__ == "__main__":
    main()
