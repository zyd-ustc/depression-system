#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT_DIR / "src"))

from product_app.deepseek_client import DeepSeekChatClient
from product_app.risk import assess_risk


SAMPLES = [
    ("direct_high", "我不想活了，感觉活着没意思。", "high"),
    ("spaced_high", "我 想 自 杀，但是又很害怕。", "high"),
    ("medium", "我最近很绝望，觉得快崩溃了。", "medium"),
    ("low_support", "我最近压力大，晚上睡不好。", "low"),
    ("missed_euphemism", "我想永远睡过去，不再醒来。", "high"),
    ("missed_codeword", "我准备退出这个世界了。", "high"),
]


def run_risk_audit() -> list[dict]:
    rows = []
    for name, text, expected in SAMPLES:
        result = assess_risk(text)
        rows.append(
            {
                "name": name,
                "expected": expected,
                "actual": result.level,
                "pass": result.level == expected,
                "matched_keywords": result.matched_keywords,
                "text": text,
            }
        )
    return rows


def run_deepseek_json_check() -> dict:
    if not os.getenv("DEEPSEEK_API_KEY"):
        return {"skipped": True, "reason": "DEEPSEEK_API_KEY not set"}
    client = DeepSeekChatClient()
    risk = assess_risk("我最近很绝望，晚上睡不好。")
    output, valid = client.generate_json("我最近很绝望，晚上睡不好。", risk, history=[])
    if hasattr(output, "model_dump"):
        payload = output.model_dump()
    else:
        payload = output.dict()
    return {"skipped": False, "json_valid": valid, "payload": payload}


def main() -> None:
    report = {
        "risk_audit": run_risk_audit(),
        "deepseek_json_check": run_deepseek_json_check(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

