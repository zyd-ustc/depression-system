from __future__ import annotations

import re

from product_app.schemas import RiskAssessment

HIGH_RISK_KEYWORDS = ["自杀", "自残", "轻生", "结束生命", "不想活", "活着没意思"]
MEDIUM_RISK_KEYWORDS = ["绝望", "消极", "无望", "崩溃", "抑郁", "严重焦虑"]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\u200b\u200c\u200d\ufeff，。！？、,.!?;；:：\"'`~（）()【】\[\]{}<>《》|-]+", "", text)


def _keyword_hits(text: str, words: list[str]) -> list[str]:
    normalized = _normalize(text)
    hits = []
    for word in words:
        if word in text or _normalize(word) in normalized:
            hits.append(word)
    return hits


def assess_risk(text: str) -> RiskAssessment:
    high_hits = _keyword_hits(text, HIGH_RISK_KEYWORDS)
    medium_hits = _keyword_hits(text, MEDIUM_RISK_KEYWORDS)

    if high_hits:
        return RiskAssessment(
            level="high",
            matched_keywords=high_hits,
            route="urgent_support",
            rationale="Matched existing high-risk self-harm keywords.",
        )
    if medium_hits:
        return RiskAssessment(
            level="medium",
            matched_keywords=medium_hits,
            route="suggest_professional_help",
            rationale="Matched existing medium-risk distress keywords.",
        )
    return RiskAssessment(
        level="low",
        matched_keywords=[],
        route="support",
        rationale="No existing medium/high risk keyword matched.",
    )


def high_risk_reply() -> str:
    return (
        "我注意到你提到了可能伤害自己的内容。这个系统不能替代紧急支持。"
        "请现在先联系身边可信任的人陪伴你，并尽快联系当地急救服务、医院急诊或心理危机援助热线。"
        "在你获得线下支持前，请尽量远离可能造成伤害的物品或环境。"
    )
