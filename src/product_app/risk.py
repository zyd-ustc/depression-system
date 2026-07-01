from __future__ import annotations

import re

from product_app.schemas import RiskAssessment

HIGH_RISK_KEYWORDS = ["自杀", "自残", "轻生", "结束生命", "不想活", "活着没意思"]
MEDIUM_RISK_KEYWORDS = ["绝望", "消极", "无望", "崩溃", "抑郁", "严重焦虑"]

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "情绪低落": ["情绪低落", "悲观", "绝望", "高兴不起来", "心情差", "情绪很低", "难过"],
    "兴趣减退": ["兴趣减退", "无愉悦感", "提不起兴趣", "什么都不想做"],
    "睡眠": ["失眠", "入睡困难", "早醒", "睡眠差", "睡不好", "多梦"],
    "精力疲劳": ["疲劳", "精力不足", "没力气", "乏力", "很累"],
    "饮食体重": ["食欲减退", "食欲下降", "暴食", "厌食", "体重减轻", "体重下降"],
    "注意力": ["注意力不集中", "反应变慢", "记忆减退", "思维迟缓"],
    "自责无价值": ["无价值感", "自责", "内疚", "没用"],
    "焦虑紧张": ["焦虑", "恐惧", "紧张", "心慌"],
    "自伤风险": ["自残", "自杀", "轻生", "不想活", "活着没意思", "消极意念"],
    "学习工作": ["学习", "考试", "工作", "上班", "绩效", "作业"],
    "人际关系": ["家人", "父母", "朋友", "同学", "同事", "关系"],
}


def _normalize(text: str) -> str:
    return re.sub(r"[\s\u200b\u200c\u200d\ufeff，。！？、,.!?;；:：\"'`~（）()【】\[\]{}<>《》|-]+", "", text)


def _keyword_hits(text: str, words: list[str]) -> list[str]:
    normalized = _normalize(text)
    hits = []
    for word in words:
        if word in text or _normalize(word) in normalized:
            hits.append(word)
    return hits


def _covered_topics(text: str) -> list[str]:
    return [topic for topic, words in TOPIC_KEYWORDS.items() if _keyword_hits(text, words)]


def _score(level: str, topics: list[str], matched_keywords: list[str]) -> int:
    base = {"low": 20, "medium": 55, "high": 85}[level]
    score = base + len(topics) * 4 + len(matched_keywords) * 3
    return max(0, min(100, score))


def assess_risk(text: str) -> RiskAssessment:
    high_hits = _keyword_hits(text, HIGH_RISK_KEYWORDS)
    medium_hits = _keyword_hits(text, MEDIUM_RISK_KEYWORDS)
    topics = _covered_topics(text)

    if high_hits:
        return RiskAssessment(
            level="high",
            score=_score("high", topics, high_hits),
            covered_topics=topics,
            matched_keywords=high_hits,
            route="urgent_support",
            rationale="Matched existing high-risk self-harm keywords.",
        )
    if medium_hits:
        return RiskAssessment(
            level="medium",
            score=_score("medium", topics, medium_hits),
            covered_topics=topics,
            matched_keywords=medium_hits,
            route="suggest_professional_help",
            rationale="Matched existing medium-risk distress keywords.",
        )
    return RiskAssessment(
        level="low",
        score=_score("low", topics, []),
        covered_topics=topics,
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
