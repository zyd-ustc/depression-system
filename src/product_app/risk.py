from __future__ import annotations

import re

from product_app.schemas import NextTopicFocus, RiskAssessment

HIGH_RISK_KEYWORDS = ["自杀", "自残", "轻生", "结束生命", "不想活", "活着没意思"]
MEDIUM_RISK_KEYWORDS = ["绝望", "消极", "无望", "崩溃", "抑郁", "严重焦虑"]

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "情绪低落": ["情绪低落", "悲观", "绝望", "高兴不起来", "心情差", "心情很低", "情绪很低", "难过"],
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

TOPIC_SEQUENCE: list[tuple[str, str, str]] = [
    (
        "最近最困扰的一件具体事",
        "建立来访者当前困扰的具体场景和触发点。",
        "如果用户还没有讲清楚具体事件，先邀请其描述最近一次最困扰的片段。",
    ),
    (
        "情绪低落",
        "了解低落、悲观或绝望情绪的持续时间和波动。",
        "围绕情绪强度、持续时间和一天中的变化追问一个核心问题。",
    ),
    (
        "兴趣减退",
        "了解愉悦感和主动性是否下降。",
        "询问用户最近是否还愿意做以前在意或喜欢的事情。",
    ),
    (
        "睡眠",
        "了解入睡、早醒、多梦和睡眠恢复感。",
        "围绕睡眠质量追问一个具体问题。",
    ),
    (
        "精力疲劳",
        "了解精力、疲劳和日常活动受影响程度。",
        "询问疲劳是否影响学习、工作或基本生活。",
    ),
    (
        "饮食体重",
        "了解食欲和体重变化。",
        "询问最近食欲或体重是否有明显变化。",
    ),
    (
        "注意力",
        "了解注意力、记忆和思维速度。",
        "询问注意力下降是否影响任务完成。",
    ),
    (
        "自责无价值",
        "了解自责、内疚、无价值感和负性自我评价。",
        "温和探索用户是否经常责怪自己或觉得自己没有价值。",
    ),
    (
        "焦虑紧张",
        "了解焦虑、紧张、心慌和躯体化表现。",
        "如果焦虑已出现，追问触发场景和身体反应。",
    ),
    (
        "学习工作",
        "了解功能损害和现实压力源。",
        "围绕学习或工作受影响的一个具体方面追问。",
    ),
    (
        "人际关系",
        "了解家庭、朋友、同学或同事支持。",
        "询问用户身边是否有可以联系或获得支持的人。",
    ),
]


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


def choose_next_topic_focus(risk: RiskAssessment) -> NextTopicFocus:
    if risk.level == "high":
        return NextTopicFocus(
            topic="安全与线下支持",
            objective="优先确认即时安全，推动用户联系现实支持和紧急资源。",
            prompt_instruction="不要继续普通探索；只做安全确认、陪伴和线下求助建议，不提供任何危险行为细节。",
        )

    covered = set(risk.covered_topics)
    for topic, objective, instruction in TOPIC_SEQUENCE:
        if topic not in covered:
            return NextTopicFocus(
                topic=topic,
                objective=objective,
                prompt_instruction=instruction,
            )

    return NextTopicFocus(
        topic="总结与一个小行动",
        objective="在已有信息基础上做简短整理，并给出一个低负担下一步。",
        prompt_instruction="用一两句话总结用户状态，再给一个今天可以完成的小行动。",
    )


def high_risk_reply() -> str:
    return (
        "先把这一刻放慢一点。你不用一个人扛着它。"
        "如果身边有任何可以马上联系的人，现在就给对方发一句很简单的话：我现在不太安全，需要你陪我一下。"
        "接下来先让自己离开最危险的地方，保持电话能打通。"
    )
