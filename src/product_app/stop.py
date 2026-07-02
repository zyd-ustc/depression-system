from __future__ import annotations

import re

from product_app.schemas import ConversationTopicState, DialogueStopDecision, NextTopicFocus, RiskAssessment

END_FOCUS = NextTopicFocus(
    topic="结束与咨询报告",
    objective="自然结束本轮咨询，并输出完整、专业、可执行的本轮小结。",
    prompt_instruction=(
        "不要继续追问新话题。先确认本轮对话到这里收束，再给出完整咨询小结："
        "已覆盖主题、主要困扰与可能触发因素、风险与求助边界、接下来一到两个低负担行动。"
    ),
)

USER_END_PATTERNS = [
    "结束对话",
    "结束咨询",
    "停止对话",
    "停止咨询",
    "先到这里",
    "今天到这里",
    "到此为止",
    "不用继续",
    "不聊了",
    "先这样",
    "总结一下",
    "给我报告",
    "输出报告",
    "生成报告",
]


def decide_stop(
    user_message: str,
    risk: RiskAssessment,
    topic_state: ConversationTopicState,
) -> DialogueStopDecision:
    if risk.level == "high":
        return DialogueStopDecision(
            should_stop=False,
            reason="continue",
            report_required=False,
            rationale="High-risk content overrides normal ending flow.",
            prompt_instruction="继续执行安全支持，不进入普通结束报告。",
        )

    if _user_requested_end(user_message):
        return DialogueStopDecision(
            should_stop=True,
            reason="user_requested_end",
            report_required=True,
            rationale="User explicitly requested ending, summary, or report.",
            prompt_instruction=(
                "尊重用户结束对话的决定，不再开启新问题。输出完整但不冗长的本轮咨询报告，"
                "包括：本轮聚焦主题、已看到的压力/情绪线索、用户可尝试的小行动、"
                "何时需要联系现实支持或专业帮助。"
            ),
        )

    if _planned_topics_covered(topic_state):
        return DialogueStopDecision(
            should_stop=True,
            reason="planned_topics_covered",
            report_required=True,
            rationale="All planned consultation topics have been actively covered.",
            prompt_instruction=(
                "本轮预热后形成的计划话题已经全部覆盖。自然说明本轮可以先收束，"
                "不要像突然下线。输出专业结束回复：先共情确认，再做主题回顾、"
                "模式整理、低负担行动建议、风险/求助边界，并说明之后可以继续回来补充。"
            ),
        )

    return DialogueStopDecision(
        should_stop=False,
        reason="continue",
        report_required=False,
        rationale="There are still planned topics to cover or the session is in warmup.",
        prompt_instruction="继续推进当前 next_topic_focus，不要主动结束。",
    )


def apply_stop_decision(
    topic_state: ConversationTopicState,
    stop_decision: DialogueStopDecision,
) -> ConversationTopicState:
    if stop_decision.should_stop:
        topic_state.session_status = "ended"
        topic_state.stop_reason = stop_decision.reason
    else:
        topic_state.session_status = "active"
        topic_state.stop_reason = None
    return topic_state


def _user_requested_end(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if "结束生命" in normalized or "结束自己" in normalized:
        return False
    return any(_normalize(pattern) in normalized for pattern in USER_END_PATTERNS)


def _planned_topics_covered(topic_state: ConversationTopicState) -> bool:
    if topic_state.stage != "planned" or not topic_state.planned_topics:
        return False
    covered = set(topic_state.covered_topics)
    return all(topic in covered for topic in topic_state.planned_topics)


def _normalize(text: str) -> str:
    return re.sub(r"[\s\u200b\u200c\u200d\ufeff，。！？、,.!?;；:：\"'`~（）()【】\[\]{}<>《》|-]+", "", text)
