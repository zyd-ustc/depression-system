from __future__ import annotations

from product_app.schemas import (
    ConversationTopicState,
    DialogueStopDecision,
    NextTopicFocus,
    RiskAssessment,
    SafetyNotice,
)


def build_safety_notice(
    risk: RiskAssessment,
    next_topic_focus: NextTopicFocus,
    topic_state: ConversationTopicState,
    stop_decision: DialogueStopDecision,
) -> SafetyNotice | None:
    if risk.level == "high":
        return SafetyNotice(
            visible=True,
            level="urgent",
            title="安全优先",
            message="如果你此刻可能伤害自己，或已经无法保证安全，请先暂停对话，立刻联系身边可信任的人、当地急救电话或线下危机支持。",
            actions=["远离可能用于伤害自己的物品或地点", "联系一个能马上回应你的人", "必要时拨打当地急救电话"],
        )

    if risk.level == "medium":
        return SafetyNotice(
            visible=True,
            level="caution",
            title="需要留意",
            message="当前对话只能做支持性梳理，不构成诊断。如果这种状态持续、加重，或明显影响睡眠、学习工作和生活，建议尽快联系线下心理咨询或精神科评估。",
            actions=["记录睡眠、情绪和功能变化", "考虑预约线下专业评估"],
        )

    if stop_decision.report_required or topic_state.warmup_completed or next_topic_focus.topic == "预热总结与话题计划":
        return SafetyNotice(
            visible=True,
            level="info",
            title="边界说明",
            message="以下内容是基于本轮对话的辅助性观察，不是诊断结论，也不能替代线下专业评估。",
            actions=[],
        )

    return None
