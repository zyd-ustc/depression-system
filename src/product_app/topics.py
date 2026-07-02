from __future__ import annotations

import json
from typing import Any

from product_app.risk import TOPIC_SEQUENCE
from product_app.schemas import ConversationTopicState, NextTopicFocus, RiskAssessment

WARMUP_MIN_TURNS = 2
WARMUP_MAX_TURNS = 3

WARMUP_SEQUENCE: list[NextTopicFocus] = [
    NextTopicFocus(
        topic="最近最困扰的一件具体事",
        objective="建立来访者当前困扰的具体场景和触发点。",
        prompt_instruction="先接住情绪，再自然邀请用户讲最近一次最困扰的片段，不要像问卷。",
    ),
    NextTopicFocus(
        topic="持续时间与影响",
        objective="了解困扰持续多久，以及已经影响到生活、学习或工作的哪一部分。",
        prompt_instruction="自然追问这种状态大概持续多久，以及现在最影响生活、学习或工作的哪一块。",
    ),
    NextTopicFocus(
        topic="本次咨询目标",
        objective="根据已听到的信息，与用户一起确定本次咨询优先覆盖的话题。",
        prompt_instruction="用选择式但不生硬的方式，帮用户在情绪、睡眠、压力、人际或学习工作中选一个最想先处理的目标。",
    ),
]

SAFETY_FOCUS = NextTopicFocus(
    topic="安全与线下支持",
    objective="优先确认即时安全，推动用户联系现实支持和紧急资源。",
    prompt_instruction="不要继续普通探索；只做安全确认、陪伴和线下求助建议，不提供任何危险行为细节。",
)

SUMMARY_FOCUS = NextTopicFocus(
    topic="总结与一个小行动",
    objective="在已有信息基础上做简短整理，并给出一个低负担下一步。",
    prompt_instruction="用一两句话总结用户状态，再给一个今天可以完成的小行动。",
)

CORE_PLAN_TOPICS = [
    "情绪低落",
    "兴趣减退",
    "睡眠",
    "焦虑紧张",
    "学习工作",
    "人际关系",
    "自伤风险",
]
SPECIAL_TOPICS = {SAFETY_FOCUS.topic, SUMMARY_FOCUS.topic, "结束与咨询报告"}

TOPIC_DETAILS = {
    topic: NextTopicFocus(topic=topic, objective=objective, prompt_instruction=instruction)
    for topic, objective, instruction in TOPIC_SEQUENCE
}


def default_topic_state() -> ConversationTopicState:
    return ConversationTopicState()


def parse_topic_state(raw: str | None) -> ConversationTopicState:
    if not raw:
        return default_topic_state()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return default_topic_state()
    if not isinstance(payload, dict):
        return default_topic_state()
    try:
        if hasattr(ConversationTopicState, "model_validate"):
            return ConversationTopicState.model_validate(payload)
        return ConversationTopicState.parse_obj(payload)
    except Exception:
        return default_topic_state()


def dump_topic_state(state: ConversationTopicState) -> str:
    return json.dumps(_as_dict(state), ensure_ascii=False)


def advance_topic_state(
    previous_state: ConversationTopicState,
    risk: RiskAssessment,
) -> tuple[ConversationTopicState, NextTopicFocus]:
    state = _clone_state(previous_state)
    _mark_previous_topic_covered(state)
    state.observed_topics = _merge_unique(state.observed_topics, risk.covered_topics)

    if risk.level == "high":
        state.current_topic = SAFETY_FOCUS.topic
        return state, SAFETY_FOCUS

    if state.stage == "warmup":
        if state.warmup_turns < WARMUP_MIN_TURNS:
            return _choose_warmup_focus(state)
        if state.warmup_turns < WARMUP_MAX_TURNS and not _ready_to_plan(state):
            return _choose_warmup_focus(state)
        state.stage = "planned"
        if not state.planned_topics:
            state.planned_topics = _build_topic_plan(state)

    if not state.planned_topics:
        state.planned_topics = _build_topic_plan(state)

    next_topic = _next_uncovered_topic(state)
    state.current_topic = next_topic.topic
    return state, next_topic


def _choose_warmup_focus(state: ConversationTopicState) -> tuple[ConversationTopicState, NextTopicFocus]:
    index = min(state.warmup_turns, len(WARMUP_SEQUENCE) - 1)
    focus = WARMUP_SEQUENCE[index]
    state.warmup_turns = min(state.warmup_turns + 1, WARMUP_MAX_TURNS)
    state.current_topic = focus.topic
    return state, focus


def _ready_to_plan(state: ConversationTopicState) -> bool:
    return len(state.observed_topics) >= 1


def _build_topic_plan(state: ConversationTopicState) -> list[str]:
    planned = _merge_unique([], state.observed_topics)
    planned = _merge_unique(planned, CORE_PLAN_TOPICS)
    return [topic for topic in planned if topic in TOPIC_DETAILS][:6]


def _next_uncovered_topic(state: ConversationTopicState) -> NextTopicFocus:
    covered = set(state.covered_topics)
    for topic in state.planned_topics:
        if topic not in covered:
            return TOPIC_DETAILS.get(
                topic,
                NextTopicFocus(
                    topic=topic,
                    objective="围绕当前用户最重要的困扰做进一步澄清。",
                    prompt_instruction="先回应用户刚才的内容，再自然追问一个具体问题。",
                ),
            )
    return SUMMARY_FOCUS


def _mark_previous_topic_covered(state: ConversationTopicState) -> None:
    topic = state.current_topic
    if not topic or topic in SPECIAL_TOPICS:
        return
    state.covered_topics = _merge_unique(state.covered_topics, [topic])


def _clone_state(state: ConversationTopicState) -> ConversationTopicState:
    payload = _as_dict(state)
    if hasattr(ConversationTopicState, "model_validate"):
        return ConversationTopicState.model_validate(payload)
    return ConversationTopicState.parse_obj(payload)


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*existing, *incoming]:
        if item and item not in merged:
            merged.append(item)
    return merged


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)
