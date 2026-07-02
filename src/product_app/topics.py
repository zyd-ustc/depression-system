from __future__ import annotations

import json
from typing import Any

from product_app.risk import TOPIC_SEQUENCE
from product_app.schemas import (
    ConversationTopicState,
    NextTopicFocus,
    PatientPreliminaryInfo,
    RiskAssessment,
    SymptomJudgment,
    WarmupResult,
)

WARMUP_MAX_TURNS = 5

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
        topic="睡眠与身体状态",
        objective="了解睡眠、精力、饮食或身体状态是否已经受到影响。",
        prompt_instruction="自然承接用户刚才的内容，只轻轻确认睡眠、精力或身体状态里最受影响的一块。",
    ),
    NextTopicFocus(
        topic="支持系统与风险边界",
        objective="了解用户身边支持资源，并排查是否存在需要升级的安全风险。",
        prompt_instruction="自然询问用户身边是否有人能知道近况，同时温和确认有没有伤害自己的念头，不要像量表。",
    ),
]

WARMUP_SUMMARY_FOCUS = NextTopicFocus(
    topic="预热总结与话题计划",
    objective="强制结束预热，输出本次咨询的话题列表、患者初步信息和症状判断。",
    prompt_instruction=(
        "预热已经达到5轮，必须结束预热。不要继续收集预热信息，不要追问新问题。"
        "用用户可读的方式输出：本次后续拟覆盖的话题列表、患者初步信息、症状判断和边界说明。"
    ),
)

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
SPECIAL_TOPICS = {SAFETY_FOCUS.topic, SUMMARY_FOCUS.topic, WARMUP_SUMMARY_FOCUS.topic, "结束与咨询报告"}

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
    source_text: str = "",
    patient_id: str = "",
) -> tuple[ConversationTopicState, NextTopicFocus]:
    state = _clone_state(previous_state)
    _mark_previous_topic_covered(state)
    state.observed_topics = _merge_unique(state.observed_topics, risk.covered_topics)

    if risk.level == "high":
        state.current_topic = SAFETY_FOCUS.topic
        return state, SAFETY_FOCUS

    if state.stage == "warmup":
        state.warmup_turns = min(state.warmup_turns + 1, WARMUP_MAX_TURNS)
        if state.warmup_turns >= WARMUP_MAX_TURNS:
            _complete_warmup(state, risk, source_text, patient_id)
            state.current_topic = WARMUP_SUMMARY_FOCUS.topic
            return state, WARMUP_SUMMARY_FOCUS
        return _choose_warmup_focus(state)

    if not state.planned_topics:
        state.planned_topics = _build_topic_plan(state)

    next_topic = _next_uncovered_topic(state)
    state.current_topic = next_topic.topic
    return state, next_topic


def _choose_warmup_focus(state: ConversationTopicState) -> tuple[ConversationTopicState, NextTopicFocus]:
    index = min(max(state.warmup_turns - 1, 0), len(WARMUP_SEQUENCE) - 1)
    focus = WARMUP_SEQUENCE[index]
    state.current_topic = focus.topic
    return state, focus


def _build_topic_plan(state: ConversationTopicState) -> list[str]:
    planned = _merge_unique([], state.observed_topics)
    planned = _merge_unique(planned, CORE_PLAN_TOPICS)
    return [topic for topic in planned if topic in TOPIC_DETAILS][:6]


def _complete_warmup(
    state: ConversationTopicState,
    risk: RiskAssessment,
    source_text: str,
    patient_id: str,
) -> None:
    state.stage = "planned"
    state.warmup_completed = True
    if not state.planned_topics:
        state.planned_topics = _build_topic_plan(state)
    state.warmup_result = _build_warmup_result(state, risk, source_text, patient_id)


def _build_warmup_result(
    state: ConversationTopicState,
    risk: RiskAssessment,
    source_text: str,
    patient_id: str,
) -> WarmupResult:
    topic_list = state.planned_topics or _build_topic_plan(state)
    info = PatientPreliminaryInfo(
        patient_id=patient_id,
        stated_context=_context_lines(state, source_text),
        main_concerns=_main_concerns(topic_list),
        functional_impacts=_functional_impacts(topic_list, source_text),
        support_context=_support_context(source_text),
        unknowns=["年龄", "性别", "既往诊疗史", "用药史", "正式量表结果"],
    )
    judgment = SymptomJudgment(
        risk_level=risk.level,
        risk_score=risk.score,
        observed_symptoms=_observed_symptoms(topic_list),
        possible_patterns=_possible_patterns(topic_list),
        risk_flags=_risk_flags(risk),
        boundary_note="仅为5轮预热后的辅助性初筛观察，不构成诊断；如症状持续、加重或出现安全风险，应转介专业评估。",
    )
    return WarmupResult(
        completed=True,
        completed_at_turn=state.warmup_turns,
        topic_list=topic_list,
        patient_preliminary_info=info,
        symptom_judgment=judgment,
    )


def _context_lines(state: ConversationTopicState, source_text: str) -> list[str]:
    lines = [f"已完成{state.warmup_turns}轮预热对话。"]
    if state.observed_topics:
        lines.append("用户主动呈现的主题包括：" + "、".join(state.observed_topics[:6]) + "。")
    if _contains_any(source_text, ["家人", "妈妈", "母亲", "父母"]):
        lines.append("对话中出现家庭互动或亲子关系线索。")
    return lines


def _main_concerns(topic_list: list[str]) -> list[str]:
    return topic_list[:6] if topic_list else ["当前困扰仍需在计划阶段继续澄清"]


def _functional_impacts(topic_list: list[str], source_text: str) -> list[str]:
    impacts: list[str] = []
    if "睡眠" in topic_list or _contains_any(source_text, ["睡不着", "失眠", "早醒", "整宿"]):
        impacts.append("睡眠质量或恢复感可能受影响")
    if "学习工作" in topic_list or _contains_any(source_text, ["工作", "学习", "上班", "任务"]):
        impacts.append("学习或工作功能可能受影响")
    if "精力疲劳" in topic_list or _contains_any(source_text, ["很累", "没力气", "疲惫"]):
        impacts.append("精力和日常启动能力可能受影响")
    if "注意力" in topic_list or _contains_any(source_text, ["反应", "注意力", "迟钝"]):
        impacts.append("注意力、反应速度或任务维持可能受影响")
    return impacts or ["功能影响仍需继续观察"]


def _support_context(source_text: str) -> list[str]:
    support: list[str] = []
    if _contains_any(source_text, ["家人", "妈妈", "母亲", "父母"]):
        support.append("家庭成员是当前叙事中的重要关系线索")
    if _contains_any(source_text, ["朋友", "同学", "同事", "伴侣"]):
        support.append("对话中出现同伴或社会关系线索")
    return support or ["现实支持资源尚未充分明确"]


def _observed_symptoms(topic_list: list[str]) -> list[str]:
    symptom_topics = [
        topic
        for topic in topic_list
        if topic in {"情绪低落", "兴趣减退", "睡眠", "精力疲劳", "饮食体重", "注意力", "自责无价值", "焦虑紧张"}
    ]
    return symptom_topics or ["症状线索不足，需继续澄清"]


def _possible_patterns(topic_list: list[str]) -> list[str]:
    patterns: list[str] = []
    if {"情绪低落", "兴趣减退"} & set(topic_list):
        patterns.append("低落情绪与动力下降可能是后续重点")
    if {"睡眠", "精力疲劳", "注意力"} & set(topic_list):
        patterns.append("睡眠、精力和认知效率可能相互影响")
    if {"自责无价值", "人际关系"} & set(topic_list):
        patterns.append("负性自我评价可能与关系压力相互强化")
    if "焦虑紧张" in topic_list:
        patterns.append("焦虑或躯体紧张需要进一步区分触发场景")
    return patterns or ["目前以探索主要压力源和功能影响为主"]


def _risk_flags(risk: RiskAssessment) -> list[str]:
    if risk.level == "high":
        return ["出现高风险关键词，需优先确认即时安全并联系线下支持"]
    if risk.level == "medium":
        return ["出现明显痛苦或消极线索，建议持续观察并考虑专业支持"]
    return ["暂未出现明确高危关键词，但仍需持续关注风险变化"]


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


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
