from __future__ import annotations

import json

from product_app.config import settings
from product_app.ragflow_client import (
    RagFlowRetrievalClient,
    build_ragflow_question,
    empty_rag_context,
)
from product_app.risk import high_risk_reply
from product_app.schemas import (
    ChatModelOutput,
    ConversationTopicState,
    DialogueStopDecision,
    NextTopicFocus,
    RiskAssessment,
)


def _validate_model_output(payload: dict) -> ChatModelOutput:
    if hasattr(ChatModelOutput, "model_validate"):
        return ChatModelOutput.model_validate(payload)
    return ChatModelOutput.parse_obj(payload)


def _as_dict(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _fallback_reply(
    user_message: str,
    risk: RiskAssessment,
    next_topic_focus: NextTopicFocus | None = None,
    topic_state: ConversationTopicState | None = None,
    stop_decision: DialogueStopDecision | None = None,
    error: str = "",
) -> ChatModelOutput:
    if stop_decision is not None and stop_decision.should_stop:
        reply = _fallback_stop_reply(risk, topic_state, stop_decision)
    elif risk.level == "high":
        reply = high_risk_reply()
    elif next_topic_focus is not None and next_topic_focus.topic == "预热总结与话题计划" and topic_state is not None:
        reply = _fallback_warmup_summary_reply(topic_state)
    elif risk.level == "medium":
        reply = (
            "我听到你现在承受了不少压力。你可以先说说最近最影响你的一个具体场景，"
            "我会帮你一起梳理情绪、睡眠、精力和现实支持。若这种状态持续或加重，建议尽快预约专业心理咨询或精神科评估。"
        )
    elif next_topic_focus is not None:
        reply = _fallback_topic_reply(next_topic_focus)
    else:
        reply = (
            "我在。你可以从最近最困扰你的事情说起，不需要一次说完整。"
            "我会先帮你把情绪、触发事件和可尝试的小步骤分开整理。"
        )
    return ChatModelOutput(assistant_reply=reply)


def _fallback_topic_reply(next_topic_focus: NextTopicFocus) -> str:
    topic = next_topic_focus.topic
    topic_replies = {
        "最近最困扰的一件具体事": (
            "我在。我们先不用一次说完整。你可以从最近一次最难受、最卡住的时刻说起："
            "当时发生了什么，你心里最明显的感受是什么？"
        ),
        "持续时间与影响": (
            "听起来这件事已经占了你不少心力。我想先帮你把范围理清一点："
            "这种状态大概持续多久了？现在最影响的是睡眠、学习工作，还是和人的相处？"
        ),
        "本次咨询目标": (
            "我们可以先给这次对话定一个小目标，不需要很大。"
            "在情绪、睡眠、压力、人际或学习工作里，你最想先处理哪一块？"
        ),
        "情绪低落": "我听到你的状态并不轻松。最近这种低落通常在一天中的什么时候更明显？它会持续多久？",
        "兴趣减退": "我想了解一下你的动力和愉悦感。最近以前还愿意做的事情，现在还有一点想做吗？",
        "睡眠": "我们先看看睡眠这块。最近更困扰你的是入睡慢、早醒、多梦，还是醒来后仍然很累？",
        "精力疲劳": "听起来身体和精神都可能在消耗。最近这种疲惫最影响你完成哪一类事情？",
        "饮食体重": "我想顺带确认一下身体状态。最近食欲或体重有没有出现比较明显的变化？",
        "注意力": "我们看看注意力这块。最近你是更难开始任务，还是开始后很难保持专注？",
        "自责无价值": "当压力上来时，人很容易把问题都压到自己身上。你最近会经常责怪自己或觉得自己不够好吗？",
        "焦虑紧张": "我们先把焦虑具体化一点。它通常被什么场景触发？出现时身体会有什么反应，比如心慌、胸闷或坐立不安？",
        "学习工作": "学习或工作上的影响很值得认真看。最近最明显受影响的是效率、沟通、任务量，还是对结果的担心？",
        "人际关系": "支持系统也很重要。你身边有没有一个相对安全、可以简单说几句近况的人？",
        "总结与一个小行动": "我们先把目前说到的内容收一下。你可以选一个今天能完成的小动作，不求解决全部，只求让压力下降一点点。",
    }
    return topic_replies.get(
        topic,
        f"我在。我们先围绕“{topic}”慢慢理清楚。你可以说说这件事最近一次出现时，最影响你的部分是什么。",
    )


def _fallback_stop_reply(
    risk: RiskAssessment,
    topic_state: ConversationTopicState | None,
    stop_decision: DialogueStopDecision,
) -> str:
    if stop_decision.reason == "already_ended":
        return "本轮对话已经结束，我不会再继续追问或开启新话题。你可以先按刚才的小结休息；如果后续要重新开始，需要进入新的会话。"

    planned = topic_state.planned_topics if topic_state else []
    covered = topic_state.covered_topics if topic_state else []
    topics = covered or planned
    topic_text = "、".join(topics[:6]) if topics else "当前困扰、情绪状态和现实影响"
    if risk.level == "medium":
        risk_text = "你提到过较强的压力或消极感受，如果状态持续、加重，建议尽快联系专业心理咨询或精神科评估。"
    else:
        risk_text = "目前没有看到需要立刻升级到危机干预的明确信号，但如果之后出现伤害自己或无法保证安全的想法，要优先联系身边可信任的人和线下紧急支持。"
    reason_text = "你提出想先结束本轮对话。" if stop_decision.reason == "user_requested_end" else "本轮预定关注的话题已经初步覆盖。"
    return (
        f"可以，我们把本轮先收束在这里。{reason_text}\n\n"
        f"本轮小结：我们主要围绕{topic_text}做了梳理。当前更值得留意的是，困扰可能同时影响情绪、身体状态和日常功能，"
        "后续不需要一次解决全部问题，可以先抓住一个最小切口。你已经把一部分模糊的压力说成了更具体的主题，"
        "这本身有助于后面继续判断什么最需要优先处理。\n\n"
        "可能的维持因素包括：压力场景反复出现、休息恢复不足、负面想法在独处或夜间变强，以及可获得的支持还没有被充分调动。"
        "这些不是诊断结论，只是基于本轮对话做出的工作假设。\n\n"
        "建议接下来先做两件低负担的事：第一，把今天最影响你的一个场景写成三句话；第二，选一个可以在24小时内完成的小行动，"
        f"比如休息、联系一个可信任的人，或减少一个压力源。{risk_text}如果之后愿意继续，我们可以从这个小行动的结果接着往下梳理。"
    )


def _fallback_warmup_summary_reply(topic_state: ConversationTopicState) -> str:
    result = topic_state.warmup_result
    info = result.patient_preliminary_info
    judgment = result.symptom_judgment
    topic_text = "、".join(result.topic_list or topic_state.planned_topics) or "情绪状态、睡眠、功能影响和支持系统"
    context_text = "；".join(info.stated_context) if info.stated_context else "已完成预热信息收集"
    concern_text = "、".join(info.main_concerns) if info.main_concerns else topic_text
    impact_text = "、".join(info.functional_impacts) if info.functional_impacts else "功能影响仍需继续观察"
    support_text = "、".join(info.support_context) if info.support_context else "现实支持资源尚未充分明确"
    symptom_text = "、".join(judgment.observed_symptoms) if judgment.observed_symptoms else "症状线索仍需继续澄清"
    pattern_text = "、".join(judgment.possible_patterns) if judgment.possible_patterns else "目前以探索主要压力源为主"
    risk_text = "、".join(judgment.risk_flags) if judgment.risk_flags else "暂未出现明确高危关键词"
    return (
        "预热先到这里，我们把本轮后续咨询的工作框架定下来。\n\n"
        f"拟覆盖话题：{topic_text}。\n\n"
        f"患者初步信息：{context_text}。当前主要困扰可先概括为：{concern_text}；"
        f"已看到的影响包括：{impact_text}；支持资源方面：{support_text}。\n\n"
        f"症状判断：目前观察到的线索包括{symptom_text}。初步工作假设是：{pattern_text}。"
        f"风险边界：{risk_text}。这些只是预热后的辅助性观察，不是诊断结论。"
        "下一轮开始，我会从计划里的第一个未覆盖话题继续推进。"
    )


class DeepSeekChatClient:
    def __init__(self) -> None:
        self.model_name = settings.DEEPSEEK_MODEL
        self.enabled = bool(settings.DEEPSEEK_API_KEY)
        self._client = None
        self._ragflow = RagFlowRetrievalClient()
        if self.enabled:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url=settings.DEEPSEEK_BASE_URL,
                    timeout=settings.DEEPSEEK_TIMEOUT_SECONDS,
                )
            except Exception:
                self.enabled = False

    @property
    def is_available(self) -> bool:
        return self.enabled and self._client is not None

    def generate_json(
        self,
        user_message: str,
        risk: RiskAssessment,
        history: list[dict[str, str]],
        patient_info: dict,
        next_topic_focus: NextTopicFocus,
        topic_state: ConversationTopicState,
        stop_decision: DialogueStopDecision,
    ) -> tuple[ChatModelOutput, bool]:
        if not self.is_available:
            return _fallback_reply(
                user_message,
                risk,
                next_topic_focus,
                topic_state,
                stop_decision,
                "deepseek_api_not_configured",
            ), True
        if risk.level == "high":
            return _fallback_reply(user_message, risk, next_topic_focus, topic_state, stop_decision), True
        if stop_decision.reason == "already_ended":
            return _fallback_reply(user_message, risk, next_topic_focus, topic_state, stop_decision), True

        rag_context = empty_rag_context()
        if not stop_decision.should_stop:
            rag_context = self._ragflow.retrieve(
                build_ragflow_question(user_message, risk, next_topic_focus, topic_state)
            )

        messages = self._build_messages(
            user_message,
            risk,
            history,
            patient_info,
            next_topic_focus,
            topic_state,
            stop_decision,
            rag_context,
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            output = _validate_model_output(payload)
            return output, True
        except Exception as exc:  # noqa: BLE001
            return _fallback_reply(
                user_message,
                risk,
                next_topic_focus,
                topic_state,
                stop_decision,
                f"deepseek_json_error: {exc}",
            ), False

    def _build_messages(
        self,
        user_message: str,
        risk: RiskAssessment,
        history: list[dict[str, str]],
        patient_info: dict,
        next_topic_focus: NextTopicFocus,
        topic_state: ConversationTopicState,
        stop_decision: DialogueStopDecision,
        rag_context: dict | None = None,
    ) -> list[dict[str, str]]:
        system = (
            "你必须扮演一名专业、温和、稳重的心理咨询师，与来访者进行心理支持对话。"
            "你的工作不是自动诊断或替代治疗，而是在边界清晰的前提下进行倾听、澄清、共情、结构化探索和低负担建议。"
            "你必须严格依据用户消息中的 JSON 输入工作，尤其是 patient_info_json、conversation_history_json、"
            "risk_assessment_json、topic_state_json、stop_decision_json 和 next_topic_focus_json。"
            "topic_state_json 表示本次咨询的话题阶段、已观察话题、计划覆盖话题和已主动覆盖话题。"
            "stop_decision_json 表示本轮是否应该结束对话，以及结束原因。"
            "每一轮只推进一个核心关注点，不要跳到多个话题；优先遵守 next_topic_focus_json.prompt_instruction。"
            "如果 topic_state_json.stage 是 warmup，提问要自然像咨询开场，不要像量表；"
            "如果 next_topic_focus_json.topic 是“预热总结与话题计划”，这条规则优先于 planned 阶段规则；说明5轮预热已结束，"
            "必须输出用户可读的结构化内容：拟覆盖话题列表、患者初步信息、症状判断、风险边界；"
            "不要继续预热，不要追问新问题。"
            "如果 stage 是 planned，下一问必须围绕 next_topic_focus_json.topic，不要重复已覆盖话题。"
            "如果 stop_decision_json.reason 是 already_ended，只做简短确认：本轮已经结束，不再追问，不再开启新话题。"
            "如果 stop_decision_json.should_stop 为 true 且 reason 不是 already_ended，不要再追问新问题，不要一句话告别；"
            "必须给出自然、完整、专业的结束回复，包含：确认收束、已覆盖主题小结、关键困扰/触发因素整理、"
            "一到两个低负担下一步、风险与线下求助边界、之后可以继续补充的说明。"
            "对话风格：中文、口语化、自然、简洁、具体；先回应情绪，再问一个问题或给一个小步骤。"
            "安全要求：不要输出自伤、自杀、药物滥用或其他危险行为的方法、步骤、工具、剂量或地点。"
            "如果风险为 high，只做安全支持和线下求助引导。"
            "如果 rag_context_json.enabled 为 true，只能把 rag_context_json.chunks 当作背景参考资料；"
            "不得逐字照搬，不得把资料包装成诊断依据。"
            "若参考资料与 risk_assessment_json、stop_decision_json、next_topic_focus_json 或安全要求冲突，"
            "必须优先遵守本系统风险与安全策略。"
            "不要向用户展示临床量表分数、诊断标签或类似“你是中度抑郁”的结论。"
            "必须只输出合法 JSON，不要使用 Markdown，不要添加 JSON 以外文字。"
            "JSON 只允许包含 assistant_reply 一个 key。"
        )
        compact_history = []
        for item in history[-settings.MAX_HISTORY_MESSAGES :]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                compact_history.append({"role": role, "content": content[:1000]})

        prompt_payload = {
            "patient_info_json": patient_info,
            "conversation_history_json": compact_history,
            "current_user_message": user_message,
            "risk_assessment_json": _as_dict(risk),
            "topic_state_json": _as_dict(topic_state),
            "stop_decision_json": _as_dict(stop_decision),
            "next_topic_focus_json": _as_dict(next_topic_focus),
            "rag_context_json": rag_context or empty_rag_context(),
            "dialogue_policy_json": {
                "role": "professional_psychological_counselor",
                "single_turn_goal": "respond_empathically_and_advance_one_focus_topic",
                "one_core_question_or_one_small_step_only": True,
                "avoid_repeating_history_questions": True,
                "no_diagnosis_or_scale_score_to_user": True,
                "no_dangerous_details": True,
                "when_stop_decision_true_output_complete_closure_report": True,
                "when_already_ended_do_not_resume": True,
                "when_warmup_summary_focus_output_structured_summary": True,
            },
            "response_contract_json": {
                "format": {"assistant_reply": "string"},
                "assistant_reply_length": (
                    "10-80 Chinese characters if stop_decision_json.reason is already_ended; "
                    "350-700 Chinese characters if stop_decision_json.should_stop is true "
                    "or next_topic_focus_json.topic is 预热总结与话题计划; "
                    "otherwise 80-220 Chinese characters"
                ),
                "json_only": True,
            },
        }
        user = "请根据以下完整 JSON 输入生成下一轮心理咨询式回复：\n" + json.dumps(
            prompt_payload,
            ensure_ascii=False,
            indent=2,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
