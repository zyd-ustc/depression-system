from __future__ import annotations

import json

from product_app.config import settings
from product_app.risk import high_risk_reply
from product_app.schemas import ChatModelOutput, NextTopicFocus, RiskAssessment


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
    error: str = "",
) -> ChatModelOutput:
    if risk.level == "high":
        reply = high_risk_reply()
    elif risk.level == "medium":
        reply = (
            "我听到你现在承受了不少压力。你可以先说说最近最影响你的一个具体场景，"
            "我会帮你一起梳理情绪、睡眠、精力和现实支持。若这种状态持续或加重，建议尽快预约专业心理咨询或精神科评估。"
        )
    elif next_topic_focus is not None:
        reply = (
            "我在。我们先不用一次说完整。"
            f"接下来可以先围绕“{next_topic_focus.topic}”慢慢说：{next_topic_focus.prompt_instruction}"
        )
    else:
        reply = (
            "我在。你可以从最近最困扰你的事情说起，不需要一次说完整。"
            "我会先帮你把情绪、触发事件和可尝试的小步骤分开整理。"
        )
    return ChatModelOutput(assistant_reply=reply)


class DeepSeekChatClient:
    def __init__(self) -> None:
        self.model_name = settings.DEEPSEEK_MODEL
        self.enabled = bool(settings.DEEPSEEK_API_KEY)
        self._client = None
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

    def generate_json(
        self,
        user_message: str,
        risk: RiskAssessment,
        history: list[dict[str, str]],
        patient_info: dict,
        next_topic_focus: NextTopicFocus,
    ) -> tuple[ChatModelOutput, bool]:
        if not self.enabled or self._client is None:
            return _fallback_reply(
                user_message,
                risk,
                next_topic_focus,
                "deepseek_api_not_configured",
            ), True
        if risk.level == "high":
            return _fallback_reply(user_message, risk, next_topic_focus), True

        messages = self._build_messages(user_message, risk, history, patient_info, next_topic_focus)
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
            return _fallback_reply(user_message, risk, next_topic_focus, f"deepseek_json_error: {exc}"), False

    def _build_messages(
        self,
        user_message: str,
        risk: RiskAssessment,
        history: list[dict[str, str]],
        patient_info: dict,
        next_topic_focus: NextTopicFocus,
    ) -> list[dict[str, str]]:
        system = (
            "你必须扮演一名专业、温和、稳重的心理咨询师，与来访者进行心理支持对话。"
            "你的工作不是自动诊断或替代治疗，而是在边界清晰的前提下进行倾听、澄清、共情、结构化探索和低负担建议。"
            "你必须严格依据用户消息中的 JSON 输入工作，尤其是 patient_info_json、conversation_history_json、"
            "risk_assessment_json 和 next_topic_focus_json。"
            "每一轮只推进一个核心关注点，不要跳到多个话题；优先遵守 next_topic_focus_json.prompt_instruction。"
            "对话风格：中文、口语化、自然、简洁、具体；先回应情绪，再问一个问题或给一个小步骤。"
            "安全要求：不要输出自伤、自杀、药物滥用或其他危险行为的方法、步骤、工具、剂量或地点。"
            "如果风险为 high，只做安全支持和线下求助引导。"
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
            "next_topic_focus_json": _as_dict(next_topic_focus),
            "dialogue_policy_json": {
                "role": "professional_psychological_counselor",
                "single_turn_goal": "respond_empathically_and_advance_one_focus_topic",
                "one_core_question_or_one_small_step_only": True,
                "avoid_repeating_history_questions": True,
                "no_diagnosis_or_scale_score_to_user": True,
                "no_dangerous_details": True,
            },
            "response_contract_json": {
                "format": {"assistant_reply": "string"},
                "assistant_reply_length": "80-220 Chinese characters",
                "json_only": True,
            },
        }
        user = "请根据以下完整 JSON 输入生成下一轮心理咨询式回复：\n" + json.dumps(
            prompt_payload,
            ensure_ascii=False,
            indent=2,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
