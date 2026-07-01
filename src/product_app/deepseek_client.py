from __future__ import annotations

import json

from product_app.config import settings
from product_app.risk import high_risk_reply
from product_app.schemas import ChatModelOutput, RiskAssessment


def _validate_model_output(payload: dict) -> ChatModelOutput:
    if hasattr(ChatModelOutput, "model_validate"):
        return ChatModelOutput.model_validate(payload)
    return ChatModelOutput.parse_obj(payload)


def _risk_json(risk: RiskAssessment) -> str:
    if hasattr(risk, "model_dump_json"):
        return risk.model_dump_json()
    return risk.json(ensure_ascii=False)


def _fallback_reply(user_message: str, risk: RiskAssessment, error: str = "") -> ChatModelOutput:
    if risk.level == "high":
        reply = high_risk_reply()
    elif risk.level == "medium":
        reply = (
            "我听到你现在承受了不少压力。你可以先说说最近最影响你的一个具体场景，"
            "我会帮你一起梳理情绪、睡眠、精力和现实支持。若这种状态持续或加重，建议尽快预约专业心理咨询或精神科评估。"
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
    ) -> tuple[ChatModelOutput, bool]:
        if not self.enabled or self._client is None:
            return _fallback_reply(user_message, risk, "deepseek_api_not_configured"), True
        if risk.level == "high":
            return _fallback_reply(user_message, risk), True

        messages = self._build_messages(user_message, risk, history)
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
            return _fallback_reply(user_message, risk, f"deepseek_json_error: {exc}"), False

    def _build_messages(
        self,
        user_message: str,
        risk: RiskAssessment,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        system = (
            "你是心理对话协助系统，不是医生，不能做诊断或替代治疗。"
            "你的目标是提供简洁、温和、可执行的情绪支持，并在风险升高时建议专业帮助。"
            "不要输出自伤、自杀、药物滥用或其他危险行为的方法、步骤、工具或剂量。"
            "不要向用户展示临床量表分数或诊断标签。"
            "必须只输出合法 JSON，不要使用 Markdown。"
            "JSON 只允许包含 assistant_reply 一个 key。"
        )
        compact_history = []
        for item in history[-settings.MAX_HISTORY_MESSAGES :]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                compact_history.append({"role": role, "content": content[:1000]})
        user = (
            "本地风险判断如下，请以它为准：\n"
            f"{_risk_json(risk)}\n\n"
            "当前用户输入：\n"
            f"{user_message}\n\n"
            "输出要求：合法 JSON，格式为 {\"assistant_reply\":\"...\"}；"
            "assistant_reply 面向用户，80-220字。"
        )
        return [{"role": "system", "content": system}, *compact_history, {"role": "user", "content": user}]
