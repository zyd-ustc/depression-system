from __future__ import annotations

import re

from product_app.schemas import ToneSkillState


SHUORENHUA_VERSION = "1.9.1"
SHUORENHUA_SOURCE = "refs/shuorenhua"

STYLE_RULES = [
    "场景固定为 chat，档位 fixed=minimal：只清模板感和姿态层，不把咨询回复改成段子、公告或教学说明。",
    "先保信息，再谈风格：不得改变用户事实、风险等级、话题推进、停止逻辑、RAG 事实和后端安全边界。",
    "删除开场套话、总结提示腔和元评论：不要用“你提到…我…”、“我听到…”、“我理解…”、“听起来…”、“如果你愿意…”作为固定起手。",
    "删除过度接住和心理判断腔：不要写“稳稳接住你”“你不是敏感”“我就在这里不躲不藏”这类姿态句。",
    "把回复落回具体动作：少复述、少解释原则，每轮只给一个自然追问或一个低负担小动作。",
    "保留专业边界但不揉进主回复：风险提示、免责声明、线下求助边界和 RAG 来源由结构化字段单独展示。",
    "回读两遍：先确认事实和术语没漂，再清理残留 AI 味；如果继续改会损害保真，就停。",
]

LEADING_CLAUSE_PATTERNS = [
    r"^(好的|好|嗯|明白|我明白|我懂了|可以)[，,。！!：:\s]+",
    r"^(我听到|听到|我理解|我能理解|我能感受到|我注意到|听起来|看起来|你提到|你说到|你刚才说|刚才你说)[^。！？!?，,\n]{0,48}[。！？!?，,]\s*",
    r"^(谢谢你愿意说|谢谢你告诉我|谢谢你分享)[^。！？!?\n]{0,36}[。！？!?，,]\s*",
]

INLINE_REPLACEMENTS = [
    (r"如果你愿意[，,]?", ""),
    (r"我们可以先", "先"),
    (r"我想了解一下", ""),
    (r"我想先帮你", "先"),
    (r"我会陪你一起", "我们先"),
    (r"稳稳地?接住你(?:的情绪|的感受|这份脆弱)?", "陪你把眼前这一小段说清楚"),
    (r"我就在这里[，,]?", ""),
    (r"不躲[，,]?不藏[，,]?不绕[，,]?不逃[，,]?", ""),
    (r"你不是敏感[，,]?你只是[^。！？!?]*[。！？!?]?", ""),
    (r"这很正常[，,]?", ""),
    (r"这不是你的错[，,]?", ""),
]

META_SENTENCE_PATTERNS = [
    r"^接下来我会[^。！？!?]*[。！？!?]\s*",
    r"^让我来[^。！？!?]*[。！？!?]\s*",
    r"^简单来说[，,：:]?\s*",
    r"^一句话总结[，,：:]?\s*",
]


def get_tone_skill_state() -> ToneSkillState:
    return ToneSkillState(
        skill_id="shuorenhua",
        version=SHUORENHUA_VERSION,
        status="active",
        profile="chat/minimal/rewrite-safe",
        rules=STYLE_RULES,
    )


def build_tone_skill_prompt(state: ToneSkillState) -> str:
    rules = "\n".join(f"- {rule}" for rule in state.rules)
    return (
        f"语气 skill：{state.skill_id}@{state.version}，状态：{state.status}，来源：{SHUORENHUA_SOURCE}。\n"
        "按“说人话”规则处理 assistant_reply：把文本从像模型在表演写作，拉回像当前场景里具体的人在回应。"
        "这不是单纯变口语，也不是加温柔词；重点是删姿态层、模板层、心理判断腔和总结提示腔。\n"
        f"{rules}"
    )


def apply_tone_skill(reply: str, state: ToneSkillState) -> str:
    if state.skill_id != "shuorenhua" or state.status != "active":
        return reply.strip()

    text = _normalize_text(reply)
    text = _drop_mechanical_opening(text)
    text = _apply_inline_replacements(text)
    text = _drop_meta_sentences(text)
    return _normalize_text(text)


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[，,]\s*([。！？!?])", r"\1", text)
    text = re.sub(r"\s+([，。！？；：])", r"\1", text)
    return text.strip()


def _drop_mechanical_opening(text: str) -> str:
    cleaned = text
    for _ in range(3):
        changed = False
        for pattern in LEADING_CLAUSE_PATTERNS:
            match = re.match(pattern, cleaned)
            if match is None:
                continue
            candidate = cleaned[match.end() :].lstrip()
            if len(candidate) >= 12:
                cleaned = candidate
                changed = True
                break
        if not changed:
            break
    return cleaned


def _apply_inline_replacements(text: str) -> str:
    cleaned = text
    for pattern, replacement in INLINE_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned


def _drop_meta_sentences(text: str) -> str:
    cleaned = text
    for pattern in META_SENTENCE_PATTERNS:
        candidate = re.sub(pattern, "", cleaned).lstrip()
        if candidate and len(candidate) >= 12:
            cleaned = candidate
    return cleaned
