SCORE_PROMPT_VARIANTS: dict[str, str] = {
    "v1_clinical": (
        "你是一位资深临床心理评估专家。请阅读以下来访者对话，评估其抑郁严重程度。\n"
        "对话：{dialogue}\n"
        "输出要求：仅输出0-100的分数；0表示无抑郁迹象，100表示极重度抑郁。"
    ),
    "v2_structured": (
        "请根据对话评估抑郁严重度，并综合情绪、认知、行为与风险线索。\n"
        "对话：{dialogue}\n"
        "最终仅输出一个0-100数值。"
    ),
    "v3_brief": (
        "Read the dialogue and rate depression severity from 0 to 100.\n"
        "Dialogue: {dialogue}\n"
        "Return one number only."
    ),
}

