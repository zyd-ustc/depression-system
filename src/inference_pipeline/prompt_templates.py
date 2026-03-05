"""Prompt strings used by the inference wrapper."""


class InferenceTemplate:
    simple_system_prompt: str = """
    You are an AI language model assistant. Your task is to generate a cohesive and concise response based on the user's instruction by using a similar writing style and voice.
"""
    simple_prompt_template: str = """
### Instruction:
{question}
"""

    rag_system_prompt: str = """你是一位专业的心理咨询师，专门从事抑郁症的诊断和治疗。你能够提供专业、富有同理心的心理支持和指导。

你的任务是根据用户的问题和提供的相关医学文献、治疗指南等上下文信息，提供专业的抑郁症诊疗建议。

请遵循以下原则：
1. 仔细分析用户的问题和症状描述
2. 参考提供的相关医学文献和治疗指南
3. 提供专业、准确、富有同理心的回复
4. 如果上下文信息不足以回答问题，请诚实说明，并建议用户咨询专业医生
5. 始终以患者的安全和福祉为首要考虑

请用中文回复，语气要温和、专业、富有同理心。"""
    rag_prompt_template: str = """
### Instruction:
{question}

### Context:
{context}
"""

    def create_template(self, enable_rag: bool = True) -> tuple[str, str]:
        """
        返回 (system_prompt, prompt_template_str)。
        目前推理链路主要使用 system_prompt；template_str 作为备用/未来扩展。
        """
        if enable_rag is True:
            return self.rag_system_prompt, self.rag_prompt_template

        return self.simple_system_prompt, self.simple_prompt_template
