"""Local inference wrapper for a fine-tuned causal LM."""
from typing import Optional

import torch
from core.config import settings
from core import logger_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_pipeline.prompt_templates import InferenceTemplate
from inference_pipeline.rag_manager import RAGManager

logger = logger_utils.get_logger(__name__)


class DepressionTreatmentSystem:
    """Chat-style inference with conversation history and optional RAG."""

    def __init__(
        self,
        model_path: str,
        mock: bool = False,
        use_local_model: bool = True,
        enable_rag: bool = False,
        enable_rerank: bool = False,
    ) -> None:
        self._mock = mock
        self._use_local_model = use_local_model
        self.model_path = model_path
        self.prompt_template_builder = InferenceTemplate()
        self.rag_manager = RAGManager(enable_rag=enable_rag, enable_rerank=enable_rerank)

        if use_local_model and not mock:
            self._load_local_model()
        else:
            logger.warning("Mock or external endpoint mode")

    def _load_local_model(self):
        logger.info(f"Loading model from {self.model_path}")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device in {"cpu", "mps"}:
            self.model = self.model.to(device)
        self.model.eval()
        logger.info(f"Model on {device}")

    def generate(
        self,
        query: str,
        user_id: str,
        session_id: str,
        enable_rag: bool = False,
        conversation_history: Optional[list] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> dict:
        context = None
        if enable_rag and self.rag_manager.is_available():
            context = self.rag_manager.retrieve_context(query)
            if context:
                logger.info(f"RAG context length: {len(context)}")
            else:
                logger.warning("RAG returned no context")

        messages = self._build_conversation_messages(query, conversation_history, context)
        answer = self._call_local_model(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        logger.debug(f"Answer: {answer}")
        return {"answer": answer, "context": context, "session_id": session_id}

    def _build_conversation_messages(
        self,
        query: str,
        conversation_history: Optional[list] = None,
        context: Optional[str] = None,
    ) -> list[dict]:
        system_prompt = (
            self.prompt_template_builder.rag_system_prompt
            if context
            else self.prompt_template_builder.simple_system_prompt
        )
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
        user_content = query
        if context:
            formatted = self.rag_manager.format_context_for_prompt(context)
            user_content = f"{formatted}\n\n**用户问题:** {query}"
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_local_model(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        if self._mock:
            return "Mock reply. Load a real model for actual inference."

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = self._format_messages_manual(messages)

        try:
            while True:
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) <= settings.MAX_INPUT_TOKENS or len(messages) <= 2:
                    break
                messages = [messages[0]] + messages[2:]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = self._format_messages_manual(messages)
        except Exception:
            pass

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated_text.strip()

    def _format_messages_manual(self, messages: list[dict]) -> str:
        formatted = ""
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted

    def get_rag_status(self) -> dict:
        if hasattr(self, "rag_manager"):
            return self.rag_manager.get_rag_info()
        return {"enabled": False, "error": "RAG manager not initialized"}


LLMTwin = DepressionTreatmentSystem
