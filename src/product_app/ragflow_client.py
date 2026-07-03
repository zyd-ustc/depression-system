from __future__ import annotations

import logging
from typing import Any

import requests

from product_app.config import settings
from product_app.schemas import ConversationTopicState, NextTopicFocus, RiskAssessment

logger = logging.getLogger(__name__)


def empty_rag_context(reason: str = "") -> dict[str, Any]:
    return {
        "enabled": False,
        "available": False,
        "chunks": [],
        "error": reason,
    }


def build_ragflow_question(
    user_message: str,
    risk: RiskAssessment,
    next_topic_focus: NextTopicFocus,
    topic_state: ConversationTopicState,
) -> str:
    observed_topics = "、".join(topic_state.observed_topics[-6:]) or "暂无"
    planned_topics = "、".join(topic_state.planned_topics[:6]) or "暂无"
    return "\n".join(
        [
            f"用户当前表达：{user_message[:800]}",
            f"当前咨询主题：{next_topic_focus.topic}",
            f"主题目标：{next_topic_focus.objective}",
            f"风险等级：{risk.level}",
            f"已观察主题：{observed_topics}",
            f"计划覆盖主题：{planned_topics}",
            "检索目标：抑郁相关心理支持、咨询提问、安全边界、低负担行为建议。",
        ]
    )


class RagFlowRetrievalClient:
    @property
    def is_enabled(self) -> bool:
        return bool(
            settings.RAGFLOW_ENABLED
            and settings.RAGFLOW_API_KEY
            and settings.RAGFLOW_DATASET_IDS
        )

    def retrieve(self, question: str) -> dict[str, Any]:
        if not self.is_enabled:
            return empty_rag_context("ragflow_disabled_or_unconfigured")

        payload = {
            "dataset_ids": settings.RAGFLOW_DATASET_IDS,
            "question": question,
            "page": 1,
            "page_size": settings.RAGFLOW_TOP_K,
            "top_k": settings.RAGFLOW_CANDIDATE_TOP_K,
            "similarity_threshold": settings.RAGFLOW_SIMILARITY_THRESHOLD,
            "vector_similarity_weight": settings.RAGFLOW_VECTOR_SIMILARITY_WEIGHT,
            "keyword": True,
        }

        try:
            response = requests.post(
                f"{settings.RAGFLOW_BASE_URL}/api/v1/retrieval",
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.RAGFLOW_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=settings.RAGFLOW_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAGFlow retrieval failed: %s", exc)
            return empty_rag_context(str(exc))

        if body.get("code") != 0:
            return empty_rag_context(str(body.get("message") or "ragflow_error"))

        chunks = (body.get("data") or {}).get("chunks") or []
        normalized_chunks: list[dict[str, Any]] = []
        used_chars = 0

        for chunk in chunks[: settings.RAGFLOW_TOP_K]:
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            remaining = settings.RAGFLOW_MAX_CONTEXT_CHARS - used_chars
            if remaining <= 0:
                break
            content = content[: min(remaining, 900)]
            used_chars += len(content)
            normalized_chunks.append(
                {
                    "content": content,
                    "source": (
                        chunk.get("document_name")
                        or chunk.get("document_keyword")
                        or chunk.get("document_id")
                        or "ragflow"
                    ),
                    "similarity": chunk.get("similarity"),
                    "dataset_id": chunk.get("dataset_id"),
                    "document_id": chunk.get("document_id"),
                }
            )

        return {
            "enabled": True,
            "available": bool(normalized_chunks),
            "chunks": normalized_chunks,
            "error": "",
        }
