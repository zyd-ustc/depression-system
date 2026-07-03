from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from product_app.config import ROOT_DIR, settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = ROOT_DIR / "data" / "knowledge_index.db"
DEFAULT_TOP_K = 4
DEFAULT_CANDIDATE_LIMIT = 10
DEFAULT_MAX_CHARS = 2400
MIN_QUERY_CHARS = 2
RAG_NOTE = "知识库内容仅供心理支持对话参考，不能替代诊断、治疗、危机干预或线下专业评估。"

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")
STOP_TOKENS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "有",
    "与",
    "或",
    "及",
    "等",
    "这",
    "那",
    "一个",
    "一些",
    "the",
    "and",
    "or",
    "for",
    "with",
    "this",
    "that",
}

try:
    import jieba  # type: ignore
except Exception:  # noqa: BLE001
    jieba = None


def generate_retrieval_query(
    user_message: str,
    current_topic: str | None = None,
    history_summary: str | None = None,
) -> str:
    parts = []
    if current_topic:
        parts.append(f"当前主题：{current_topic}")
    if history_summary:
        parts.append(f"历史摘要：{history_summary}")
    parts.append(f"用户表达：{user_message}")
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _tokenize(text: str) -> list[str]:
    if jieba is not None:
        raw_tokens = jieba.cut(text, cut_all=False)
    else:
        raw_tokens = TOKEN_RE.findall(text)

    tokens: list[str] = []
    for raw in raw_tokens:
        token = raw.strip().lower()
        if not token or token in STOP_TOKENS:
            continue
        if not TOKEN_RE.search(token):
            continue
        tokens.append(token)
    return tokens


def _escape_fts_token(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def _build_match_query(text: str, max_tokens: int = 24) -> str:
    tokens = []
    seen = set()
    for token in _tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return " OR ".join(_escape_fts_token(token) for token in tokens)


class MiniRAG:
    def __init__(
        self,
        db_path: str | Path | None = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        max_chars: int = DEFAULT_MAX_CHARS,
        enable_embedding: bool = False,
        enable_rerank: bool = False,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.max_chars = max_chars
        self.enable_embedding = enable_embedding
        self.enable_rerank = enable_rerank
        self._conn: sqlite3.Connection | None = None

    @property
    def is_available(self) -> bool:
        return self.db_path.exists()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def retrieve(
        self,
        user_message: str,
        current_topic: str | None = None,
        history_summary: str | None = None,
        chunk_type: str | Sequence[str] | None = None,
        top_k: int | None = None,
        candidate_limit: int | None = None,
        max_chars: int | None = None,
    ) -> dict[str, Any]:
        query = generate_retrieval_query(user_message, current_topic, history_summary)
        query_basis = " ".join(
            part.strip()
            for part in [user_message, current_topic or "", history_summary or ""]
            if part and part.strip()
        )
        limit = top_k or self.top_k
        candidate_limit = candidate_limit or self.candidate_limit
        max_chars = max_chars or self.max_chars

        if len(query_basis) < MIN_QUERY_CHARS:
            return self._empty_result(query, "query_too_short", max_chars)
        if not self.is_available:
            return self._empty_result(query, "index_missing", max_chars)

        match_query = _build_match_query(query)
        if not match_query:
            return self._empty_result(query, "query_too_short", max_chars)

        try:
            rows = self._search(match_query, chunk_type, candidate_limit)
            if self.enable_embedding:
                rows = self._merge_hybrid_candidates(query=query, lexical_rows=rows)
            if self.enable_rerank:
                rows = self._rerank_candidates(query=query, rows=rows)
        except sqlite3.Error as exc:
            logger.warning("MiniRAG retrieval failed: %s", exc)
            result = self._empty_result(query, "error", max_chars)
            result["error"] = str(exc)
            return result

        if not rows:
            return self._empty_result(query, "no_match", max_chars)

        retrieved: list[dict[str, Any]] = []
        total_chars = 0
        for row in rows:
            if len(retrieved) >= limit:
                break
            content = str(row["content"])
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining].rstrip()
            if not content:
                continue
            total_chars += len(content)
            retrieved.append(
                {
                    "id": row["id"],
                    "source": row["source_file"],
                    "section": row["section_title"],
                    "content": content,
                    "type": row["chunk_type"],
                    "rank": len(retrieved) + 1,
                    "score": float(row["score"]),
                    "char_count": len(content),
                }
            )

        if not retrieved:
            return self._empty_result(query, "no_match", max_chars)

        return {
            "query": query,
            "retrieved_chunks": retrieved,
            "total_chunks_returned": len(retrieved),
            "total_chars": total_chars,
            "max_chars_limit": max_chars,
            "status": "success",
            "note": RAG_NOTE,
        }

    def _search(
        self,
        match_query: str,
        chunk_type: str | Sequence[str] | None,
        candidate_limit: int,
    ) -> list[sqlite3.Row]:
        where_extra = ""
        params: list[Any] = [match_query]

        chunk_types = self._normalize_chunk_types(chunk_type)
        if chunk_types:
            placeholders = ", ".join("?" for _ in chunk_types)
            where_extra = f" AND c.chunk_type IN ({placeholders})"
            params.extend(chunk_types)

        params.append(candidate_limit)
        conn = self._connect()
        rows = conn.execute(
            f"""
            SELECT
                c.id,
                c.source_file,
                c.section_title,
                c.content,
                c.char_count,
                c.chunk_type,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?{where_extra}
            ORDER BY score ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return list(rows)

    def _merge_hybrid_candidates(self, query: str, lexical_rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Future hook for FTS5 + embedding hybrid retrieval.

        Planned path:
        - Generate vectors with sentence-transformers and BAAI/bge-small-zh-v1.5.
        - Store vectors in local `.npy` files or a `faiss-cpu` index.
        - Retrieve semantic candidates alongside FTS5 candidates.
        - Fuse lexical and semantic ranks using weighted scores or reciprocal rank fusion.

        MVP keeps this hook disabled by default and returns lexical results unchanged.
        """
        _ = query
        return lexical_rows

    def _rerank_candidates(self, query: str, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Future hook for candidate reranking.

        Planned path:
        - Reorder by chunk_type priority, query intent, normalized length, or a learned reranker.
        - Preserve detailed rank metadata for a retrieval debug endpoint.

        MVP keeps this hook disabled by default and returns input order unchanged.
        """
        _ = query
        return rows

    @staticmethod
    def _normalize_chunk_types(chunk_type: str | Sequence[str] | None) -> list[str]:
        if chunk_type is None:
            return []
        if isinstance(chunk_type, str):
            return [chunk_type]
        return [item for item in chunk_type if item]

    @staticmethod
    def _empty_result(query: str, status: str, max_chars: int) -> dict[str, Any]:
        return {
            "query": query,
            "retrieved_chunks": [],
            "total_chunks_returned": 0,
            "total_chars": 0,
            "max_chars_limit": max_chars,
            "status": status,
            "note": RAG_NOTE,
        }

    @staticmethod
    def format_for_prompt(rag_result: dict[str, Any]) -> str:
        chunks = rag_result.get("retrieved_chunks") or []
        if not chunks:
            return f"【本地知识库检索】状态：{rag_result.get('status', 'unknown')}。{RAG_NOTE}"

        lines = [
            "【本地知识库参考】",
            RAG_NOTE,
            f"检索状态：{rag_result.get('status')}；返回片段：{rag_result.get('total_chunks_returned')}；字符数：{rag_result.get('total_chars')}/{rag_result.get('max_chars_limit')}",
        ]
        for chunk in chunks:
            lines.extend(
                [
                    "",
                    f"[{chunk['rank']}] 来源：{chunk['source']}；章节：{chunk['section']}；类型：{chunk['type']}",
                    str(chunk["content"]).strip(),
                ]
            )
        return "\n".join(lines)


_DEFAULT_MINI_RAG: MiniRAG | None = None


def get_mini_rag() -> MiniRAG:
    global _DEFAULT_MINI_RAG
    if _DEFAULT_MINI_RAG is None:
        _DEFAULT_MINI_RAG = MiniRAG(
            db_path=settings.MINI_RAG_DB_PATH,
            top_k=settings.MINI_RAG_TOP_K,
            candidate_limit=settings.MINI_RAG_CANDIDATE_LIMIT,
            max_chars=settings.MINI_RAG_MAX_CHARS,
            enable_embedding=settings.MINI_RAG_ENABLE_EMBEDDING,
            enable_rerank=settings.MINI_RAG_ENABLE_RERANK,
        )
    return _DEFAULT_MINI_RAG
