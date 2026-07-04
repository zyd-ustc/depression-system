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
DEFAULT_EMBEDDING_CANDIDATE_LIMIT = 30
DEFAULT_VECTOR_INDEX_PATH = ROOT_DIR / "data" / "knowledge_vectors.npz"
DEFAULT_FAISS_INDEX_PATH = ROOT_DIR / "data" / "knowledge_vectors.faiss"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
DEFAULT_EMBEDDING_WEIGHT = 0.65
DEFAULT_RERANK_WEIGHT = 0.75
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


def _lazy_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("MiniRAG embedding search requires numpy") from exc
    return np


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rank_score(rank: int) -> float:
    return 1.0 / max(rank, 1)


def _cosine_to_relevance(score: float) -> float:
    return _clamp01((score + 1.0) / 2.0)


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [_clamp01((score - low) / (high - low)) for score in scores]


def _candidate_from_row(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())
    return {
        "id": row["id"],
        "source_file": row["source_file"],
        "section_title": row["section_title"],
        "content": row["content"],
        "char_count": row["char_count"],
        "chunk_type": row["chunk_type"],
        "metadata": row["metadata"] if "metadata" in keys else "",
        "score": float(row["score"]) if "score" in keys else 0.0,
        "lexical_score": 0.0,
        "semantic_score": 0.0,
        "rerank_score": 0.0,
    }


class _VectorIndex:
    def __init__(self, index_path: Path, faiss_index_path: Path | None = None) -> None:
        self.index_path = index_path
        self.faiss_index_path = faiss_index_path
        self.ids: list[str] = []
        self.model_name = ""
        self.backend = "numpy"
        self._embeddings: Any = None
        self._faiss_index: Any = None
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Vector index missing: {self.index_path}")

        np = _lazy_numpy()
        data = np.load(str(self.index_path), allow_pickle=False)
        self.ids = [str(item) for item in data["ids"].tolist()]
        self._embeddings = data["embeddings"].astype("float32")
        if "model_name" in data:
            self.model_name = str(data["model_name"].item())
        if self._embeddings.ndim != 2:
            raise ValueError("Vector index embeddings must be a 2D matrix")
        if self._embeddings.shape[0] != len(self.ids):
            raise ValueError("Vector index ids and embeddings length mismatch")

        if self.faiss_index_path and self.faiss_index_path.exists():
            try:
                import faiss  # type: ignore

                faiss_index = faiss.read_index(str(self.faiss_index_path))
                if faiss_index.ntotal == len(self.ids):
                    self._faiss_index = faiss_index
                    self.backend = "faiss"
                else:
                    logger.warning(
                        "Ignoring FAISS index with ntotal=%s for %s vector ids",
                        faiss_index.ntotal,
                        len(self.ids),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("FAISS index load failed, falling back to numpy search: %s", exc)

    def search(self, query_embedding: Any, top_k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []

        np = _lazy_numpy()
        limit = min(max(top_k, 1), len(self.ids))
        vector = np.asarray(query_embedding, dtype="float32")
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(vector.reshape(1, -1), limit)
            results: list[tuple[str, float]] = []
            for score, index in zip(distances[0].tolist(), indices[0].tolist(), strict=False):
                if index < 0:
                    continue
                results.append((self.ids[index], float(score)))
            return results

        scores = self._embeddings @ vector
        order = np.argsort(scores)[::-1][:limit]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in order]


class MiniRAG:
    def __init__(
        self,
        db_path: str | Path | None = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        max_chars: int = DEFAULT_MAX_CHARS,
        enable_embedding: bool = False,
        enable_rerank: bool = False,
        vector_index_path: str | Path | None = None,
        faiss_index_path: str | Path | None = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        rerank_model_name: str = DEFAULT_RERANK_MODEL,
        embedding_candidate_limit: int = DEFAULT_EMBEDDING_CANDIDATE_LIMIT,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        rerank_weight: float = DEFAULT_RERANK_WEIGHT,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.max_chars = max_chars
        self.enable_embedding = enable_embedding
        self.enable_rerank = enable_rerank
        self.vector_index_path = Path(vector_index_path) if vector_index_path is not None else DEFAULT_VECTOR_INDEX_PATH
        self.faiss_index_path = Path(faiss_index_path) if faiss_index_path is not None else DEFAULT_FAISS_INDEX_PATH
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        self.embedding_candidate_limit = embedding_candidate_limit
        self.embedding_weight = _clamp01(embedding_weight)
        self.rerank_weight = _clamp01(rerank_weight)
        self._conn: sqlite3.Connection | None = None
        self._embedding_model: Any = None
        self._vector_index: _VectorIndex | None = None
        self._reranker_model: Any = None

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
        if not match_query and not self.enable_embedding:
            return self._empty_result(query, "query_too_short", max_chars)

        try:
            rows = self._search(match_query, chunk_type, candidate_limit) if match_query else []
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
                    "score": float(row.get("score", 0.0)),
                    "lexical_score": row.get("lexical_score"),
                    "semantic_score": row.get("semantic_score"),
                    "rerank_score": row.get("rerank_score"),
                    "retrieval_backend": row.get("retrieval_backend", "fts5"),
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
    ) -> list[dict[str, Any]]:
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
                c.metadata,
                bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?{where_extra}
            ORDER BY score ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        candidates = []
        for rank, row in enumerate(rows, start=1):
            candidate = _candidate_from_row(row)
            candidate["lexical_rank"] = rank
            candidate["lexical_score"] = _rank_score(rank)
            candidate["score"] = candidate["lexical_score"]
            candidate["retrieval_backend"] = "fts5"
            candidates.append(candidate)
        return candidates

    def _merge_hybrid_candidates(self, query: str, lexical_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge FTS5 candidates with semantic vector candidates."""
        try:
            vector_index = self._get_vector_index()
            query_embedding = self._embed_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiniRAG embedding search unavailable, using FTS5 only: %s", exc)
            return lexical_rows

        vector_limit = max(self.embedding_candidate_limit, len(lexical_rows), self.candidate_limit)
        vector_hits = vector_index.search(query_embedding, vector_limit)
        if not vector_hits:
            return lexical_rows

        candidates: dict[str, dict[str, Any]] = {}
        for row in lexical_rows:
            candidates[str(row["id"])] = dict(row)

        missing_ids = [chunk_id for chunk_id, _ in vector_hits if chunk_id not in candidates]
        for row in self._fetch_chunks_by_ids(missing_ids):
            candidates[str(row["id"])] = row

        for vector_rank, (chunk_id, cosine_score) in enumerate(vector_hits, start=1):
            candidate = candidates.get(chunk_id)
            if candidate is None:
                continue
            candidate["vector_rank"] = vector_rank
            candidate["vector_score"] = cosine_score
            candidate["semantic_score"] = _cosine_to_relevance(cosine_score)
            if candidate.get("retrieval_backend") == "fts5":
                candidate["retrieval_backend"] = vector_index.backend + "+fts5"
            else:
                candidate["retrieval_backend"] = vector_index.backend

        lexical_weight = 1.0 - self.embedding_weight
        for candidate in candidates.values():
            candidate["score"] = (
                lexical_weight * float(candidate.get("lexical_score") or 0.0)
                + self.embedding_weight * float(candidate.get("semantic_score") or 0.0)
            )

        return sorted(
            candidates.values(),
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("semantic_score") or 0.0),
                -int(item.get("vector_rank") or 10_000),
                -int(item.get("lexical_rank") or 10_000),
            ),
            reverse=True,
        )

    def _rerank_candidates(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rerank candidates with a CrossEncoder when available, with heuristic fallback."""
        if len(rows) <= 1:
            return rows

        rerank_scores: list[float] | None = None
        try:
            rerank_scores = self._cross_encoder_scores(query, rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiniRAG learned reranker unavailable, using heuristic rerank: %s", exc)

        if rerank_scores is None:
            rerank_scores = [self._heuristic_rerank_score(query, row) for row in rows]
        else:
            rerank_scores = _normalize_scores(rerank_scores)

        base_weight = 1.0 - self.rerank_weight
        reranked = []
        for row, rerank_score in zip(rows, rerank_scores, strict=False):
            candidate = dict(row)
            candidate["rerank_score"] = _clamp01(float(rerank_score))
            candidate["score"] = (
                base_weight * float(candidate.get("score") or 0.0)
                + self.rerank_weight * candidate["rerank_score"]
            )
            reranked.append(candidate)

        return sorted(
            reranked,
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("rerank_score") or 0.0),
                float(item.get("semantic_score") or 0.0),
                float(item.get("lexical_score") or 0.0),
            ),
            reverse=True,
        )

    def _get_vector_index(self) -> _VectorIndex:
        if self._vector_index is None:
            self._vector_index = _VectorIndex(self.vector_index_path, self.faiss_index_path)
            if self._vector_index.model_name and self._vector_index.model_name != self.embedding_model_name:
                raise RuntimeError(
                    "Vector index model mismatch: "
                    f"index={self._vector_index.model_name}, runtime={self.embedding_model_name}"
                )
        return self._vector_index

    def _embed_query(self, query: str) -> Any:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Install sentence-transformers to enable MiniRAG embeddings") from exc
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

        embeddings = self._embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        np = _lazy_numpy()
        return np.asarray(embeddings[0], dtype="float32")

    def _fetch_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        placeholders = ", ".join("?" for _ in chunk_ids)
        conn = self._connect()
        rows = conn.execute(
            f"""
            SELECT
                id,
                source_file,
                section_title,
                content,
                char_count,
                chunk_type,
                metadata
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            list(chunk_ids),
        ).fetchall()
        by_id = {str(row["id"]): _candidate_from_row(row) for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def _cross_encoder_scores(self, query: str, rows: list[dict[str, Any]]) -> list[float]:
        if self._reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("Install sentence-transformers to enable MiniRAG learned rerank") from exc
            self._reranker_model = CrossEncoder(self.rerank_model_name)

        pairs = [(query, str(row["content"])) for row in rows]
        scores = self._reranker_model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]

    def _heuristic_rerank_score(self, query: str, row: dict[str, Any]) -> float:
        query_tokens = set(_tokenize(query))
        content_tokens = set(_tokenize(str(row["content"])))
        if query_tokens and content_tokens:
            overlap = len(query_tokens & content_tokens) / max(len(query_tokens), 1)
        else:
            overlap = 0.0

        chunk_type_score = self._chunk_type_intent_score(query, str(row.get("chunk_type") or "general"))
        char_count = int(row.get("char_count") or len(str(row.get("content") or "")))
        if 220 <= char_count <= 1200:
            length_score = 1.0
        elif char_count < 220:
            length_score = _clamp01(char_count / 220)
        else:
            length_score = _clamp01(1200 / char_count)
        authority_score = self._source_authority_score(str(row.get("source_file") or ""))

        return _clamp01(
            0.45 * overlap
            + 0.25 * chunk_type_score
            + 0.15 * float(row.get("score") or 0.0)
            + 0.10 * length_score
            + 0.05 * authority_score
        )

    @staticmethod
    def _chunk_type_intent_score(query: str, chunk_type: str) -> float:
        text = query.lower()
        if any(word in text for word in ["自杀", "自伤", "危机", "热线", "急诊", "不想活"]):
            return 1.0 if chunk_type == "safety" else 0.35
        if any(word in text for word in ["phq", "量表", "评分", "筛查", "评估"]):
            return 1.0 if chunk_type == "assessment" else 0.45
        if any(word in text for word in ["睡眠", "行为激活", "认知", "记录", "问题解决", "练习"]):
            return 1.0 if chunk_type == "technique" else 0.55
        if any(word in text for word in ["症状", "低落", "兴趣", "食欲", "精力", "注意力"]):
            return 1.0 if chunk_type == "symptom" else 0.55
        return {"safety": 0.75, "assessment": 0.85, "technique": 0.85, "symptom": 0.85}.get(chunk_type, 0.70)

    @staticmethod
    def _source_authority_score(source_file: str) -> float:
        lowered = source_file.lower()
        if any(part in lowered for part in ["safety", "triage", "phq", "assessment", "guideline"]):
            return 1.0
        if any(part in lowered for part in ["cbt", "symptom"]):
            return 0.85
        return 0.70

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
            vector_index_path=settings.MINI_RAG_VECTOR_INDEX_PATH,
            faiss_index_path=settings.MINI_RAG_FAISS_INDEX_PATH,
            embedding_model_name=settings.MINI_RAG_EMBEDDING_MODEL,
            rerank_model_name=settings.MINI_RAG_RERANK_MODEL,
            embedding_candidate_limit=settings.MINI_RAG_EMBEDDING_CANDIDATE_LIMIT,
            embedding_weight=settings.MINI_RAG_EMBEDDING_WEIGHT,
            rerank_weight=settings.MINI_RAG_RERANK_WEIGHT,
        )
    return _DEFAULT_MINI_RAG
