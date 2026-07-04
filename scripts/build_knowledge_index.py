#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_DIR = ROOT_DIR / "data" / "knowledge"
DEFAULT_DB_PATH = ROOT_DIR / "data" / "knowledge_index.db"
DEFAULT_VECTOR_INDEX_PATH = ROOT_DIR / "data" / "knowledge_vectors.npz"
DEFAULT_FAISS_INDEX_PATH = ROOT_DIR / "data" / "knowledge_vectors.faiss"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_CHUNK_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 160
DEFAULT_EMBEDDING_BATCH_SIZE = 16

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
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


def tokenize(text: str) -> str:
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
    return " ".join(tokens)


def iter_markdown_files(knowledge_dir: Path) -> Iterable[Path]:
    for path in sorted(knowledge_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        yield path


def parse_sections(markdown: str, fallback_title: str) -> list[tuple[str, str]]:
    matches = list(SECTION_RE.finditer(markdown))
    if not matches:
        body = markdown.strip()
        return [(fallback_title, body)] if body else []

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = normalize_heading(match.group(1))
        body = markdown[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def normalize_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("#")).strip()


def safety_prefix(content: str) -> str:
    lines = []
    for line in content.splitlines()[:10]:
        if "免责声明" in line or "安全边界" in line:
            lines.append(line.strip())
    return "\n\n".join(lines)


def split_content(content: str, max_chars: int, overlap_chars: int) -> list[str]:
    content = content.strip()
    if len(content) <= max_chars:
        return [content]

    prefix = safety_prefix(content)
    chunks: list[str] = []
    start = 0
    step = max(1, max_chars - overlap_chars)
    while start < len(content):
        end = min(len(content), start + max_chars)
        chunk = content[start:end].strip()
        if prefix and prefix not in chunk:
            chunk = f"{prefix}\n\n{chunk}"
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start += step
    return chunks


def infer_chunk_type(source_file: str, section_title: str, content: str) -> str:
    primary_text = f"{source_file} {section_title}".lower()
    if any(word in primary_text for word in ["safety", "安全", "风险", "自杀", "自伤", "危机", "988", "急诊", "热线"]):
        return "safety"
    if any(word in primary_text for word in ["phq", "assessment", "量表", "评分", "筛查", "评估", "分层"]):
        return "assessment"
    if any(word in primary_text for word in ["cbt", "technique", "行为激活", "认知", "睡眠卫生", "问题解决", "技术"]):
        return "technique"
    if any(word in primary_text for word in ["symptom", "症状", "表现", "低落", "兴趣", "睡眠", "食欲", "精力", "注意力"]):
        return "symptom"

    body_without_boilerplate = "\n".join(
        line
        for line in content.splitlines()
        if "免责声明" not in line and "安全边界" not in line
    ).lower()
    if any(word in body_without_boilerplate for word in ["自杀", "自伤", "危机", "988", "急诊", "热线"]):
        return "safety"
    if any(word in body_without_boilerplate for word in ["phq", "量表", "评分", "筛查", "评估", "分层"]):
        return "assessment"
    if any(word in body_without_boilerplate for word in ["cbt", "行为激活", "认知", "睡眠卫生", "问题解决", "技术"]):
        return "technique"
    if any(word in body_without_boilerplate for word in ["症状", "表现", "低落", "兴趣", "睡眠", "食欲", "精力", "注意力"]):
        return "symptom"
    return "general"


def make_chunk_id(source_file: str, section_title: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha1(f"{source_file}\n{section_title}\n{chunk_index}\n{content}".encode("utf-8")).hexdigest()
    return digest[:20]


def collect_chunks(knowledge_dir: Path, max_chars: int, overlap_chars: int) -> list[dict]:
    chunks: list[dict] = []
    for path in iter_markdown_files(knowledge_dir):
        source_file = path.relative_to(knowledge_dir).as_posix()
        markdown = path.read_text(encoding="utf-8")
        source_sha1 = hashlib.sha1(markdown.encode("utf-8")).hexdigest()
        source_modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        sections = parse_sections(markdown, path.stem.replace("_", " "))
        for section_index, (section_title, content) in enumerate(sections):
            content_chunks = split_content(content, max_chars, overlap_chars)
            for chunk_index, chunk_content in enumerate(content_chunks):
                chunk_type = infer_chunk_type(source_file, section_title, chunk_content)
                chunk_id = make_chunk_id(source_file, section_title, chunk_index, chunk_content)
                metadata = {
                    "source_file": source_file,
                    "section_title": section_title,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "chunks_in_section": len(content_chunks),
                    "chunk_type": chunk_type,
                    "content_sha1": hashlib.sha1(chunk_content.encode("utf-8")).hexdigest(),
                    "source_sha1": source_sha1,
                    "source_modified_at": source_modified_at,
                }
                chunks.append(
                    {
                        "id": chunk_id,
                        "source_file": source_file,
                        "section_title": section_title,
                        "content": chunk_content,
                        "char_count": len(chunk_content),
                        "chunk_type": chunk_type,
                        "metadata": metadata,
                    }
                )
    return chunks


def ensure_fts5_available(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_check USING fts5(content)")
        conn.execute("DROP TABLE temp.__fts5_check")
    except sqlite3.Error as exc:
        raise RuntimeError("SQLite FTS5 is not available in this Python runtime") from exc


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=DELETE;
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            section_title TEXT NOT NULL,
            content TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            metadata TEXT NOT NULL
        );

        CREATE INDEX idx_chunks_source_file ON chunks(source_file);
        CREATE INDEX idx_chunks_chunk_type ON chunks(chunk_type);

        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            source_file,
            section_title,
            content_tokens
        );
        """
    )


def write_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> None:
    with conn:
        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO chunks(id, source_file, section_title, content, char_count, chunk_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    chunk["source_file"],
                    chunk["section_title"],
                    chunk["content"],
                    chunk["char_count"],
                    chunk["chunk_type"],
                    json.dumps(chunk["metadata"], ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.execute(
                """
                INSERT INTO chunks_fts(chunk_id, source_file, section_title, content_tokens)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chunk["id"],
                    tokenize(chunk["source_file"]),
                    tokenize(chunk["section_title"]),
                    tokenize(chunk["content"]),
                ),
            )


def write_index_metadata(
    conn: sqlite3.Connection,
    *,
    knowledge_dir: Path,
    chunks: list[dict],
    max_chars: int,
    overlap_chars: int,
) -> dict:
    sources: dict[str, dict] = {}
    for chunk in chunks:
        metadata = chunk["metadata"]
        source_file = metadata["source_file"]
        sources[source_file] = {
            "sha1": metadata["source_sha1"],
            "modified_at": metadata["source_modified_at"],
        }

    manifest = {
        "schema_version": 2,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_dir": str(knowledge_dir),
        "chunk_count": len(chunks),
        "chunk_chars": max_chars,
        "overlap_chars": overlap_chars,
        "sources": dict(sorted(sources.items())),
    }
    with conn:
        conn.execute(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            ("manifest", json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
        )
    return manifest


def embedding_text(chunk: dict) -> str:
    return "\n".join(
        [
            f"来源：{chunk['source_file']}",
            f"章节：{chunk['section_title']}",
            f"类型：{chunk['chunk_type']}",
            str(chunk["content"]),
        ]
    )


def build_vector_index(
    *,
    chunks: list[dict],
    vector_index_path: Path,
    faiss_index_path: Path,
    embedding_model: str,
    vector_store: str,
    batch_size: int,
) -> dict:
    try:
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Embedding index build requires numpy and sentence-transformers. "
            "Install optional RAG dependencies first."
        ) from exc

    model = SentenceTransformer(embedding_model)
    texts = [embedding_text(chunk) for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    vector_index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{vector_index_path.name}.",
        suffix=".tmp",
        dir=vector_index_path.parent,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            np.savez_compressed(
                temp_file,
                ids=np.array([chunk["id"] for chunk in chunks], dtype=str),
                embeddings=embeddings,
                model_name=np.array(embedding_model),
                schema_version=np.array("1"),
                built_at=np.array(datetime.now(timezone.utc).isoformat()),
                content_sha1=np.array([chunk["metadata"]["content_sha1"] for chunk in chunks], dtype=str),
            )
        os.replace(temp_path, vector_index_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    faiss_written = False
    if vector_store in {"auto", "faiss"}:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
            temp_faiss = tempfile.NamedTemporaryFile(
                prefix=f".{faiss_index_path.name}.",
                suffix=".tmp",
                dir=faiss_index_path.parent,
                delete=False,
            )
            temp_faiss_path = Path(temp_faiss.name)
            temp_faiss.close()
            try:
                faiss.write_index(index, str(temp_faiss_path))
                os.replace(temp_faiss_path, faiss_index_path)
                faiss_written = True
            finally:
                if temp_faiss_path.exists():
                    temp_faiss_path.unlink()
        except Exception as exc:  # noqa: BLE001
            if vector_store == "faiss":
                raise RuntimeError("FAISS vector store requested but faiss-cpu is unavailable") from exc
            print(f"FAISS unavailable; kept numpy vector index only: {exc}", file=sys.stderr)

    if not faiss_written and faiss_index_path.exists():
        faiss_index_path.unlink()

    return {
        "vector_index_path": str(vector_index_path),
        "vector_index_size_bytes": vector_index_path.stat().st_size,
        "embedding_model": embedding_model,
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "vector_store": "faiss" if faiss_written else "numpy",
        "faiss_index_path": str(faiss_index_path) if faiss_written else None,
        "faiss_index_size_bytes": faiss_index_path.stat().st_size if faiss_written else 0,
    }


def build_index(
    knowledge_dir: Path,
    db_path: Path,
    max_chars: int,
    overlap_chars: int,
    *,
    enable_embedding: bool = False,
    vector_index_path: Path = DEFAULT_VECTOR_INDEX_PATH,
    faiss_index_path: Path = DEFAULT_FAISS_INDEX_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_store: str = "auto",
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> dict:
    if not knowledge_dir.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {knowledge_dir}")

    chunks = collect_chunks(knowledge_dir, max_chars, overlap_chars)
    if not chunks:
        raise RuntimeError(f"No Markdown chunks found under {knowledge_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent, delete=False)
    temp_file.close()
    temp_path = Path(temp_file.name)
    embedding_stats = None

    try:
        conn = sqlite3.connect(str(temp_path))
        ensure_fts5_available(conn)
        create_schema(conn)
        write_chunks(conn, chunks)
        manifest = write_index_metadata(
            conn,
            knowledge_dir=knowledge_dir,
            chunks=chunks,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        conn.execute("VACUUM")
        conn.close()
        if enable_embedding:
            embedding_stats = build_vector_index(
                chunks=chunks,
                vector_index_path=vector_index_path,
                faiss_index_path=faiss_index_path,
                embedding_model=embedding_model,
                vector_store=vector_store,
                batch_size=embedding_batch_size,
            )
        os.replace(temp_path, db_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    docs = sorted({chunk["source_file"] for chunk in chunks})
    type_counts = Counter(chunk["chunk_type"] for chunk in chunks)
    stats = {
        "db_path": str(db_path),
        "db_size_bytes": db_path.stat().st_size,
        "document_count": len(docs),
        "chunk_count": len(chunks),
        "chunk_type_counts": dict(sorted(type_counts.items())),
        "tokenizer": "jieba" if jieba is not None else "fallback",
        "manifest_built_at": manifest["built_at"],
    }
    if embedding_stats is not None:
        stats["embedding_index"] = embedding_stats
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local SQLite FTS5 knowledge index and optional vector index.")
    parser.add_argument("--knowledge-dir", type=Path, default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--enable-embedding", action="store_true", help="Build a sentence-transformers vector index.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--vector-index-path", type=Path, default=DEFAULT_VECTOR_INDEX_PATH)
    parser.add_argument("--faiss-index-path", type=Path, default=DEFAULT_FAISS_INDEX_PATH)
    parser.add_argument("--embedding-batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument(
        "--vector-store",
        choices=["auto", "faiss", "numpy"],
        default="auto",
        help="Use FAISS when available, require FAISS, or keep only the numpy vector index.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Perform a full rebuild. MVP always rebuilds, this flag documents the intended workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overlap_chars >= args.chunk_chars:
        raise SystemExit("--overlap-chars must be smaller than --chunk-chars")

    stats = build_index(
        knowledge_dir=args.knowledge_dir.resolve(),
        db_path=args.db_path.resolve(),
        max_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        enable_embedding=args.enable_embedding,
        vector_index_path=args.vector_index_path.resolve(),
        faiss_index_path=args.faiss_index_path.resolve(),
        embedding_model=args.embedding_model,
        vector_store=args.vector_store,
        embedding_batch_size=args.embedding_batch_size,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
