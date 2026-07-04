# Knowledge Source Guidelines

This directory is the single auditable source of truth for the mini RAG index used by the depression support product.

## Scope

- MVP supports Markdown files only: `*.md`.
- `README.md` is documentation and is skipped by the index builder.
- Do not place PDFs, images, OCR outputs, or raw clinical records here.
- Future formats such as `.txt` and structured `.json` should be added only after the index builder has explicit support for them.

## Required Markdown Structure

Each knowledge file should use:

- One `#` title for the document.
- Multiple `##` sections. The index builder treats each `##` section as the primary natural chunk boundary.
- Clear, conservative, reviewable prose.
- Source links or source names inside the relevant section.

Each `##` section must include:

- `免责声明`: the content is informational, does not diagnose, and does not replace professional care.
- `安全边界`: what the assistant must not do, when to escalate, and how to handle urgent risk.

## Content Standard

Materials should be:

- Evidence-informed and conservative.
- Focused on symptoms, screening use, risk boundary, low-intensity support, and referral guidance.
- Written without detailed self-harm methods, medication dosing instructions, or instructions for dangerous behavior.
- Suitable for prompt grounding, not for direct display as medical advice.

Acceptable topics include:

- Depression-related symptom descriptions.
- PHQ-9 use and limitations.
- Safety risk signals and escalation boundaries.
- CBT-informed low-intensity techniques such as behavioral activation, cognitive reframing, sleep hygiene, and problem solving.

## Update Workflow

1. Edit or add Markdown files in this directory.
2. Review every changed section manually for accuracy and safety.
3. Rebuild the local index:

```bash
python scripts/build_knowledge_index.py --rebuild
```

To enable semantic retrieval, install the optional RAG dependencies and build the vector index:

```bash
pip install -r src/product_app/requirements-rag.txt
python scripts/build_knowledge_index.py --rebuild --enable-embedding --vector-store auto
```

This writes:

- `data/knowledge_index.db`: SQLite + FTS5 lexical index.
- `data/knowledge_vectors.npz`: local normalized embedding matrix and chunk ids.
- `data/knowledge_vectors.faiss`: optional FAISS inner-product index when `faiss-cpu` is available.

Runtime switches:

```bash
export MINI_RAG_ENABLE_EMBEDDING=1
export MINI_RAG_ENABLE_RERANK=1
export MINI_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
export MINI_RAG_RERANK_MODEL=BAAI/bge-reranker-base
```

4. Run a targeted retrieval smoke test:

```bash
PYTHONPATH=src python - <<'PY'
from product_app.mini_rag import MiniRAG

rag = MiniRAG()
result = rag.retrieve("最近睡不好，也没有精神", current_topic="睡眠")
print(result["status"], result["total_chunks_returned"], result["total_chars"])
print(rag.format_for_prompt(result))
PY
```

Embedding smoke test:

```bash
MINI_RAG_ENABLE_EMBEDDING=1 PYTHONPATH=src python - <<'PY'
from product_app.mini_rag import MiniRAG

rag = MiniRAG(enable_embedding=True)
result = rag.retrieve("我最近入睡困难，白天也没有精神", current_topic="睡眠")
print(result["status"], result["total_chunks_returned"])
for chunk in result["retrieved_chunks"]:
    print(chunk["rank"], chunk["retrieval_backend"], chunk["source"], chunk["section"])
PY
```

## Chunk Strategy

- Primary boundary: `##` Markdown headings.
- Metadata keeps `source_file` and `section_title`.
- If a section exceeds the configured character threshold, the builder splits it by character window with overlap.
- Default MVP values are hardcoded in `scripts/build_knowledge_index.py`:
  - chunk size: about 1200 characters
  - overlap: about 160 characters
- Every chunk is stored with original content in SQLite and searchable token text in FTS5.
- When embedding is enabled, every chunk is encoded as `source + section + type + content`, normalized, and stored in a local vector index. Runtime retrieval uses FTS5 candidates plus vector candidates, then fuses lexical and semantic scores.
- When rerank is enabled, runtime attempts `sentence-transformers` `CrossEncoder` reranking. If the model is unavailable, it falls back to a deterministic multi-factor rerank using token overlap, chunk type intent, source authority, and chunk length.

## Review Rules

- Prefer adding a new section over editing multiple unrelated claims in one section.
- Keep sources close to the claim they support.
- If a source changes materially, update both the content and the source note.
- If a section involves suicide or self-harm risk, use only high-level risk language and escalation guidance.
- After each rebuild, inspect `index_metadata.manifest` in `data/knowledge_index.db` for source hashes and build time before enabling the new index in a user-facing environment.
