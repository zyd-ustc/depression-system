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

## Chunk Strategy

- Primary boundary: `##` Markdown headings.
- Metadata keeps `source_file` and `section_title`.
- If a section exceeds the configured character threshold, the builder splits it by character window with overlap.
- Default MVP values are hardcoded in `scripts/build_knowledge_index.py`:
  - chunk size: about 1200 characters
  - overlap: about 160 characters
- Every chunk is stored with original content in SQLite and searchable token text in FTS5.

## Review Rules

- Prefer adding a new section over editing multiple unrelated claims in one section.
- Keep sources close to the claim they support.
- If a source changes materially, update both the content and the source note.
- If a section involves suicide or self-harm risk, use only high-level risk language and escalation guidance.
