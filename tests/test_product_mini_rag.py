from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except Exception:  # noqa: BLE001
    np = None

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from product_app.mini_rag import MiniRAG  # noqa: E402
from scripts.build_knowledge_index import build_index  # noqa: E402


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):  # noqa: ANN001, ANN003
        assert np is not None
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


class MiniRAGTest(unittest.TestCase):
    def _build_test_index(self, tmp_dir: Path) -> tuple[Path, list[str]]:
        knowledge_dir = tmp_dir / "knowledge"
        knowledge_dir.mkdir()
        (knowledge_dir / "guide.md").write_text(
            """# Test Guide

## Alpha Support

免责声明：测试内容。
安全边界：测试边界。
alpha grounding text for lexical retrieval.

## Beta Sleep

免责声明：测试内容。
安全边界：测试边界。
sleep hygiene content for semantic retrieval target.
""",
            encoding="utf-8",
        )
        db_path = tmp_dir / "knowledge_index.db"
        build_index(knowledge_dir=knowledge_dir, db_path=db_path, max_chars=1200, overlap_chars=100)

        rag = MiniRAG(db_path=db_path, top_k=10, candidate_limit=10)
        rows = rag._fetch_chunks_by_ids([])  # noqa: SLF001
        self.assertEqual(rows, [])
        conn = rag._connect()  # noqa: SLF001
        ids = [row["id"] for row in conn.execute("SELECT id FROM chunks ORDER BY section_title").fetchall()]
        rag.close()
        return db_path, ids

    def test_fts_retrieval_still_works_without_optional_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path, _ = self._build_test_index(Path(tmp))
            rag = MiniRAG(db_path=db_path, top_k=1, candidate_limit=5)

            result = rag.retrieve("alpha lexical", current_topic="support")

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["total_chunks_returned"], 1)
            self.assertEqual(result["retrieved_chunks"][0]["section"], "Alpha Support")
            self.assertEqual(result["retrieved_chunks"][0]["retrieval_backend"], "fts5")

    @unittest.skipIf(np is None, "numpy is required for vector index smoke test")
    def test_embedding_retrieval_can_add_vector_only_candidate(self) -> None:
        assert np is not None
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            db_path, ids = self._build_test_index(tmp_dir)
            vector_path = tmp_dir / "vectors.npz"
            np.savez_compressed(
                vector_path,
                ids=np.array(ids, dtype=str),
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
                model_name=np.array("fake-model"),
            )

            rag = MiniRAG(
                db_path=db_path,
                top_k=1,
                candidate_limit=1,
                enable_embedding=True,
                vector_index_path=vector_path,
                faiss_index_path=tmp_dir / "missing.faiss",
                embedding_model_name="fake-model",
            )
            rag._embedding_model = _FakeEmbeddingModel()  # noqa: SLF001

            result = rag.retrieve("semantic-only-query", current_topic="none")

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["total_chunks_returned"], 1)
            self.assertEqual(result["retrieved_chunks"][0]["section"], "Alpha Support")
            self.assertIn("numpy", result["retrieved_chunks"][0]["retrieval_backend"])


if __name__ == "__main__":
    unittest.main()
