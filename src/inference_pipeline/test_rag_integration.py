#!/usr/bin/env python3
"""RAG integration test (mock model)."""
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

from core import logger_utils
from inference_pipeline.llm_twin import DepressionTreatmentSystem

logger = logger_utils.get_logger(__name__)

TEST_QUERIES = [
    "我最近总感觉胸口很闷，但医生检查过说身体没问题。可我就是觉得喘不过气来，尤其是看到那些旧照片，想起过去的日子",
    "我现在处于高三阶段，感到非常迷茫和害怕。我觉得自己从出生以来就是多余的，没有必要存在于这个世界。",
    "我最近心情非常差，有什么解决办法吗？",
]


def test_rag_integration():
    configs = [
        {"enable_rag": False, "enable_rerank": False, "name": "No RAG"},
        {"enable_rag": True, "enable_rerank": False, "name": "RAG"},
        {"enable_rag": True, "enable_rerank": True, "name": "RAG + rerank"},
    ]
    for config in configs:
        print(f"\n--- {config['name']} ---")
        try:
            system = DepressionTreatmentSystem(
                model_path="mock_model",
                mock=True,
                use_local_model=False,
                enable_rag=config["enable_rag"],
                enable_rerank=config["enable_rerank"],
            )
            rag_status = system.get_rag_status()
            print("RAG status:", rag_status)
            for i, query in enumerate(TEST_QUERIES, 1):
                try:
                    response = system.generate(
                        query=query,
                        user_id="test_user",
                        session_id="test_session",
                        enable_rag=config["enable_rag"],
                        conversation_history=None,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                    print(f"Q{i} answer: {response['answer'][:80]}...")
                    if response.get("context"):
                        print(f"  context len: {len(response['context'])}")
                except Exception as e:
                    print(f"  Error: {e}")
        except Exception as e:
            print(f"Init failed: {e}")
            logger.exception("Init failed")


def test_rag_manager_only():
    try:
        from inference_pipeline.rag_manager import RAGManager
        rag_manager = RAGManager(enable_rag=True, enable_rerank=False)
        print("RAG info:", rag_manager.get_rag_info())
        print("Available:", rag_manager.is_available())
        if rag_manager.is_available():
            ctx = rag_manager.retrieve_context("我最近心情很差，总是睡不好")
            print("Context length:", len(ctx) if ctx else 0)
    except Exception as e:
        print("RAG manager test failed:", e)
        logger.exception("RAG manager test failed")


if __name__ == "__main__":
    test_rag_manager_only()
    test_rag_integration()
