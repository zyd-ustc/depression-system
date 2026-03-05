"""RAG manager: document retrieval and context injection for inference."""
import os
import sys
from typing import Optional, Dict, Any
from loguru import logger

rag_src_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "rag", "src"
)
sys.path.insert(0, rag_src_path)

try:
    from data_processing import Data_process
    from config.config import prompt_template, select_num, retrieval_num
except ImportError as e:
    logger.warning(f"RAG module import failed: {e}")
    Data_process = None


class RAGManager:
    def __init__(self, enable_rag: bool = True, enable_rerank: bool = False):
        self.enable_rag = enable_rag
        self.enable_rerank = enable_rerank
        self.vectorstores = None
        self.data_processing_obj = None

        if self.enable_rag and Data_process is not None:
            self._initialize_rag()
        else:
            logger.warning("RAG disabled or module unavailable")

    def _initialize_rag(self):
        try:
            logger.info("Initializing RAG...")
            self.data_processing_obj = Data_process()
            self.vectorstores = self._load_vector_db()
            logger.info("RAG ready")
        except Exception as e:
            logger.error(f"RAG init failed: {e}")
            self.enable_rag = False

    def _load_vector_db(self):
        try:
            vectorstores = self.data_processing_obj.load_vector_db()
            logger.info("Vector DB loaded")
            return vectorstores
        except Exception as e:
            logger.error(f"Vector DB load failed: {e}")
            return None

    def retrieve_context(self, query: str, k: int = None) -> Optional[str]:
        if not self.enable_rag or self.vectorstores is None:
            return None
        try:
            k = k or retrieval_num
            documents = self.vectorstores.similarity_search(query, k=k)
            content = [doc.page_content for doc in documents]
            if self.enable_rerank and self.data_processing_obj:
                try:
                    reranked_docs, _ = self.data_processing_obj.rerank(
                        query, documents, select_num
                    )
                    content = reranked_docs
                except Exception as e:
                    logger.warning(f"Rerank failed, using raw results: {e}")
            return "\n\n".join(content)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return None

    def format_context_for_prompt(self, context: str) -> str:
        if not context:
            return ""
        return f"**参考资料:**\n{context}\n\n**请基于以上参考资料回答问题:**"

    def is_available(self) -> bool:
        return self.enable_rag and self.vectorstores is not None

    def get_rag_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enable_rag,
            "vectorstore_available": self.vectorstores is not None,
            "rerank_enabled": self.enable_rerank,
            "data_processor_available": self.data_processing_obj is not None,
        }
