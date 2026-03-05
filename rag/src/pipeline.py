from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers.utils import logging

from rag.src.data_processing import Data_process
from rag.src.config.config import prompt_template 
logger = logging.get_logger(__name__)


class EmoLLMRAG(object):
    def __init__(self, model, retrieval_num=3, rerank_flag=False, select_num=3) -> None:
        self.model = model
        self.data_processing_obj = Data_process()
        self.vectorstores = self._load_vector_db()
        self.prompt_template = prompt_template
        self.retrieval_num = retrieval_num
        self.rerank_flag = rerank_flag
        self.select_num = select_num

    def _load_vector_db(self):
        vectorstores = self.data_processing_obj.load_vector_db()

        return vectorstores 

    def get_retrieval_content(self, query) -> str:
        content = []
        documents = self.vectorstores.similarity_search(query, k=self.retrieval_num)
        for doc in documents:
            content.append(doc.page_content)
        if self.rerank_flag:
            documents, _ = self.data_processing_obj.rerank(documents, self.select_num)

            content = []
            for doc in documents:
                content.append(doc)
        logger.info(f'Retrieval data: {content}')
        return content
    
    def generate_answer(self, query, content) -> str:
        prompt = PromptTemplate(
                template=self.prompt_template,
                input_variables=["query", "content"],
                )

        rag_chain = prompt | self.model | StrOutputParser()

        generation = rag_chain.invoke(
                {
                    "query": query,
                    "content": content,
                }
            )
        return generation
    
    def main(self, query) -> str:
        content = self.get_retrieval_content(query)
        response = self.generate_answer(query, content)

        return response
