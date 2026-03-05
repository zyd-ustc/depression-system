import json
import pickle
import os

from loguru import logger
from langchain_community.vectorstores import FAISS
from rag.src.config.config import (
    embedding_path,
    embedding_model_name,
    doc_dir, qa_dir,
    knowledge_pkl_path,
    data_dir,
    vector_db_dir,
    rerank_path,
    rerank_model_name,
    chunk_size,
    chunk_overlap
)
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from FlagEmbedding import FlagReranker

class Data_process():

    def __init__(self):

        self.chunk_size: int=chunk_size
        self.chunk_overlap: int=chunk_overlap
        
    def load_embedding_model(self, model_name=embedding_model_name, device='cpu', normalize_embeddings=True):
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path, exist_ok=True)
        embedding_model_path = os.path.join(embedding_path,model_name.split('/')[1] + '.pkl')
        logger.info('Loading embedding model...')
        if os.path.exists(embedding_model_path):
            try:
                with open(embedding_model_path , 'rb') as f:
                    embeddings = pickle.load(f)
                    logger.info('Embedding model loaded.')
                    return embeddings
            except Exception as e:
                logger.error(f'Failed to load embedding model from {embedding_model_path}')
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': normalize_embeddings})
            logger.info('Embedding model loaded.')    
            with open(embedding_model_path, 'wb') as file:
                pickle.dump(embeddings, file)
        except Exception as e:
            logger.error(f'Failed to load embedding model: {e}')
            return None
        return embeddings
    
    def load_rerank_model(self, model_name=rerank_model_name): 
        if not os.path.exists(rerank_path):
            os.makedirs(rerank_path, exist_ok=True)
        rerank_model_path = os.path.join(rerank_path, model_name.split('/')[1] + '.pkl')   
        logger.info('Loading rerank model...')
        if os.path.exists(rerank_model_path):
            try:
                with open(rerank_model_path , 'rb') as f:
                    reranker_model = pickle.load(f)
                    logger.info('Rerank model loaded.')
                    return reranker_model
            except Exception as e:
                logger.error(f'Failed to load embedding model from {rerank_model_path}') 
        try:
            reranker_model = FlagReranker(model_name, use_fp16=True)
            logger.info('Rerank model loaded.')
            with open(rerank_model_path, 'wb') as file:
                pickle.dump(reranker_model, file)
        except Exception as e:
            logger.error(f'Failed to load rerank model: {e}')
            raise

        return reranker_model
    
    def extract_text_from_json(self, obj, content=None):
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    content = self.extract_text_from_json(value, content)
                except Exception as e:
                    print(f"Error processing value: {e}")
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                try:
                    content = self.extract_text_from_json(item, content)
                except Exception as e:
                    print(f"Error processing item: {e}")
        elif isinstance(obj, str):
            content += obj
        return content

    def split_document(self, data_path):
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) 
        split_docs = []
        logger.info(f'Loading txt files from {data_path}')
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="**/*.txt",show_progress=True)
            docs = loader.load()
            split_docs = text_spliter.split_documents(docs)
        elif data_path.endswith('.txt'): 
            file_path = data_path
            logger.info(f'splitting file {file_path}')
            text_loader = TextLoader(file_path, encoding='utf-8')        
            text = text_loader.load()
            splits = text_spliter.split_documents(text)
            split_docs = splits
        logger.info(f'split_docs size {len(split_docs)}')
        return split_docs
  
    def split_conversation(self, path): 
        logger.info(f'Loading json files from {path}')
        split_qa = []
        if os.path.isdir(path):
            # loader = DirectoryLoader(path, glob="**/*.json",show_progress=True)
            # jsons = loader.load()
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.json'): 
                        file_path = os.path.join(root, file)
                        logger.info(f'splitting file {file_path}')
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f.readlines():
                                content = self.extract_text_from_json(line,'')
                                split_qa.append(Document(page_content = content))

                            #data = json.load(f)
                            #for conversation in data:
                            #    #for dialog in conversation['conversation']:
                            #    #    #按qa对切分,将每一轮qa转换为langchain_core.documents.base.Document
                            #    #    content = self.extract_text_from_json(dialog,'')
                            #    #    split_qa.append(Document(page_content = content))
                            #    #按conversation块切分
                            #    content = self.extract_text_from_json(conversation['conversation'], '')
                            #    #logger.info(f'content====={content}')
                            #    split_qa.append(Document(page_content = content))    
            # logger.info(f'split_qa size====={len(split_qa)}')
        return split_qa

    def create_vector_db(self, emb_model):
        logger.info(f'Creating index...')
        split_doc = self.split_document(doc_dir)
        split_qa = self.split_conversation(qa_dir)
        # logger.info(f'split_doc == {len(split_doc)}')
        # logger.info(f'split_qa == {len(split_qa)}')
        # logger.info(f'split_doc type == {type(split_doc[0])}')
        # logger.info(f'split_qa type== {type(split_qa[0])}')
        db = FAISS.from_documents(split_doc + split_qa, emb_model)
        db.save_local(vector_db_dir)
        return db
        
    def load_vector_db(self, knowledge_pkl_path=knowledge_pkl_path, doc_dir=doc_dir, qa_dir=qa_dir):       
        emb_model = self.load_embedding_model()
        if not os.path.exists(vector_db_dir) or not os.listdir(vector_db_dir):
            db = self.create_vector_db(emb_model)
        else:
            db = FAISS.load_local(vector_db_dir, emb_model)
        return db
    
if __name__ == "__main__":
    logger.info(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    dp = Data_process()
    vector_db = dp.load_vector_db()
