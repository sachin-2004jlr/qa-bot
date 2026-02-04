import sys
# --- CLOUD DATABASE FIX ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ---------------------------

import os
import shutil
import re
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings,
    get_response_synthesizer
)
# Switched to Standard Splitter (FAST)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb

load_dotenv()

class AdvancedRAG:
    def __init__(self):
        # 1. SETUP MODELS
        
        # Optimization 1: Use MiniLM (Fastest & Most Reliable for CPU)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            trust_remote_code=True
        )
        
        # LLM: Qwen 3 (32B)
        self.llm = Groq(
            model="qwen/qwen3-32b", 
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )

        # Optimization 2: Use Base Reranker (Faster than Large)
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=5
        )

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

    def process_documents(self, file_dir, db_path):
        try:
            reader = SimpleDirectoryReader(
                input_dir=file_dir,
                recursive=True
            )
            documents = reader.load_data()

            if not documents:
                return "No documents found."

            # Optimization 3: Standard Chunking (Instant)
            # Semantic chunking is too heavy for free cloud tiers
            splitter = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=200
            )
            
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_or_create_collection("user_data")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[splitter],
                show_progress=True
            )
            
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def query(self, query_text, db_path):
        try:
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_or_create_collection("user_data")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )

            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=10
            )

            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[self.reranker], 
                response_synthesizer=get_response_synthesizer(
                    response_mode="compact",
                )
            )

            response = query_engine.query(query_text)
            raw_output = str(response)

            # Clean <think> tags
            clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            
            if not clean_output:
                return raw_output
                
            return clean_output

        except Exception as e:
            return f"Error during query: {str(e)}"