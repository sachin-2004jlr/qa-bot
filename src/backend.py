import sys
import os
import shutil
from dotenv import load_dotenv

# --- CLOUD DATABASE FIX ---
# This forces the system to use the correct SQLite version on Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# --------------------------

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb

load_dotenv()

class AdvancedRAG:
    def __init__(self):
        # 1. EMBEDDING: Load once (Heavy operation)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
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
            
            # Filter empty docs
            documents = [doc for doc in documents if doc.text and len(doc.text.strip()) > 0]
            if not documents:
                return "No valid text found in documents."

            # Chunking
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            
            # Database Connection
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

    def query(self, query_text, db_path, model_name):
        try:
            # 2. LLM: Initialize dynamically based on user selection
            llm = Groq(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1
            )
            Settings.llm = llm

            # Connect to DB
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_or_create_collection("user_data")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )

            # Retrieve top 5 chunks
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5
            )

            query_engine = RetrieverQueryEngine(
                retriever=retriever
            )

            response = query_engine.query(query_text)
            return str(response)

        except Exception as e:
            return f"Error during query: {str(e)}"