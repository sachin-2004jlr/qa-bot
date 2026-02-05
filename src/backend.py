import sys
import os
import shutil
from dotenv import load_dotenv

# --- Streamlit Cloud SQLite Fix ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------

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
        # 1. EMBEDDING: Load once (Heavy operation, but shared safely)
        # We use MiniLM because it is fast and CPU-friendly for free cloud
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Set embedding model globally
        Settings.embed_model = self.embed_model

    def process_documents(self, file_dir, db_path):
        """
        Processes documents and creates a USER-SPECIFIC ChromaDB.
        """
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
            
            # Database Connection (ISOLATED via db_path)
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
        """
        Retrieves context and generates answer using the SELECTED Model.
        """
        try:
            # 2. LLM: Initialize dynamically based on user selection
            # This ensures the generation step uses the specific model requested
            llm = Groq(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3 # Slightly creative but focused
            )
            
            # Force LlamaIndex to use this specific LLM for this query
            Settings.llm = llm

            # Connect to the User's specific DB
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_or_create_collection("user_data")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=self.embed_model
            )

            # Retrieve top 5 most relevant chunks (Context)
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5
            )

            # The Query Engine combines: Retriever + LLM (Generator)
            query_engine = RetrieverQueryEngine(
                retriever=retriever
            )

            response = query_engine.query(query_text)
            return str(response)

        except Exception as e:
            return f"Error during query: {str(e)}"