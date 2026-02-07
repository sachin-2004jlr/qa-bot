import sys
import os
from dotenv import load_dotenv

# --- CLOUD DATABASE FIX ---
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
    Settings,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.file import PyMuPDFReader
import chromadb

load_dotenv()

class AdvancedRAG:
    def __init__(self):
        # 1. Improved Embedding Model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = self.embed_model
        
        # 2. Refined Chunking Logic
        # Smaller chunks (512) help the model find more specific information
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        Settings.node_parser = self.node_parser

    def process_documents(self, file_dir, db_path):
        try:
            # Using PyMuPDFReader for better table and structure extraction
            file_extractor = {".pdf": PyMuPDFReader()}
            reader = SimpleDirectoryReader(
                input_dir=file_dir,
                recursive=True,
                file_extractor=file_extractor
            )
            documents = reader.load_data()

            if not documents:
                return "No documents found."
            
            # Database Connection
            chroma_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = chroma_client.get_or_create_collection("user_data")
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            return "Success"
            
        except Exception as e:
            return f"Error: {str(e)}"

    def query(self, query_text, db_path, model_name):
        try:
            # 3. System Prompt for Response Control
            # This forces the model to analyze query intent for length
            system_prompt = (
                "You are an expert assistant. Use the provided context to answer the user's question.\n"
                "RESPONSE RULES:\n"
                "1. If the query is simple, provide a concise 2-line response.\n"
                "2. If the query asks for details or complex analysis, provide a comprehensive answer.\n"
                "3. If you don't know the answer based on context, say you don't know. Don't hallucinate."
            )

            llm = Groq(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1, # Low temperature for high precision
                system_prompt=system_prompt 
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

            # 4. Custom QA Prompt Template
            qa_prompt_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the query.\n"
                "Query: {query_str}\n"
                "Answer: "
            )
            qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

            # Retrieve top 5 most relevant chunks
            retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                text_qa_template=qa_prompt_tmpl
            )

            response = query_engine.query(query_text)
            return str(response)

        except Exception as e:
            return f"Error during query: {str(e)}"