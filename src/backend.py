import os
import sys
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
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb

# SQLite Fix
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

load_dotenv()

class AdvancedRAG:
    def __init__(self):
        # 1. SETUP MODELS
        
        # Embedding
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            trust_remote_code=True
        )
        
        # LLM: Qwen 3 (32B)
        self.llm = Groq(
            model="qwen/qwen3-32b", 
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )

        # Reranker
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-large",
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

            splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95, 
                embed_model=self.embed_model
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
            
            # --- UPDATED: Short simple message ---
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

            # Custom Prompt
            custom_prompt = (
                "You are a precise data assistant.\n"
                "Guidelines:\n"
                "1. If the user asks for a specific fact (Name, Date, Company, Cost), output ONLY that fact. No full sentences.\n"
                "2. If the user asks for an explanation, provide a detailed response.\n"
                "3. NEVER output your internal thought process or <think> tags.\n"
                "4. Answer based strictly on the context provided."
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

            # Remove <think> tags
            clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
            
            if not clean_output:
                return raw_output
                
            return clean_output

        except Exception as e:
            return f"Error during query: {str(e)}"