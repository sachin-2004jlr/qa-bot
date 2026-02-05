import streamlit as st
import os
import shutil
import uuid
from src.backend import AdvancedRAG

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi Model RAG", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. CLEAN TITLE (Formal, no icons, no extra words)
st.markdown("<h1 style='text-align: center;'>Multi Model RAG</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECURITY & ISOLATION
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# MODEL REGISTRY
# -----------------------------------------------------------------------------
model_map = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
    "Llama 4 (Scout 17B)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Qwen 3 32B": "qwen/qwen3-32b",
    "GPT-OSS 20B": "openai/gpt-oss-20b"
}

# -----------------------------------------------------------------------------
# CACHED BACKEND
# -----------------------------------------------------------------------------
@st.cache_resource
def get_rag_engine():
    return AdvancedRAG()

rag_engine = get_rag_engine()

# -----------------------------------------------------------------------------
# SIDEBAR: CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    # Formal Header (No Icons)
    st.header("Model Selection")
    
    selected_model_friendly = st.selectbox("Select AI Model", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    # Formal Info Box
    st.info(f"**Current Model:** `{selected_model_friendly}`")
    st.divider()
    
    # Formal Upload Section
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF/Docx", accept_multiple_files=True)
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Clean old files for THIS session
                if os.path.exists(FILES_DIR):
                    shutil.rmtree(FILES_DIR)
                os.makedirs(FILES_DIR)
                
                # Save new files
                for file in uploaded_files:
                    file_path = os.path.join(FILES_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Process
                status = rag_engine.process_documents(FILES_DIR, DB_DIR)
                
                if status == "Success":
                    st.success("Documents processed successfully.")
                    st.session_state.db_ready = True
                else:
                    st.error(f"Error: {status}")
        else:
            st.warning("Please upload files first.")

# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'Unknown')})"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg["role"] == "assistant":
        st.markdown("---")

# Chat Input
if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")
    
    if st.session_state.get("db_ready"):
        with st.spinner(f"Processing with {selected_model_friendly}..."):
            
            response = rag_engine.query(
                query_text=prompt, 
                db_path=DB_DIR, 
                model_name=selected_model_id
            )
            
            st.markdown(f"**AI ({selected_model_friendly}):** {response}")
            st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "model_name": selected_model_friendly
            })
    else:
        st.error("Please upload and process documents first.")