import streamlit as st
import os
import shutil
import uuid
from src.backend import AdvancedRAG

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-LLM RAG", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>Multi-Model RAG (Secure & Isolated)</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SECURITY & ISOLATION
# -----------------------------------------------------------------------------
# 1. Generate a unique Session ID for this specific user/tab
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# 2. Define isolated paths based on that Session ID
BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")

# 3. Ensure these folders exist immediately
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# MODEL REGISTRY (VERIFIED ACTIVE)
# -----------------------------------------------------------------------------
# Updated to replace decommissioned models with DeepSeek R1
model_map = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",          # Best General Purpose
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",              # Fastest
    "Llama 4 (Scout 17B)": "meta-llama/llama-4-scout-17b-16e-instruct", # State-of-the-Art Preview
    "Qwen 3 32B": "qwen/qwen3-32b",                                # Best Non-Llama Open Model
    "DeepSeek R1 (Distill 70B)": "deepseek-r1-distill-llama-70b"   # <--- REPLACED BROKEN MIXTRAL WITH THIS
}

# -----------------------------------------------------------------------------
# CACHED BACKEND
# -----------------------------------------------------------------------------
@st.cache_resource
def get_rag_engine():
    return AdvancedRAG()

rag_engine = get_rag_engine()

# -----------------------------------------------------------------------------
# SIDEBAR: CONTROLS & DATA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🧠 Neural Core")
    
    # Model Selector
    selected_model_friendly = st.selectbox("Select AI Model", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    st.info(f"**Active Brain:** `{selected_model_friendly}`")
    st.divider()
    
    st.header("📂 Secure Data Upload")
    uploaded_files = st.file_uploader("Upload PDF/Docx", accept_multiple_files=True)
    
    if st.button("Process & Index Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Encrypting, Chunking & Indexing..."):
                # Clean old files for THIS session only
                if os.path.exists(FILES_DIR):
                    shutil.rmtree(FILES_DIR)
                os.makedirs(FILES_DIR)
                
                # Save new files
                for file in uploaded_files:
                    file_path = os.path.join(FILES_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Process into the isolated ChromaDB
                status = rag_engine.process_documents(FILES_DIR, DB_DIR)
                
                if status == "Success":
                    st.success("✅ Data Successfully Indexed!")
                    st.session_state.db_ready = True
                else:
                    st.error(f"❌ Error: {status}")
        else:
            st.warning("⚠️ Please upload files first.")

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
if prompt := st.chat_input(f"Ask questions using {selected_model_friendly}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")
    
    if st.session_state.get("db_ready"):
        with st.spinner(f"Thinking with {selected_model_friendly}..."):
            
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
        st.error("⚠️ Please upload and process documents first.")