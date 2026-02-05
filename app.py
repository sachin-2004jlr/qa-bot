import streamlit as st
import os
import shutil
import uuid
from src.backend import AdvancedRAG

st.set_page_config(page_title="Multi-LLM RAG", layout="wide")
st.markdown("<h1 style='text-align: center;'>Multi-Model RAG (Secure & Isolated)</h1>", unsafe_allow_html=True)

# --- SESSION ISOLATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

base_dir = "temp_data"
user_session_path = os.path.join(base_dir, st.session_state.session_id)
files_path = os.path.join(user_session_path, "files")
db_path = os.path.join(user_session_path, "db")

os.makedirs(files_path, exist_ok=True)
os.makedirs(db_path, exist_ok=True)

# --- MODEL REGISTRY ---
# I have removed the broken Gemma model and added Mixtral
model_map = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
    "Llama 4 (Scout 17B)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Qwen 3 32B": "qwen/qwen3-32b",
    "Mixtral 8x7b": "mixtral-8x7b-32768"  # <--- Replaced Gemma with this stable model
}

@st.cache_resource
def get_rag_engine():
    return AdvancedRAG()

rag_engine = get_rag_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.header("🧠 Neural Core")
    selected_model_friendly = st.selectbox("Select AI Model", list(model_map.keys()))
    selected_model_id = model_map[selected_model_friendly]
    
    st.info(f"Active Model ID: `{selected_model_id}`")
    st.divider()
    
    st.header("📂 Secure Data Upload")
    uploaded_files = st.file_uploader("Upload PDF/Docx", accept_multiple_files=True)
    
    if st.button("Process & Index"):
        if uploaded_files:
            with st.spinner("Indexing..."):
                if os.path.exists(files_path):
                    shutil.rmtree(files_path)
                os.makedirs(files_path)
                
                for file in uploaded_files:
                    with open(os.path.join(files_path, file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                status = rag_engine.process_documents(files_path, db_path)
                
                if status == "Success":
                    st.success("Indexed Successfully!")
                    st.session_state.db_ready = True
                else:
                    st.error(status)
        else:
            st.warning("Please upload files first.")

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'Unknown')})"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg["role"] == "assistant":
        st.markdown("---")

if prompt := st.chat_input(f"Ask {selected_model_friendly}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")
    
    if st.session_state.get("db_ready"):
        with st.spinner(f"Generating with {selected_model_friendly}..."):
            response = rag_engine.query(prompt, db_path, selected_model_id)
            
            st.markdown(f"**AI ({selected_model_friendly}):** {response}")
            st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "model_name": selected_model_friendly
            })
    else:
        st.error("⚠️ Please upload and process documents first.")