import streamlit as st
import os
import shutil
import uuid
import io
from docx import Document
from src.backend import AdvancedRAG

# 1. Page Configuration
st.set_page_config(
    page_title="Multi Model RAG", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Refined Professional Dark Theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }

    /* Main Title Styling */
    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: #f3f4f6;
        padding: 30px 0px 10px 0px;
        font-weight: 700;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }
    
    .title-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.75rem;
        letter-spacing: 0.2em;
        margin-bottom: 30px;
    }

    /* Sidebar - High Contrast Slate */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #9ca3af;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 25px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 8px;
    }

    /* Message Containers */
    .chat-container {
        padding: 24px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #1f2937;
        font-family: 'Inter', sans-serif;
    }
    
    .user-box {
        background-color: #1f2937;
        border-left: 4px solid #475569;
    }
    
    .ai-box {
        background-color: #111827;
        border-left: 4px solid #065f46;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .role-header {
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 12px;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
    }
    
    .content-text {
        color: #d1d5db;
        line-height: 1.8;
        font-size: 0.95rem;
    }

    /* Column alignment for download button */
    .download-col {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_title = "New Chat"
    st.session_state.db_ready = False

# 4. Directory Setup
BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 5. Document Helper
def generate_document(messages):
    doc = Document()
    doc.add_heading('Formal Conversation Log', 0)
    for msg in messages:
        role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'System')})"
        p = doc.add_paragraph()
        p.add_run(f"{role}:").bold = True
        doc.add_paragraph(msg["content"])
        doc.add_paragraph("-" * 20)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

model_map = {
    "Llama 3.3 70B (Versatile)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Instant)": "llama-3.1-8b-instant",
    "Llama 4 (Scout 17B)": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Qwen 3 32B": "qwen/qwen3-32b",
    "GPT-OSS 20B": "openai/gpt-oss-20b"
}

@st.cache_resource
def get_rag_engine():
    return AdvancedRAG()
rag_engine = get_rag_engine()

# 6. Sidebar Implementation
with st.sidebar:
    st.header("New Chat")
    if st.button("Start New Chat", type="primary", use_container_width=True):
        if st.session_state.messages:
            st.session_state.chat_history.append({
                "id": st.session_state.session_id,
                "title": st.session_state.chat_title,
                "messages": st.session_state.messages,
                "db_ready": st.session_state.db_ready
            })
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_title = "New Chat"
        st.session_state.db_ready = False
        st.rerun()

    st.markdown("---")
    st.header("Chat History")
    for chat in reversed(st.session_state.chat_history):
        if st.button(f"{chat['title']}", key=f"hist_{chat['id']}", use_container_width=True):
            st.session_state.session_id = chat['id']
            st.session_state.messages = chat['messages']
            st.session_state.chat_title = chat['title']
            st.session_state.db_ready = chat['db_ready']
            st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat['id']]
            st.rerun()

    st.markdown("---")
    st.header("Settings")
    selected_model_friendly = st.selectbox("Select Model", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Drop files here", accept_multiple_files=True, key=f"uploader_{st.session_state.session_id}")
    
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Indexing..."):
                if os.path.exists(FILES_DIR): shutil.rmtree(FILES_DIR)
                os.makedirs(FILES_DIR)
                for file in uploaded_files:
                    with open(os.path.join(FILES_DIR, file.name), "wb") as f: f.write(file.getbuffer())
                status = rag_engine.process_documents(FILES_DIR, DB_DIR)
                if status == "Success":
                    st.success("Ready")
                    st.session_state.db_ready = True
                else: st.error(f"Error: {status}")

# 7. Main Interface & Export Logic (Right Side)
st.markdown("<h1 class='main-title'>Multi Model RAG</h1>", unsafe_allow_html=True)
st.markdown("<p class='title-subtitle'>ENTERPRISE INTELLIGENCE SYSTEM</p>", unsafe_allow_html=True)

if st.session_state.messages:
    # Use columns to push the download button to the far right
    header_col, download_col = st.columns([8, 2])
    with download_col:
        docx_file = generate_document(st.session_state.messages)
        st.download_button(
            label="Download Log (DOCX)",
            data=docx_file,
            file_name=f"chat_log_{st.session_state.session_id[:8]}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )

# Chat Display
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-container user-box"><div class="role-header">User</div><div class="content-text">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-container ai-box"><div class="role-header">Response | {msg.get("model_name", "System")}</div><div class="content-text">{msg["content"]}</div></div>', unsafe_allow_html=True)

# 8. Input Processing (Fixed Parameter Name)
if prompt := st.chat_input("Enter your query..."):
    if not st.session_state.messages:
        st.session_state.chat_title = " ".join(prompt.split()[:5]) + "..."
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.session_state.get("db_ready"):
        with st.spinner("Processing..."):
            # FIX: Use query_text instead of prompt
            response = rag_engine.query(
                query_text=st.session_state.messages[-1]["content"], 
                db_path=DB_DIR, 
                model_name=selected_model_id
            )
            st.session_state.messages.append({"role": "assistant", "content": response, "model_name": selected_model_friendly})
            st.rerun()
    else:
        st.error("Please upload and process documents first.")