import streamlit as st
import os
import shutil
import uuid
import io
from docx import Document
from src.backend import AdvancedRAG

# 1. Page Configuration
st.set_page_config(
    page_title="Multi Model RAG | Enterprise Console", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Enterprise Dark Theme CSS
st.markdown("""
    <style>
    /* Import Premium San-Serif Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Transitions & Background */
    .stApp {
        background-color: #020617;
        font-family: 'Inter', sans-serif;
    }

    /* Elite Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
        padding: 2rem 1rem;
    }

    /* Section Headers in Sidebar */
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #64748b;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Main Title - Modern Minimalist */
    .main-title {
        text-align: center;
        color: #f8fafc;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.05em;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .title-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        margin-bottom: 3rem;
    }

    /* Professional Message Containers */
    .chat-container {
        max-width: 850px;
        margin: auto;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .user-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8; /* Slate Blue */
    }
    
    .ai-box {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #10b981; /* Emerald */
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    .role-header {
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
    }
    
    .content-text {
        color: #e2e8f0;
        line-height: 1.8;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Sidebar Button Overrides */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #1e293b;
        background-color: #1e293b;
        color: #f8fafc;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #334155;
        border-color: #38bdf8;
    }

    /* Custom Chat Input Focus */
    .stChatInputContainer input {
        border-radius: 12px !important;
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Enhanced Title Section
st.markdown("<div class='main-title'>MULTI MODEL RAG</div>", unsafe_allow_html=True)
st.markdown("<div class='title-subtitle'>Enterprise Intelligence Console</div>", unsafe_allow_html=True)

# 4. Initialize Core Engine
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_title = "New Session"
    st.session_state.db_ready = False

BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 5. Helper Functions
def generate_document(messages):
    doc = Document()
    doc.add_heading('Enterprise Intelligence Log', 0)
    for msg in messages:
        role = "CLIENT" if msg["role"] == "user" else f"AGENT ({msg.get('model_name', 'SYSTEM')})"
        p = doc.add_paragraph()
        p.add_run(f"{role}:").bold = True
        doc.add_paragraph(msg["content"])
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
    st.header("Control Center")
    if st.button("Initialize New Session", type="primary"):
        if st.session_state.messages:
            st.session_state.chat_history.append({
                "id": st.session_state.session_id,
                "title": st.session_state.chat_title,
                "messages": st.session_state.messages,
                "db_ready": st.session_state.db_ready
            })
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chat_title = "New Session"
        st.session_state.db_ready = False
        st.rerun()

    st.header("Archived Intelligence")
    for chat in reversed(st.session_state.chat_history):
        if st.button(f"{chat['title']}", key=f"hist_{chat['id']}"):
            st.session_state.session_id = chat['id']
            st.session_state.messages = chat['messages']
            st.session_state.chat_title = chat['title']
            st.session_state.db_ready = chat['db_ready']
            st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat['id']]
            st.rerun()

    st.header("Model Parameters")
    selected_model_friendly = st.selectbox("Intelligence Core", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    st.header("Data Ingestion")
    uploaded_files = st.file_uploader("Upload Knowledge Assets", accept_multiple_files=True, key=f"uploader_{st.session_state.session_id}")
    
    if st.button("Build Knowledge Index"):
        if uploaded_files:
            with st.spinner("Processing Corporate Assets..."):
                if os.path.exists(FILES_DIR):
                    shutil.rmtree(FILES_DIR)
                os.makedirs(FILES_DIR)
                for file in uploaded_files:
                    file_path = os.path.join(FILES_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                status = rag_engine.process_documents(FILES_DIR, DB_DIR)
                if status == "Success":
                    st.success("Indexing Complete")
                    st.session_state.db_ready = True
                else:
                    st.error(f"Error: {status}")

    st.header("Governance")
    if st.session_state.messages:
        docx_file = generate_document(st.session_state.messages)
        st.download_button("Export Session (DOCX)", data=docx_file, file_name=f"session_{st.session_state.session_id[:8]}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

# 7. Elite Chat Display Logic
for msg in st.session_state.messages:
    box_class = "user-box" if msg["role"] == "user" else "ai-box"
    role_label = "CLIENT INQUIRY" if msg["role"] == "user" else f"AGENT RESPONSE | {msg.get('model_name', 'SYSTEM')}"
    
    st.markdown(f"""
        <div class="chat-container {box_class}">
            <div class="role-header">
                <span>{role_label}</span>
            </div>
            <div class="content-text">{msg['content']}</div>
        </div>
    """, unsafe_allow_html=True)

# 8. Optimized Chat Input
if prompt := st.chat_input("Input inquiry..."):
    if not st.session_state.messages:
        st.session_state.chat_title = " ".join(prompt.split()[:4]) + "..."
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    if st.session_state.get("db_ready"):
        with st.spinner(f"Consulting {selected_model_friendly}..."):
            response = rag_engine.query(query_text=st.session_state.messages[-1]["content"], db_path=DB_DIR, model_name=selected_model_id)
            st.session_state.messages.append({"role": "assistant", "content": response, "model_name": selected_model_friendly})
            st.rerun()
    else:
        st.error("Knowledge base not initialized. Please ingestion documents via the control center.")