import streamlit as st
import os
import shutil
import uuid
import io
from docx import Document
from src.backend import AdvancedRAG

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi Model RAG", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>Multi Model RAG</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_title = "New Chat"
    st.session_state.db_ready = False

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")

os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# HELPER: WORD DOCUMENT GENERATOR
# -----------------------------------------------------------------------------
def generate_document(messages):
    doc = Document()
    doc.add_heading('Chat Conversation Log', 0)
    
    for msg in messages:
        role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'Unknown')})"
        content = msg["content"]
        
        # Add Role as Bold Heading
        p = doc.add_paragraph()
        runner = p.add_run(f"{role}:")
        runner.bold = True
        
        # Add Content
        doc.add_paragraph(content)
        doc.add_paragraph("-" * 20) # Separator
        
    # Save to memory buffer
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

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
# SIDEBAR LOGIC
# -----------------------------------------------------------------------------
with st.sidebar:
    # --- NEW CHAT BUTTON ---
    # Removed "➕" icon
    if st.button("New Chat", type="primary", use_container_width=True):
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
    
    # --- HISTORY SECTION ---
    st.header("Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        # Removed "📄" icon
        if st.button(f"{chat['title']}", key=f"hist_{chat['id']}"):
            if st.session_state.messages and st.session_state.session_id != chat['id']:
                 st.session_state.chat_history.append({
                    "id": st.session_state.session_id,
                    "title": st.session_state.chat_title,
                    "messages": st.session_state.messages,
                    "db_ready": st.session_state.db_ready
                })
            
            st.session_state.session_id = chat['id']
            st.session_state.messages = chat['messages']
            st.session_state.chat_title = chat['title']
            st.session_state.db_ready = chat['db_ready']
            
            st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat['id']]
            st.rerun()

    st.markdown("---")
    
    # --- DOWNLOAD SECTION ---
    if st.session_state.messages:
        docx_file = generate_document(st.session_state.messages)
        # Removed "📥" icon
        st.download_button(
            label="Download Conversation (DOCX)",
            data=docx_file,
            file_name=f"chat_log_{st.session_state.session_id[:8]}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
        st.markdown("---")

    # --- MODEL & UPLOAD SECTION ---
    st.header("Settings")
    
    selected_model_friendly = st.selectbox("Select Model", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        accept_multiple_files=True, 
        key=f"uploader_{st.session_state.session_id}"
    )
    
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                if os.path.exists(FILES_DIR):
                    shutil.rmtree(FILES_DIR)
                os.makedirs(FILES_DIR)
                
                for file in uploaded_files:
                    file_path = os.path.join(FILES_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                status = rag_engine.process_documents(FILES_DIR, DB_DIR)
                
                if status == "Success":
                    st.success("Ready")
                    st.session_state.db_ready = True
                else:
                    st.error(f"Error: {status}")
        else:
            st.warning("Upload files first.")

# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'Unknown')})"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg["role"] == "assistant":
        st.markdown("---")

if prompt := st.chat_input("Enter your query..."):
    if not st.session_state.messages:
        st.session_state.chat_title = " ".join(prompt.split()[:5]) + "..."
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")
    
    if st.session_state.get("db_ready"):
        with st.spinner("Processing..."):
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
        st.error("Please upload documents for this new chat.")