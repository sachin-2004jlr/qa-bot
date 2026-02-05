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

st.markdown("<h1 style='text-align: center;'>Multi Model RAG</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STATE MANAGEMENT (Crucial for History)
# -----------------------------------------------------------------------------
# 1. Initialize Global History List
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Initialize Current Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.chat_title = "New Chat"
    st.session_state.db_ready = False

# -----------------------------------------------------------------------------
# PATH CONFIGURATION (Dynamic based on Current Session ID)
# -----------------------------------------------------------------------------
BASE_DIR = "temp_data"
USER_SESSION_DIR = os.path.join(BASE_DIR, st.session_state.session_id)
FILES_DIR = os.path.join(USER_SESSION_DIR, "files")
DB_DIR = os.path.join(USER_SESSION_DIR, "db")

# Ensure folders exist for the CURRENT active chat
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
# SIDEBAR LOGIC
# -----------------------------------------------------------------------------
with st.sidebar:
    # --- NEW CHAT BUTTON ---
    if st.button("➕ New Chat", type="primary", use_container_width=True):
        # 1. Save current chat to history (if it has messages)
        if st.session_state.messages:
            st.session_state.chat_history.append({
                "id": st.session_state.session_id,
                "title": st.session_state.chat_title,
                "messages": st.session_state.messages,
                "db_ready": st.session_state.db_ready
            })
        
        # 2. Reset State for Fresh Start
        st.session_state.session_id = str(uuid.uuid4()) # NEW UUID = NEW DATA FOLDER
        st.session_state.messages = []
        st.session_state.chat_title = "New Chat"
        st.session_state.db_ready = False
        st.rerun()

    st.markdown("---")
    
    # --- HISTORY SECTION ---
    st.header("Chat History")
    
    # Display previous chats as buttons
    # We reverse the list to show newest on top
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        if st.button(f"📄 {chat['title']}", key=f"hist_{chat['id']}"):
            # 1. Save CURRENT chat before switching
            if st.session_state.messages and st.session_state.session_id != chat['id']:
                 st.session_state.chat_history.append({
                    "id": st.session_state.session_id,
                    "title": st.session_state.chat_title,
                    "messages": st.session_state.messages,
                    "db_ready": st.session_state.db_ready
                })
            
            # 2. Load the CLICKED chat
            st.session_state.session_id = chat['id']
            st.session_state.messages = chat['messages']
            st.session_state.chat_title = chat['title']
            st.session_state.db_ready = chat['db_ready']
            
            # 3. Remove the loaded chat from history list (it's now active)
            # We filter it out by ID
            st.session_state.chat_history = [c for c in st.session_state.chat_history if c['id'] != chat['id']]
            st.rerun()

    st.markdown("---")

    # --- MODEL & UPLOAD SECTION ---
    st.header("Settings")
    
    # Model Selector
    selected_model_friendly = st.selectbox("Select Model", list(model_map.keys()), index=0)
    selected_model_id = model_map[selected_model_friendly]
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    # Process Button
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Clean old files for THIS SPECIFIC SESSION only
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
                    st.success("Ready!")
                    st.session_state.db_ready = True
                else:
                    st.error(f"Error: {status}")
        else:
            st.warning("Upload files first.")

# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
# Display Chat History for Current Session
for msg in st.session_state.messages:
    role = "User" if msg["role"] == "user" else f"AI ({msg.get('model_name', 'Unknown')})"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg["role"] == "assistant":
        st.markdown("---")

# Chat Input
if prompt := st.chat_input("Enter your query..."):
    # 1. Set Title (if first message)
    if not st.session_state.messages:
        # Generate a title from the first 4-5 words
        st.session_state.chat_title = " ".join(prompt.split()[:5]) + "..."
    
    # 2. Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**User:** {prompt}")
    
    # 3. Process with AI
    if st.session_state.get("db_ready"):
        with st.spinner("Thinking..."):
            response = rag_engine.query(
                query_text=prompt, 
                db_path=DB_DIR, # Uses CURRENT session's DB folder
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