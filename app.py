import streamlit as st
import os
import shutil
import uuid
from src.backend import AdvancedRAG

st.set_page_config(page_title="Q/A Bot", layout="wide")
st.markdown("<h1 style='text-align: center;'>Q/A Bot</h1>", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    base_dir = "temp_data"
    os.makedirs(os.path.join(base_dir, st.session_state.session_id, "files"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, st.session_state.session_id, "db"), exist_ok=True)

@st.cache_resource
def get_rag_engine():
    return AdvancedRAG()

rag_engine = get_rag_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

user_session_path = os.path.join("temp_data", st.session_state.session_id)
files_path = os.path.join(user_session_path, "files")
db_path = os.path.join(user_session_path, "db")

with st.sidebar:
    st.header("Data Center")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                if os.path.exists(files_path): shutil.rmtree(files_path)
                os.makedirs(files_path)
                
                for file in uploaded_files:
                    with open(os.path.join(files_path, file.name), "wb") as f:
                        f.write(file.getbuffer())
                
                status = rag_engine.process_documents(files_path, db_path)
                st.markdown(f'<p style="color:#28a745; font-weight:bold; font-size:16px;">{status}</p>', unsafe_allow_html=True)
                st.session_state.db_ready = True
        else:
            st.warning("Please upload files first.")

for msg in st.session_state.messages:
    role = "Your Query" if msg["role"] == "user" else "AI Answer"
    st.markdown(f"**{role}:** {msg['content']}")
    if msg["role"] == "assistant":
        st.markdown("---")

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"**Your Query:** {prompt}")
    
    if st.session_state.get("db_ready"):
        with st.spinner("Reasoning..."):
            response = rag_engine.query(prompt, db_path)
            st.markdown(f"**AI Answer:** {response}")
            st.markdown("---")
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please process documents first.")