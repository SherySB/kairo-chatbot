import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Kairo AI Assistant", layout="wide")

# 2. Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "messages" not in st.session_state:
    st.session_state.messages = []

BACKEND_URL = "http://127.0.0.1:8000"

# --- SIDEBAR: AUTH & DOCUMENT INGESTION ---
with st.sidebar:
    st.title("🤖 Kairo Assistant")
    st.caption("Multimodal Authentication & Intelligence")
    st.divider()

    # 1. Facial Login / Signup Section
    st.subheader("🛡️ Identity Verification")
    auth_mode = st.radio("Mode", ["Login", "Signup"])
    camera_photo = st.camera_input("Verify face with Kairo")

    if camera_photo:
        st.info("Kairo is analyzing facial features...")
        # TODO: Member A hooks up face_auth logic here
        if st.button("Simulate Successful Login"):
            st.session_state.authenticated = True
            st.session_state.user_id = "test_user"
            st.success("Welcome back! Authenticated by Kairo.")

    st.divider()

    # 2. RAG Document Ingestion Section
    st.subheader("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Feed PDFs to Kairo", type=["pdf"])

    if uploaded_file:
        if "last_uploaded_doc" not in st.session_state or st.session_state.last_uploaded_doc != uploaded_file.name:
            with st.spinner("Kairo is parsing & indexing document into ChromaDB..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/api/rag/upload-pdf", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.last_uploaded_doc = uploaded_file.name
                        st.success(f"✅ {data.get('message', 'Indexed successfully!')}")
                    else:
                        st.error(f"❌ Failed to index PDF: {res.text}")
                except Exception as e:
                    st.error(f"❌ Backend connection error: {e}")
        else:
            st.success(f"✅ Indexed in Memory: {uploaded_file.name}")

# --- MAIN CHAT INTERFACE ---
st.title("💬 Chat with Kairo")
st.caption("Ask questions via text or voice, powered by RAG and speech synthesis.")

if not st.session_state.authenticated:
    st.warning(
        "Please verify your identity with Kairo using the webcam in the sidebar.")
else:
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Inputs
    audio_input = st.file_uploader(
        "🎤 Voice Prompt to Kairo", type=["wav", "mp3"])
    user_query = st.chat_input("Ask Kairo anything about your documents...")

    if user_query or audio_input:
        prompt_text = user_query if user_query else "[Audio Query Transcribed]"

        st.session_state.messages.append(
            {"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Kairo is searching document knowledge base..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/api/chat", data={"prompt": prompt_text})
                    if res.status_code == 200:
                        response_text = res.json().get("response", "No response received.")
                    else:
                        response_text = f"Error from backend: {res.text}"
                except Exception as e:
                    response_text = f"Could not connect to backend ({e}). Please ensure FastAPI is running via `uvicorn main:app --reload`."

            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text})
