import streamlit as st

# Custom Page Title & Icon for Kairo
st.set_page_config(
    page_title="Kairo | Multimodal AI Assistant",
    page_icon="🤖",
    layout="wide"
)

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
        st.info("Kairo is indexing document...")
        # TODO: Member B hooks up PyMuPDF / ChromaDB logic here
        st.success(f"Indexed into Kairo Memory: {uploaded_file.name}")

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
            st.info("Kairo is searching document knowledge base...")
            # TODO: Member B hooks up RAG LLM query response
            response_text = "Hello! I am Kairo. Connect my RAG engine to start receiving real document answers."
            st.markdown(response_text)

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text})
