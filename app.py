import io

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Kairo AI Assistant", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

BACKEND_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Sidebar: Auth & Document Ingestion
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Kairo Assistant")
    st.caption("Multimodal Authentication & Intelligence")
    st.divider()

    # ---- Identity Verification ----
    st.subheader("🛡️ Identity Verification")
    auth_mode = st.radio("Mode", ["Login", "Signup"])

    user_id_input = st.text_input(
        "User ID",
        placeholder="Enter your user ID",
        help="Must match the ID used during enrolment.",
    )

    camera_photo = st.camera_input("Capture your face")

    if camera_photo and user_id_input:
        btn_label = "Verify & Login" if auth_mode == "Login" else "Enrol Face"
        if st.button(btn_label):
            image_bytes = camera_photo.getvalue()
            endpoint = (
                f"{BACKEND_URL}/api/auth/verify-face"
                if auth_mode == "Login"
                else f"{BACKEND_URL}/api/auth/enrol"
            )

            with st.spinner("Communicating with Kairo backend..."):
                try:
                    response = requests.post(
                        endpoint,
                        files={"file": ("capture.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                        data={"user_id": user_id_input},
                        timeout=30,
                    )
                    result = response.json()
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Is the FastAPI server running?")
                    result = None
                except Exception as exc:
                    st.error(f"Request failed: {exc}")
                    result = None

            if result is not None:
                if auth_mode == "Login":
                    if result.get("authenticated"):
                        st.session_state.authenticated = True
                        st.session_state.user_id = result.get("user_id", user_id_input)
                        st.success(f"Welcome back, {st.session_state.user_id}!")
                        st.rerun()
                    else:
                        err = result.get("error") or "Authentication failed."
                        st.error(f"Login failed: {err}")
                else:  # Signup / Enrol
                    if result.get("success"):
                        st.success(
                            f"Face enrolled for '{result.get('user_id', user_id_input)}'. "
                            "You can now log in."
                        )
                    else:
                        err = result.get("error") or "Enrolment failed."
                        st.error(f"Enrolment failed: {err}")
    elif camera_photo and not user_id_input:
        st.warning("Please enter a User ID before verifying.")

    if st.session_state.authenticated:
        st.success(f"Logged in as: {st.session_state.user_id}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.rerun()

    st.divider()

    # ---- RAG Document Ingestion (Member B stub — preserved) ----
    st.subheader("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Feed PDFs to Kairo", type=["pdf"])

    if uploaded_file:
        st.info("Kairo is indexing document...")
        # TODO: Member B hooks up PyMuPDF / ChromaDB logic here
        st.success(f"Indexed into Kairo Memory: {uploaded_file.name}")

# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------
st.title("💬 Chat with Kairo")
st.caption("Ask questions via text or voice, powered by RAG and speech synthesis.")

if not st.session_state.authenticated:
    st.warning("Please verify your identity with Kairo using the webcam in the sidebar.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---- Voice input ----
    audio_input = st.file_uploader(
        "🎤 Voice Prompt to Kairo", type=["wav", "mp3", "ogg", "webm", "flac"]
    )

    # ---- Text input ----
    user_query = st.chat_input("Ask Kairo anything about your documents...")

    prompt_text = None

    if audio_input:
        with st.spinner("Transcribing audio..."):
            try:
                audio_bytes = audio_input.read()
                resp = requests.post(
                    f"{BACKEND_URL}/api/voice/transcribe",
                    files={
                        "file": (
                            audio_input.name,
                            io.BytesIO(audio_bytes),
                            audio_input.type or "audio/wav",
                        )
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                prompt_text = resp.json().get("text", "").strip()
                if not prompt_text:
                    st.warning("No speech detected in the uploaded audio.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend for transcription.")
            except Exception as exc:
                st.error(f"Transcription error: {exc}")

    if user_query:
        prompt_text = user_query.strip()

    if prompt_text:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            st.info("Kairo is searching document knowledge base...")
            # TODO: Member B hooks up RAG LLM query response
            response_text = (
                "Hello! I am Kairo. Connect my RAG engine to start receiving "
                "real document answers."
            )
            st.markdown(response_text)

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

            # ---- Optional TTS playback ----
            if st.toggle("🔊 Read response aloud", key=f"tts_{len(st.session_state.messages)}"):
                with st.spinner("Generating audio..."):
                    try:
                        tts_resp = requests.post(
                            f"{BACKEND_URL}/api/voice/tts",
                            data={"text": response_text},
                            timeout=30,
                        )
                        tts_resp.raise_for_status()
                        st.audio(tts_resp.content, format="audio/mp3")
                    except requests.exceptions.ConnectionError:
                        st.warning("Cannot reach backend for TTS.")
                    except Exception as exc:
                        st.warning(f"TTS error: {exc}")
