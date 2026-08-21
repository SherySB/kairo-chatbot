import io
import os
from pathlib import Path
from typing import Any
import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

API_BASE_URL = os.getenv("KAIRO_API_URL", "http://127.0.0.1:8000").rstrip("/")
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Kairo AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- EXTERNAL CSS LOADER (Kairo light theme) ---
def load_css(file_path: Path) -> None:
    """Injects the Kairo custom theme CSS into the Streamlit app."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(
            f"Theme file not found at {file_path}. Falling back to default Streamlit styling."
        )


load_css(BASE_DIR / "assets" / "kairo_custom.css")


# --- SESSION STATE INITIALISATION ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None
if "last_processed_recording" not in st.session_state:
    st.session_state.last_processed_recording = None


def post_request(url: str, **kwargs: Any) -> tuple[bool, dict[str, Any] | str]:
    """Helper for JSON endpoint responses."""
    try:
        response = requests.post(
            url, timeout=kwargs.pop("timeout", 60), **kwargs)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.ok and isinstance(payload, dict):
            return True, payload
        detail = payload.get("detail") if isinstance(payload, dict) else None
        return False, f"Backend HTTP {response.status_code}: {detail or response.text[:300]}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot reach Kairo backend at {API_BASE_URL}. Is FastAPI running?"
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def get_binary_request(url: str, **kwargs: Any) -> tuple[bool, bytes | str]:
    """Helper specifically for raw binary streams (e.g. MP3 audio from TTS)."""
    try:
        response = requests.post(
            url, timeout=kwargs.pop("timeout", 60), **kwargs)
        if response.ok:
            return True, response.content
        return False, f"Backend HTTP {response.status_code}: {response.text[:300]}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot reach Kairo backend at {API_BASE_URL}."
    except Exception as exc:
        return False, f"Request failed: {exc}"


# --- SIDEBAR: AUTH & DOCUMENT INGESTION ---
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">🤖 KAIRO AI</div>
            <div class="brand-sub">Multimodal Intelligence Core</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🛡️ Identity Verification")
    auth_mode = st.radio("Mode", ["Login", "Signup"], horizontal=True)

    user_id_input = st.text_input("User ID", placeholder="Enter your user ID")
    camera_photo = st.camera_input("Capture Face")

    if camera_photo and user_id_input:
        btn_label = "Verify & Login" if auth_mode == "Login" else "Enroll Face"
        if st.button(btn_label):
            image_bytes = camera_photo.getvalue()
            endpoint = (
                f"{API_BASE_URL}/api/auth/verify-face"
                if auth_mode == "Login"
                else f"{API_BASE_URL}/api/auth/enrol"
            )

            with st.spinner("Communicating with backend..."):
                ok, result = post_request(
                    endpoint,
                    files={"file": ("capture.jpg", io.BytesIO(
                        image_bytes), "image/jpeg")},
                    data={"user_id": user_id_input},
                )

            if ok and isinstance(result, dict):
                if auth_mode == "Login":
                    if result.get("authenticated"):
                        st.session_state.authenticated = True
                        st.session_state.user_id = result.get(
                            "user_id", user_id_input)
                        st.success(
                            f"Welcome back, {st.session_state.user_id}!")
                        st.rerun()
                    else:
                        st.error(result.get("error")
                                 or "Authentication failed.")
                else:
                    if result.get("success"):
                        st.success("Face enrolled! You can now log in.")
                    else:
                        st.error(result.get("error") or "Enrolment failed.")
            else:
                st.error(str(result))

    if st.session_state.authenticated:
        st.success(f"Authenticated as: {st.session_state.user_id}")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.rerun()

    st.divider()

    # ---- RAG Document Ingestion ----
    st.subheader("📄 Document Ingestion")
    pdf_file = st.file_uploader("Upload PDF for RAG indexing", type=["pdf"])

    if st.button("📥 Index Document", disabled=pdf_file is None):
        with st.spinner(f"Indexing {pdf_file.name}..."):
            ok, result = post_request(
                f"{API_BASE_URL}/api/rag/upload-pdf",
                files={
                    "file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
                timeout=120,
            )
            if ok and isinstance(result, dict):
                st.success(
                    f"Indexed {result.get('chunks_indexed', 'N/A')} chunks into vector store!"
                )
            else:
                st.error(str(result))

    st.divider()

    # ---- DEV MODE TOGGLE ----
    # dev_mode = st.checkbox("⚙️ Enable Dev Mode (Bypass Auth)")
    # if dev_mode:
    #     st.session_state.authenticated = True

# --- MAIN CHAT INTERFACE ---
st.markdown("<h1>💬 Chat Workspace</h1>", unsafe_allow_html=True)
st.caption("Interact with your indexed knowledge base using text or voice.")

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="auth-card-warning">
            <strong>🔒 Access Restricted</strong><br/>
            Please complete webcam verification or enable Dev Mode in the sidebar to proceed.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # 1. Always render existing chat history cleanly
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if st.toggle("🔊 Read response aloud", key=f"tts_toggle_{idx}"):
                    if "audio_bytes" not in msg:
                        with st.spinner("Synthesizing audio..."):
                            tts_ok, audio_res = get_binary_request(
                                f"{API_BASE_URL}/api/voice/tts",
                                data={"text": msg["content"]},
                            )
                            if tts_ok and isinstance(audio_res, bytes):
                                msg["audio_bytes"] = audio_res
                            else:
                                st.error(f"TTS Failed: {audio_res}")

                    if "audio_bytes" in msg:
                        st.audio(msg["audio_bytes"], format="audio/mp3")

    prompt_text = None

    # 2. Audio Inputs Section
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("🎙️ **Live Mic Recording**")
        recorded_audio = audio_recorder(
            text="",
            recording_color="#1f252a",
            neutral_color="#75777b",
            icon_size="2x"
        )

        # Prevent re-transcribing live recording on rerun loop
        if recorded_audio and recorded_audio != st.session_state.last_processed_recording:
            st.session_state.last_processed_recording = recorded_audio
            with st.spinner("Transcribing recording..."):
                ok, result = post_request(
                    f"{API_BASE_URL}/api/voice/transcribe",
                    files={
                        "file": ("recording.wav", recorded_audio, "audio/wav")},
                )
                if ok and isinstance(result, dict):
                    prompt_text = result.get("text", "").strip()
                else:
                    st.error(f"Transcription error: {result}")

    with col2:
        audio_input = st.file_uploader(
            "📁 Upload Audio File Query",
            type=["wav", "mp3", "ogg", "webm", "flac"],
            key="voice_uploader"
        )
        # Prevent re-transcribing uploaded file on rerun loop
        if audio_input and audio_input.file_id != st.session_state.last_processed_audio:
            st.session_state.last_processed_audio = audio_input.file_id
            with st.spinner("Transcribing audio query..."):
                ok, result = post_request(
                    f"{API_BASE_URL}/api/voice/transcribe",
                    files={
                        "file": (
                            audio_input.name,
                            audio_input.getvalue(),
                            audio_input.type or "audio/wav",
                        )
                    },
                )
                if ok and isinstance(result, dict):
                    prompt_text = result.get("text", "").strip()
                else:
                    st.error(f"Transcription error: {result}")

    user_query = st.chat_input("Ask Kairo anything about your documents...")
    if user_query:
        prompt_text = user_query.strip()

    # 3. Process new message on input
    if prompt_text:
        st.session_state.messages.append(
            {"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            with st.spinner("Searching document knowledge base..."):
                ok, result = post_request(
                    f"{API_BASE_URL}/api/chat", data={"prompt": prompt_text}
                )
                if ok and isinstance(result, dict):
                    response_text = result.get("response", "")
                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    st.rerun()
                else:
                    st.error(f"Chat error: {result}")
