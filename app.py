"""Kairo AI Assistant — Streamlit app.

Run with: streamlit run app.py

Optional FastAPI endpoints:
  POST {KAIRO_API_URL}/api/chat          form field: prompt
  POST {KAIRO_API_URL}/api/rag/upload-pdf multipart field: file
"""

import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("KAIRO_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Kairo AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ============================================================
       KAIRO — "Aperture & Instrument" identity
       A dark optics-bench palette (ink + brass) built around one
       recurring signature: a lens-aperture ring. It appears in the
       brand mark, in status indicators (a focus-pulse for "pending",
       echoing a camera hunting for focus), and in the camera-input
       bezel — reinforcing that Kairo verifies identity by sight.
       ============================================================ */
    :root {
      --ink:#0B0E14; --panel:#151A24; --panel-raised:#1D2432; --hairline:#2B3242;
      --paper:#EDEFF3; --mist:#9BA4B6;
      --brass:#CC9A4E; --brass-bright:#E3B36A; --brass-dim:#8A6B3A;
      --teal:#4FB89F; --rose:#E2685E; --rose-dim:#B84C43;
      --radius-sm:8px; --radius-md:12px; --radius-lg:18px;
      --ease:cubic-bezier(.4,0,.2,1);
      --shadow-xs: 0 1px 2px rgba(0,0,0,0.28);
      --shadow-sm: 0 2px 10px rgba(0,0,0,0.30), 0 1px 2px rgba(0,0,0,0.35);
      --shadow-md: 0 10px 28px rgba(0,0,0,0.40), 0 2px 6px rgba(0,0,0,0.30);
      --shadow-brass: 0 0 0 3px rgba(204,154,78,0.20);
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration:0.001ms !important; animation-iteration-count:1 !important;
        transition-duration:0.001ms !important; scroll-behavior:auto !important;
      }
    }

    html, body, [class*="css"] { font-family:'Inter', sans-serif; }
    * { scroll-behavior:smooth; }

    .stApp {
      background:
        radial-gradient(1100px 620px at 12% -6%, #1A2033 0%, transparent 60%),
        var(--ink);
      color:var(--paper);
    }

    /* Slim, unobtrusive scrollbars */
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:var(--hairline); border-radius:8px; }
    ::-webkit-scrollbar-thumb:hover { background:var(--brass-dim); }

    /* Center and cap the width of the main content for readability */
    .block-container { max-width:920px; padding-top:2rem; padding-bottom:3rem; }

    /* Readable defaults for native text/markdown */
    p, li, span, label, .stMarkdown { color:var(--paper); }
    ::selection { background:rgba(204,154,78,0.35); color:var(--paper); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
      background:var(--panel); border-right:1px solid var(--hairline);
    }
    section[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p { color:var(--paper); }

    /* Brand mark: an aperture ring rendered in pure CSS (conic-gradient
       blades over an ink core) — the page's one recurring signature. */
    .kairo-brand { display:flex; align-items:center; gap:.75rem;
      padding:0 0 1.1rem; border-bottom:1px solid var(--hairline);
      margin-bottom:1rem; }
    .kairo-brand .kairo-logo {
      display:flex; align-items:center; justify-content:center;
      width:40px; height:40px; border-radius:50%; flex-shrink:0;
      background:
        radial-gradient(circle at center, var(--ink) 0 33%, transparent 34%),
        conic-gradient(from 0deg,
          var(--brass) 0deg 34deg, transparent 34deg 90deg,
          var(--brass) 90deg 124deg, transparent 124deg 180deg,
          var(--brass) 180deg 214deg, transparent 214deg 270deg,
          var(--brass) 270deg 304deg, transparent 304deg 360deg);
      box-shadow:0 0 0 1px var(--hairline), var(--shadow-sm);
      font-size:1.05rem; filter:saturate(1.05);
    }
    .kairo-brand h1 {
      font-family:'Fraunces', serif; font-optical-sizing:auto;
      color:var(--paper); font-size:1.32rem; font-weight:650; margin:0; letter-spacing:-.01em;
    }
    .kairo-subtitle {
      font-family:'JetBrains Mono', monospace; color:var(--mist);
      font-size:.66rem; text-transform:uppercase; letter-spacing:.13em;
      margin:.5rem 0 1.5rem; line-height:1.6;
    }
    .kairo-main-subtext { color:var(--mist); font-size:.98rem; margin:-.2rem 0 1.8rem; line-height:1.6; }

    .kairo-label {
      display:flex; align-items:center; gap:.55rem; color:var(--mist);
      font-family:'JetBrains Mono', monospace;
      font-size:.68rem; font-weight:500; letter-spacing:.13em; margin:1.6rem 0 .7rem;
      text-transform:uppercase;
    }
    .kairo-label::after { content:''; flex:1; height:1px; background:var(--hairline); }

    /* Native bordered containers used as "cards" */
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius:var(--radius-lg) !important; border-color:var(--hairline) !important;
      background:var(--panel-raised) !important; box-shadow:var(--shadow-sm);
      transition:box-shadow .2s var(--ease), border-color .2s var(--ease);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
      box-shadow:var(--shadow-md), var(--shadow-brass); border-color:rgba(204,154,78,0.35) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { padding:.15rem; }

    /* Status badges: a hollow ring stands in for the filled dot — the
       same aperture idea in miniature. "Pending" pulses like autofocus
       hunting for a lock, which is literally what it's waiting on. */
    .kairo-badge {
      display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .8rem;
      border-radius:999px; font-family:'JetBrains Mono', monospace;
      font-size:.7rem; font-weight:500; letter-spacing:.03em;
    }
    .kairo-badge::before {
      content:''; width:7px; height:7px; border-radius:50%;
      border:1.5px solid currentColor; background:transparent; flex-shrink:0;
    }
    .success { color:var(--teal); border:1px solid rgba(79,184,159,0.4); background:rgba(79,184,159,0.10); }
    .locked  { color:var(--rose); border:1px solid rgba(226,104,94,0.4); background:rgba(226,104,94,0.10); }
    .info    { color:var(--brass-bright); border:1px solid rgba(204,154,78,0.4); background:rgba(204,154,78,0.12); }
    .pending {
      color:var(--brass-bright); border:1px solid rgba(204,154,78,0.4); background:rgba(204,154,78,0.12);
    }
    .pending::before { animation:kairo-focus-pulse 1.8s ease-out infinite; }

    @keyframes kairo-focus-pulse {
      0%   { box-shadow:0 0 0 0 rgba(227,179,106,0.55); }
      70%  { box-shadow:0 0 0 7px rgba(227,179,106,0); }
      100% { box-shadow:0 0 0 0 rgba(227,179,106,0); }
    }

    .kairo-status-row { display:flex; justify-content:flex-end; margin-top:.65rem; }

    /* Camera input gets a lens-bezel treatment */
    [data-testid="stCameraInput"] {
      border:1px solid var(--hairline); border-radius:var(--radius-lg);
      padding:.6rem; background:var(--panel); transition:box-shadow .2s var(--ease), border-color .2s var(--ease);
    }
    [data-testid="stCameraInput"]:focus-within {
      border-color:rgba(204,154,78,0.45); box-shadow:var(--shadow-brass);
    }

    .auth-banner { display:flex; gap:.9rem; align-items:flex-start;
      color:#FBDAD6; border:1px solid rgba(226,104,94,0.35);
      border-left:3px solid var(--rose); border-radius:var(--radius-md); padding:1.05rem 1.3rem;
      background:rgba(226,104,94,0.08); box-shadow:var(--shadow-sm); margin-bottom:1.5rem; }
    .auth-banner .icon {
      display:flex; align-items:center; justify-content:center; flex-shrink:0;
      width:34px; height:34px; border-radius:50%; background:rgba(226,104,94,0.18); font-size:1.15rem; line-height:1;
    }
    .auth-banner strong { color:#FFEAE8; font-size:.98rem; }
    .auth-banner > div { font-size:.92rem; line-height:1.6; color:#F3C9C5; }

    /* ---------- Header ---------- */
    .kairo-header { display:flex; align-items:center; gap:.7rem; margin-bottom:.25rem; }
    .kairo-header h1 {
      font-family:'Fraunces', serif; font-optical-sizing:auto;
      font-size:2.05rem; font-weight:650; margin:0; letter-spacing:-.02em; color:var(--paper);
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
      background:linear-gradient(135deg, var(--brass-bright), var(--brass));
      color:#20140A; font-weight:700; border:0;
      border-radius:var(--radius-sm); padding:.55rem 1.1rem; box-shadow:var(--shadow-xs);
      transition:filter .15s var(--ease), transform .1s var(--ease), box-shadow .15s var(--ease);
    }
    .stButton > button:hover { filter:brightness(1.08); box-shadow:var(--shadow-sm); transform:translateY(-1px); }
    .stButton > button:active { transform:scale(0.98) translateY(0); }
    .stButton > button:focus-visible { outline:2px solid var(--brass-bright); outline-offset:2px; }
    .stButton > button:disabled { background:var(--panel-raised); color:#5A6274; box-shadow:none; transform:none; }

    /* ---------- Inputs ---------- */
    input, textarea, [data-baseweb="select"] > div {
      background:var(--panel-raised) !important; color:var(--paper) !important;
      border-color:var(--hairline) !important; border-radius:var(--radius-sm) !important;
      transition:border-color .15s var(--ease), box-shadow .15s var(--ease);
    }
    input::placeholder, textarea::placeholder { color:#6E7789 !important; }
    input:focus, textarea:focus { border-color:var(--brass) !important; box-shadow:var(--shadow-brass) !important; }
    [data-testid="stFileUploaderDropzone"] {
      background:var(--panel) !important; border:1.5px dashed var(--hairline) !important;
      border-radius:var(--radius-md) !important; transition:border-color .15s var(--ease), background .15s var(--ease);
    }
    [data-testid="stFileUploaderDropzone"]:hover {
      border-color:var(--brass) !important; background:var(--panel-raised) !important;
    }
    [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span { color:var(--mist) !important; }
    div[data-testid="stChatInput"] {
      border-top:1px solid var(--hairline); background:var(--ink); padding-top:.75rem;
    }
    div[data-testid="stChatInput"] textarea { border-radius:var(--radius-md) !important; }

    /* Radio as a segmented control */
    div[role="radiogroup"] { gap:.4rem; }
    div[role="radiogroup"] label p { color:var(--paper) !important; }

    /* Chat message bubbles — assistant replies carry a thin brass
       instrument-rule, echoing the same accent used for verification. */
    div[data-testid="stChatMessage"] {
      border-radius:var(--radius-lg); padding:.6rem .85rem; margin-bottom:.4rem;
      animation:kairo-fade-in .25s var(--ease);
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
      background:var(--panel);
    }
    div[data-testid="stChatMessage"]:not(:has([data-testid="chatAvatarIcon-user"])) {
      background:var(--panel-raised); border-left:2px solid var(--brass);
    }
    div[data-testid="stChatMessage"] p { color:var(--paper); }

    @keyframes kairo-fade-in {
      from { opacity:0; transform:translateY(4px); }
      to { opacity:1; transform:translateY(0); }
    }

    hr { border-color:var(--hairline); margin:1.3rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    for key, value in {
        "authenticated": False,
        "messages": [],
        "auth_status": None,
        "rag_status": None,
    }.items():
        st.session_state.setdefault(key, value)


def post_request(url: str, **kwargs: Any) -> tuple[bool, dict[str, Any] | str]:
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
        return False, f"Backend returned HTTP {response.status_code}: {detail or response.text[:300]}"
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to Kairo backend at {API_BASE_URL}."
    except requests.exceptions.Timeout:
        return False, "The request timed out. Please try again."
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}"


def send_chat(prompt: str) -> tuple[bool, str]:
    ok, result = post_request(
        f"{API_BASE_URL}/api/chat", data={"prompt": prompt})
    if not ok:
        return False, str(result)
    return True, str(result.get("response", "")).strip()


def index_pdf(uploaded_file: Any) -> tuple[bool, dict[str, Any] | str]:
    files = {"file": (uploaded_file.name,
                      uploaded_file.getvalue(), "application/pdf")}
    return post_request(
        f"{API_BASE_URL}/api/rag/upload-pdf",
        files=files,
        timeout=120,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="kairo-brand"><div class="kairo-logo">🤖</div>'
            '<h1>Kairo AI Assistant</h1></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="kairo-subtitle">Multimodal intelligence, secured by face.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="kairo-label">🔐 Identity Verification</div>', unsafe_allow_html=True)
        with st.container(border=True):
            auth_mode = st.radio(
                "Authentication mode",
                ["Login", "Signup"],
                horizontal=True,
                label_visibility="collapsed",
            )
            camera_frame = st.camera_input(
                f"Face scan — {auth_mode}", key="kairo_camera")
            if camera_frame is not None:
                st.session_state.auth_status = (
                    f"Frame captured for {auth_mode.lower()}. "
                    "Connect a face-auth endpoint to confirm identity."
                )
            if st.session_state.auth_status:
                st.markdown(
                    '<span class="kairo-badge pending">⏳ Pending Verification</span>',
                    unsafe_allow_html=True,
                )
                st.caption(st.session_state.auth_status)
            status_class = "success" if st.session_state.authenticated else "locked"
            status_text = "✅ Authenticated" if st.session_state.authenticated else "🔒 Not Authenticated"
            st.markdown(
                f'<div class="kairo-status-row"><span class="kairo-badge {status_class}">{status_text}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="kairo-label">📄 Document Ingestion</div>', unsafe_allow_html=True)
        with st.container(border=True):
            pdf_file = st.file_uploader("Upload PDF for RAG indexing", type=[
                                        "pdf"], key="kairo_pdf")
            if st.button("📥 Index Document", use_container_width=True, disabled=pdf_file is None):
                with st.spinner(f"Indexing {pdf_file.name}..."):
                    ok, result = index_pdf(pdf_file)
                    st.session_state.rag_status = {
                        "success": ok, "result": result}
            if st.session_state.rag_status:
                rag = st.session_state.rag_status
                if rag["success"]:
                    data = rag["result"]
                    st.success(
                        f"Indexed — {data.get('chunks_indexed', 'N/A')} chunks")
                    st.caption(
                        f"File: {data.get('filename', pdf_file.name if pdf_file else 'document.pdf')}")
                else:
                    st.error(str(rag["result"]))

        st.divider()
        st.markdown(
            '<div class="kairo-label">⚙️ Developer Controls</div>', unsafe_allow_html=True)
        dev_mode = st.checkbox(
            "⚙️ Dev Mode (Bypass Auth)", key="kairo_dev_mode")
        st.session_state.authenticated = dev_mode
        if dev_mode:
            st.markdown(
                '<span class="kairo-badge info">🛠️ Dev Mode Active</span>',
                unsafe_allow_html=True,
            )


def render_workspace() -> None:
    st.markdown(
        '<div class="kairo-header"><h1>💬 Chat with Kairo</h1></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="kairo-main-subtext">Ask questions about indexed documents or start a conversation.</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.authenticated:
        st.markdown(
            '<div class="auth-banner"><span class="icon">🔒</span>'
            '<div><strong>Verification required.</strong><br/>'
            "Complete webcam verification in the sidebar or enable Dev Mode.</div></div>",
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.expander("🎙️ Voice prompt"):
        audio_file = st.file_uploader(
            "Upload WAV or MP3",
            type=["wav", "mp3"],
            key="kairo_audio",
        )
        if audio_file:
            st.audio(audio_file)
            st.info("Voice transcription needs a dedicated backend endpoint.")

    prompt = st.chat_input("Message Kairo...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Kairo is thinking..."):
                ok, response = send_chat(prompt)
            if ok and response:
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})
            else:
                message = response or "The backend returned an empty response."
                st.error(f"⚠️ {message}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"_Error: {message}_"}
                )


init_state()
render_sidebar()
render_workspace()
