import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from typing import Optional

from src.auth import face_service, firebase_service
from src.auth.auth_helpers import get_current_user, format_auth_response, verify_token
from src.voice import voice_service
from src.rag.parser_service import extract_text_from_pdf, chunk_text
from src.rag.rag_service import RAGService

app = FastAPI(
    title="Kairo AI Assistant API",
    description="Backend API endpoints for authentication, RAG search, and voice handling.",
    version="1.0.0",
)

# Enable CORS for local Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGService()


@app.get("/")
def read_root():
    return {"status": "online", "message": "Kairo Backend API is running!"}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class UserRegistrationRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None


class GoogleAuthRequest(BaseModel):
    """Client sends a Firebase ID token obtained via Google Sign-In on the
    frontend (Firebase JS SDK).  The backend verifies the token and returns
    the authenticated identity — it never handles the raw Google OAuth flow
    itself."""
    id_token: str


class TokenVerifyRequest(BaseModel):
    id_token: str


class UserResponse(BaseModel):
    uid: str
    email: str
    display_name: Optional[str] = None
    email_verified: bool = False


# ---------------------------------------------------------------------------
# Auth endpoints — email / password
# ---------------------------------------------------------------------------


@app.post("/api/auth/signup", status_code=status.HTTP_201_CREATED)
async def signup(body: UserRegistrationRequest):
    """Register a new user with email and password.

    Uses Firebase Admin SDK to create the account server-side.

    Request JSON
    ------------
    email        : valid email address
    password     : ≥ 6 characters (Firebase minimum)
    display_name : optional

    Response JSON
    -------------
    success      : bool
    authenticated: bool
    uid          : Firebase UID on success
    email        : registered email on success
    display_name : display name if provided
    provider     : "password"
    error        : error message on failure
    """
    if not body.password or len(body.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 6 characters.",
        )

    result = firebase_service.create_user(
        email=body.email,
        password=body.password,
        display_name=body.display_name,
    )

    if not result["success"]:
        # Distinguish common error cases for the client
        error_msg = result.get("error", "User creation failed.")
        if "already exists" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    return format_auth_response(
        success=True,
        authenticated=True,
        uid=result["uid"],
        email=result["email"],
        display_name=result.get("display_name"),
        provider="password",
    )


@app.post("/api/auth/login")
async def login(body: TokenVerifyRequest):
    """Verify a Firebase ID token obtained after client-side email/password
    sign-in (via Firebase JS / REST SDK) and return the authenticated identity.

    The backend does **not** accept raw email/password credentials — that
    exchange happens on the client using the Firebase Authentication REST API
    (``signInWithEmailAndPassword``).  The resulting ID token is sent here for
    server-side verification.

    Request JSON
    ------------
    id_token : Firebase ID token string

    Response JSON
    -------------
    success      : bool
    authenticated: bool
    uid          : Firebase UID
    email        : user email
    display_name : display name (if set)
    provider     : authentication provider string
    error        : error message on failure
    """
    if not body.id_token or not body.id_token.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id_token must not be empty.",
        )

    try:
        decoded = firebase_service.verify_id_token(body.id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error.",
        ) from exc

    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired ID token.",
        )

    uid = decoded.get("uid")
    # Fetch full user record for display_name and provider info
    try:
        user = firebase_service.get_user_by_uid(uid)
    except Exception:
        user = None

    provider = "password"
    if user and user.get("provider_data"):
        provider = user["provider_data"][0].get("provider_id", "password")

    return format_auth_response(
        success=True,
        authenticated=True,
        uid=uid,
        email=decoded.get("email"),
        display_name=decoded.get("name") or (user.get("display_name") if user else None),
        provider=provider,
    )


# ---------------------------------------------------------------------------
# Auth endpoint — Google Sign-In
# ---------------------------------------------------------------------------


@app.post("/api/auth/google")
async def google_auth(body: GoogleAuthRequest):
    """Verify a Firebase ID token obtained after Google Sign-In on the client.

    The frontend performs the Google OAuth flow via the Firebase JS SDK
    (``signInWithPopup`` / ``signInWithRedirect``), then sends the resulting
    Firebase ID token to this endpoint.  The backend verifies the token,
    extracts the authenticated identity, and returns it — no raw Google
    credentials are ever handled here.

    Firebase Console prerequisite
    ------------------------------
    Google must be enabled as a Sign-In Provider in:
    Firebase Console → Authentication → Sign-in method → Google

    Request JSON
    ------------
    id_token : Firebase ID token (NOT the raw Google access/ID token)

    Response JSON
    -------------
    success      : bool
    authenticated: bool
    uid          : Firebase UID
    email        : user email
    display_name : user display name
    provider     : "google.com"
    error        : error message on failure
    """
    if not body.id_token or not body.id_token.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id_token must not be empty.",
        )

    try:
        decoded = firebase_service.verify_id_token(body.id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error.",
        ) from exc

    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired ID token.",
        )

    # Confirm the token came from Google
    firebase_identities = decoded.get("firebase", {}).get("identities", {})
    if "google.com" not in firebase_identities:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token was not issued by Google Sign-In.",
        )

    return format_auth_response(
        success=True,
        authenticated=True,
        uid=decoded.get("uid"),
        email=decoded.get("email"),
        display_name=decoded.get("name"),
        provider="google.com",
    )


# ---------------------------------------------------------------------------
# Auth endpoint — token verification / /me
# ---------------------------------------------------------------------------


@app.post("/api/auth/verify-token")
async def verify_token_endpoint(body: TokenVerifyRequest):
    """Verify any Firebase ID token and return the decoded identity.

    Useful for the frontend to validate a stored token is still active.

    Request JSON
    ------------
    id_token : Firebase ID token string

    Response JSON
    -------------
    success      : bool
    authenticated: bool
    uid          : Firebase UID
    email        : user email
    display_name : display name (if available)
    provider     : primary provider
    error        : error message on failure
    """
    if not body.id_token or not body.id_token.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id_token must not be empty.",
        )

    try:
        decoded = firebase_service.verify_id_token(body.id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error.",
        ) from exc

    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired ID token.",
        )

    return format_auth_response(
        success=True,
        authenticated=True,
        uid=decoded.get("uid"),
        email=decoded.get("email"),
        display_name=decoded.get("name"),
        provider=decoded.get("firebase", {}).get("sign_in_provider"),
    )


@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile.

    Requires a valid Firebase ID token in the ``Authorization: Bearer <token>``
    header.

    Response JSON
    -------------
    uid            : Firebase UID
    email          : user email
    display_name   : display name
    email_verified : whether the email is verified
    """
    return current_user


# ---------------------------------------------------------------------------
# Face enrolment / verification endpoints
# ---------------------------------------------------------------------------


@app.post("/api/auth/enrol")
async def enrol_face(file: UploadFile = File(...), user_id: str | None = Form(None)):
    """Enrol a user's face by extracting an embedding and storing it against
    their Firebase UID.

    The ``user_id`` field must be a valid Firebase UID.  The caller is
    responsible for authenticating the user first (e.g. via
    ``/api/auth/login``) and passing the real UID — arbitrary strings are
    accepted here but will only be useful if they match a real Firebase user.

    Form fields
    -----------
    file    : image file (JPEG / PNG / etc.)
    user_id : Firebase UID

    Response JSON
    -------------
    success : bool
    user_id : the UID the embedding was stored against
    error   : error message on failure
    """
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id must not be empty.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    try:
        embedding = face_service.extract_face_embedding(image=image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding extraction failed: {exc}",
        ) from exc

    if embedding is None:
        return {"success": False, "user_id": user_id, "error": "No face detected in the uploaded image."}

    try:
        firebase_service.save_face_embedding(user_id=user_id, embedding=embedding)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save embedding: {exc}",
        ) from exc

    return {"success": True, "user_id": user_id, "error": None}


@app.post("/api/auth/verify-face")
async def verify_face(file: UploadFile = File(...), user_id: str | None = Form(None)):
    """Verify a user's identity by comparing a submitted image against the
    stored face embedding for the given Firebase UID.

    Form fields
    -----------
    file    : image file (JPEG / PNG / etc.)
    user_id : Firebase UID to compare against

    Response JSON
    -------------
    authenticated : bool
    user_id       : the UID that was checked
    error         : error message on failure
    """
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id must not be empty.",
        )

    image_bytes = await file.read()
    return face_service.verify_user_face(image_bytes, user_id)


# ---------------------------------------------------------------------------
# Voice endpoints
# ---------------------------------------------------------------------------


@app.post("/api/voice/transcribe")
async def transcribe_voice(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file and return the text."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty.")

    try:
        text = voice_service.process_voice_input(audio_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc

    return {"text": text}


@app.post("/api/voice/tts")
async def text_to_speech_endpoint(text: str = Form(...)):
    """Convert text to speech and return the MP3 audio file."""
    if not text or not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="text must not be empty.")

    try:
        mp3_path = voice_service.text_to_speech(text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS failed: {exc}",
        ) from exc

    return FileResponse(
        path=mp3_path,
        media_type="audio/mpeg",
        filename="response.mp3",
        background=_cleanup_file(mp3_path),
    )


def _cleanup_file(path: str):
    """Return a BackgroundTask that deletes *path* after the response is sent."""
    from starlette.background import BackgroundTask

    return BackgroundTask(lambda: os.remove(path) if os.path.exists(path) else None)


# ---------------------------------------------------------------------------
# RAG endpoints (Member B stubs — do not modify)
# ---------------------------------------------------------------------------


@app.post("/api/rag/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise HTTPException(
                status_code=400, detail="Failed to extract text from PDF.")

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        rag.index_chunks(chunks, doc_id=file.filename)

        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_query(prompt: str = Form(...)):
    try:
        answer = rag.query(prompt)
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
