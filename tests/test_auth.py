"""
tests/test_auth.py
------------------
Unit + integration tests for all authentication endpoints.
All Firebase Admin SDK and DeepFace calls are mocked — no real credentials,
network access, or GPU required.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so the modules import without real Firebase / CV2 / DeepFace
# ---------------------------------------------------------------------------

# Stub firebase_admin before any project import touches it
_fa_stub = types.ModuleType("firebase_admin")
_fa_stub.get_app = MagicMock(side_effect=ValueError("not initialised"))
_fa_stub.initialize_app = MagicMock()
_fa_credentials = types.ModuleType("firebase_admin.credentials")
_fa_credentials.Certificate = MagicMock(return_value=object())
_fa_auth = types.ModuleType("firebase_admin.auth")
_fa_firestore = types.ModuleType("firebase_admin.firestore")
_fa_stub.credentials = _fa_credentials
_fa_stub.auth = _fa_auth
_fa_stub.firestore = _fa_firestore
sys.modules.setdefault("firebase_admin", _fa_stub)
sys.modules.setdefault("firebase_admin.credentials", _fa_credentials)
sys.modules.setdefault("firebase_admin.auth", _fa_auth)
sys.modules.setdefault("firebase_admin.firestore", _fa_firestore)

# Stub deepface
_df_stub = types.ModuleType("deepface")
_df_stub.DeepFace = MagicMock()
sys.modules.setdefault("deepface", _df_stub)

# Stub cv2
_cv2_stub = types.ModuleType("cv2")
_cv2_stub.VideoCapture = MagicMock()
_cv2_stub.imdecode = MagicMock(return_value=None)
_cv2_stub.IMREAD_COLOR = 1
sys.modules.setdefault("cv2", _cv2_stub)

# Stub numpy
import numpy as _np_real
sys.modules.setdefault("numpy", _np_real)

# Stub faster_whisper
_fw_stub = types.ModuleType("faster_whisper")
_fw_stub.WhisperModel = MagicMock()
sys.modules.setdefault("faster_whisper", _fw_stub)

# Stub gtts
_gtts_pkg = types.ModuleType("gtts")
_gtts_pkg.gTTS = MagicMock()
sys.modules.setdefault("gtts", _gtts_pkg)

# ---------------------------------------------------------------------------
# Now import project modules
# ---------------------------------------------------------------------------
from src.auth import firebase_service, face_service  # noqa: E402
from src.voice import voice_service  # noqa: E402

# ---------------------------------------------------------------------------
# Patch service layers BEFORE importing FastAPI app
# ---------------------------------------------------------------------------

_FAKE_UID = "uid_abc123"
_FAKE_EMAIL = "user@example.com"
_FAKE_DISPLAY = "Test User"
_FAKE_EMBEDDING = [0.1] * 128

# --- firebase_service mocks ---
firebase_service.create_user = MagicMock(return_value={
    "success": True,
    "uid": _FAKE_UID,
    "email": _FAKE_EMAIL,
    "display_name": _FAKE_DISPLAY,
    "error": None,
})

firebase_service.verify_id_token = MagicMock(return_value={
    "uid": _FAKE_UID,
    "email": _FAKE_EMAIL,
    "name": _FAKE_DISPLAY,
    "firebase": {
        "sign_in_provider": "password",
        "identities": {},
    },
})

firebase_service.get_user_by_uid = MagicMock(return_value={
    "uid": _FAKE_UID,
    "email": _FAKE_EMAIL,
    "display_name": _FAKE_DISPLAY,
    "email_verified": True,
    "provider_data": [{"provider_id": "password", "uid": _FAKE_EMAIL, "email": _FAKE_EMAIL}],
})

firebase_service.save_face_embedding = MagicMock()
firebase_service.get_face_embedding = MagicMock(return_value=_FAKE_EMBEDDING)

# --- face_service mocks ---
face_service.extract_face_embedding = MagicMock(return_value=_FAKE_EMBEDDING)
face_service.verify_user_face = MagicMock(return_value={
    "authenticated": True,
    "user_id": _FAKE_UID,
    "error": None,
})

# --- voice_service mocks ---
voice_service.process_voice_input = MagicMock(return_value="hello kairo")
voice_service.text_to_speech = MagicMock()

# --- import app after all patches ---
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)

FAKE_IMAGE = b"\xff\xd8\xff\xe0" + b"\x00" * 100
FAKE_AUDIO = b"RIFF" + b"\x00" * 100


# ===========================================================================
# Helper
# ===========================================================================

def _mp3_tmp() -> str:
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.write(fd, b"\xff\xfb\x90\x00" * 10)
    os.close(fd)
    return path


# ===========================================================================
# Tests
# ===========================================================================

class TestHealthCheck(unittest.TestCase):
    def test_root_200(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("online", r.json()["status"])


class TestSignup(unittest.TestCase):
    def setUp(self):
        firebase_service.create_user.reset_mock()

    def test_signup_success(self):
        firebase_service.create_user.return_value = {
            "success": True, "uid": _FAKE_UID, "email": _FAKE_EMAIL,
            "display_name": _FAKE_DISPLAY, "error": None,
        }
        r = client.post("/api/auth/signup", json={
            "email": _FAKE_EMAIL, "password": "secret123", "display_name": _FAKE_DISPLAY,
        })
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertTrue(body["success"])
        self.assertTrue(body["authenticated"])
        self.assertEqual(body["uid"], _FAKE_UID)
        self.assertEqual(body["provider"], "password")

    def test_signup_short_password(self):
        r = client.post("/api/auth/signup", json={
            "email": _FAKE_EMAIL, "password": "abc",
        })
        self.assertEqual(r.status_code, 422)

    def test_signup_duplicate_email(self):
        firebase_service.create_user.return_value = {
            "success": False, "uid": None, "email": None, "display_name": None,
            "error": "An account with this email already exists.",
        }
        r = client.post("/api/auth/signup", json={
            "email": _FAKE_EMAIL, "password": "secret123",
        })
        self.assertEqual(r.status_code, 409)

    def test_signup_invalid_email(self):
        r = client.post("/api/auth/signup", json={
            "email": "not-an-email", "password": "secret123",
        })
        self.assertEqual(r.status_code, 422)

    def test_signup_firebase_error(self):
        firebase_service.create_user.return_value = {
            "success": False, "uid": None, "email": None, "display_name": None,
            "error": "Firebase quota exceeded.",
        }
        r = client.post("/api/auth/signup", json={
            "email": _FAKE_EMAIL, "password": "secret123",
        })
        self.assertEqual(r.status_code, 400)


class TestLogin(unittest.TestCase):
    def setUp(self):
        firebase_service.verify_id_token.reset_mock()
        firebase_service.get_user_by_uid.reset_mock()

    def test_login_valid_token(self):
        firebase_service.verify_id_token.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL, "name": _FAKE_DISPLAY,
            "firebase": {"sign_in_provider": "password", "identities": {}},
        }
        firebase_service.get_user_by_uid.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL, "display_name": _FAKE_DISPLAY,
            "email_verified": True,
            "provider_data": [{"provider_id": "password", "uid": _FAKE_EMAIL, "email": _FAKE_EMAIL}],
        }
        r = client.post("/api/auth/login", json={"id_token": "valid.jwt.token"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        self.assertTrue(body["authenticated"])
        self.assertEqual(body["uid"], _FAKE_UID)

    def test_login_invalid_token(self):
        firebase_service.verify_id_token.return_value = None
        r = client.post("/api/auth/login", json={"id_token": "bad.token"})
        self.assertEqual(r.status_code, 401)

    def test_login_empty_token(self):
        r = client.post("/api/auth/login", json={"id_token": ""})
        self.assertEqual(r.status_code, 400)

    def test_login_missing_body(self):
        r = client.post("/api/auth/login", json={})
        self.assertEqual(r.status_code, 422)

    def test_login_firebase_exception(self):
        firebase_service.verify_id_token.side_effect = RuntimeError("SDK crash")
        r = client.post("/api/auth/login", json={"id_token": "some.token"})
        self.assertEqual(r.status_code, 500)
        firebase_service.verify_id_token.side_effect = None


class TestGoogleAuth(unittest.TestCase):
    def setUp(self):
        firebase_service.verify_id_token.reset_mock()

    def test_google_auth_success(self):
        firebase_service.verify_id_token.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL, "name": _FAKE_DISPLAY,
            "firebase": {
                "sign_in_provider": "google.com",
                "identities": {"google.com": ["google_sub_123"]},
            },
        }
        r = client.post("/api/auth/google", json={"id_token": "google.firebase.token"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["authenticated"])
        self.assertEqual(body["provider"], "google.com")
        self.assertEqual(body["uid"], _FAKE_UID)

    def test_google_auth_invalid_token(self):
        firebase_service.verify_id_token.return_value = None
        r = client.post("/api/auth/google", json={"id_token": "bad.token"})
        self.assertEqual(r.status_code, 401)

    def test_google_auth_wrong_provider(self):
        # Token is valid but not from Google
        firebase_service.verify_id_token.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL, "name": _FAKE_DISPLAY,
            "firebase": {"sign_in_provider": "password", "identities": {}},
        }
        r = client.post("/api/auth/google", json={"id_token": "password.token"})
        self.assertEqual(r.status_code, 401)

    def test_google_auth_empty_token(self):
        r = client.post("/api/auth/google", json={"id_token": ""})
        self.assertEqual(r.status_code, 400)

    def test_google_auth_firebase_exception(self):
        firebase_service.verify_id_token.side_effect = RuntimeError("SDK crash")
        r = client.post("/api/auth/google", json={"id_token": "some.token"})
        self.assertEqual(r.status_code, 500)
        firebase_service.verify_id_token.side_effect = None


class TestVerifyToken(unittest.TestCase):
    def setUp(self):
        firebase_service.verify_id_token.reset_mock()

    def test_valid_token(self):
        firebase_service.verify_id_token.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL, "name": _FAKE_DISPLAY,
            "firebase": {"sign_in_provider": "password", "identities": {}},
        }
        r = client.post("/api/auth/verify-token", json={"id_token": "valid.token"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["authenticated"])

    def test_invalid_token(self):
        firebase_service.verify_id_token.return_value = None
        r = client.post("/api/auth/verify-token", json={"id_token": "bad"})
        self.assertEqual(r.status_code, 401)

    def test_empty_token(self):
        r = client.post("/api/auth/verify-token", json={"id_token": "  "})
        self.assertEqual(r.status_code, 400)


class TestGetMe(unittest.TestCase):
    def test_me_valid_token(self):
        firebase_service.verify_id_token.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL,
            "firebase": {"sign_in_provider": "password", "identities": {}},
        }
        firebase_service.get_user_by_uid.return_value = {
            "uid": _FAKE_UID, "email": _FAKE_EMAIL,
            "display_name": _FAKE_DISPLAY, "email_verified": True,
            "provider_data": [],
        }
        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer valid.token"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["uid"], _FAKE_UID)

    def test_me_missing_header(self):
        r = client.get("/api/auth/me")
        self.assertEqual(r.status_code, 401)

    def test_me_invalid_token(self):
        firebase_service.verify_id_token.return_value = None
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer bad.token"})
        self.assertEqual(r.status_code, 401)

    def test_me_user_not_found(self):
        firebase_service.verify_id_token.return_value = {"uid": "ghost_uid", "email": "x@y.com"}
        firebase_service.get_user_by_uid.return_value = None
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer ghost.token"})
        self.assertEqual(r.status_code, 404)


class TestFaceEnrol(unittest.TestCase):
    def setUp(self):
        face_service.extract_face_embedding.reset_mock()
        firebase_service.save_face_embedding.reset_mock()

    def test_enrol_success(self):
        face_service.extract_face_embedding.return_value = _FAKE_EMBEDDING
        firebase_service.save_face_embedding.return_value = None
        r = client.post(
            "/api/auth/enrol",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["user_id"], _FAKE_UID)
        firebase_service.save_face_embedding.assert_called_once()

    def test_enrol_no_face_detected(self):
        face_service.extract_face_embedding.return_value = None
        r = client.post(
            "/api/auth/enrol",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["success"])
        self.assertIn("No face", body["error"])

    def test_enrol_empty_user_id(self):
        r = client.post(
            "/api/auth/enrol",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": ""},
        )
        self.assertEqual(r.status_code, 400)

    def test_enrol_save_failure(self):
        face_service.extract_face_embedding.return_value = _FAKE_EMBEDDING
        firebase_service.save_face_embedding.side_effect = RuntimeError("Firestore down")
        r = client.post(
            "/api/auth/enrol",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 500)
        firebase_service.save_face_embedding.side_effect = None

    def test_enrol_extraction_failure(self):
        face_service.extract_face_embedding.side_effect = RuntimeError("DeepFace crash")
        r = client.post(
            "/api/auth/enrol",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 500)
        face_service.extract_face_embedding.side_effect = None


class TestFaceVerify(unittest.TestCase):
    def setUp(self):
        face_service.verify_user_face.reset_mock()

    def test_verify_success(self):
        face_service.verify_user_face.return_value = {
            "authenticated": True, "user_id": _FAKE_UID, "error": None,
        }
        r = client.post(
            "/api/auth/verify-face",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["authenticated"])

    def test_verify_no_match(self):
        face_service.verify_user_face.return_value = {
            "authenticated": False, "user_id": _FAKE_UID,
            "error": "Face did not match stored embedding.",
        }
        r = client.post(
            "/api/auth/verify-face",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["authenticated"])

    def test_verify_no_embedding(self):
        face_service.verify_user_face.return_value = {
            "authenticated": False, "user_id": _FAKE_UID,
            "error": "No face embedding registered for user.",
        }
        r = client.post(
            "/api/auth/verify-face",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": _FAKE_UID},
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.json()["authenticated"])

    def test_verify_empty_user_id(self):
        r = client.post(
            "/api/auth/verify-face",
            files={"file": ("face.jpg", io.BytesIO(FAKE_IMAGE), "image/jpeg")},
            data={"user_id": ""},
        )
        self.assertEqual(r.status_code, 400)


class TestVoiceTranscribe(unittest.TestCase):
    def setUp(self):
        voice_service.process_voice_input.reset_mock()

    def test_transcribe_success(self):
        voice_service.process_voice_input.return_value = "hello kairo"
        r = client.post(
            "/api/voice/transcribe",
            files={"file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "hello kairo")

    def test_transcribe_empty_file(self):
        r = client.post(
            "/api/voice/transcribe",
            files={"file": ("audio.wav", io.BytesIO(b""), "audio/wav")},
        )
        self.assertEqual(r.status_code, 400)

    def test_transcribe_service_error(self):
        voice_service.process_voice_input.side_effect = RuntimeError("Whisper crash")
        r = client.post(
            "/api/voice/transcribe",
            files={"file": ("audio.wav", io.BytesIO(FAKE_AUDIO), "audio/wav")},
        )
        self.assertEqual(r.status_code, 500)
        voice_service.process_voice_input.side_effect = None


class TestVoiceTTS(unittest.TestCase):
    def setUp(self):
        voice_service.text_to_speech.reset_mock()

    def test_tts_success(self):
        tmp = _mp3_tmp()
        voice_service.text_to_speech.return_value = tmp
        r = client.post("/api/voice/tts", data={"text": "hello kairo"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("audio", r.headers.get("content-type", ""))
        if os.path.exists(tmp):
            os.remove(tmp)

    def test_tts_empty_text(self):
        r = client.post("/api/voice/tts", data={"text": "  "})
        self.assertEqual(r.status_code, 400)

    def test_tts_service_error(self):
        voice_service.text_to_speech.side_effect = RuntimeError("gTTS crash")
        r = client.post("/api/voice/tts", data={"text": "hello"})
        self.assertEqual(r.status_code, 500)
        voice_service.text_to_speech.side_effect = None


class TestFirebaseServiceUnit(unittest.TestCase):
    """Unit tests for firebase_service helpers — no real Firebase calls."""

    def test_euclidean_distance_identical(self):
        from src.auth.firebase_service import _euclidean_distance
        d = _euclidean_distance([1.0, 0.0], [1.0, 0.0])
        self.assertAlmostEqual(d, 0.0)

    def test_euclidean_distance_known(self):
        from src.auth.firebase_service import _euclidean_distance
        d = _euclidean_distance([0.0, 0.0], [3.0, 4.0])
        self.assertAlmostEqual(d, 5.0)

    def test_euclidean_distance_length_mismatch(self):
        from src.auth.firebase_service import _euclidean_distance
        with self.assertRaises(ValueError):
            _euclidean_distance([1.0], [1.0, 2.0])

    def test_authenticate_face_no_embedding(self):
        """authenticate_face returns not-authenticated when no embedding stored."""
        firebase_service.get_face_embedding = MagicMock(return_value=None)
        # Call the real function (not mocked at module level for this test)
        from src.auth import firebase_service as fs
        original = fs.get_face_embedding
        fs.get_face_embedding = MagicMock(return_value=None)
        result = fs.authenticate_face([0.1] * 128, "uid_no_embedding")
        self.assertFalse(result["success"])
        self.assertFalse(result["authenticated"])
        fs.get_face_embedding = original

    def test_authenticate_face_match(self):
        from src.auth import firebase_service as fs
        emb = [0.1] * 128
        original = fs.get_face_embedding
        fs.get_face_embedding = MagicMock(return_value=emb)
        result = fs.authenticate_face(emb, "uid_match", threshold=0.4)
        self.assertTrue(result["success"])
        self.assertTrue(result["authenticated"])
        fs.get_face_embedding = original

    def test_authenticate_face_no_match(self):
        from src.auth import firebase_service as fs
        stored = [0.0] * 128
        query  = [1.0] * 128   # large distance
        original = fs.get_face_embedding
        fs.get_face_embedding = MagicMock(return_value=stored)
        result = fs.authenticate_face(query, "uid_no_match", threshold=0.4)
        self.assertTrue(result["success"])
        self.assertFalse(result["authenticated"])
        fs.get_face_embedding = original


class TestInitializeFirebase(unittest.TestCase):
    """Test initialize_firebase error paths without touching real SDK."""

    def test_missing_env_var(self):
        import firebase_admin as fa
        # Simulate no app initialised
        fa.get_app = MagicMock(side_effect=ValueError)
        from src.auth import firebase_service as fs
        original_env = os.environ.pop("FIREBASE_CREDENTIALS_PATH", None)
        try:
            with self.assertRaises(EnvironmentError):
                fs.initialize_firebase()
        finally:
            if original_env:
                os.environ["FIREBASE_CREDENTIALS_PATH"] = original_env

    def test_file_not_found(self):
        import firebase_admin as fa
        fa.get_app = MagicMock(side_effect=ValueError)
        from src.auth import firebase_service as fs
        os.environ["FIREBASE_CREDENTIALS_PATH"] = "/nonexistent/path.json"
        try:
            with self.assertRaises(FileNotFoundError):
                fs.initialize_firebase()
        finally:
            del os.environ["FIREBASE_CREDENTIALS_PATH"]


if __name__ == "__main__":
    unittest.main()
