"""
face_service.py
---------------
Vision component: webcam frame capture and face embedding extraction.

Public API
----------
capture_frame_from_webcam(camera_index, warmup_frames) -> np.ndarray | None
extract_face_embedding(image, model_name)              -> list[float] | None
get_face_embedding_from_webcam(camera_index, model_name) -> dict
verify_user_face(image_bytes, user_id)                 -> dict
"""

from __future__ import annotations

import logging
from typing import Union

import cv2
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def capture_frame_from_webcam(
    camera_index: int = 0,
    warmup_frames: int = 5,
) -> np.ndarray | None:
    """Capture a single BGR frame from the webcam.

    Opens the camera identified by *camera_index*, discards *warmup_frames*
    frames so that auto-exposure / auto-focus can settle, then captures and
    returns one usable frame.  The camera is always released, even when an
    error occurs.

    Parameters
    ----------
    camera_index:
        Index of the video capture device (default ``0`` for the primary
        webcam).
    warmup_frames:
        Number of frames to read and discard before capturing the real frame.
        Defaults to ``5``.

    Returns
    -------
    np.ndarray | None
        A BGR image array on success, or ``None`` when the camera cannot be
        opened or a frame cannot be captured.
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            logger.warning(
                "capture_frame_from_webcam: could not open camera index %d.",
                camera_index,
            )
            return None

        # Discard warmup frames so exposure/focus can stabilise.
        for _ in range(warmup_frames):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(
                "capture_frame_from_webcam: failed to read a frame from camera index %d.",
                camera_index,
            )
            return None

        return frame
    finally:
        cap.release()


def extract_face_embedding(
    image: Union[np.ndarray, str, bytes],
    model_name: str = "Facenet",
) -> list[float] | None:
    """Extract a face embedding vector from an image.

    Uses ``DeepFace.represent()`` with the OpenCV detector backend.

    Parameters
    ----------
    image:
        A BGR NumPy / OpenCV image array, raw image bytes, **or** a file path
        accepted by DeepFace (string).
    model_name:
        The DeepFace recognition model to use.  Defaults to ``"Facenet"``.

    Returns
    -------
    list[float] | None
        The embedding as a plain Python list of floats when exactly one face
        is found, ``None`` when no face is detected.

    Raises
    ------
    Exception
        Any unexpected failure (model loading, TensorFlow runtime error, etc.)
        is **not** swallowed — it propagates to the caller so that problems
        remain visible during development and production debugging.
    """
    if isinstance(image, bytes):
        nparr = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.debug(
                "extract_face_embedding: could not decode raw image bytes.")
            return None

    try:
        results = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend="opencv",
            enforce_detection=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError when no face is found and
        # enforce_detection=True.  Treat this as an expected condition.
        no_face_phrases = ("face could not be detected", "no face detected")
        message = str(exc).lower()
        if any(phrase in message for phrase in no_face_phrases):
            logger.debug("extract_face_embedding: no face detected — %s", exc)
            return None
        # A ValueError with an unrecognised message is unexpected; re-raise.
        raise

    if not results:
        # represent() returned an empty list — no embedding available.
        logger.debug(
            "extract_face_embedding: DeepFace.represent returned empty list.")
        return None

    # results is a list of dicts; take the first detected face.
    embedding: list[float] = results[0]["embedding"]
    return embedding


def get_face_embedding_from_webcam(
    camera_index: int = 0,
    model_name: str = "Facenet",
) -> dict:
    """Capture a webcam frame and return a face embedding.

    Combines :func:`capture_frame_from_webcam` and
    :func:`extract_face_embedding` into a single convenience call.

    Parameters
    ----------
    camera_index:
        Index of the video capture device (default ``0``).
    model_name:
        DeepFace recognition model (default ``"Facenet"``).

    Returns
    -------
    dict
        Always returns a dict with the following keys:

        ``success`` (*bool*)
            ``True`` when an embedding was successfully extracted.
        ``embedding`` (*list[float] | None*)
            The face embedding, or ``None`` on failure.
        ``error`` (*str | None*)
            Human-readable error description, or ``None`` on success.
    """
    frame = capture_frame_from_webcam(camera_index=camera_index)
    if frame is None:
        return {
            "success": False,
            "embedding": None,
            "error": f"Could not capture a frame from camera index {camera_index}.",
        }

    try:
        embedding = extract_face_embedding(image=frame, model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "get_face_embedding_from_webcam: unexpected error during embedding extraction."
        )
        return {
            "success": False,
            "embedding": None,
            "error": f"Embedding extraction failed: {exc}",
        }

    if embedding is None:
        return {
            "success": False,
            "embedding": None,
            "error": "No face detected in the captured frame.",
        }

    return {
        "success": True,
        "embedding": embedding,
        "error": None,
    }


def verify_user_face(image_bytes: bytes, user_id: str) -> dict:
    """Verify a user's identity from raw image bytes.

    Decodes *image_bytes* into a NumPy array, extracts a face embedding
    with :func:`extract_face_embedding`, then calls
    ``firebase_service.authenticate_face`` to compare the
    embedding against the stored embedding for *user_id* in Firestore.

    Parameters
    ----------
    image_bytes:
        Raw bytes of a JPEG, PNG, or other image format supported by
        OpenCV's ``imdecode``.
    user_id:
        The Firestore document ID of the user to authenticate against.

    Returns
    -------
    dict
        Always contains exactly these keys:

        ``authenticated`` (*bool*)
            ``True`` when the face embedding matched the stored embedding.
        ``user_id`` (*str*)
            The *user_id* that was checked.
        ``error`` (*str | None*)
            Human-readable explanation on failure, ``None`` on success.
    """
    # Safe import handling for both modular and flat project structure
    try:
        from . import firebase_service
    except ImportError:
        import firebase_service

    if not user_id or not user_id.strip():
        return {
            "authenticated": False,
            "user_id": user_id,
            "error": "user_id must not be empty.",
        }

    # Decode bytes -> NumPy BGR image.
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "authenticated": False,
            "user_id": user_id,
            "error": "Could not decode image bytes into a valid image.",
        }

    # Extract the face embedding.
    try:
        embedding = extract_face_embedding(image=image)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "verify_user_face: unexpected error during embedding extraction.")
        return {
            "authenticated": False,
            "user_id": user_id,
            "error": f"Embedding extraction failed: {exc}",
        }

    if embedding is None:
        return {
            "authenticated": False,
            "user_id": user_id,
            "error": "No face detected in the supplied image.",
        }

    # Authenticate against Firestore.
    try:
        result = firebase_service.authenticate_face(
            embedding=embedding,
            user_id=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "verify_user_face: unexpected error during Firestore authentication.")
        return {
            "authenticated": False,
            "user_id": user_id,
            "error": f"Authentication failed: {exc}",
        }

    return {
        "authenticated": result.get("authenticated", False),
        "user_id": user_id,
        "error": result.get("error"),
    }
