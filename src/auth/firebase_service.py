"""
firebase_service.py
-------------------
Firebase initialisation and Firestore persistence layer for the Kairo chatbot.

Reads the service-account credential path from the environment variable
``FIREBASE_CREDENTIALS_PATH`` and provides a Firestore client accessor,
face-embedding persistence, and similarity-based face authentication.
No credentials, project IDs, or file paths are hardcoded here.

Public API
----------
initialize_firebase()                               -> None
get_firestore_client()                              -> google.cloud.firestore.Client
save_face_embedding(user_id, embedding)             -> None
get_face_embedding(user_id)                         -> list[float] | None
delete_face_embedding(user_id)                      -> None
authenticate_face(embedding, user_id, threshold)    -> dict
"""

from __future__ import annotations

import logging
import math
import os

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

# Name of the environment variable that must point to the
# Firebase service-account JSON file.
_ENV_VAR = "FIREBASE_CREDENTIALS_PATH"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def initialize_firebase() -> None:
    """Initialise the Firebase Admin SDK from the environment.

    Reads the service-account JSON path from the environment variable
    ``FIREBASE_CREDENTIALS_PATH``, validates that the variable is set and
    that the file exists, then initialises the Firebase Admin SDK.

    If the default Firebase app has already been initialised this call is a
    no-op, making it safe to call multiple times across modules.

    Raises
    ------
    EnvironmentError
        If ``FIREBASE_CREDENTIALS_PATH`` is not set or is an empty string.
    FileNotFoundError
        If the path in ``FIREBASE_CREDENTIALS_PATH`` does not point to an
        existing file.
    Exception
        Any error raised by the Firebase SDK (malformed JSON, invalid
        credentials, etc.) propagates to the caller unchanged.
    """
    # Check for an existing default app first — cheap and avoids reading the
    # file system unnecessarily on repeated calls.
    try:
        firebase_admin.get_app()
        logger.debug("initialize_firebase: Firebase already initialised, skipping.")
        return
    except ValueError:
        # ValueError means no app has been initialised yet — proceed.
        pass

    # Validate the environment variable.
    cred_path = os.environ.get(_ENV_VAR, "").strip()
    if not cred_path:
        raise EnvironmentError(
            f"Environment variable '{_ENV_VAR}' is not set or is empty. "
            "Set it to the path of your Firebase service-account JSON file."
        )

    # Validate that the file actually exists before handing it to the SDK.
    if not os.path.isfile(cred_path):
        raise FileNotFoundError(
            f"Firebase service-account file not found at '{cred_path}' "
            f"(from environment variable '{_ENV_VAR}')."
        )

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    logger.info("initialize_firebase: Firebase initialised successfully.")


def get_firestore_client():
    """Return the Firestore client, initialising Firebase first if needed.

    Calls :func:`initialize_firebase` to ensure the Admin SDK is ready,
    then returns the default Firestore client instance.

    Returns
    -------
    google.cloud.firestore.Client
        The Firestore client for the initialised Firebase project.

    Raises
    ------
    EnvironmentError
        If ``FIREBASE_CREDENTIALS_PATH`` is not set (propagated from
        :func:`initialize_firebase`).
    FileNotFoundError
        If the credential file does not exist (propagated from
        :func:`initialize_firebase`).
    Exception
        Any Firebase or Firestore SDK error propagates to the caller.
    """
    initialize_firebase()
    return firestore.client()


# ---------------------------------------------------------------------------
# Firestore constants
# ---------------------------------------------------------------------------

# Collection that holds per-user face embeddings.
_COLLECTION = "face_embeddings"

# Document field that stores the embedding vector.
_EMBEDDING_FIELD = "embedding"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Return the Euclidean (L2) distance between two equal-length vectors.

    Parameters
    ----------
    a, b:
        Plain Python lists of floats of identical length.

    Returns
    -------
    float
        L2 distance between *a* and *b*.

    Raises
    ------
    ValueError
        If *a* and *b* have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding length mismatch: {len(a)} vs {len(b)}."
        )
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Firestore persistence functions
# ---------------------------------------------------------------------------


def save_face_embedding(user_id: str, embedding: list[float]) -> None:
    """Persist a face embedding in Firestore.

    Creates or overwrites the document for *user_id* in the
    ``face_embeddings`` collection.  Only the embedding vector is stored —
    no images or temporary files are written.

    Parameters
    ----------
    user_id:
        Unique identifier for the user (used as the Firestore document ID).
    embedding:
        The face embedding as a plain Python list of floats.

    Raises
    ------
    Exception
        Any Firestore error propagates to the caller.
    """
    db = get_firestore_client()
    db.collection(_COLLECTION).document(user_id).set({_EMBEDDING_FIELD: embedding})
    logger.info("save_face_embedding: embedding saved for user '%s'.", user_id)


def get_face_embedding(user_id: str) -> list[float] | None:
    """Retrieve a stored face embedding from Firestore.

    Parameters
    ----------
    user_id:
        Unique identifier for the user.

    Returns
    -------
    list[float] | None
        The stored embedding as a plain Python list of floats, or ``None``
        when no document exists for *user_id* or the document contains no
        embedding field.

    Raises
    ------
    Exception
        Any Firestore error propagates to the caller.
    """
    db = get_firestore_client()
    doc = db.collection(_COLLECTION).document(user_id).get()

    if not doc.exists:
        logger.debug("get_face_embedding: no document found for user '%s'.", user_id)
        return None

    data = doc.to_dict() or {}
    raw = data.get(_EMBEDDING_FIELD)

    if raw is None:
        logger.warning(
            "get_face_embedding: document for user '%s' exists but has no "
            "embedding field.",
            user_id,
        )
        return None

    return list(raw)


def delete_face_embedding(user_id: str) -> None:
    """Delete a user's stored face embedding from Firestore.

    Completes silently when no document exists for *user_id*; the Firestore
    SDK delete operation is a no-op in that case.

    Parameters
    ----------
    user_id:
        Unique identifier for the user whose embedding should be removed.

    Raises
    ------
    Exception
        Any Firestore error propagates to the caller.
    """
    db = get_firestore_client()
    db.collection(_COLLECTION).document(user_id).delete()
    logger.info("delete_face_embedding: embedding deleted for user '%s'.", user_id)


def authenticate_face(
    embedding: list[float],
    user_id: str,
    threshold: float = 0.4,
) -> dict:
    """Compare a query embedding against the stored embedding for *user_id*.

    Computes the Euclidean (L2) distance between *embedding* and the stored
    vector.  A distance strictly below *threshold* is treated as a match.

    The default threshold of ``0.4`` is appropriate for Facenet 128-d
    embeddings; adjust if a different model is used.

    Parameters
    ----------
    embedding:
        The query face embedding produced by
        ``face_service.extract_face_embedding``.
    user_id:
        Identifier of the user to authenticate against.
    threshold:
        Maximum L2 distance to accept as a match.  Defaults to ``0.4``.

    Returns
    -------
    dict
        Always contains exactly these keys:

        ``success`` (*bool*)
            ``True`` when the comparison was performed without error.
        ``authenticated`` (*bool*)
            ``True`` when the distance is strictly below *threshold*.
        ``error`` (*str | None*)
            Human-readable explanation when ``success`` is ``False``,
            otherwise ``None``.
    """
    stored = get_face_embedding(user_id)

    if stored is None:
        return {
            "success": False,
            "authenticated": False,
            "error": (
                f"No face embedding registered for user '{user_id}'. "
                "Please enrol before attempting authentication."
            ),
        }

    try:
        distance = _euclidean_distance(embedding, stored)
    except ValueError as exc:
        return {
            "success": False,
            "authenticated": False,
            "error": str(exc),
        }

    authenticated = distance < threshold
    logger.debug(
        "authenticate_face: user='%s' distance=%.4f threshold=%.4f authenticated=%s",
        user_id,
        distance,
        threshold,
        authenticated,
    )

    return {
        "success": True,
        "authenticated": authenticated,
        "error": None,
    }
