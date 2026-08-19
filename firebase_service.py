"""
firebase_service.py
-------------------
Firebase initialisation layer for the Kairo chatbot.

Reads the service-account credential path from the environment variable
``FIREBASE_CREDENTIALS_PATH`` and provides a single Firestore client
accessor.  No credentials, project IDs, or file paths are hardcoded here.

Public API
----------
initialize_firebase() -> None
get_firestore_client() -> google.cloud.firestore.Client
"""

from __future__ import annotations

import logging
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
