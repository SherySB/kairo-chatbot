"""
voice_service.py
----------------
Voice component: speech-to-text transcription and text-to-speech synthesis.

Uses faster-whisper for local, offline STT and gTTS for TTS.  No API keys
are required; gTTS makes outbound HTTP requests to Google's TTS endpoint
without requiring authentication credentials.

Public API
----------
transcribe_audio(audio_file_path)   -> str
text_to_speech(text_response)       -> str
process_voice_input(audio_bytes)    -> str
"""

from __future__ import annotations

import logging
import os
import tempfile

from faster_whisper import WhisperModel
from gtts import gTTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whisper model configuration
#
# Model size is read from the environment so it can be overridden without
# touching source code.  Defaults to "base" which balances speed and
# accuracy on CPU.  Valid values: tiny, base, small, medium, large-v2, etc.
# ---------------------------------------------------------------------------
_WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
_WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
_WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Module-level singleton — loaded once on first use.
_whisper_model: WhisperModel | None = None


def _get_whisper_model() -> WhisperModel:
    """Return the shared WhisperModel instance, loading it on first call."""
    global _whisper_model  # noqa: PLW0603
    if _whisper_model is None:
        logger.info(
            "_get_whisper_model: loading Whisper '%s' on %s (%s).",
            _WHISPER_MODEL_SIZE,
            _WHISPER_DEVICE,
            _WHISPER_COMPUTE,
        )
        _whisper_model = WhisperModel(
            _WHISPER_MODEL_SIZE,
            device=_WHISPER_DEVICE,
            compute_type=_WHISPER_COMPUTE,
        )
    return _whisper_model


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe a local audio file to text using faster-whisper.

    Parameters
    ----------
    audio_file_path:
        Path to an audio file (WAV, MP3, FLAC, etc.) supported by
        faster-whisper / ffmpeg.

    Returns
    -------
    str
        The transcribed text.  Returns an empty string when the audio
        contains no recognisable speech.

    Raises
    ------
    FileNotFoundError
        If *audio_file_path* does not exist.
    Exception
        Any unexpected faster-whisper or ffmpeg error propagates to the
        caller.
    """
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(
            f"Audio file not found: '{audio_file_path}'."
        )

    model = _get_whisper_model()
    segments, _info = model.transcribe(audio_file_path, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    logger.debug("transcribe_audio: transcribed %d chars.", len(text))
    return text


def text_to_speech(text_response: str) -> str:
    """Convert text to speech and save the result as a temporary MP3 file.

    Uses gTTS (Google Text-to-Speech).  The caller is responsible for
    playing or further processing the returned file path, and for deleting
    the file when it is no longer needed.

    Parameters
    ----------
    text_response:
        The text to synthesise.

    Returns
    -------
    str
        Absolute path to the generated MP3 file.

    Raises
    ------
    ValueError
        If *text_response* is empty or contains only whitespace.
    Exception
        Any gTTS or I/O error propagates to the caller.
    """
    if not text_response or not text_response.strip():
        raise ValueError("text_response must not be empty.")

    tts = gTTS(text=text_response.strip(), lang="en")

    # Write to a named temporary file so the caller has a stable path.
    # delete=False is required on Windows; the caller owns the file.
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3", prefix="kairo_tts_")
    os.close(fd)  # close the OS-level file descriptor before gTTS writes

    try:
        tts.save(tmp_path)
    except Exception:
        # Clean up the empty temp file if saving failed.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    logger.debug("text_to_speech: saved MP3 to '%s'.", tmp_path)
    return tmp_path


def process_voice_input(audio_bytes: bytes) -> str:
    """Accept raw audio bytes, transcribe them, and return the text.

    Writes *audio_bytes* to a temporary file, calls
    :func:`transcribe_audio`, and guarantees cleanup of the temporary file
    regardless of success or failure.

    Parameters
    ----------
    audio_bytes:
        Raw bytes of an audio file (WAV, MP3, etc.).

    Returns
    -------
    str
        The transcribed text, or an empty string when no speech is detected.

    Raises
    ------
    ValueError
        If *audio_bytes* is empty.
    Exception
        Any transcription error propagates to the caller.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes must not be empty.")

    # Use a neutral extension so we do not misrepresent the format.
    # faster-whisper detects the actual codec from the file content.
    fd, tmp_path = tempfile.mkstemp(suffix=".audio", prefix="kairo_voice_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(audio_bytes)
        return transcribe_audio(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning(
                "process_voice_input: could not remove temp file '%s'.", tmp_path
            )
