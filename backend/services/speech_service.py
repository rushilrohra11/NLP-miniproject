"""Speech-to-text helpers built on Google Gemini API.

The main entry point is `speech_to_text(file_path)`, which accepts a local audio
file path and uses Google's Gemini API for transcription. No FFmpeg required!
"""

from __future__ import annotations

import logging
from pathlib import Path

from google.genai import Client
from google.genai.types import Part

from backend.core.config import settings


logger = logging.getLogger(__name__)


def speech_to_text(file_path: str) -> str:
    """Transcribe an audio file into text using Google Gemini API.

    Args:
        file_path: Path to an audio file (wav, mp3, m4a, flac, ogg, etc.).

    Returns:
        Clean transcribed text.

    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        ValueError: If GEMINI_API_KEY is not configured.
    """
    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY not configured. Set it via environment variable or .env file. "
            "Get your free API key from: https://aistudio.google.com/app/apikey"
        )

    try:
        client = Client(api_key=settings.gemini_api_key)
        
        logger.info(f"Processing audio file with Gemini: {file_path}")
        
        # Determine MIME type based on file extension
        suffix = audio_path.suffix.lower()
        mime_type_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        mime_type = mime_type_map.get(suffix, "audio/mpeg")
        
        # Read audio file and create Part object
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        # Create Part with correct structure
        audio_part = Part.from_bytes(
            data=audio_data,
            mime_type=mime_type,
        )
        
        # Generate response with audio transcription request
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Please transcribe this audio file. Return only the transcribed text with no additional commentary or formatting.",
                audio_part,
            ],
        )
        
        transcript = response.text.strip()
        logger.info("Transcription complete")
        
        return transcript
    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise
