import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Get the project root directory (parent of backend)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseModel):
    app_name: str = "Clinical NLP System"
    environment: str = os.getenv("ENVIRONMENT", "development")
    hf_model_name: str = "google/flan-t5-small"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_transcription_model: str = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-2.5-flash")


settings = Settings()
