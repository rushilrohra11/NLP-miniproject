import logging
import shutil
import tempfile
from os import unlink
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Body

from backend.schemas.note import ProcessResponse
from backend.services.pipeline_service import process_audio_pipeline, process_text_pipeline


logger = logging.getLogger(__name__)


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/process", response_model=ProcessResponse)
def process_audio(file: UploadFile = File(...)) -> ProcessResponse:
    """Process audio file and generate transcription, summary, and SOAP note.
    
    Uses Google Gemini API for transcription (no FFmpeg needed!).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    temp_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        logger.info(f"Processing audio file: {file.filename}")
        result = process_audio_pipeline(temp_path)
        
        return ProcessResponse(
            transcription=result["transcription"],
            transcription_dialogue=result.get("transcription_dialogue"),
            summary=result["summary"],
            soap_note=result["soap_note"],
            transcription_entities=result.get("transcription_entities"),
        )
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if temp_path:
            try:
                unlink(temp_path)
            except OSError:
                pass


@router.post("/process-text", response_model=ProcessResponse)
def process_text(text: str = Body(..., embed=True)) -> ProcessResponse:
    """Process clinical text and generate transcription, summary, and SOAP note."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        logger.info("Processing text input")
        result = process_text_pipeline(text)
        
        return ProcessResponse(
            transcription="",  # No transcription for text input
            transcription_dialogue=None,
            summary=result["summary"],
            soap_note=result["soap_note"],
            transcription_entities=None,
        )
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
