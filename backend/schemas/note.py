from typing import Any, Dict

from pydantic import BaseModel, Field


class NoteRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Clinical text or transcript input")


class NoteResponse(BaseModel):
    input_text: str
    generated_note: Dict[str, Any]


class TextRequest(BaseModel):
    input_text: str = Field(..., min_length=1, description="Clinical text input")


class SummaryResponse(BaseModel):
    input_text: str
    summary: str


class TranscriptionResponse(BaseModel):
    transcript: str


class SOAPResponse(BaseModel):
    input_text: str
    soap: Dict[str, Any]
    extracted_entities: Dict[str, Any]


class ProcessResponse(BaseModel):
    transcription: str
    transcription_dialogue: str | None = None
    summary: str
    soap_note: Dict[str, Any]
    transcription_entities: Dict[str, Any] | None = None
