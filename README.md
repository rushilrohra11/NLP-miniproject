# Clinical NLP System for Automated Medical Note Generation

Beginner-friendly starter project for building a clinical NLP system that can accept speech or text, transcribe audio with Whisper, process the text with HuggingFace Transformers, and generate structured medical notes.

## Project Title

Clinical NLP System for Automated Medical Note Generation

## Problem Statement

Clinical documentation is time-consuming and often repetitive. This project builds a lightweight clinical NLP pipeline that accepts audio or text, removes sensitive information, converts speech to text when needed, summarizes the content, and generates structured SOAP notes for faster medical documentation.

## Features

- Audio support with Google Gemini AI (no FFmpeg needed!)
- Text input for clinical notes and doctor-patient conversations
- Text sanitization to mask patient names, phone numbers, and IDs
- Clinical text summarization with HuggingFace Transformers
- SOAP note generation with rule-based and NLP-driven extraction
- Summarization quality evaluation with ROUGE and BLEU
- FastAPI backend with async audio processing
- Browser-based frontend with tab-based input selection
- Standalone command-line pipeline runner

## Tech Stack

- Python
- FastAPI
- Google Gemini API (for audio transcription)
- HuggingFace Transformers (for text summarization)
- spaCy (for NLP and entity extraction)
- NLTK (for tokenization)
- rouge-score (for evaluation)
- HTML, CSS, JavaScript (for frontend)

## Installation Steps

1. Clone or open the project in VS Code.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up Gemini API key (required for audio transcription):
   - Get your free API key from: https://aistudio.google.com/app/apikey
   - Create a `.env` file in the project root:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```
   - Or set it as an environment variable:
   ```bash
   $env:GEMINI_API_KEY="your_api_key_here"  # PowerShell
   set GEMINI_API_KEY=your_api_key_here     # CMD
   export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
   ```

5. (Optional) If you want spaCy entity detection to work better, install the English model:

```bash
python -m spacy download en_core_web_sm
```

## How to Run

### Run the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Run the frontend

Open `frontend/index.html` in a browser, or serve the folder with a local static server (e.g., Live Server extension in VS Code).

Then navigate to `http://127.0.0.1:5500` (or your local server port) and choose to either:
- Upload an audio file (WAV, MP3, M4A, FLAC, OGG) - uses Google Gemini AI for transcription
- Enter clinical text directly

### Run the full pipeline from the command line

Process raw text:

```bash
python run_pipeline.py --text "Patient reports fever, cough, and fatigue for two days."
```

Process text and save the output to a file:

```bash
python run_pipeline.py --text "Patient has been experiencing chest pain and shortness of breath." --output outputs\pipeline_result.json
```

### Run tests

```bash
pytest -q
```

## Sample Output

```json
{
   "input_type": "audio",
   "input_path": "data\\raw\\sample.wav",
   "sanitized_text": "Patient reports fever and cough. Taking ibuprofen.",
   "transcription": "Patient reports fever and cough. Taking ibuprofen.",
   "summary": "Patient reports fever and cough. Taking ibuprofen.",
   "soap_note": {
      "SOAP": {
         "Subjective": "Patient reports fever and cough.",
         "Objective": "Vitals stable.",
         "Assessment": "Suspected upper respiratory infection.",
         "Plan": "Continue ibuprofen and monitor symptoms."
      },
      "ExtractedEntities": {
         "Symptoms": ["fever", "cough"],
         "Medicines": ["ibuprofen"],
         "Conditions": ["infection"]
      }
   }
}
```

## Project Structure

- `backend/` - FastAPI app, API routes, schemas, and services
- `frontend/` - Simple HTML, CSS, and JavaScript UI
- `models/` - Saved models and downloaded artifacts
- `data/` - Raw and processed data
- `utils/` - Shared helper functions
- `outputs/` - Generated files and pipeline results
- `tests/` - Pytest-based validation and sample fixtures

## Notes

- The pipeline is designed to be easy to extend with clinical templates and domain-specific rules.
- Privacy masking is applied before summarization and SOAP generation.
- The output format is JSON-friendly for integration with APIs and downstream systems.

## Project Structure

- `backend/` - FastAPI app, API routes, schemas, and services
- `frontend/` - Simple HTML, CSS, and JavaScript UI
- `models/` - Saved models, checkpoints, and downloaded artifacts
- `data/` - Raw and processed data
- `utils/` - Shared helper functions
- `outputs/` - Generated notes, logs, and export files

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
uvicorn backend.main:app --reload
```

4. Open the frontend:
   - Open `frontend/index.html` in your browser, or serve it with a simple local server.

## Install and Run

If you are starting from scratch, these are the main commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Then open `frontend/index.html` in a browser.

## Run Tests

Use pytest to run the clinical NLP test cases:

```bash
pytest -q
```

The tests include:

- Sample doctor-patient conversation text
- A generated sample audio file fixture
- Expected summary output
- Expected SOAP format

## Summarization Evaluation

You can evaluate a generated summary against a reference summary with ROUGE and BLEU:

```python
from utils.evaluation import evaluate_summarization, explain_evaluation_results

results = evaluate_summarization(reference_summary, generated_summary)
print(explain_evaluation_results(results))
```

How to read the scores:

- Higher ROUGE means the generated summary shares more content with the reference.
- Higher BLEU means the generated summary uses phrasing closer to the reference.
- The interpretation field gives a plain-English quality check.

## Privacy Layer

Clinical text is sanitized before summarization and SOAP generation.

```python
from utils.privacy import sanitize_text

sanitized = sanitize_text(raw_text)
```

It masks patient names, phone numbers, and ID-like values using regex plus spaCy-based name detection.

## Run the Full Pipeline Script

Use the standalone runner to process either raw text or an audio file:

```bash
python run_pipeline.py --text "Patient has fever and cough for two days."
python run_pipeline.py --audio data\raw\sample.wav --output outputs\pipeline_result.json
```

The script prints JSON with sanitized text, transcription when applicable, summary, and SOAP note.

## What the Starter Includes

- FastAPI backend with a health endpoint
- Placeholder NLP service for HuggingFace model integration
- Placeholder Whisper transcription service
- Simple frontend form for text or audio workflow
- Organized folders for data, models, and outputs

## Suggested Next Steps

- Connect a real HuggingFace text-generation or summarization model
- Add Whisper audio upload and transcription handling
- Save generated notes into `outputs/`
- Add validation and clinical note templates
