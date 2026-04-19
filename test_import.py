#!/usr/bin/env python
"""Quick test to verify speech_service imports correctly."""

try:
    from backend.services.speech_service import speech_to_text
    print("SUCCESS: speech_service imports correctly")
except Exception as e:
    print(f"ERROR: {e}")
