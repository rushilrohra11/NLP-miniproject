from __future__ import annotations

from utils.privacy import sanitize_text


def test_sanitize_text_masks_names_phone_and_ids() -> None:
    text = (
        "Patient John Smith called from 555-123-4567. "
        "MRN: 12345678. Dr. Jane Doe noted the follow-up plan."
    )

    sanitized = sanitize_text(text)

    assert "John Smith" not in sanitized
    assert "Jane Doe" not in sanitized
    assert "555-123-4567" not in sanitized
    assert "12345678" not in sanitized
    assert "[REDACTED_NAME]" in sanitized
    assert "[REDACTED_PHONE]" in sanitized
    assert "[REDACTED_ID]" in sanitized
