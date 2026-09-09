"""
pytest tests for extract_text_from_pdf().
"""
import pytest
from pathlib import Path
from utils.pdf_reader import extract_text_from_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "sample_resume" / "resume1.pdf"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_resume/resume1.pdf not found")
def test_pdf_extraction_returns_string():
    """PDF extraction should return a non-empty string."""
    with open(SAMPLE_PDF, "rb") as f:
        text = extract_text_from_pdf(f)
    assert isinstance(text, str)
    assert len(text) > 0, "Extracted text should not be empty"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_resume/resume1.pdf not found")
def test_pdf_extraction_contains_text():
    """Extracted text should contain readable characters."""
    with open(SAMPLE_PDF, "rb") as f:
        text = extract_text_from_pdf(f)
    # Should contain at least some alphabetic characters
    assert any(c.isalpha() for c in text), "Extracted text should contain letters"


# Keep original print output for manual inspection
if __name__ == "__main__":
    if SAMPLE_PDF.exists():
        with open(SAMPLE_PDF, "rb") as f:
            print(extract_text_from_pdf(f))
    else:
        print(f"PDF not found at {SAMPLE_PDF}")