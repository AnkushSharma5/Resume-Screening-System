"""
pytest tests for preprocess_text().
"""
import pytest
from utils.preprocessing import preprocess_text


def test_lowercase():
    """Output should be all lowercase."""
    result = preprocess_text("Hello World PYTHON")
    assert result == result.lower()


def test_removes_punctuation():
    """Punctuation should be stripped."""
    result = preprocess_text("Hello!! My name is Ankush.")
    assert "!" not in result
    assert "." not in result


def test_removes_digits():
    """Digits should be removed."""
    result = preprocess_text("I have 5 years of experience in 2025.")
    assert not any(char.isdigit() for char in result)


def test_removes_stopwords():
    """Common stopwords like 'my', 'is', 'in' should be removed."""
    result = preprocess_text("My name is John and I am a developer")
    tokens = result.split()
    for sw in ["my", "is", "and", "i", "a"]:
        assert sw not in tokens, f"Stopword '{sw}' should be removed"


def test_output_is_string():
    """Output must be a string."""
    result = preprocess_text("Sample text for testing")
    assert isinstance(result, str)


# Keep original print output for manual inspection
if __name__ == "__main__":
    sample_text = """
    Hello!! My Name is Ankush Sharma.
    I have completed 5 Machine Learning Projects in 2025.
    """
    clean_text = preprocess_text(sample_text)
    print("Original Text:\n")
    print(sample_text)
    print("\n" + "=" * 50 + "\n")
    print("Cleaned Text:\n")
    print(clean_text)