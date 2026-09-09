"""
pytest tests for TF-IDF similarity (calculate_similarity).
"""
import pytest
from utils.preprocessing import preprocess_text
from utils.similarity import calculate_similarity

RESUME = """
Python SQL Machine Learning Git Docker
Worked on Machine Learning projects using Python.
"""

JD = """
Looking for a Python Developer.
Required Skills: Python SQL Machine Learning Git Docker
"""


def test_similarity_high_for_matching_content():
    """Closely matching resume and JD should give a high TF-IDF similarity score."""
    score = calculate_similarity(preprocess_text(RESUME), preprocess_text(JD))
    assert isinstance(score, float)
    assert 0 <= score <= 100
    assert score > 50, f"Expected > 50 for matching content, got {score}"


def test_similarity_low_for_unrelated_content():
    """Unrelated texts should give a low TF-IDF similarity score."""
    resume = "Pastry chef experienced in baking croissants and sourdough bread."
    jd = "Senior Kubernetes DevOps engineer with Terraform and Go experience."
    score = calculate_similarity(preprocess_text(resume), preprocess_text(jd))
    assert 0 <= score <= 100
    assert score < 50, f"Expected < 50 for unrelated content, got {score}"


def test_similarity_returns_float():
    """calculate_similarity must always return a float."""
    score = calculate_similarity("python developer", "python engineer")
    assert isinstance(score, float)


# Keep original print-based output for manual inspection
if __name__ == "__main__":
    clean_resume = preprocess_text(RESUME)
    clean_jd = preprocess_text(JD)
    score = calculate_similarity(clean_resume, clean_jd)
    print(f"Resume Match Score: {score}%")