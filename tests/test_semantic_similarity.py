"""
pytest tests for the semantic similarity function.

Two cases:
  1. Clearly related texts → score should be high (> 50)
  2. Clearly unrelated texts → score should be low (< 50)

The sentence-transformer model (all-MiniLM-L6-v2) is loaded once at module
import time; subsequent test calls are fast.
"""

import pytest
from utils.similarity import calculate_semantic_similarity


RELATED_RESUME = """
Experienced Python developer with 3 years of hands-on machine learning and
data science experience. Proficient in scikit-learn, TensorFlow, pandas,
and SQL. Worked on NLP projects including text classification and named
entity recognition. Strong background in data preprocessing and model evaluation.
"""

RELATED_JD = """
We are looking for a Machine Learning Engineer with strong Python skills.
The ideal candidate has experience with scikit-learn or TensorFlow, data
preprocessing pipelines, and NLP tasks such as text classification.
Familiarity with SQL and model evaluation is required.
"""

UNRELATED_RESUME = """
Certified pastry chef with 10 years of experience in high-end bakeries.
Specialised in French patisserie, bread baking, and sugar art. Managed
kitchen inventory, trained junior chefs, and created seasonal dessert menus.
"""

UNRELATED_JD = """
We are hiring a senior software engineer to build distributed cloud
infrastructure. Requirements: Kubernetes, Terraform, Go or Rust, CI/CD
pipelines, microservices architecture, and AWS or GCP certification.
"""


def test_high_semantic_similarity():
    """Clearly related resume and JD should produce a high similarity score."""
    score = calculate_semantic_similarity(RELATED_RESUME, RELATED_JD)
    assert isinstance(score, float), "Score must be a float"
    assert 0 <= score <= 100, f"Score must be in [0, 100], got {score}"
    assert score > 50, (
        f"Expected high similarity (> 50) for related texts, got {score:.2f}"
    )


def test_low_semantic_similarity():
    """Clearly unrelated resume and JD should produce a low similarity score."""
    score = calculate_semantic_similarity(UNRELATED_RESUME, UNRELATED_JD)
    assert isinstance(score, float), "Score must be a float"
    assert 0 <= score <= 100, f"Score must be in [0, 100], got {score}"
    assert score < 50, (
        f"Expected low similarity (< 50) for unrelated texts, got {score:.2f}"
    )


def test_identical_texts_give_max_score():
    """Identical texts should return 100.0 (perfect semantic match)."""
    text = "Python machine learning data science deep learning NLP"
    score = calculate_semantic_similarity(text, text)
    assert score >= 99.0, f"Identical texts should score ~100, got {score:.2f}"


def test_score_is_within_bounds():
    """Score must always be between 0 and 100 inclusive."""
    score = calculate_semantic_similarity("hello", "completely different topic entirely")
    assert 0 <= score <= 100
