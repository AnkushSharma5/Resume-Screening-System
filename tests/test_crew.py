"""
pytest end-to-end test for the CrewAI pipeline.

The LLM call inside generate_suggestions() is effectively stubbed by ensuring
no LLM_API_KEY is set, which triggers the rule-based fallback path.
This means the test runs in CI without any paid API key.

The test asserts that:
  - analyze_resume_with_crew() returns a dict
  - All expected keys are present
  - Score values are numeric and within [0, 100]
  - Skills lists are lists
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

# Remove any LLM key so we always get the rule-based fallback
os.environ.pop("LLM_API_KEY", None)
os.environ.pop("OPENAI_API_KEY", None)

from agents.crew import analyze_resume_with_crew

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "sample_resume" / "resume1.pdf"

SAMPLE_JD = """
We are looking for a Python developer with experience in machine learning,
data science, and SQL. Knowledge of scikit-learn, pandas, and Git is required.
Experience with Docker and cloud platforms is a plus.
"""

EXPECTED_KEYS = [
    "score",
    "similarity_score",
    "tfidf_similarity_score",
    "semantic_similarity_score",
    "skill_match_score",
    "resume_skills",
    "job_skills",
    "matched_skills",
    "missing_skills",
]


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_resume/resume1.pdf not found")
def test_crew_result_has_expected_keys():
    """
    End-to-end crew run returns a dict with all expected keys.
    LLM is stubbed via missing API key → rule-based fallback ensures no real API call.
    """
    with open(SAMPLE_PDF, "rb") as f:
        result = analyze_resume_with_crew(f, SAMPLE_JD)

    assert isinstance(result, dict), "Result must be a dict"

    for key in EXPECTED_KEYS:
        assert key in result, f"Expected key '{key}' missing from result: {list(result.keys())}"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_resume/resume1.pdf not found")
def test_crew_result_score_in_range():
    """Overall score and similarity scores must be numeric and within [0, 100]."""
    with open(SAMPLE_PDF, "rb") as f:
        result = analyze_resume_with_crew(f, SAMPLE_JD)

    for score_key in ["score", "similarity_score", "tfidf_similarity_score",
                      "semantic_similarity_score", "skill_match_score"]:
        val = result.get(score_key, -1)
        assert isinstance(val, (int, float)), f"{score_key} must be numeric, got {type(val)}"
        assert 0 <= val <= 100, f"{score_key} must be in [0, 100], got {val}"


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample_resume/resume1.pdf not found")
def test_crew_result_skills_are_lists():
    """Skill fields must be lists."""
    with open(SAMPLE_PDF, "rb") as f:
        result = analyze_resume_with_crew(f, SAMPLE_JD)

    for key in ["resume_skills", "job_skills", "matched_skills", "missing_skills"]:
        assert isinstance(result[key], list), f"{key} must be a list"
