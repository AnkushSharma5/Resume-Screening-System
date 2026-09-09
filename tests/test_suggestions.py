"""
pytest tests for resume improvement suggestions (utils/suggestions.py).
Tests both rule-based fallback and structured LLM parsing.
"""

import json
import pytest
from unittest.mock import patch

from utils.suggestions import (
    generate_suggestions,
    _rule_based_suggestions,
    _parse_llm_response,
)


@pytest.fixture
def sample_result():
    return {
        "score": 65.0,
        "similarity_score": 60.0,
        "tfidf_similarity_score": 58.0,
        "semantic_similarity_score": 62.0,
        "skill_match_score": 50.0,
        "resume_skills": ["python", "sql"],
        "job_skills": ["python", "sql", "docker", "aws"],
        "matched_skills": ["python", "sql"],
        "missing_skills": ["docker", "aws"],
        "resume_text": "Experienced Python and SQL developer.",
    }


def test_rule_based_suggestions_has_content(sample_result):
    """Rule-based suggestions should return actionable items."""
    suggestions = _rule_based_suggestions(sample_result)
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    # Must mention missing skills
    text = " ".join(suggestions)
    assert "docker" in text
    assert "aws" in text


def test_generate_suggestions_fallback_when_no_api_key(sample_result):
    """Without LLM_API_KEY, generate_suggestions should fallback to rule-based strings."""
    with patch.dict("os.environ", {}, clear=True):
        suggestions = generate_suggestions(sample_result)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)


def test_parse_llm_response_valid_json():
    """Should correctly parse a standard JSON array of suggestions."""
    raw = json.dumps([
        {"title": "Add Docker", "detail": "Highlight containerization experience."},
        {"title": "Quantify Impact", "detail": "Add percentage improvements to projects."}
    ])
    parsed = _parse_llm_response(raw)
    assert len(parsed) == 2
    assert parsed[0]["title"] == "Add Docker"
    assert parsed[0]["detail"] == "Highlight containerization experience."


def test_parse_llm_response_markdown_fences():
    """Should strip markdown code fences from LLM output."""
    raw = """```json
[
    {"title": "Cloud Skills", "detail": "Add AWS or GCP certifications."}
]
```"""
    parsed = _parse_llm_response(raw)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Cloud Skills"


def test_parse_llm_response_invalid_format():
    """Should raise ValueError when response is not a valid list of dicts."""
    with pytest.raises(ValueError):
        _parse_llm_response('{"not": "a list"}')

    with pytest.raises(ValueError):
        _parse_llm_response('["plain string item"]')


def test_generate_suggestions_with_mocked_llm(sample_result):
    """With an active LLM provider mocked, generate_suggestions returns structured dicts."""
    mock_response = json.dumps([
        {"title": "Add Docker and AWS", "detail": "Include hands-on container and cloud experience."},
        {"title": "Action Verbs", "detail": "Begin bullet points with verbs like Engineered or Deployed."}
    ])

    with patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_PROVIDER": "openai"}):
        with patch("utils.suggestions._call_openai", return_value=mock_response):
            suggestions = generate_suggestions(
                sample_result,
                resume_text=sample_result["resume_text"],
                job_description="Looking for Python, Docker, AWS developer."
            )
            assert len(suggestions) == 2
            assert suggestions[0]["title"] == "Add Docker and AWS"
            assert isinstance(suggestions[0], dict)
