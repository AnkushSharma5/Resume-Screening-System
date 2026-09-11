"""
pytest unit tests for rank_resumes() in utils/ranking.py.
"""

from unittest.mock import MagicMock, patch
import pytest

from utils.ranking import rank_resumes


def _make_mock_file(name: str):
    mock = MagicMock()
    mock.name = name
    return mock


@patch("utils.ranking.analyze_resume")
def test_rank_resumes_returns_expected_keys(mock_analyze):
    mock_analyze.return_value = {
        "score": 85.0,
        "similarity_score": 80.0,
        "tfidf_similarity_score": 78.5,
        "semantic_similarity_score": 81.5,
        "skill_match_score": 90.0,
        "resume_skills": ["python", "sql"],
        "job_skills": ["python", "sql", "docker"],
        "matched_skills": ["python", "sql"],
        "missing_skills": ["docker"],
        "resume_text": "Sample text",
    }

    files = [_make_mock_file("candidate1.pdf")]
    results = rank_resumes(files, "Looking for Python SQL developer")

    assert isinstance(results, list)
    assert len(results) == 1
    row = results[0]

    expected_keys = [
        "Resume",
        "ATS Score",
        "TF-IDF Score",
        "Semantic Score",
        "Matched Skills",
        "Missing Skills",
    ]
    for key in expected_keys:
        assert key in row, f"Missing key: {key}"

    assert row["Resume"] == "candidate1.pdf"
    assert row["ATS Score"] == 85.0
    assert row["TF-IDF Score"] == 78.5
    assert row["Semantic Score"] == 81.5
    assert row["Matched Skills"] == ["python", "sql"]
    assert row["Missing Skills"] == ["docker"]


@patch("utils.ranking.analyze_resume")
def test_rank_resumes_sorted_descending(mock_analyze):
    def side_effect(file, jd):
        scores = {
            "low.pdf": {"score": 50.0, "tfidf_similarity_score": 40.0, "semantic_similarity_score": 45.0, "matched_skills": [], "missing_skills": ["python"]},
            "high.pdf": {"score": 95.0, "tfidf_similarity_score": 90.0, "semantic_similarity_score": 92.0, "matched_skills": ["python"], "missing_skills": []},
            "mid.pdf": {"score": 75.0, "tfidf_similarity_score": 70.0, "semantic_similarity_score": 72.0, "matched_skills": ["python"], "missing_skills": []},
        }
        return scores[file.name]

    mock_analyze.side_effect = side_effect

    files = [_make_mock_file("low.pdf"), _make_mock_file("high.pdf"), _make_mock_file("mid.pdf")]
    results = rank_resumes(files, "Sample JD")

    assert [r["Resume"] for r in results] == ["high.pdf", "mid.pdf", "low.pdf"]
    assert [r["ATS Score"] for r in results] == [95.0, 75.0, 50.0]


def test_rank_resumes_empty_list():
    results = rank_resumes([], "Sample JD")
    assert results == []


@patch("utils.ranking.analyze_resume")
def test_rank_resumes_score_types(mock_analyze):
    mock_analyze.return_value = {
        "score": 82.34,
        "tfidf_similarity_score": 80.12,
        "semantic_similarity_score": 84.56,
        "matched_skills": ["git"],
        "missing_skills": [],
    }

    files = [_make_mock_file("resume.pdf")]
    results = rank_resumes(files, "JD")

    assert isinstance(results[0]["ATS Score"], float)
    assert isinstance(results[0]["TF-IDF Score"], float)
    assert isinstance(results[0]["Semantic Score"], float)


@patch("utils.ranking.analyze_resume")
def test_rank_resumes_preserves_all_resumes(mock_analyze):
    mock_analyze.return_value = {
        "score": 70.0,
        "tfidf_similarity_score": 65.0,
        "semantic_similarity_score": 68.0,
        "matched_skills": [],
        "missing_skills": [],
    }

    files = [_make_mock_file(f"res_{i}.pdf") for i in range(5)]
    results = rank_resumes(files, "JD")

    assert len(results) == 5


@patch("utils.ranking.analyze_resume")
def test_rank_resumes_ties_handled(mock_analyze):
    mock_analyze.return_value = {
        "score": 80.0,
        "tfidf_similarity_score": 75.0,
        "semantic_similarity_score": 78.0,
        "matched_skills": ["python"],
        "missing_skills": [],
    }

    files = [_make_mock_file("a.pdf"), _make_mock_file("b.pdf")]
    results = rank_resumes(files, "JD")

    assert len(results) == 2
    assert results[0]["ATS Score"] == results[1]["ATS Score"]
