"""
Comprehensive verification test suite for Fixes 1, 2, 3, and 4.
"""
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from utils.skill_extractor import extract_skills, normalize_skill, SKILLS
from utils.preprocessing import preprocess_text
from utils.matching import calculate_skill_match
from utils.similarity import calculate_similarity, calculate_semantic_similarity
from analyzer import analyze_resume


def test_fix1_skills_count_and_required_skills():
    """Verify skills.csv has ~68 skills and includes all 17 required skills."""
    assert len(SKILLS) == 68, f"Expected 68 skills, got {len(SKILLS)}"

    required = [
        "nlp", "llm", "agentic ai", "prompt engineering", "angular",
        "javascript", "react", "rest", "oracle", "pl/sql", "gcp",
        "crewai", "langchain", "rag", "mongodb", "jwt", "duckdb"
    ]
    for req in required:
        assert req in SKILLS, f"Required skill '{req}' not found in skills.csv"


def test_fix1_new_skills_extracted_from_text():
    """Verify newly added skills are extracted properly from text."""
    sample_text = """
    Experienced with NLP and LLM applications using LangChain, RAG, and CrewAI.
    Built Agentic AI workflows with Prompt Engineering techniques.
    Full stack web development in Angular, React, and JavaScript with REST APIs.
    Database experience with Oracle, PL/SQL, MongoDB, DuckDB, and JWT authentication on GCP.
    """
    extracted = extract_skills(sample_text)
    expected = [
        "agentic ai", "angular", "crewai", "duckdb", "gcp",
        "javascript", "jwt", "langchain", "llm", "mongodb",
        "nlp", "oracle", "pl/sql", "prompt engineering",
        "rag", "react", "rest"
    ]
    for skill in expected:
        assert skill in extracted, f"Expected '{skill}' to be extracted, got: {extracted}"


def test_fix2_hyphen_and_casing_normalization():
    """Verify hyphen, underscore, whitespace, and casing variations match consistently."""
    variations = [
        "scikit-learn",
        "Scikit-Learn",
        "scikit_learn",
        "SCIKIT-LEARN",
        "scikit learn",
        "scikitlearn",
    ]
    for var in variations:
        norm = normalize_skill(var)
        assert norm == "scikit-learn", f"normalize_skill('{var}') should be 'scikit-learn', got '{norm}'"

        # Also test extraction from text with variation
        extracted = extract_skills(f"Experience in {var} library.")
        assert "scikit-learn" in extracted, f"extract_skills failed for variant '{var}'"


def test_fix2_other_complex_skills_normalization():
    """Verify normalization and extraction for pl/sql, node.js, agentic ai, c++, etc."""
    tests = [
        (["PL/SQL", "pl_sql", "pl-sql", "plsql"], "pl/sql"),
        (["Node.js", "node.js", "nodejs", "node_js", "node js"], "node.js"),
        (["Next.js", "next.js", "nextjs", "next_js"], "next.js"),
        (["CI/CD", "ci/cd", "cicd", "ci_cd", "ci cd"], "ci/cd"),
        (["Agentic AI", "agentic-ai", "agentic_ai", "agentic ai"], "agentic ai"),
        (["Prompt Engineering", "prompt-engineering", "prompt_engineering"], "prompt engineering"),
        (["Machine Learning", "machine-learning", "machine_learning"], "machine learning"),
        (["Power BI", "power-bi", "power_bi", "powerbi"], "power bi"),
    ]
    for variants, canonical in tests:
        for var in variants:
            assert normalize_skill(var) == canonical, f"normalize_skill('{var}') failed, expected '{canonical}'"
            extracted = extract_skills(f"Skills include {var} for backend.")
            assert canonical in extracted, f"extract_skills failed for '{var}' -> '{canonical}'"


def test_fix2_skill_matching_across_representations():
    """Verify calculate_skill_match matches across different representations."""
    resume_skills = ["Scikit-Learn", "PL/SQL", "Agentic AI", "Docker"]
    jd_skills = ["scikit_learn", "pl-sql", "agentic_ai", "AWS"]

    score, matched, missing = calculate_skill_match(resume_skills, jd_skills)
    assert score == pytest.approx(75.0)  # 3 out of 4
    assert matched == ["agentic ai", "pl/sql", "scikit-learn"]
    assert missing == ["aws"]


def test_fix3_combined_similarity_formula():
    """Verify combined similarity calculation is 0.70 * semantic + 0.30 * tfidf."""
    semantic = 80.0
    tfidf = 40.0
    expected_combined = round((0.70 * semantic) + (0.30 * tfidf), 2)
    assert expected_combined == 68.0  # 56.0 + 12.0

    with patch("analyzer.extract_text_from_pdf", return_value="Python SQL scikit-learn"), \
         patch("analyzer.calculate_similarity", return_value=tfidf), \
         patch("analyzer.calculate_semantic_similarity", return_value=semantic):
        res = analyze_resume("dummy_pdf", "Python SQL scikit-learn")
        assert res["tfidf_similarity_score"] == 40.0
        assert res["semantic_similarity_score"] == 80.0
        assert res["similarity_score"] == 68.0
