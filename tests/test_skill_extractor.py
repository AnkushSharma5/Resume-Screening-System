"""
pytest tests for extract_skills().
"""
import pytest
from utils.preprocessing import preprocess_text
from utils.skill_extractor import extract_skills

SAMPLE_RESUME = """
Hello, my name is Ankush Sharma.
Skills: Python Java SQL Machine Learning Power BI Git Docker
I have completed several Machine Learning projects using Python and SQL.
"""


def test_known_skills_detected():
    """Skills present in the resume should be detected."""
    clean_text = preprocess_text(SAMPLE_RESUME)
    skills = extract_skills(clean_text)
    for skill in ["python", "sql", "git", "docker"]:
        assert skill in skills, f"'{skill}' should be detected"


def test_returns_list():
    """extract_skills must return a list."""
    skills = extract_skills(preprocess_text(SAMPLE_RESUME))
    assert isinstance(skills, list)


def test_no_duplicate_skills():
    """Each skill should appear at most once."""
    skills = extract_skills(preprocess_text(SAMPLE_RESUME))
    assert len(skills) == len(set(skills)), "Duplicate skills found"


def test_skills_are_sorted():
    """Skills list should be sorted alphabetically."""
    skills = extract_skills(preprocess_text(SAMPLE_RESUME))
    assert skills == sorted(skills)


def test_empty_text_returns_empty_list():
    """Empty text should yield no skills."""
    skills = extract_skills("")
    assert skills == []


# Keep original print output for manual inspection
if __name__ == "__main__":
    clean_text = preprocess_text(SAMPLE_RESUME)
    skills = extract_skills(clean_text)
    print("Detected Skills:\n")
    print(skills)