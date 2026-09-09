"""
pytest tests for calculate_skill_match().
"""
import pytest
from utils.matching import calculate_skill_match

RESUME = ["python", "sql", "git", "docker"]
JD = ["python", "sql", "git", "docker", "aws"]


def test_skill_match_score():
    """4 out of 5 JD skills matched → 80%."""
    score, matched, missing = calculate_skill_match(RESUME, JD)
    assert isinstance(score, float)
    assert score == pytest.approx(80.0), f"Expected 80.0, got {score}"


def test_matched_skills():
    """Matched skills should be the intersection of resume and JD."""
    _, matched, _ = calculate_skill_match(RESUME, JD)
    assert set(matched) == {"python", "sql", "git", "docker"}


def test_missing_skills():
    """Missing skills are those in JD but not in resume."""
    _, _, missing = calculate_skill_match(RESUME, JD)
    assert missing == ["aws"], f"Expected ['aws'], got {missing}"


def test_empty_jd_returns_zero():
    """If JD has no skills, score should be 0."""
    score, matched, missing = calculate_skill_match(["python"], [])
    assert score == 0
    assert matched == []
    assert missing == []


# Keep original print output for manual inspection
if __name__ == "__main__":
    score, matched, missing = calculate_skill_match(RESUME, JD)
    print("Skill Match:", score, "%")
    print()
    print("Matched Skills:")
    print(matched)
    print()
    print("Missing Skills:")
    print(missing)