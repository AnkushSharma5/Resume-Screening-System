from utils.skill_extractor import normalize_skill


def calculate_skill_match(resume_skills, jd_skills):
    """
    Calculate skill matching percentage.

    Parameters:
        resume_skills (list): Skills extracted from or present in resume.
        jd_skills (list): Skills extracted from or present in job description.

    Returns:
        tuple:
            percentage (float): Match percentage (0–100),
            matched_skills (list): Sorted list of matched canonical skills,
            missing_skills (list): Sorted list of missing canonical skills
    """
    normalized_resume = [normalize_skill(s) for s in resume_skills if s]
    normalized_jd = [normalize_skill(s) for s in jd_skills if s]

    resume_set = set(normalized_resume)
    jd_set = set(normalized_jd)

    matched = sorted(resume_set.intersection(jd_set))
    missing = sorted(jd_set - resume_set)

    if len(jd_set) == 0:
        percentage = 0.0
    else:
        percentage = round(
            (len(matched) / len(jd_set)) * 100,
            2
        )

    return percentage, matched, missing