def calculate_skill_match(resume_skills, jd_skills):
    """
    Calculate skill matching percentage.

    Parameters:
        resume_skills (list)
        jd_skills (list)

    Returns:
        tuple:
            percentage,
            matched_skills,
            missing_skills
    """

    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = sorted(resume_set.intersection(jd_set))

    missing = sorted(jd_set - resume_set)

    if len(jd_set) == 0:
        percentage = 0

    else:
        percentage = round(
            (len(matched) / len(jd_set)) * 100,
            2
        )

    return percentage, matched, missing