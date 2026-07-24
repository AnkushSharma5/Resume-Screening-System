def generate_suggestions(result):
    """
    Generate resume improvement suggestions based on analysis.
    """

    suggestions = []

    # Missing Skills
    if result["missing_skills"]:
        suggestions.append(
            "Add the following skills (if you have experience): "
            + ", ".join(result["missing_skills"])
        )

    # Similarity Score
    if result["similarity_score"] < 70:
        suggestions.append(
            "Tailor your resume to better match the job description keywords."
        )

    # Skill Match Score
    if result["skill_match_score"] < 70:
        suggestions.append(
            "Include more relevant technical skills and tools."
        )

    # Overall Score
    if result["overall_score"] < 80:
        suggestions.append(
            "Add more projects and achievements related to this role."
        )

    # General Suggestions
    suggestions.append(
        "Use action verbs such as Developed, Built, Designed, Implemented."
    )

    suggestions.append(
        "Quantify your achievements whenever possible (e.g., Improved model accuracy by 15%)."
    )

    suggestions.append(
        "Keep your resume concise and ATS-friendly."
    )

    return suggestions