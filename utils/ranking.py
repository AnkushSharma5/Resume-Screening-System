from analyzer import analyze_resume


def rank_resumes(uploaded_files, job_description):
    """
    Analyze multiple resumes and rank them by ATS score.
    """

    results = []

    for resume in uploaded_files:

        result = analyze_resume(
            resume,
            job_description
        )

        results.append(
            {
                "Resume": resume.name,
                "ATS Score": result["score"],
                "Matched Skills": result["matched_skills"],
                "Missing Skills": result["missing_skills"]
            }
        )

    results.sort(
        key=lambda x: x["ATS Score"],
        reverse=True
    )

    return results