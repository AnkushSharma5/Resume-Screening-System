from utils.pdf_reader import extract_text_from_pdf
from utils.preprocessing import preprocess_text
from utils.skill_extractor import extract_skills
from utils.similarity import calculate_similarity
from utils.matching import calculate_skill_match


def analyze_resume(pdf_file, job_description):

    resume_text = extract_text_from_pdf(pdf_file)


    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(job_description)

    resume_skills = extract_skills(clean_resume)
    jd_skills = extract_skills(clean_jd)

    similarity_score = calculate_similarity(
        clean_resume,
        clean_jd
    )

    skill_score, matched, missing = calculate_skill_match(
        resume_skills,
        jd_skills
    )

    overall_score = round(
        (0.7 * similarity_score) +
        (0.3 * skill_score),
        2
    )

    return {
        "score": overall_score,
        "similarity_score": similarity_score,
        "skill_match_score": skill_score,
        "resume_skills": resume_skills,
        "job_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing
    }