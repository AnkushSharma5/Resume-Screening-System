from utils.pdf_reader import extract_text_from_pdf
from utils.preprocessing import preprocess_text
from utils.skill_extractor import extract_skills
from utils.similarity import calculate_similarity, calculate_semantic_similarity
from utils.matching import calculate_skill_match


def analyze_resume(pdf_file, job_description):
    """
    Orchestrate the full resume analysis pipeline.

    Computes:
    - TF-IDF similarity (keyword overlap)
    - Semantic similarity (sentence-transformer embeddings)
    - Combined similarity (70% Semantic + 30% TF-IDF)
    - Skill match score
    - Weighted overall ATS score (70% Combined Similarity + 30% Skill Match)

    Returns a result dict with both individual and combined scores so
    the UI can display all three side by side.
    """

    resume_text = extract_text_from_pdf(pdf_file)

    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(job_description)

    resume_skills = extract_skills(clean_resume)
    jd_skills = extract_skills(clean_jd)

    # --- Dual similarity scores ---
    tfidf_score = calculate_similarity(clean_resume, clean_jd)
    semantic_score = calculate_semantic_similarity(clean_resume, clean_jd)

    # Combined: 70% Semantic Similarity + 30% TF-IDF Similarity
    combined_similarity = round((0.70 * semantic_score) + (0.30 * tfidf_score), 2)

    skill_score, matched, missing = calculate_skill_match(
        resume_skills,
        jd_skills
    )

    overall_score = round(
        (0.7 * combined_similarity) +
        (0.3 * skill_score),
        2
    )

    return {
        "score": overall_score,
        # Combined similarity (70% semantic + 30% tf-idf, used for overall ATS score)
        "similarity_score": combined_similarity,
        # Individual technique scores — shown separately in the UI
        "tfidf_similarity_score": tfidf_score,
        "semantic_similarity_score": semantic_score,
        "skill_match_score": skill_score,
        "resume_skills": resume_skills,
        "job_skills": jd_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        # Raw resume text (needed by LLM suggestions)
        "resume_text": resume_text,
    }
