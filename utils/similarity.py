from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    st_util = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

_embedding_model = None


def _get_embedding_model():
    """
    Lazy-load the SentenceTransformer model once at module level.
    all-MiniLM-L6-v2: small (80 MB), fast, no GPU required.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def calculate_similarity(resume_text, job_description):
    """
    Calculate TF-IDF cosine similarity between resume and job description.

    Parameters:
        resume_text (str)
        job_description (str)

    Returns:
        float: Match percentage (0–100)
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = sk_cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


def calculate_semantic_similarity(resume_text, job_description):
    """
    Calculate semantic cosine similarity using sentence-transformer embeddings.

    Uses the module-level ``all-MiniLM-L6-v2`` model so the expensive model
    load happens only once regardless of how many calls are made.

    Parameters:
        resume_text (str)
        job_description (str)

    Returns:
        float: Match percentage (0–100)
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return calculate_similarity(resume_text, job_description)

    model = _get_embedding_model()
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_description, convert_to_tensor=True)
    score = st_util.cos_sim(resume_emb, jd_emb).item()
    # cos_sim can be slightly negative for unrelated texts; clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return round(score * 100, 2)