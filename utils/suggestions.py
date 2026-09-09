"""
Resume improvement suggestions.

Tries to call an LLM (OpenAI / Anthropic / Google Gemini) to generate
structured, resume-specific feedback as JSON.  Falls back to rule-based
suggestions if no API key is set or the API call fails — the app never
crashes and still works for a live demo without a paid key.

Environment variables:
    LLM_PROVIDER   – one of: "openai", "anthropic", "gemini"  (default: "openai")
    LLM_API_KEY    – your API key for the chosen provider
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based fallback (original logic, preserved unchanged)
# ---------------------------------------------------------------------------

def _rule_based_suggestions(result):
    """Return a list of plain-string suggestions using if/else rules."""
    suggestions = []

    if result.get("missing_skills"):
        suggestions.append(
            "Add the following skills (if you have experience): "
            + ", ".join(result["missing_skills"])
        )

    if result.get("similarity_score", 100) < 70:
        suggestions.append(
            "Tailor your resume to better match the job description keywords."
        )

    if result.get("skill_match_score", 100) < 70:
        suggestions.append(
            "Include more relevant technical skills and tools."
        )

    if result.get("score", 100) < 80:
        suggestions.append(
            "Add more projects and achievements related to this role."
        )

    suggestions.append(
        "Use action verbs such as Developed, Built, Designed, Implemented."
    )
    suggestions.append(
        "Quantify your achievements whenever possible "
        "(e.g., Improved model accuracy by 15%)."
    )
    suggestions.append(
        "Keep your resume concise and ATS-friendly."
    )

    return suggestions


# ---------------------------------------------------------------------------
# LLM provider adapters
# ---------------------------------------------------------------------------

def _build_prompt(result, resume_text, job_description):
    """Build the structured prompt sent to the LLM."""
    matched = ", ".join(result.get("matched_skills", [])) or "none"
    missing = ", ".join(result.get("missing_skills", [])) or "none"

    return f"""You are an expert resume coach. Analyze the resume against the job description and return ONLY a JSON array of improvement suggestions. Each item must have exactly two keys: "title" (short label, max 8 words) and "detail" (1–2 sentence actionable advice). Return 4–6 suggestions.

JOB DESCRIPTION (excerpt, first 600 chars):
{job_description[:600]}

RESUME TEXT (excerpt, first 800 chars):
{resume_text[:800]}

MATCHED SKILLS: {matched}
MISSING SKILLS: {missing}
OVERALL ATS SCORE: {result.get("score", 0)}%

Respond with ONLY valid JSON, no markdown, no explanation. Example format:
[{{"title": "Add missing skills", "detail": "Consider adding Python and SQL to your skills section."}}]"""


def _call_openai(prompt, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_gemini(prompt, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def _get_caller(provider: str):
    """Dynamically resolve caller function so mocks in tests apply properly."""
    callers = {
        "openai": _call_openai,
        "anthropic": _call_anthropic,
        "gemini": _call_gemini,
    }
    return callers.get(provider)


def _parse_llm_response(raw_text):
    """
    Parse and validate the LLM JSON response.
    Returns a list of dicts with 'title' and 'detail', or raises ValueError.
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")
    validated = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each suggestion must be a JSON object")
        if "title" not in item or "detail" not in item:
            raise ValueError("Each suggestion must have 'title' and 'detail'")
        validated.append({"title": str(item["title"]), "detail": str(item["detail"])})
    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_suggestions(result, resume_text="", job_description=""):
    """
    Generate resume improvement suggestions.

    Attempts to use an LLM (provider/key from env vars) to produce structured
    JSON suggestions.  Falls back gracefully to rule-based suggestions on any
    error.

    Parameters:
        result          (dict) – output from analyze_resume()
        resume_text     (str)  – raw resume text (for LLM prompt context)
        job_description (str)  – job description text (for LLM prompt context)

    Returns:
        list[dict | str] – list of {"title": ..., "detail": ...} dicts if LLM
                           succeeded, or list of plain strings as fallback.
    """
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    provider = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

    if not api_key:
        logger.debug("LLM_API_KEY not set — using rule-based suggestions.")
        return _rule_based_suggestions(result)

    caller = _get_caller(provider)
    if caller is None:
        logger.warning("Unknown LLM_PROVIDER '%s' — using rule-based suggestions.", provider)
        return _rule_based_suggestions(result)

    try:
        # Use result's resume_text if caller didn't pass one explicitly
        r_text = resume_text or result.get("resume_text", "")
        prompt = _build_prompt(result, r_text, job_description)
        raw = caller(prompt, api_key)
        suggestions = _parse_llm_response(raw)
        return suggestions
    except Exception as exc:
        logger.warning("LLM suggestion call failed (%s) — falling back to rule-based.", exc)
        return _rule_based_suggestions(result)