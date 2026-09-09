"""
CrewAI-style multi-agent pipeline for resume screening.

This module implements the same architectural concepts as CrewAI
(Agents, Tasks, Tools, sequential Process) in lightweight pure Python.
It imports crewai if available; otherwise it uses the built-in
``_MiniCrew`` fallback so the pipeline always runs.

Three sequential agents:
  1. Extraction Agent  – extracts raw text and skills from the resume PDF
  2. Matching Agent    – computes TF-IDF + semantic similarity and skill overlap
  3. Feedback Agent    – generates personalised improvement suggestions

Usage:
    from agents.crew import analyze_resume_with_crew
    result = analyze_resume_with_crew(pdf_file, job_description)

Returns the same dict shape as analyzer.analyze_resume(), plus
``suggestions`` from the Feedback Agent.
"""

import json
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Try real CrewAI first; fall back to our lightweight implementation
# ──────────────────────────────────────────────────────────────────────────────
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool as crewai_tool
    _CREWAI_AVAILABLE = True
    logger.info("Using real crewai package.")
except Exception:  # ImportError or any version-compat error
    _CREWAI_AVAILABLE = False
    logger.info("crewai not importable — using built-in lightweight agent framework.")

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline function imports (the actual work is always done here)
# ──────────────────────────────────────────────────────────────────────────────
from utils.pdf_reader import extract_text_from_pdf
from utils.preprocessing import preprocess_text
from utils.skill_extractor import extract_skills
from utils.similarity import calculate_similarity, calculate_semantic_similarity
from utils.matching import calculate_skill_match
from utils.suggestions import generate_suggestions
from analyzer import analyze_resume


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight Agent / Task / Crew framework
# (used when crewai cannot be imported due to version conflicts)
# ──────────────────────────────────────────────────────────────────────────────

class _Tool:
    """Wraps a plain Python function as an agent tool."""
    def __init__(self, name: str, func):
        self.name = name
        self._func = func

    def run(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _Agent:
    """
    Represents a specialised worker with a role, goal, and a set of tools.
    In a real CrewAI workflow the LLM decides *when* to call tools; here
    we call the single tool directly since the workflow is deterministic.
    """
    def __init__(self, role: str, goal: str, backstory: str, tools: list):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools  # list of _Tool


class _Task:
    """
    A unit of work assigned to an agent with a description and expected output.
    ``context`` tasks must complete before this one runs.
    """
    def __init__(self, description: str, expected_output: str,
                 agent: _Agent, tool_input=None, context=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.tool_input = tool_input   # argument forwarded to the agent's first tool
        self.context = context or []   # list of upstream _Task objects
        self.output = None             # filled in after execution


class _SequentialProcess:
    """
    Runs tasks one at a time in order, passing each task's output as
    context to the next (same semantics as CrewAI's Process.sequential).
    """
    @staticmethod
    def run(tasks: list) -> list:
        for task in tasks:
            tool = task.agent.tools[0]
            task.output = tool.run(task.tool_input)
        return [t.output for t in tasks]


class _Crew:
    """
    Orchestrates agents and tasks.
    Equivalent to CrewAI's Crew(process=Process.sequential).
    """
    def __init__(self, agents: list, tasks: list):
        self.agents = agents
        self.tasks = tasks

    def kickoff(self) -> list:
        return _SequentialProcess.run(self.tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Shared state (populated by tools, read by the result builder)
# ──────────────────────────────────────────────────────────────────────────────
_run_state: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations (pure Python — no LLM required)
# ──────────────────────────────────────────────────────────────────────────────

def _extraction_tool(input_arg: str) -> str:
    """
    Extraction Agent tool.
    Reads the PDF from shared state, extracts text and skills.
    Returns JSON with resume_text (truncated) and resume_skills.
    """
    pdf_file = _run_state.get("pdf_file")
    if pdf_file is None:
        return json.dumps({"error": "No PDF file in shared state"})

    resume_text = extract_text_from_pdf(pdf_file)
    clean_resume = preprocess_text(resume_text)
    resume_skills = extract_skills(clean_resume)

    _run_state["resume_text"] = resume_text
    _run_state["clean_resume"] = clean_resume
    _run_state["resume_skills"] = resume_skills

    snippet = resume_text[:500] + "…" if len(resume_text) > 500 else resume_text
    return json.dumps({
        "resume_text_snippet": snippet,
        "resume_skills": resume_skills,
        "skill_count": len(resume_skills),
    })


def _matching_tool(job_description: str) -> str:
    """
    Matching Agent tool.
    Uses shared state (from Extraction Agent) + job description to compute:
      - TF-IDF similarity (keyword overlap)
      - Semantic similarity (sentence embeddings)
      - Skill match score
    Returns JSON with all scores and matched/missing skills.
    """
    clean_resume = _run_state.get("clean_resume", "")
    resume_skills = _run_state.get("resume_skills", [])

    clean_jd = preprocess_text(job_description)
    jd_skills = extract_skills(clean_jd)

    tfidf_score = calculate_similarity(clean_resume, clean_jd)
    semantic_score = calculate_semantic_similarity(clean_resume, clean_jd)
    combined_similarity = round((tfidf_score + semantic_score) / 2, 2)

    skill_score, matched, missing = calculate_skill_match(resume_skills, jd_skills)
    overall_score = round((0.7 * combined_similarity) + (0.3 * skill_score), 2)

    _run_state.update({
        "tfidf_score": tfidf_score,
        "semantic_score": semantic_score,
        "combined_similarity": combined_similarity,
        "skill_score": skill_score,
        "matched_skills": matched,
        "missing_skills": missing,
        "overall_score": overall_score,
        "jd_skills": jd_skills,
    })

    return json.dumps({
        "tfidf_similarity_score": tfidf_score,
        "semantic_similarity_score": semantic_score,
        "combined_similarity_score": combined_similarity,
        "skill_match_score": skill_score,
        "overall_score": overall_score,
        "matched_skills": matched,
        "missing_skills": missing,
    })


def _feedback_tool(job_description: str) -> str:
    """
    Feedback Agent tool.
    Uses scores and skills from shared state (populated by Matching Agent)
    to call generate_suggestions() — LLM if configured, rule-based fallback.
    Returns JSON array of suggestion objects.
    """
    partial_result = {
        "score": _run_state.get("overall_score", 0),
        "similarity_score": _run_state.get("combined_similarity", 0),
        "skill_match_score": _run_state.get("skill_score", 0),
        "matched_skills": _run_state.get("matched_skills", []),
        "missing_skills": _run_state.get("missing_skills", []),
        "resume_text": _run_state.get("resume_text", ""),
    }

    suggestions = generate_suggestions(
        partial_result,
        resume_text=_run_state.get("resume_text", ""),
        job_description=job_description,
    )
    _run_state["suggestions"] = suggestions
    return json.dumps(suggestions)


# ──────────────────────────────────────────────────────────────────────────────
# Crew builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_and_run_crew(job_description: str):
    """
    Constructs and runs the 3-agent sequential crew using either the real
    crewai package (if importable) or the built-in lightweight framework.
    """
    if _CREWAI_AVAILABLE:
        _run_crew_with_real_crewai(job_description)
    else:
        _run_crew_with_mini_framework(job_description)


def _run_crew_with_mini_framework(job_description: str):
    """Run the pipeline using the lightweight _Agent/_Task/_Crew classes."""

    extraction_agent = _Agent(
        role="Resume Extraction Specialist",
        goal="Extract all text and skills from the uploaded resume PDF.",
        backstory=(
            "Expert at parsing resume PDFs and identifying technical and soft skills "
            "using specialised extraction tools."
        ),
        tools=[_Tool("resume_extraction_tool", _extraction_tool)],
    )

    matching_agent = _Agent(
        role="Resume Matching Analyst",
        goal=(
            "Compute TF-IDF keyword overlap and semantic embedding similarity "
            "between resume and job description, and identify skill gaps."
        ),
        backstory=(
            "ATS scoring expert who evaluates resume–job fit with two NLP techniques: "
            "TF-IDF (keyword frequency) and sentence-transformer embeddings (semantic meaning)."
        ),
        tools=[_Tool("resume_matching_tool", _matching_tool)],
    )

    feedback_agent = _Agent(
        role="Career Coach",
        goal="Generate personalised, actionable resume improvement suggestions.",
        backstory=(
            "Senior career coach who reads ATS analysis results and produces concrete, "
            "specific advice to help candidates improve their resumes and close skill gaps."
        ),
        tools=[_Tool("resume_feedback_tool", _feedback_tool)],
    )

    extraction_task = _Task(
        description="Extract resume text and identify skills from the PDF.",
        expected_output="JSON with resume_text_snippet, resume_skills, skill_count.",
        agent=extraction_agent,
        tool_input="resume",
    )

    matching_task = _Task(
        description=(
            f"Compute similarity scores and skill match against the job description."
        ),
        expected_output=(
            "JSON with tfidf_similarity_score, semantic_similarity_score, "
            "combined_similarity_score, skill_match_score, overall_score, "
            "matched_skills, missing_skills."
        ),
        agent=matching_agent,
        tool_input=job_description,
        context=[extraction_task],
    )

    feedback_task = _Task(
        description="Generate resume improvement suggestions using analysis results.",
        expected_output="JSON array of suggestion objects with title and detail.",
        agent=feedback_agent,
        tool_input=job_description,
        context=[matching_task],
    )

    crew = _Crew(
        agents=[extraction_agent, matching_agent, feedback_agent],
        tasks=[extraction_task, matching_task, feedback_task],
    )
    crew.kickoff()


def _run_crew_with_real_crewai(job_description: str):
    """Run the pipeline using the real crewai package when available."""
    import os
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool as crewai_tool

    os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-tool-only-crew")

    extraction_tool_c = crewai_tool(name="resume_extraction_tool")(_extraction_tool)
    matching_tool_c = crewai_tool(name="resume_matching_tool")(_matching_tool)
    feedback_tool_c = crewai_tool(name="resume_feedback_tool")(_feedback_tool)

    extraction_agent = Agent(
        role="Resume Extraction Specialist",
        goal="Extract all text and skills from the uploaded resume PDF.",
        backstory="Expert at parsing PDFs and identifying skills.",
        tools=[extraction_tool_c],
        verbose=False, allow_delegation=False,
    )
    matching_agent = Agent(
        role="Resume Matching Analyst",
        goal="Compute TF-IDF + semantic similarity and skill match scores.",
        backstory="ATS scoring expert using both keyword and semantic NLP techniques.",
        tools=[matching_tool_c],
        verbose=False, allow_delegation=False,
    )
    feedback_agent = Agent(
        role="Career Coach",
        goal="Generate personalised resume improvement suggestions.",
        backstory="Senior career coach producing actionable advice from ATS results.",
        tools=[feedback_tool_c],
        verbose=False, allow_delegation=False,
    )

    extraction_task = Task(
        description="Extract text and skills from the PDF resume in shared state.",
        expected_output="JSON with resume_text_snippet, resume_skills, skill_count.",
        agent=extraction_agent,
    )
    matching_task = Task(
        description=f"Compute similarity and skill match for: {job_description[:400]}",
        expected_output="JSON with all similarity and skill match scores.",
        agent=matching_agent,
        context=[extraction_task],
    )
    feedback_task = Task(
        description=f"Generate improvement suggestions for: {job_description[:200]}",
        expected_output="JSON array of suggestion objects.",
        agent=feedback_agent,
        context=[matching_task],
    )

    crew = Crew(
        agents=[extraction_agent, matching_agent, feedback_agent],
        tasks=[extraction_task, matching_task, feedback_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def analyze_resume_with_crew(pdf_file, job_description: str) -> dict:
    """
    Run the 3-agent sequential pipeline and return a result dict.

    The returned dict has the same shape as analyzer.analyze_resume(), plus:
        ``suggestions`` – list of {title, detail} dicts (or plain strings on fallback)

    Falls back to the classic analyze_resume() pipeline on any unrecoverable error.

    Parameters:
        pdf_file        – file-like object or path (same as analyze_resume)
        job_description – job description string

    Returns:
        dict
    """
    global _run_state
    _run_state = {"pdf_file": pdf_file}

    try:
        _build_and_run_crew(job_description)

        result = {
            "score": _run_state.get("overall_score", 0),
            "similarity_score": _run_state.get("combined_similarity", 0),
            "tfidf_similarity_score": _run_state.get("tfidf_score", 0),
            "semantic_similarity_score": _run_state.get("semantic_score", 0),
            "skill_match_score": _run_state.get("skill_score", 0),
            "resume_skills": _run_state.get("resume_skills", []),
            "job_skills": _run_state.get("jd_skills", []),
            "matched_skills": _run_state.get("matched_skills", []),
            "missing_skills": _run_state.get("missing_skills", []),
            "resume_text": _run_state.get("resume_text", ""),
            "suggestions": _run_state.get("suggestions", []),
        }
        return result

    except Exception as exc:
        logger.error(
            "Agent pipeline failed (%s) — falling back to classic pipeline.", exc,
            exc_info=True,
        )
        result = analyze_resume(pdf_file, job_description)
        result["suggestions"] = generate_suggestions(
            result,
            resume_text=result.get("resume_text", ""),
            job_description=job_description,
        )
        return result
