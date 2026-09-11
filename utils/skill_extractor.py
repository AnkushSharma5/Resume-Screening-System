import re
from pathlib import Path
import pandas as pd

# -----------------------------
# Locate project root directory
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -----------------------------
# Path to skills.csv
# -----------------------------
SKILLS_PATH = BASE_DIR / "data" / "skills.csv"

# -----------------------------
# Load skills
# -----------------------------
skills_df = pd.read_csv(SKILLS_PATH)
SKILLS = sorted(set(skills_df["skill"].dropna().astype(str).str.strip().str.lower().tolist()))
SKILLS_SET = set(SKILLS)

# Explicit alias dictionary mapping non-canonical or variant formats to canonical names
SKILL_ALIASES = {
    # Scikit-learn
    "scikit-learn": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikitlearn": "scikit-learn",
    "sklearn": "scikit-learn",
    # PL/SQL
    "pl/sql": "pl/sql",
    "pl-sql": "pl/sql",
    "pl_sql": "pl/sql",
    "pl sql": "pl/sql",
    "plsql": "pl/sql",
    # Node.js
    "node.js": "node.js",
    "nodejs": "node.js",
    "node-js": "node.js",
    "node_js": "node.js",
    "node js": "node.js",
    # Next.js
    "next.js": "next.js",
    "nextjs": "next.js",
    "next-js": "next.js",
    "next_js": "next.js",
    "next js": "next.js",
    # CI/CD
    "ci/cd": "ci/cd",
    "ci-cd": "ci/cd",
    "ci_cd": "ci/cd",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    # C++
    "c++": "c++",
    "cpp": "c++",
    # C#
    "c#": "c#",
    "c-sharp": "c#",
    "c_sharp": "c#",
    "c sharp": "c#",
    # Power BI
    "power bi": "power bi",
    "power-bi": "power bi",
    "power_bi": "power bi",
    "powerbi": "power bi",
    # Agentic AI
    "agentic ai": "agentic ai",
    "agentic-ai": "agentic ai",
    "agentic_ai": "agentic ai",
    # Prompt Engineering
    "prompt engineering": "prompt engineering",
    "prompt-engineering": "prompt engineering",
    "prompt_engineering": "prompt engineering",
    # Machine Learning
    "machine learning": "machine learning",
    "machine-learning": "machine learning",
    "machine_learning": "machine learning",
    # Deep Learning
    "deep learning": "deep learning",
    "deep-learning": "deep learning",
    "deep_learning": "deep learning",
    # Data Analysis
    "data analysis": "data analysis",
    "data-analysis": "data analysis",
    "data_analysis": "data analysis",
    # Spring Boot
    "spring boot": "spring boot",
    "spring-boot": "spring boot",
    "spring_boot": "spring boot",
    # REST
    "rest": "rest",
    "restful": "rest",
    "rest api": "rest",
    "rest apis": "rest",
    "rest-api": "rest",
    "rest_api": "rest",
    # PostgreSQL
    "postgres": "postgresql",
    "postgresql": "postgresql",
}

# Add auto-generated aliases for all skills in skills.csv
for s in SKILLS:
    SKILL_ALIASES[s] = s
    SKILL_ALIASES[s.replace("-", " ")] = s
    SKILL_ALIASES[s.replace(" ", "-")] = s
    SKILL_ALIASES[s.replace("-", "_")] = s
    SKILL_ALIASES[s.replace(" ", "_")] = s
    SKILL_ALIASES[re.sub(r"[\s\-_./]+", "", s)] = s


def normalize_skill(skill: str) -> str:
    """
    Normalize any skill string (handling casing, whitespace, hyphens, underscores, etc.)
    to its canonical representation in data/skills.csv.
    """
    if not isinstance(skill, str):
        return str(skill)

    s = skill.strip().lower()

    if s in SKILL_ALIASES:
        return SKILL_ALIASES[s]

    variant_hyphen = re.sub(r"[\s_]+", "-", s)
    if variant_hyphen in SKILL_ALIASES:
        return SKILL_ALIASES[variant_hyphen]
    if variant_hyphen in SKILLS_SET:
        return variant_hyphen

    variant_space = re.sub(r"[\-_]+", " ", s)
    if variant_space in SKILL_ALIASES:
        return SKILL_ALIASES[variant_space]
    if variant_space in SKILLS_SET:
        return variant_space

    variant_clean = re.sub(r"[\s\-_./]+", "", s)
    if variant_clean in SKILL_ALIASES:
        return SKILL_ALIASES[variant_clean]

    return s


def _build_skill_patterns():
    """
    Build compiled regex patterns for all skills to support extraction from
    both raw and preprocessed text accurately without false positives.
    """
    patterns = {}

    # Custom regex for skills with special characters or specific abbreviations
    custom_regex = {
        "scikit-learn": r"(?:\b|_)(?:scikit[\s\-_]?learn|scikitlearn|sklearn)(?:\b|_)",
        "pl/sql": r"(?:\b|_)(?:pl[\s\-_/]?sql|plsql)(?:\b|_)",
        "node.js": r"(?:\b|_)(?:node[\s\-_.]?js|nodejs)(?:\b|_)",
        "next.js": r"(?:\b|_)(?:next[\s\-_.]?js|nextjs)(?:\b|_)",
        "ci/cd": r"(?:\b|_)(?:ci[\s\-_/]?cd|cicd)(?:\b|_)",
        "c++": r"(?:(?<=\s)|(?<=[^a-zA-Z0-9])|^)(?:c\+\+|cpp)(?:(?=\s)|(?=[^a-zA-Z0-9])|$)",
        "c#": r"(?:(?<=\s)|(?<=[^a-zA-Z0-9])|^)(?:c\#|c[\s\-_]?sharp)(?:(?=\s)|(?=[^a-zA-Z0-9])|$)",
        "c": r"(?:(?<=\s)|(?<=[,;/()[\]{}])|^)c(?:(?=\s)|(?=[,;/()[\]{}])|$)",
        "r": r"(?:(?<=\s)|(?<=[,;/()[\]{}])|^)r(?:(?=\s)|(?=[,;/()[\]{}])|$)",
        "power bi": r"(?:\b|_)(?:power[\s\-_]?bi|powerbi)(?:\b|_)",
        "agentic ai": r"(?:\b|_)(?:agentic[\s\-_]+ai)(?:\b|_)",
        "prompt engineering": r"(?:\b|_)(?:prompt[\s\-_]+engineering)(?:\b|_)",
        "machine learning": r"(?:\b|_)(?:machine[\s\-_]+learning)(?:\b|_)",
        "deep learning": r"(?:\b|_)(?:deep[\s\-_]+learning)(?:\b|_)",
        "data analysis": r"(?:\b|_)(?:data[\s\-_]+analysis)(?:\b|_)",
        "spring boot": r"(?:\b|_)(?:spring[\s\-_]+boot)(?:\b|_)",
        "rest": r"(?:\b|_)(?:rest|restful|rest[\s\-_]?api[s]?)(?:\b|_)",
        "llm": r"(?:\b|_)(?:llm|llms)(?:\b|_)",
        "postgresql": r"(?:\b|_)(?:postgresql|postgres)(?:\b|_)",
    }

    for skill in SKILLS:
        if skill in custom_regex:
            patterns[skill] = re.compile(custom_regex[skill], re.IGNORECASE)
        elif " " in skill or "-" in skill:
            parts = re.split(r"[\s\-]+", skill)
            pattern_str = r"(?:\b|_)" + r"[\s\-_]+".join(re.escape(p) for p in parts) + r"(?:\b|_)"
            patterns[skill] = re.compile(pattern_str, re.IGNORECASE)
        else:
            patterns[skill] = re.compile(r"(?:\b|_)" + re.escape(skill) + r"(?:\b|_)", re.IGNORECASE)

    return patterns


_SKILL_PATTERNS = _build_skill_patterns()


def extract_skills(text):
    """
    Extract canonical skills from text (works on both raw and preprocessed text).

    Returns:
        list[str]: Alphabetically sorted list of canonical skill names.
    """
    if not text:
        return []

    found_skills = set()

    for skill, pattern in _SKILL_PATTERNS.items():
        if pattern.search(text):
            found_skills.add(skill)

    return sorted(found_skills)