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

SKILLS = skills_df["skill"].str.lower().tolist()


def extract_skills(text):
    """
    Extract skills from preprocessed text.
    """

    text = text.lower()

    found_skills = []

    for skill in SKILLS:
        if skill in text:
            found_skills.append(skill)

    return sorted(set(found_skills))