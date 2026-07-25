import re


def extract_candidate_info(text):

    info = {}

    # Remove line breaks for better matching
    clean_text = text.replace("\n", " ")

    # Email
    email = re.search(
        r'[\w\.-]+@[\w\.-]+\.\w+',
        clean_text
    )

    info["email"] = email.group(0) if email else "Not Found"

    # Phone
    phone = re.search(
        r'(\+91[\s-]?)?[6-9]\d{9}',
        clean_text
    )

    info["phone"] = phone.group(0) if phone else "Not Found"

    # LinkedIn
    linkedin = re.search(
        r'(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9_-]+',
        clean_text,
        re.IGNORECASE
    )

    info["linkedin"] = linkedin.group(0) if linkedin else "Not Found"

    # GitHub
    github = re.search(
        r'(https?://)?(www\.)?github\.com/[A-Za-z0-9_-]+',
        clean_text,
        re.IGNORECASE
    )

    info["github"] = github.group(0) if github else "Not Found"

    return info