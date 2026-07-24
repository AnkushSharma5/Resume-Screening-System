from utils.preprocessing import preprocess_text
from utils.similarity import calculate_similarity

resume = """
Python
SQL
Machine Learning
Git
Docker

Worked on Machine Learning projects using Python.
"""

job_description = """
Looking for a Python Developer.

Required Skills:
Python
SQL
Machine Learning
Git
Docker
"""

clean_resume = preprocess_text(resume)
clean_jd = preprocess_text(job_description)

score = calculate_similarity(clean_resume, clean_jd)

print(f"Resume Match Score: {score}%")