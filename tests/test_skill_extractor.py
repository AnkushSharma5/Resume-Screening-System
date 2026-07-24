from utils.preprocessing import preprocess_text
from utils.skill_extractor import extract_skills

sample_resume = """
Hello,

My name is Ankush Sharma.

Skills:
Python
Java
SQL
Machine Learning
Power BI
Git
Docker

I have completed several Machine Learning projects using Python and SQL.
"""

clean_text = preprocess_text(sample_resume)

skills = extract_skills(clean_text)

print("Detected Skills:\n")
print(skills)