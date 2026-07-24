from utils.matching import calculate_skill_match

resume = [
    "python",
    "sql",
    "git",
    "docker"
]

jd = [
    "python",
    "sql",
    "git",
    "docker",
    "aws"
]

score, matched, missing = calculate_skill_match(resume, jd)

print("Skill Match:", score, "%")
print()

print("Matched Skills:")
print(matched)

print()

print("Missing Skills:")
print(missing)