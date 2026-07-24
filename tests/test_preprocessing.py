from utils.preprocessing import preprocess_text

sample_text = """
Hello!! My Name is Ankush Sharma.
I have completed 5 Machine Learning Projects in 2025.
"""

clean_text = preprocess_text(sample_text)

print("Original Text:\n")
print(sample_text)

print("\n" + "=" * 50 + "\n")

print("Cleaned Text:\n")
print(clean_text)