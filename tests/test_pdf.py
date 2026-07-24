from utils.pdf_reader import extract_text_from_pdf

pdf_path = "resumes/resume1.pdf"

resume_text = extract_text_from_pdf(pdf_path)

print(resume_text)