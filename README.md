# 🤖 AI Resume Screening System

### Intelligent ATS-Based Resume Analyzer & Candidate Ranking Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-Agents-6366F1?style=for-the-badge)
![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-10B981?style=for-the-badge)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Open-success?style=for-the-badge)](https://YOUR-STREAMLIT-APP.streamlit.app)

---
### 📌 ATS Scoring • Resume Ranking • Skill Matching • AI Suggestions

> An AI-powered Resume Screening System that analyzes resumes against
> job descriptions, calculates ATS compatibility scores, ranks
> candidates, identifies missing skills, and generates personalized
> improvement suggestions.
:::

------------------------------------------------------------------------

# 📖 Overview

The **AI Resume Screening System** is an AI-powered recruitment
assistant that automates resume evaluation using ATS-inspired scoring
techniques.

The application extracts resume content from PDF files, compares it with
a given Job Description, calculates ATS compatibility scores, identifies
matched and missing skills, ranks multiple candidates, and generates
personalized suggestions to improve resume quality.

With interactive visualizations and automated candidate ranking, the
system enables recruiters to efficiently shortlist the most suitable
candidates while helping job seekers optimize their resumes.

------------------------------------------------------------------------

# ✨ Features

### 📄 Single Resume Analysis

-   ATS Compatibility Score
-   Resume Parsing (PDF)
-   Job Description Matching
-   Skill Extraction
-   Missing Skill Identification
-   Personalized AI Suggestions

### 🏆 Multiple Resume Ranking

-   Upload Multiple Resumes
-   Automatic Candidate Ranking
-   ATS Score Comparison
-   Best Candidate Identification
-   Recruiter-Friendly Candidate Comparison

### 📊 Interactive Dashboard

-   ATS Score Visualization
-   Resume Comparison Charts
-   Interactive Plotly Graphs
-   Downloadable PDF Report

### 🤖 AI-Powered Evaluation

-   Keyword Matching
-   Skill Gap Analysis
-   Resume Quality Assessment
-   Resume Strength Detection
-   Actionable Resume Recommendations

------------------------------------------------------------------------

# 🖼️ Screenshots

## 🏠 Home Page

![Home](screenshots/home.png)

## 📄 Resume Analysis

![Resume Analysis](screenshots/resume-analysis.png)

## 🎯 Skills & AI Suggestions

![Skills](screenshots/skills-suggestions.png)

## 🏆 Multiple Resume Ranking

![Ranking](screenshots/resume-ranking.png)

## 👑 Best Candidate Selection

![Best Candidate](screenshots/best-candidate.png)

------------------------------------------------------------------------

# ⚙️ Tech Stack

  Category           Technologies
  ------------------ ---------------
  Language           Python
  Framework          Streamlit
  Data Analysis      Pandas, NumPy
  Machine Learning   Scikit-learn
  Visualization      Plotly
  PDF Processing     PyPDF2
  Styling            HTML, CSS
  Version Control    Git, GitHub

------------------------------------------------------------------------

# 🧠 Workflow

``` text
Resume Upload
      │
      ▼
PDF Text Extraction
      │
      ▼
Skill Extraction
      │
      ▼
Job Description Matching
      │
      ▼
ATS Score Calculation
      │
      ├────────► Resume Suggestions
      │
      ▼
Multiple Resume Ranking
      │
      ▼
Best Candidate Selection
```

------------------------------------------------------------------------

# 🤖 Agentic AI Architecture

## Dual NLP Similarity

Resumes are scored against the job description using **two complementary NLP techniques** run in parallel:

| Technique | Method | Strength |
|-----------|--------|----------|
| **TF-IDF Similarity** | Keyword frequency vectors + cosine similarity | Catches exact keyword overlap |
| **Semantic Similarity** | `all-MiniLM-L6-v2` sentence embeddings + cosine similarity | Catches meaning matches even without exact keywords |

Both scores are averaged into a **Combined Similarity** score, then weighted with Skill Match (`0.7 × similarity + 0.3 × skill_score`) to produce the final ATS score.

## 3-Agent CrewAI Pipeline

An optional multi-agent pipeline (toggle in the sidebar) orchestrates the analysis as three sequential CrewAI agents:

``` text
┌─────────────────────────┐
│   1. Extraction Agent   │  Wraps: extract_text_from_pdf() + extract_skills()
│   (PDF → Skills)        │  Output: resume text, skill list
└────────────┬────────────┘
             │ context
             ▼
┌─────────────────────────┐
│   2. Matching Agent     │  Wraps: calculate_similarity() + calculate_semantic_similarity()
│   (Scores + Gaps)       │         + calculate_skill_match()
└────────────┬────────────┘  Output: TF-IDF score, semantic score, matched/missing skills
             │ context
             ▼
┌─────────────────────────┐
│   3. Feedback Agent     │  Wraps: generate_suggestions() (LLM or rule-based fallback)
│   (Suggestions)         │  Output: personalised improvement suggestions
└─────────────────────────┘
```

**Process:** `Process.sequential` — each agent's output is passed as context to the next.

## LLM-Powered Suggestions

- Set `LLM_PROVIDER` (`openai` / `anthropic` / `gemini`) and `LLM_API_KEY` env vars to enable LLM feedback.
- The prompt returns **structured JSON** (`[{"title": ..., "detail": ...}]`) for clean rendering.
- **Fallback:** If no API key is set or the call fails, rule-based suggestions are used automatically — the app never crashes.

------------------------------------------------------------------------

# 📁 Project Structure

``` text
Resume-Screening-System
├── screenshots/
│   ├── home.png
│   ├── resume-analysis.png
│   ├── skills-suggestions.png
│   ├── resume-ranking.png
│   └── best-candidate.png
├── assets/
│   └── style.css
├── utils/
├── analyzer.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# 🚀 Installation

``` bash
git clone https://github.com/AnkushSharma5/Resume-Screening-System.git
cd Resume-Screening-System
python -m venv venv
```

**Windows**

``` bash
venv\Scripts\activate
```

**Linux/macOS**

``` bash
source venv/bin/activate
```

``` bash
pip install -r requirements.txt
https://ankush-resume-screening.streamlit.app
```

------------------------------------------------------------------------

# 📊 Core Functionalities

-   ✅ Single Resume ATS Analysis
-   ✅ Multiple Resume Ranking
-   ✅ Job Description Matching
-   ✅ Skill Extraction
-   ✅ Missing Skill Detection
-   ✅ ATS Compatibility Score
-   ✅ Candidate Comparison Dashboard
-   ✅ Interactive Plotly Visualizations
-   ✅ Personalized Resume Suggestions
-   ✅ Downloadable Analysis Report

------------------------------------------------------------------------

# 🎯 Future Enhancements

-   ✅ LLM-powered Resume Feedback *(implemented)*
-   ✅ Semantic Similarity with Sentence Embeddings *(implemented)*
-   ✅ Multi-agent CrewAI Pipeline *(implemented)*
-   📄 DOCX Resume Support
-   🖼 OCR-based Resume Parsing
-   🔐 Recruiter Authentication
-   🗄 Database Integration
-   📧 Email Report Generation
-   💬 AI Interview Question Generator
-   📈 Resume Analytics Dashboard
-   🌐 Multi-language Resume Support

------------------------------------------------------------------------

# 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit your changes.
4.  Open a Pull Request.

------------------------------------------------------------------------

# 📜 License

This project is licensed under the **MIT License**.

------------------------------------------------------------------------


### ⭐ If you found this project useful, consider giving it a Star!

**Made with ❤️ by Ankush Sharma**
:::
