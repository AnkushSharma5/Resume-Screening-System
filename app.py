from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analyzer import analyze_resume
from agents.crew import analyze_resume_with_crew
from utils.ranking import rank_resumes
from utils.report_generator import generate_report
from utils.suggestions import generate_suggestions


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# LOAD CSS & SVG ICONS
# =====================================================

css_path = Path("assets/style.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Clean, professional vector SVG icon for resume screening
DOC_ICON_SVG = """
<svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">
    <path d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="rgba(99,102,241,0.18)"/>
    <path d="M14 2V8H20" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16 13H8" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16 17H8" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M10 9H8" stroke="#A78BFA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

DOC_ICON_LARGE = """
<svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;">
    <path d="M14 2H6C4.89543 2 4 2.89543 4 4V20C4 21.1046 4.89543 22 6 22H18C19.1046 22 20 21.1046 20 20V8L14 2Z" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="rgba(99,102,241,0.2)"/>
    <path d="M14 2V8H20" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16 13H8" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M16 17H8" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M10 9H8" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


# =====================================================
# HERO HEADER
# =====================================================

st.markdown(
    f"""
<div class="hero-header">
    <div class="hero-title-wrapper">
        {DOC_ICON_LARGE}
        <h1>AI Resume Screening System</h1>
    </div>
    <p class="hero-subtitle">
        Enterprise ATS Analyzer · 70% Semantic &amp; 30% TF-IDF NLP · CrewAI Multi-Agent Pipeline
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown(
        f"""
    <div class="sidebar-brand">
        <div class="brand-icon-box">{DOC_ICON_SVG}</div>
        <div>
            <div class="brand-title">Resume Screener</div>
            <div class="brand-version">v3.0 • AI + AGENTS</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        '<div class="sidebar-label">⚙️ Analysis Mode</div>', unsafe_allow_html=True
    )

    mode = st.radio(
        "Select Mode",
        ["Single Resume Analysis", "Multiple Resume Ranking"],
        label_visibility="collapsed",
    )

    st.markdown("")

    # ── CrewAI toggle ──────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-label">🤖 Pipeline</div>', unsafe_allow_html=True
    )
    use_crew = st.checkbox(
        "Use multi-agent pipeline (CrewAI)",
        value=False,
        help=(
            "When enabled, runs a 3-agent CrewAI workflow: "
            "Extraction Agent → Matching Agent → Feedback Agent. "
            "Falls back to the classic pipeline if CrewAI is unavailable."
        ),
    )
    # ──────────────────────────────────────────────────────────

    st.markdown("")

    st.markdown(
        '<div class="sidebar-label">📂 Upload Resume</div>', unsafe_allow_html=True
    )

    if mode == "Single Resume Analysis":
        uploaded_resume = st.file_uploader(
            "📂 Upload Resume", type=["pdf"], label_visibility="collapsed"
        )
    else:
        uploaded_resume = st.file_uploader(
            "📂 Upload Resumes",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    st.markdown("")

    st.markdown(
        '<div class="sidebar-label">📝 Job Description</div>', unsafe_allow_html=True
    )

    job_description = st.text_area(
        "📝 Job Description",
        height=240,
        placeholder="Paste the Job Description here (e.g. required skills, responsibilities, tools)...",
        label_visibility="collapsed",
    )

    st.markdown("")

    analyze_button = st.button("🚀 Analyze Resume", use_container_width=True)

    st.markdown("")

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Select Mode** — Single Analysis or Multi-Resume Ranking.
        2. **Upload** resume PDF(s) from your computer.
        3. **Paste** the target Job Description.
        4. *(Optional)* Check **CrewAI** for autonomous multi-agent analysis.
        5. Click **🚀 Analyze Resume** to generate your ATS score and breakdown.
        """)


# =====================================================
# DASHBOARD CONTENT
# =====================================================

if analyze_button:
    # ── Validation ────────────────────────────────────────
    if mode == "Single Resume Analysis":
        if uploaded_resume is None:
            st.warning("⚠ Please upload a resume PDF before analyzing.")
            st.stop()
    else:
        if uploaded_resume is None or len(uploaded_resume) == 0:
            st.warning("⚠ Please upload at least one resume PDF to rank.")
            st.stop()

    if job_description.strip() == "":
        st.warning("⚠ Please paste or enter a Job Description.")
        st.stop()

    # =====================================================
    # MULTIPLE RESUME RANKING MODE
    # =====================================================
    if mode == "Multiple Resume Ranking":
        with st.spinner("Screening and ranking all uploaded resumes…"):
            ranking = rank_resumes(uploaded_resume, job_description)

        st.success("✅ Candidate Ranking Completed Successfully")

        # ---- Ranking Table ----
        st.markdown(
            """
        <div class="section-card">
            <div class="section-header">
                <span class="icon">🏆</span>
                <h3>Candidate Leaderboard</h3>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        ranking_df = pd.DataFrame(ranking)
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Resume": st.column_config.TextColumn("Resume File", width="medium"),
                "ATS Score": st.column_config.NumberColumn("ATS Score", format="%.2f%%"),
                "TF-IDF Score": st.column_config.NumberColumn("TF-IDF Score (30%)", format="%.2f%%"),
                "Semantic Score": st.column_config.NumberColumn("Semantic Score (70%)", format="%.2f%%"),
            },
        )

        st.markdown("")

        # ---- ATS Score Comparison Bar Chart ----
        st.markdown(
            """
        <div class="section-card" style="padding-bottom:8px;">
            <div class="section-header">
                <span class="icon">📊</span>
                <h3>Candidate Comparison Chart</h3>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        bar_fig = px.bar(
            ranking_df,
            x="Resume",
            y="ATS Score",
            text="ATS Score",
            hover_data=["TF-IDF Score", "Semantic Score"],
            color="Resume",
            color_discrete_sequence=[
                "#4F46E5",
                "#7C3AED",
                "#10B981",
                "#F59E0B",
                "#0EA5E9",
                "#EC4899",
            ],
        )

        bar_fig.update_traces(textposition="outside", texttemplate="%{text:.2f}%")
        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
            height=400,
            xaxis_title="Candidate Resume",
            yaxis_title="Overall ATS Score (%)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("")

        # ---- Best Candidate Card ----
        best_resume = ranking[0]

        st.markdown(
            f"""
        <div class="best-candidate-card">
            <div class="trophy">🏆</div>
            <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#A78BFA;margin-bottom:4px;">Top Ranked Candidate</div>
            <div class="candidate-name">{best_resume["Resume"]}</div>
            <div class="candidate-score">{best_resume["ATS Score"]:.2f}%</div>
            <div style="display:flex;justify-content:center;gap:14px;margin-top:12px;font-size:13px;color:#94A3B8;flex-wrap:wrap;">
                <span style="background:rgba(79,70,229,0.15);padding:5px 14px;border-radius:20px;border:1px solid rgba(99,102,241,0.3);">📝 TF-IDF: <strong style="color:#818CF8;">{best_resume.get("TF-IDF Score", 0):.2f}%</strong></span>
                <span style="background:rgba(16,185,129,0.15);padding:5px 14px;border-radius:20px;border:1px solid rgba(16,185,129,0.3);">🔮 Semantic: <strong style="color:#34D399;">{best_resume.get("Semantic Score", 0):.2f}%</strong></span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ---- Skills Tabs for Top Candidate ----
        st.markdown(
            """
        <div class="section-card" style="padding-bottom:8px;">
            <div class="section-header">
                <span class="icon">🎯</span>
                <h3>Top Candidate Skill Breakdown</h3>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        tab_matched, tab_missing = st.tabs([
            f"✅ Matched Skills ({len(best_resume['Matched Skills'])})",
            f"❌ Missing Skills ({len(best_resume['Missing Skills'])})",
        ])

        with tab_matched:
            if best_resume["Matched Skills"]:
                chips_html = "".join(
                    f'<span class="skill-chip matched">✔ {skill}</span>'
                    for skill in best_resume["Matched Skills"]
                )
                st.markdown(f'<div style="padding:8px 0;">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No matching skills detected for this role.")

        with tab_missing:
            if best_resume["Missing Skills"]:
                chips_html = "".join(
                    f'<span class="skill-chip missing">✘ {skill}</span>'
                    for skill in best_resume["Missing Skills"]
                )
                st.markdown(f'<div style="padding:8px 0;">{chips_html}</div>', unsafe_allow_html=True)
            else:
                st.success("Candidate matches 100% of required JD skills!")

        st.stop()

    # =====================================================
    # SINGLE RESUME ANALYSIS MODE
    # =====================================================
    if use_crew:
        pipeline_label = "🤖 CrewAI Multi-Agent Pipeline"
        with st.spinner("Running 3-agent CrewAI pipeline (Extraction → Matching → Feedback)…"):
            result = analyze_resume_with_crew(uploaded_resume, job_description)
        crew_suggestions = result.get("suggestions", [])
    else:
        pipeline_label = "⚡ Classic Dual-NLP Pipeline"
        with st.spinner("Analyzing resume with TF-IDF & Semantic Embeddings…"):
            result = analyze_resume(uploaded_resume, job_description)
        crew_suggestions = None

    st.success(f"✅ Resume Analyzed Successfully  [{pipeline_label}]")

    # ---- Primary Metrics (Row 1) ----
    st.markdown(
        """
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">📊</span>
            <h3>ATS Performance Metrics</h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
        <div class="metric-card blue">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Overall ATS Score</div>
            <div class="metric-value">{result["score"]}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
        <div class="metric-card purple">
            <div class="metric-icon">🔗</div>
            <div class="metric-label">Combined Similarity</div>
            <div class="metric-value">{result["similarity_score"]}%</div>
            <div style="font-size:11px;color:#94A3B8;margin-top:4px;">70% Semantic + 30% TF-IDF</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
        <div class="metric-card emerald">
            <div class="metric-icon">🛠</div>
            <div class="metric-label">Skill Match Score</div>
            <div class="metric-value">{result["skill_match_score"]}%</div>
            <div style="font-size:11px;color:#94A3B8;margin-top:4px;">Matched vs Required Skills</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- Dual NLP Similarity Breakdown (Row 2) ----
    st.markdown(
        """
    <div class="section-card" style="padding-bottom:4px;">
        <div class="section-header">
            <span class="icon">🧠</span>
            <h3>Dual NLP Similarity Breakdown</h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_tfidf, col_semantic = st.columns(2)

    with col_tfidf:
        st.markdown(
            f"""
        <div class="metric-card" style="border-top:3px solid #6366F1;">
            <div class="metric-icon">📝</div>
            <div class="metric-label">TF-IDF Keyword Similarity <span style="font-size:11px;background:rgba(99,102,241,0.2);padding:2px 8px;border-radius:10px;margin-left:6px;color:#A5B4FC;">30% Weight</span></div>
            <div class="metric-value" style="font-size:1.6rem;color:#A5B4FC;">{result.get("tfidf_similarity_score", "—")}%</div>
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:4px;">Exact keyword frequency overlap</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_semantic:
        st.markdown(
            f"""
        <div class="metric-card" style="border-top:3px solid #10B981;">
            <div class="metric-icon">🔮</div>
            <div class="metric-label">Semantic Embedding Similarity <span style="font-size:11px;background:rgba(16,185,129,0.2);padding:2px 8px;border-radius:10px;margin-left:6px;color:#6EE7B7;">70% Weight</span></div>
            <div class="metric-value" style="font-size:1.6rem;color:#6EE7B7;">{result.get("semantic_similarity_score", "—")}%</div>
            <div style="font-size:0.75rem;color:#94A3B8;margin-top:4px;">Dense vector embeddings (all-MiniLM-L6-v2)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- Score Overview Chart ----
    st.markdown(
        """
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">📈</span>
            <h3>Score Overview</h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    score_data = pd.DataFrame(
        {
            "Category": [
                "ATS Score",
                "TF-IDF Similarity",
                "Semantic Similarity",
                "Combined Similarity",
                "Skill Match",
            ],
            "Score": [
                result["score"],
                result.get("tfidf_similarity_score", 0),
                result.get("semantic_similarity_score", 0),
                result["similarity_score"],
                result["skill_match_score"],
            ],
            "Type": [
                "Overall",
                "NLP Breakdown",
                "NLP Breakdown",
                "Combined NLP",
                "Skill Match",
            ],
        }
    )

    score_fig = px.bar(
        score_data,
        x="Category",
        y="Score",
        text="Score",
        color="Type",
        color_discrete_map={
            "Overall": "#4F46E5",
            "NLP Breakdown": "#0EA5E9",
            "Combined NLP": "#7C3AED",
            "Skill Match": "#10B981",
        },
    )

    score_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    score_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
        height=380,
        yaxis_range=[0, 115],
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="Metric Type",
    )

    st.plotly_chart(score_fig, use_container_width=True)

    st.markdown("")

    # ---- Skills Breakdown Tabs ----
    st.markdown(
        """
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">🎯</span>
            <h3>Skills Analysis</h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    matched_list = result.get("matched_skills", [])
    missing_list = result.get("missing_skills", [])

    tab_m, tab_x = st.tabs([
        f"✅ Matched Skills ({len(matched_list)})",
        f"❌ Missing Skills ({len(missing_list)})",
    ])

    with tab_m:
        if matched_list:
            chips_html = "".join(
                f'<span class="skill-chip matched">✔ {skill}</span>'
                for skill in matched_list
            )
            st.markdown(f'<div style="padding:8px 0;">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No matching skills detected.")

    with tab_x:
        if missing_list:
            chips_html = "".join(
                f'<span class="skill-chip missing">✘ {skill}</span>'
                for skill in missing_list
            )
            st.markdown(f'<div style="padding:8px 0;">{chips_html}</div>', unsafe_allow_html=True)
        else:
            st.success("Candidate matches 100% of required job skills.")

    st.markdown("")

    # ---- Improvement Suggestions ----
    if crew_suggestions is not None:
        suggestions = crew_suggestions
    else:
        suggestions = generate_suggestions(
            result,
            resume_text=result.get("resume_text", ""),
            job_description=job_description,
        )

    # Dynamic source badge
    is_llm = bool(suggestions and isinstance(suggestions[0], dict))
    if is_llm:
        source_label = "🤖 Feedback Agent (LLM)" if use_crew else "🤖 LLM-Generated Feedback"
        badge_style = "color:#34D399;background:rgba(16,185,129,0.12);padding:3px 10px;border-radius:12px;border:1px solid rgba(16,185,129,0.3);"
    else:
        source_label = "📋 Rule-Based Fallback" if use_crew else "📋 Rule-Based Feedback"
        badge_style = "color:#F59E0B;background:rgba(245,158,11,0.12);padding:3px 10px;border-radius:12px;border:1px solid rgba(245,158,11,0.3);"

    st.markdown(
        f"""
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">💡</span>
            <h3>Improvement Suggestions
                <span style="font-size:0.75rem;font-weight:600;margin-left:10px;{badge_style}">
                    {source_label}
                </span>
            </h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if suggestions:
        for idx, suggestion in enumerate(suggestions, 1):
            if isinstance(suggestion, dict):
                title = suggestion.get("title", "")
                detail = suggestion.get("detail", "")
                st.markdown(
                    f"""
                <div class="suggestion-item">
                    <div class="suggestion-num">{idx}</div>
                    <div class="suggestion-text">
                        <strong>{title}</strong><br>{detail}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="suggestion-item">
                    <div class="suggestion-num">{idx}</div>
                    <div class="suggestion-text">{suggestion}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.success("Your resume is well optimized for this role.")

    st.markdown("")

    # ---- Download PDF Report CTA ----
    st.markdown(
        """
    <div class="download-cta">
        <div class="cta-icon">📄</div>
        <div class="cta-title">Download Resume Report</div>
        <div class="cta-desc">Get a detailed PDF analysis of your resume and ATS fit</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    report = generate_report(result)

    st.download_button(
        label="⬇ Download PDF Report",
        data=report,
        file_name="Resume_Analysis_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

else:
    # =====================================================
    # DEFAULT LANDING STATE (Clean & Minimal)
    # =====================================================
    st.markdown(
        f"""
    <div class="landing-card">
        <div class="landing-icon-box">{DOC_ICON_LARGE}</div>
        <div class="landing-title">Ready for Resume Screening</div>
        <div class="landing-desc">
            Upload candidate resumes on the left sidebar, paste the target job description, and click <strong>🚀 Analyze Resume</strong> to evaluate ATS match scores and skill gaps.
        </div>
        <div>
            <span class="landing-pill">✨ 68 Technical Skills</span>
            <span class="landing-pill">🧠 70% Semantic + 30% TF-IDF</span>
            <span class="landing-pill">🤖 Optional CrewAI Pipeline</span>
            <span class="landing-pill">📄 PDF Report Export</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
<div class="premium-footer">
    <div class="footer-title">🚀 AI Resume Screening System</div>
    <div class="footer-tech">
        <span class="tech-badge">🐍 Python</span>
        <span class="tech-badge">🧠 NLP &amp; Embeddings</span>
        <span class="tech-badge">🤖 CrewAI Agents</span>
        <span class="tech-badge">💬 LLM Suggestions</span>
        <span class="tech-badge">📄 ReportLab</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
