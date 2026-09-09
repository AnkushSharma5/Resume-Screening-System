import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

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
# LOAD CSS
# =====================================================

css_path = Path("assets/style.css")

if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class="hero-header">

<h1>
📄 AI Resume Screening System
</h1>

<p class="hero-subtitle">
Smart ATS Resume Analyzer · TF-IDF &amp; Semantic NLP · CrewAI Multi-Agent Pipeline
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
        """
    <div class="sidebar-brand">
        <div class="brand-icon">📄</div>
        <div class="brand-title">Resume Screener</div>
        <div class="brand-version">v3.0 • AI + Agents</div>
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
        height=250,
        placeholder="Paste the Job Description here...",
        label_visibility="collapsed",
    )

    st.markdown("")

    analyze_button = st.button("🚀 Analyze Resume", use_container_width=True)

    st.markdown("")

    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Select Mode** — Single or Multiple
        2. **Upload** your resume PDF(s)
        3. **Paste** the job description
        4. (Optional) Enable **CrewAI** pipeline
        5. Click **Analyze Resume**
        """)


# =====================================================
# INPUT VALIDATION
# =====================================================

if analyze_button:
    if mode == "Single Resume Analysis":
        if uploaded_resume is None:
            st.warning("⚠ Please upload a resume.")

            st.stop()

    else:
        if uploaded_resume is None or len(uploaded_resume) == 0:
            st.warning("⚠ Please upload resumes.")

            st.stop()

    if job_description.strip() == "":
        st.warning("⚠ Please enter a Job Description.")

        st.stop()
        # =====================================================
    # MULTIPLE RESUME RANKING
    # =====================================================

    if mode == "Multiple Resume Ranking":
        ranking = rank_resumes(uploaded_resume, job_description)

        st.success("✅ Ranking Completed Successfully")

        # ---- Ranking Table ----
        st.markdown(
            """
        <div class="section-card">
            <div class="section-header">
                <span class="icon">🏆</span>
                <h3>Resume Ranking</h3>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        ranking_df = pd.DataFrame(ranking)

        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

        st.markdown("")

        # ==========================================
        # ATS Score Comparison
        # ==========================================

        st.markdown(
            """
        <div class="section-card" style="padding-bottom:8px;">
            <div class="section-header">
                <span class="icon">📊</span>
                <h3>ATS Score Comparison</h3>
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
            color="Resume",
            color_discrete_sequence=[
                "#2563EB",
                "#4F46E5",
                "#10B981",
                "#F59E0B",
                "#EF4444",
            ],
        )

        bar_fig.update_traces(textposition="outside")

        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94A3B8"),
            height=450,
            xaxis_title="Resume",
            yaxis_title="ATS Score",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("")

        # ==========================================
        # Best Candidate
        # ==========================================

        best_resume = ranking[0]

        st.markdown(
            f"""
        <div class="best-candidate-card">
            <div class="trophy">🏆</div>
            <div style="font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#94A3B8;margin-bottom:4px;">Best Matching Candidate</div>
            <div class="candidate-name">📄 {best_resume["Resume"]}</div>
            <div class="candidate-score">{best_resume["ATS Score"]:.2f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        # ---- Skills Tabs ----
        st.markdown(
            """
        <div class="section-card" style="padding-bottom:8px;">
            <div class="section-header">
                <span class="icon">🎯</span>
                <h3>Skills Analysis — Best Candidate</h3>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        tab_matched, tab_missing = st.tabs(["✅ Matched Skills", "❌ Missing Skills"])

        with tab_matched:
            if best_resume["Matched Skills"]:
                chips_html = ""
                for skill in best_resume["Matched Skills"]:
                    chips_html += f'<span class="skill-chip matched">✔ {skill}</span>'
                st.markdown(
                    f'<div style="padding:8px 0;">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.info("No matched skills found.")

        with tab_missing:
            if best_resume["Missing Skills"]:
                chips_html = ""
                for skill in best_resume["Missing Skills"]:
                    chips_html += f'<span class="skill-chip missing">✘ {skill}</span>'
                st.markdown(
                    f'<div style="padding:8px 0;">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.success("No missing skills.")

        st.stop()

    # =====================================================
    # SINGLE RESUME ANALYSIS
    # =====================================================

    # ── Choose pipeline ──────────────────────────────────────
    if use_crew:
        pipeline_label = "🤖 CrewAI Multi-Agent"
        with st.spinner("Running CrewAI pipeline (Extraction → Matching → Feedback)…"):
            result = analyze_resume_with_crew(uploaded_resume, job_description)
        # Suggestions already in result["suggestions"] from the Feedback Agent
        crew_suggestions = result.get("suggestions", [])
    else:
        pipeline_label = "⚡ Classic NLP"
        with st.spinner("Analyzing resume…"):
            result = analyze_resume(uploaded_resume, job_description)
        crew_suggestions = None
    # ─────────────────────────────────────────────────────────

    st.success(f"✅ Resume Analyzed Successfully  [{pipeline_label}]")

    # ---- Score Metric Cards ----
    st.markdown(
        """
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">📊</span>
            <h3>Resume Analysis</h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Row 1: ATS / Combined Similarity / Skill Match
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
        <div class="metric-card blue">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">ATS Score</div>
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
        </div>
        """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
        <div class="metric-card emerald">
            <div class="metric-icon">🛠</div>
            <div class="metric-label">Skill Match</div>
            <div class="metric-value">{result["skill_match_score"]}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Row 2: TF-IDF vs Semantic — the two individual NLP scores
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
        <div class="metric-card" style="background:linear-gradient(135deg,#1e3a5f,#2563EB22);border:1px solid #2563EB44;">
            <div class="metric-icon">📝</div>
            <div class="metric-label">TF-IDF Similarity</div>
            <div class="metric-value" style="font-size:1.6rem;">{result.get("tfidf_similarity_score", "—")}%</div>
            <div style="font-size:0.7rem;color:#94A3B8;margin-top:4px;">Keyword overlap (classical NLP)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_semantic:
        st.markdown(
            f"""
        <div class="metric-card" style="background:linear-gradient(135deg,#1e3a2f,#10B98122);border:1px solid #10B98144;">
            <div class="metric-icon">🔮</div>
            <div class="metric-label">Semantic Similarity</div>
            <div class="metric-value" style="font-size:1.6rem;">{result.get("semantic_similarity_score", "—")}%</div>
            <div style="font-size:0.7rem;color:#94A3B8;margin-top:4px;">Sentence embeddings (all-MiniLM-L6-v2)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # =====================================================
    # SCORE VISUALIZATION
    # =====================================================

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
                "NLP",
                "NLP",
                "NLP",
                "Skills",
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
            "Overall": "#2563EB",
            "NLP": "#10B981",
            "Skills": "#F59E0B",
        },
    )

    score_fig.update_traces(texttemplate="%{text}%", textposition="outside")

    score_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8"),
        height=400,
        yaxis_range=[0, 115],
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text="Score Type",
    )

    st.plotly_chart(score_fig, use_container_width=True)

    st.markdown("")

    # =====================================================
    # SKILLS ANALYSIS
    # =====================================================

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

    tab_m, tab_x = st.tabs(["✅ Matched Skills", "❌ Missing Skills"])

    with tab_m:
        if result.get("matched_skills"):
            chips_html = ""
            for skill in result["matched_skills"]:
                chips_html += f'<span class="skill-chip matched">✔ {skill}</span>'

            st.markdown(
                f'<div style="padding:8px 0;">{chips_html}</div>',
                unsafe_allow_html=True,
            )

        else:
            st.info("No matched skills found.")

    with tab_x:
        if result.get("missing_skills"):
            chips_html = ""
            for skill in result["missing_skills"]:
                chips_html += f'<span class="skill-chip missing">✘ {skill}</span>'

            st.markdown(
                f'<div style="padding:8px 0;">{chips_html}</div>',
                unsafe_allow_html=True,
            )

        else:
            st.success("No missing skills.")

    st.markdown("")

    # =====================================================
    # AI SUGGESTIONS
    # =====================================================

    st.markdown(
        f"""
    <div class="section-card" style="padding-bottom:8px;">
        <div class="section-header">
            <span class="icon">💡</span>
            <h3>AI Improvement Suggestions
                <span style="font-size:0.7rem;font-weight:400;color:#64748B;margin-left:8px;">
                    {"🤖 LLM-powered (with fallback)" if not use_crew else "🤖 Feedback Agent"}
                </span>
            </h3>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Use crew suggestions (already generated by Feedback Agent) or call generate_suggestions
    if crew_suggestions is not None:
        suggestions = crew_suggestions
    else:
        suggestions = generate_suggestions(
            result,
            resume_text=result.get("resume_text", ""),
            job_description=job_description,
        )

    if suggestions:
        for idx, suggestion in enumerate(suggestions, 1):
            # Handle both {"title": ..., "detail": ...} dicts and plain strings
            if isinstance(suggestion, dict):
                title = suggestion.get("title", "")
                detail = suggestion.get("detail", "")
                st.markdown(
                    f"""
                <div class="suggestion-item" style="animation-delay: {idx * 0.1}s;">
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
                <div class="suggestion-item" style="animation-delay: {idx * 0.1}s;">
                    <div class="suggestion-num">{idx}</div>
                    <div class="suggestion-text">{suggestion}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    else:
        st.success("Your resume is already well optimized.")

    st.markdown("")

    # =====================================================
    # RESUME REPORT GENERATION
    # =====================================================

    st.markdown(
        """
    <div class="download-cta">
        <div class="cta-icon">📄</div>
        <div class="cta-title">Download Resume Report</div>
        <div class="cta-desc">Get a detailed PDF analysis of your resume</div>
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


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
<div class="premium-footer">

<div class="footer-title">
🚀 AI Resume Screening System
</div>

<div class="footer-tech">
    <span class="tech-badge">🐍 Python</span>
    <span class="tech-badge">🧠 NLP + Embeddings</span>
    <span class="tech-badge">🤖 CrewAI Agents</span>
    <span class="tech-badge">💬 LLM Feedback</span>
    <span class="tech-badge">🎨 Streamlit</span>
</div>

</div>
""",
    unsafe_allow_html=True,
)
