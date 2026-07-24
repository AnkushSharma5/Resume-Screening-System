import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from analyzer import analyze_resume
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
    initial_sidebar_state="expanded"
)


# =====================================================
# LOAD CSS
# =====================================================

css_path = Path("assets/style.css")

if css_path.exists():

    with open(css_path) as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True

        )


# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div style="background:linear-gradient(90deg,#2563EB,#4F46E5);
padding:28px;
border-radius:18px;
text-align:center;
margin-bottom:25px;
box-shadow:0px 8px 20px rgba(0,0,0,.15);">

<h1 style="color:white;margin:0;">
📄 AI Resume Screening System
</h1>

<p style="color:white;
font-size:18px;
margin-top:10px;">

Smart ATS Resume Analyzer using NLP & Machine Learning

</p>

</div>
""",
unsafe_allow_html=True
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙ Control Panel")

    st.markdown("---")

    mode = st.radio(

        "Select Mode",

        [

            "Single Resume Analysis",

            "Multiple Resume Ranking"

        ]

    )

    st.markdown("---")


    if mode == "Single Resume Analysis":

        uploaded_resume = st.file_uploader(

            "📂 Upload Resume",

            type=["pdf"]

        )

    else:

        uploaded_resume = st.file_uploader(

            "📂 Upload Resumes",

            type=["pdf"],

            accept_multiple_files=True

        )


    st.markdown("---")


    job_description = st.text_area(

        "📝 Job Description",

        height=250,

        placeholder="Paste the Job Description here..."

    )


    st.markdown("---")


    analyze_button = st.button(

        "🚀 Analyze Resume",

        use_container_width=True

    )


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

        ranking = rank_resumes(
            uploaded_resume,
            job_description
        )

        st.success("✅ Ranking Completed Successfully")

        st.markdown("## 🏆 Resume Ranking")

        ranking_df = pd.DataFrame(ranking)

        ranking_df.insert(
            0,
            "Rank",
            range(1, len(ranking_df) + 1)
        )

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # ==========================================
        # ATS Score Comparison
        # ==========================================

        st.subheader("📊 ATS Score Comparison")

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

        "#EF4444"

    ]

)

        bar_fig.update_traces(

            textposition="outside"

        )

        bar_fig.update_layout(

        template="plotly_white",

        font=dict(
            color="#1F2937"
        ),

            height=450,

            xaxis_title="Resume",

            yaxis_title="ATS Score",

            coloraxis_showscale=False,

            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            )

        )

        st.plotly_chart(

            bar_fig,

            use_container_width=True

        )

        st.markdown("---")

        # ==========================================
        # Best Candidate
        # ==========================================

        st.subheader("🥇 Best Matching Candidate")

        best_resume = ranking[0]

        card1, card2 = st.columns([3,1])

        with card1:

            st.info(

                f"""

### 📄 {best_resume['Resume']}

**ATS Score :**

{best_resume['ATS Score']:.2f}%

"""

            )

        with card2:

            st.metric(

                "ATS Score",

                f"{best_resume['ATS Score']:.2f}%"

            )

        st.markdown("---")

        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:

            st.subheader("✅ Matched Skills")

            if best_resume["Matched Skills"]:

                for skill in best_resume["Matched Skills"]:

                    st.success(skill)

            else:

                st.info("No matched skills found.")

        with skill_col2:

            st.subheader("❌ Missing Skills")

            if best_resume["Missing Skills"]:

                for skill in best_resume["Missing Skills"]:

                    st.error(skill)

            else:

                st.success("No missing skills.")

        st.stop()


    # =====================================================
    # SINGLE RESUME ANALYSIS
    # =====================================================

    result = analyze_resume(

        uploaded_resume,

        job_description

    )

    st.success("✅ Resume Analyzed Successfully")

    st.markdown("## 📊 Resume Analysis")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(

            "🎯 ATS Score",

            f"{result['score']}%"

        )

    with metric2:

        st.metric(

            "📄 Similarity",

            f"{result['similarity_score']}%"

        )

    with metric3:

        st.metric(

            "🛠 Skill Match",

            f"{result['skill_match_score']}%"

        )

    st.markdown("---")
        # =====================================================
    # SCORE VISUALIZATION
    # =====================================================

    st.subheader("📈 Resume Score Overview")


    score_data = pd.DataFrame(

        {

            "Category":[

                "ATS Score",

                "Similarity",

                "Skill Match"

            ],

            "Score":[

                result["score"],

                result["similarity_score"],

                result["skill_match_score"]

            ]

        }

    )


    score_fig = px.bar(

        score_data,

        x="Category",

        y="Score",

        text="Score",

        color="Score",

        color_continuous_scale="Blues"

    )


    score_fig.update_traces(

        texttemplate="%{text}%",

        textposition="outside"

    )


    score_fig.update_layout(

        template="plotly_white",

        height=400,

        yaxis_range=[0,100],

        coloraxis_showscale=False,

        margin=dict(

            l=20,

            r=20,

            t=30,

            b=20

        )

    )


    st.plotly_chart(

        score_fig,

        use_container_width=True

    )


    st.markdown("---")


    # =====================================================
    # SKILLS ANALYSIS
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("✅ Matched Skills")


        if result.get["matched_skills"]:

            for skill in result["matched_skills"]:

                st.success(

                    f"✔ {skill}"

                )

        else:

            st.info(

                "No matched skills found."

            )


    with col2:

        st.subheader("❌ Missing Skills")


        if result.get["missing_skills"]:

            for skill in result["missing_skills"]:

                st.error(

                    f"✘ {skill}"

                )

        else:

            st.success(

                "No missing skills."

            )


    st.markdown("---")


    # =====================================================
    # AI SUGGESTIONS
    # =====================================================

    st.subheader("💡 AI Improvement Suggestions")


    suggestions = generate_suggestions(

        result

    )


    if suggestions:

        for suggestion in suggestions:

            st.info(

                f"🔹 {suggestion}"

            )

    else:

        st.success(

            "Your resume is already well optimized."

        )


    st.markdown("---")


    # =====================================================
    # RESUME REPORT GENERATION
    # =====================================================

    st.subheader("📄 Download Resume Report")


    report = generate_report(

        result

    )


    st.download_button(

        label="⬇ Download PDF Report",

        data=report,

        file_name="Resume_Analysis_Report.pdf",

        mime="application/pdf",

        use_container_width=True

    )


# =====================================================
# FOOTER
# =====================================================

st.markdown(

"""
<div style="
text-align:center;
padding:20px;
margin-top:40px;
background:#f8fafc;
border-radius:15px;
">

<h4>
🚀 AI Resume Screening System
</h4>

<p>
Built using Python • NLP • Machine Learning • Streamlit
</p>

</div>
""",

unsafe_allow_html=True

)