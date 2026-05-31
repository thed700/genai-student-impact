"""
╔══════════════════════════════════════════════════════════════════╗
║         AI STUDENT IMPACT ANALYSER  —  Streamlit App             ║
║  Tab 1 : Macro Insights & Analytics Dashboard                    ║
║  Tab 2 : Personalised AI Safe-Usage Recommendations              ║
║  Tab 3 : AI Student Impact Predictor (ML Inference)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AI Student Impact Analyser",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL THEME & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
ACCENT   = "#6C63FF"
ACCENT2  = "#FF6584"
ACCENT3  = "#43D9AD"
BG_DARK  = "#0E1117"
BG_CARD  = "#1A1D27"
BG_CARD2 = "#22263A"
FONT     = "'DM Sans', sans-serif"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family=FONT, color="#E2E8F0"),
        colorway=[ACCENT, ACCENT2, ACCENT3, "#F6AD55", "#68D391", "#FC8181"],
        xaxis=dict(gridcolor="#2D3748", linecolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748", linecolor="#2D3748"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Root ── */
html, body, [class*="css"]  {{ font-family: {FONT}; }}
.stApp {{ background: {BG_DARK}; }}

/* ── Hide default header ── */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ── Tab bar ── */
div[data-testid="stTabs"] > div:first-child {{
    background: {BG_CARD};
    border-radius: 14px;
    padding: 4px 8px;
    gap: 4px;
    border: 1px solid #2D3748;
}}
button[data-baseweb="tab"] {{
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #94A3B8 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: {ACCENT} !important;
    color: #fff !important;
    box-shadow: 0 0 18px {ACCENT}66 !important;
}}
div[data-testid="stTabContent"] {{
    padding-top: 1.5rem;
}}

/* ── Metric cards ── */
div[data-testid="stMetric"] {{
    background: {BG_CARD};
    border: 1px solid #2D3748;
    border-radius: 16px;
    padding: 20px 24px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
div[data-testid="stMetric"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.15);
}}
div[data-testid="stMetric"] label {{
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: #F1F5F9 !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}}
div[data-testid="stMetricDelta"] {{
    font-size: 0.82rem !important;
}}

/* ── Selectbox / Sliders ── */
.stSelectbox > div > div,
.stSlider > div {{
    background: {BG_CARD2} !important;
    border-radius: 10px;
    border: 1px solid #2D3748 !important;
}}

/* ── Primary button ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {ACCENT}, #8B5CF6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 36px !important;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 20px {ACCENT}55 !important;
    transition: all 0.2s ease !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px {ACCENT}77 !important;
}}

/* ── Card containers ── */
.card {{
    background: {BG_CARD};
    border: 1px solid #2D3748;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 18px;
}}
.card-accent {{
    border-left: 4px solid {ACCENT};
}}

/* ── Page hero ── */
.hero {{
    background: linear-gradient(135deg, {BG_CARD} 0%, #161929 100%);
    border: 1px solid #2D3748;
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, {ACCENT}22 0%, transparent 70%);
    pointer-events: none;
}}
.hero h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.03em;
    margin: 0 0 8px 0;
    line-height: 1.2;
}}
.hero p {{
    color: #94A3B8;
    font-size: 1rem;
    margin: 0;
    max-width: 600px;
}}
.hero .badge {{
    display: inline-block;
    background: {ACCENT}22;
    color: {ACCENT};
    border: 1px solid {ACCENT}55;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 16px;
}}

/* ── Section header ── */
.section-header {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #CBD5E1;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 28px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: #2D3748;
}}

/* ── Recommendation cards ── */
.rec-card {{
    background: {BG_CARD2};
    border-radius: 14px;
    padding: 20px 24px;
    border: 1px solid #2D3748;
    margin-bottom: 14px;
}}
.rec-card .label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748B;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.rec-card .value {{
    font-size: 1.4rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 4px;
}}
.rec-card .sub {{
    font-size: 0.85rem;
    color: #94A3B8;
}}

/* ── Prediction result ── */
.pred-high {{
    background: linear-gradient(135deg, #7f1d1d22, #991b1b11);
    border: 1px solid #EF444455;
    border-radius: 18px;
    padding: 28px;
}}
.pred-medium {{
    background: linear-gradient(135deg, #78350f22, #92400e11);
    border: 1px solid #F59E0B55;
    border-radius: 18px;
    padding: 28px;
}}
.pred-low {{
    background: linear-gradient(135deg, #14532d22, #16653411);
    border: 1px solid #22C55E55;
    border-radius: 18px;
    padding: 28px;
}}
.pred-label {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 8px;
}}
.pred-value {{
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 6px;
}}
.pred-desc {{
    font-size: 0.88rem;
    color: #94A3B8;
}}

/* ── Warning box ── */
.warning-box {{
    background: #7c2d1222;
    border: 1px solid #EA580C55;
    border-radius: 12px;
    padding: 18px 22px;
    margin-top: 18px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}}
.warning-box .icon {{ font-size: 1.4rem; }}
.warning-box .text {{ font-size: 0.9rem; color: #FDBA74; line-height: 1.5; }}

/* ── Info box ── */
.info-box {{
    background: {ACCENT}11;
    border: 1px solid {ACCENT}44;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 14px;
    font-size: 0.88rem;
    color: #A5B4FC;
    line-height: 1.6;
}}

/* ── Divider ── */
hr.styled {{ border: none; border-top: 1px solid #2D3748; margin: 24px 0; }}

/* ── Plotly chart tweaks ── */
.js-plotly-plot .plotly .modebar {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# Get script directory for robust path resolution (works locally & on Streamlit Cloud)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "data/ai_student_impact_dataset.csv")
BURNOUT_PKL  = os.path.join(SCRIPT_DIR, "models/burnout_model.pkl")
GPA_PKL      = os.path.join(SCRIPT_DIR, "models/gpa_model.pkl")
LE_PKL       = os.path.join(SCRIPT_DIR, "models/label_encoder.pkl")
PIPE_PKL     = os.path.join(SCRIPT_DIR, "models/feature_pipeline.pkl")

ORDINAL_FEATURES = {
    "Year_of_Study"           : ["Freshman","Sophomore","Junior","Senior","Graduate"],
    "Prompt_Engineering_Skill": ["Beginner","Intermediate","Advanced"],
}
NOMINAL_FEATURES = ["Major_Category","Primary_Use_Case","Institutional_Policy"]
NUMERIC_FEATURES = [
    "Pre_Semester_GPA","Weekly_GenAI_Hours","Tool_Diversity",
    "Paid_Subscription","Traditional_Study_Hours","Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams","Skill_Retention_Score",
]

MAJOR_META = {
    "STEM"      : {"icon":"⚗️",  "color":"#6C63FF"},
    "Business"  : {"icon":"💼",  "color":"#F6AD55"},
    "Humanities": {"icon":"📚",  "color":"#68D391"},
    "Medical"   : {"icon":"🏥",  "color":"#FC8181"},
    "Arts"      : {"icon":"🎨",  "color":"#76E4F7"},
}

BURNOUT_META = {
    "Low"   : {"color":"#22C55E", "emoji":"🟢", "css":"pred-low"},
    "Medium": {"color":"#F59E0B", "emoji":"🟡", "css":"pred-medium"},
    "High"  : {"color":"#EF4444", "emoji":"🔴", "css":"pred-high"},
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["GPA_Change"]       = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
    df["Paid_Subscription"]= df["Paid_Subscription"].astype(int)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    clf  = joblib.load(BURNOUT_PKL)
    reg  = joblib.load(GPA_PKL)
    le   = joblib.load(LE_PKL)
    pipe = joblib.load(PIPE_PKL)
    return clf, reg, le, pipe


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — plotly figure defaults
# ─────────────────────────────────────────────────────────────────────────────
def _fig_defaults(fig, height=420):
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — MACRO INSIGHTS & ANALYTICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def tab_dashboard(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero">
        <div class="badge">📊 Dataset · 50,000 Students</div>
        <h1>Macro Insights Dashboard</h1>
        <p>Explore how Generative AI usage patterns relate to academic performance,
        burnout risk, and skill retention across the full student population.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics Row ──────────────────────────────────────────────────────
    avg_ai_hrs   = df["Weekly_GenAI_Hours"].mean()
    avg_gpa_chg  = df["GPA_Change"].mean()
    pct_high_brn = (df["Burnout_Risk_Level"] == "High").mean() * 100
    avg_skill_ret= df["Skill_Retention_Score"].mean()
    pct_low_brn  = (df["Burnout_Risk_Level"] == "Low").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg AI Hours / Week",   f"{avg_ai_hrs:.1f} hrs",
              delta=f"Range 0–40 hrs")
    c2.metric("Avg GPA Change",        f"+{avg_gpa_chg:.3f}",
              delta="Post vs Pre semester", delta_color="normal")
    c3.metric("High Burnout Rate",     f"{pct_high_brn:.1f}%",
              delta=f"{pct_low_brn:.1f}% Low risk", delta_color="inverse")
    c4.metric("Avg Skill Retention",   f"{avg_skill_ret:.1f}",
              delta="out of 100")
    c5.metric("Total Records",         "50,000",
              delta="5 major categories")

    st.markdown('<hr class="styled">', unsafe_allow_html=True)

    # ── Chart Row 1: GenAI Hours vs Burnout & Anxiety ────────────────────────
    st.markdown('<div class="section-header">🔥 AI Exposure & Burnout Dynamics</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        # Violin: Weekly GenAI Hours distribution by Burnout Level
        order = ["Low", "Medium", "High"]
        colors= {"Low": ACCENT3, "Medium": "#F6AD55", "High": ACCENT2}
        fig = go.Figure()
        for lvl in order:
            sub = df[df["Burnout_Risk_Level"] == lvl]["Weekly_GenAI_Hours"]
            fig.add_trace(go.Violin(
                y=sub, name=lvl,
                box_visible=True, meanline_visible=True,
                fillcolor=colors[lvl], opacity=0.75,
                line_color=colors[lvl], points=False,
            ))
        fig.update_layout(
            title="Weekly AI Hours by Burnout Level",
            yaxis_title="Weekly GenAI Hours",
            violingap=0.3, violinmode="overlay",
        )
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Heatmap: Burnout Level × Anxiety Bin → avg AI hours
        df["Anxiety_Bin"] = pd.cut(
            df["Anxiety_Level_During_Exams"],
            bins=[0,2,4,6,8,10], labels=["1-2","3-4","5-6","7-8","9-10"]
        )
        heat = (df.groupby(["Burnout_Risk_Level","Anxiety_Bin"], observed=True)
                  ["Weekly_GenAI_Hours"].mean()
                  .reset_index()
                  .pivot(index="Burnout_Risk_Level",
                         columns="Anxiety_Bin",
                         values="Weekly_GenAI_Hours")
                  .reindex(["Low","Medium","High"]))

        fig2 = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale=[[0,"#1A1D27"],[0.5,"#6C63FF"],[1.0,"#FF6584"]],
            text=np.round(heat.values, 1),
            texttemplate="%{text}h",
            colorbar=dict(thickness=12, tickfont=dict(color="#94A3B8")),
        ))
        fig2.update_layout(
            title="Avg AI Hours: Burnout × Exam Anxiety",
            xaxis_title="Exam Anxiety Level (binned)",
            yaxis_title="Burnout Risk Level",
        )
        _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart Row 2: Sweet Spot ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 The AI Sweet Spot — Optimal Hours</div>',
                unsafe_allow_html=True)

    df["AI_Hour_Bin"] = pd.cut(df["Weekly_GenAI_Hours"],
                               bins=10, precision=0)
    sweet = (df.groupby("AI_Hour_Bin", observed=True)
               .agg(
                   avg_skill   =("Skill_Retention_Score","mean"),
                   pct_high_brn=("Burnout_Risk_Level",
                                 lambda x: (x=="High").mean()*100),
                   count       =("Student_ID","count"),
                )
               .reset_index())
    sweet["bin_mid"] = sweet["AI_Hour_Bin"].apply(lambda iv: iv.mid).astype(float)
    sweet = sweet.sort_values("bin_mid")

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Bar(
        x=sweet["bin_mid"], y=sweet["avg_skill"],
        name="Avg Skill Retention",
        marker_color=ACCENT, opacity=0.8,
        hovertemplate="Hours: %{x:.0f}<br>Skill Retention: %{y:.1f}<extra></extra>",
    ), secondary_y=False)
    fig3.add_trace(go.Scatter(
        x=sweet["bin_mid"], y=sweet["pct_high_brn"],
        name="High Burnout %",
        mode="lines+markers",
        line=dict(color=ACCENT2, width=3),
        marker=dict(size=8, color=ACCENT2, symbol="circle"),
        hovertemplate="Hours: %{x:.0f}<br>High Burnout: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    # Sweet spot band (8-12 hrs)
    fig3.add_vrect(x0=8, x1=12,
                   fillcolor=ACCENT3, opacity=0.07,
                   annotation_text="Sweet Spot",
                   annotation_position="top left",
                   annotation_font_color=ACCENT3,
                   line_width=1, line_color=ACCENT3)

    fig3.update_layout(
        title="Skill Retention vs High Burnout Rate by Weekly AI Hours",
        xaxis_title="Weekly GenAI Hours",
        legend=dict(orientation="h", y=1.05),
        **PLOTLY_TEMPLATE["layout"],
        height=420, margin=dict(l=0,r=0,t=40,b=0),
    )
    fig3.update_yaxes(title_text="Avg Skill Retention Score",
                      secondary_y=False,
                      gridcolor="#2D3748", color="#E2E8F0")
    fig3.update_yaxes(title_text="High Burnout Rate (%)",
                      secondary_y=True,
                      gridcolor="rgba(0,0,0,0)", color=ACCENT2)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        💡 <strong>Reading the chart:</strong> The shaded green band (8–12 hrs/week) represents the
        empirical sweet spot — Skill Retention peaks here before declining sharply, while the High
        Burnout rate begins its steep climb beyond ~12 hrs. Students spending more than
        <strong>20 hrs/week</strong> on GenAI tools show &gt;50% high-burnout incidence with
        diminishing skill retention returns.
    </div>
    """, unsafe_allow_html=True)

    # ── Chart Row 3: Breakdown by Major ──────────────────────────────────────
    st.markdown('<div class="section-header">🏫 Breakdown by Academic Major</div>',
                unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="large")

    with col_c:
        major_stats = (df.groupby("Major_Category")
                         .agg(avg_gpa  =("Post_Semester_GPA","mean"),
                              avg_hours =("Weekly_GenAI_Hours","mean"),
                              avg_skill =("Skill_Retention_Score","mean"))
                         .reset_index()
                         .sort_values("avg_gpa", ascending=True))
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            y=major_stats["Major_Category"],
            x=major_stats["avg_gpa"],
            orientation="h",
            marker_color=[MAJOR_META[m]["color"]
                          for m in major_stats["Major_Category"]],
            text=major_stats["avg_gpa"].round(3),
            textposition="outside",
            hovertemplate="%{y}: GPA %{x:.3f}<extra></extra>",
        ))
        fig4.update_layout(title="Average Post-Semester GPA by Major",
                           xaxis_title="Avg GPA")
        _fig_defaults(fig4, height=340)
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        brn_major = (df.groupby(["Major_Category","Burnout_Risk_Level"])
                       .size().reset_index(name="count"))
        total_major= df.groupby("Major_Category").size().reset_index(name="total")
        brn_major  = brn_major.merge(total_major, on="Major_Category")
        brn_major["pct"] = brn_major["count"] / brn_major["total"] * 100

        color_map = {"High": ACCENT2, "Medium": "#F6AD55", "Low": ACCENT3}
        fig5 = px.bar(brn_major, x="Major_Category", y="pct",
                      color="Burnout_Risk_Level",
                      color_discrete_map=color_map,
                      barmode="stack",
                      text=brn_major["pct"].round(0).astype(str)+"%",
                      title="Burnout Distribution by Major (%)")
        fig5.update_traces(textposition="inside", textfont_size=11)
        fig5.update_layout(yaxis_title="Student Share (%)",
                           xaxis_title="",
                           legend_title="Burnout Level")
        _fig_defaults(fig5, height=340)
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PERSONALISED RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def tab_recommendations(df: pd.DataFrame) -> None:
    st.markdown("""
    <div class="hero">
        <div class="badge">🎯 Data-Driven Guidance</div>
        <h1>Personalised AI Safe-Usage Recommendations</h1>
        <p>Select your academic profile and discover evidence-based habits shared
        by the highest-performing, lowest-burnout students in your cohort.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        major = st.selectbox("🏫 Your Major Category",
                             sorted(df["Major_Category"].unique()))
    with c2:
        year  = st.selectbox("📅 Year of Study",
                             ["Freshman","Sophomore","Junior","Senior","Graduate"])
    with c3:
        policy = st.selectbox("🏛️ Institutional AI Policy",
                              sorted(df["Institutional_Policy"].unique()))

    # Filter
    filtered = df[
        (df["Major_Category"]  == major) &
        (df["Year_of_Study"]   == year)  &
        (df["Institutional_Policy"] == policy)
    ]

    if len(filtered) < 20:
        # Relax policy filter
        filtered = df[(df["Major_Category"]==major) & (df["Year_of_Study"]==year)]
        st.info(f"ℹ️ Fewer than 20 records matched all three filters — "
                f"recommendations are based on {major} + {year} (all policies).")

    total_n = len(filtered)
    # Top 20% by GPA with Low/Medium burnout
    top_cut  = filtered[filtered["Burnout_Risk_Level"] != "High"]
    top_cut  = top_cut.nlargest(max(30, int(len(top_cut)*0.20)),
                                "Post_Semester_GPA")

    # ── Cohort Summary ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="section-header">👥 Your Cohort — {major} · {year}</div>
    """, unsafe_allow_html=True)

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Students in cohort",   f"{total_n:,}")
    s2.metric("Avg GPA (cohort)",     f"{filtered['Post_Semester_GPA'].mean():.3f}")
    s3.metric("Avg AI hrs/week",      f"{filtered['Weekly_GenAI_Hours'].mean():.1f}")
    s4.metric("High Burnout rate",    f"{(filtered['Burnout_Risk_Level']=='High').mean()*100:.1f}%")

    st.markdown('<hr class="styled">', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    opt_hours   = top_cut["Weekly_GenAI_Hours"].median()
    opt_trad    = top_cut["Traditional_Study_Hours"].median()
    top_use     = top_cut["Primary_Use_Case"].value_counts().idxmax()
    top_skill   = top_cut["Prompt_Engineering_Skill"].value_counts().idxmax()
    low_brn_pct = (top_cut["Burnout_Risk_Level"]=="Low").mean()*100
    avg_top_gpa = top_cut["Post_Semester_GPA"].mean()
    ideal_ratio = opt_trad / max(opt_hours, 0.1)

    icon   = MAJOR_META.get(major, {}).get("icon","🎓")
    color  = MAJOR_META.get(major, {}).get("color", ACCENT)

    st.markdown(f"""
    <div class="section-header">🏆 Habits of Top-Performing {icon} {major} {year}s</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{color}11; border:1px solid {color}44;
                border-radius:18px; padding:28px 32px; margin-bottom:20px;">
        <p style="color:{color}; font-weight:600; font-size:0.8rem;
                  letter-spacing:0.08em; text-transform:uppercase; margin:0 0 12px 0;">
            Evidence-based from top {len(top_cut)} students · avg GPA {avg_top_gpa:.3f}
        </p>
        <p style="color:#E2E8F0; font-size:1.05rem; line-height:1.7; margin:0;">
            The highest-achieving <strong>{major}</strong> students in <strong>{year}</strong>
            keep their GenAI use to <strong>≈{opt_hours:.1f} hrs/week</strong> while investing
            <strong>≈{opt_trad:.1f} hrs/week</strong> in traditional study — a
            <strong>{ideal_ratio:.1f}:1 traditional-to-AI ratio</strong>. They primarily use AI
            for <strong>{top_use.replace("_"," ")}</strong> and describe themselves as
            <strong>{top_skill}</strong> prompt engineers. Among this group,
            <strong>{low_brn_pct:.0f}%</strong> report low burnout.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Four rec cards
    r1,r2,r3,r4 = st.columns(4)

    def rec_card(col, label, value, sub):
        col.markdown(f"""
        <div class="rec-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    rec_card(r1, "⏱️ Optimal AI Hours",
             f"{opt_hours:.1f} hrs/wk",
             f"Cohort avg: {filtered['Weekly_GenAI_Hours'].mean():.1f} hrs")
    rec_card(r2, "📖 Traditional Study",
             f"{opt_trad:.1f} hrs/wk",
             f"Ratio traditional:AI = {ideal_ratio:.1f}:1")
    rec_card(r3, "🛠️ Best Use Case",
             top_use.replace("_"," "),
             "Most common among top performers")
    rec_card(r4, "🧠 Prompt Skill Level",
             top_skill,
             f"{low_brn_pct:.0f}% of top performers = Low burnout")

    # ── Comparison chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Top Performers vs Full Cohort</div>',
                unsafe_allow_html=True)

    ca, cb = st.columns(2, gap="large")

    with ca:
        # Scatter: hours vs GPA coloured by burnout
        sample = filtered.sample(min(2000, len(filtered)), random_state=42)
        color_map = {"High":ACCENT2, "Medium":"#F6AD55", "Low":ACCENT3}
        fig = px.scatter(
            sample, x="Weekly_GenAI_Hours", y="Post_Semester_GPA",
            color="Burnout_Risk_Level", color_discrete_map=color_map,
            opacity=0.55, size_max=8,
            title=f"AI Hours vs GPA — {major} {year}",
            labels={"Weekly_GenAI_Hours":"Weekly AI Hours",
                    "Post_Semester_GPA" :"Post-Semester GPA"},
        )
        # Optimal hours line
        fig.add_vline(x=opt_hours, line_dash="dash",
                      line_color=ACCENT3, opacity=0.9,
                      annotation_text=f"Optimal {opt_hours:.1f}h",
                      annotation_font_color=ACCENT3)
        _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        # Bar: use-case frequency among top vs rest
        top_uc = top_cut["Primary_Use_Case"].value_counts(normalize=True)*100
        all_uc = filtered["Primary_Use_Case"].value_counts(normalize=True)*100
        compare_df = pd.DataFrame({"Top Performers %": top_uc,
                                   "Full Cohort %"   : all_uc}).fillna(0)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Top Performers",
            x=compare_df.index, y=compare_df["Top Performers %"],
            marker_color=ACCENT, opacity=0.85,
        ))
        fig2.add_trace(go.Bar(
            name="Full Cohort",
            x=compare_df.index, y=compare_df["Full Cohort %"],
            marker_color="#475569", opacity=0.7,
        ))
        fig2.update_layout(
            title="AI Use-Case: Top Performers vs Cohort",
            barmode="group", yaxis_title="Share (%)",
            xaxis_tickangle=-15,
        )
        _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ML INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def tab_predictor(df: pd.DataFrame) -> None:
    try:
        clf, reg, le, pipe = load_models()
        models_ok = True
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}\n\n"
                 "Run the Phase 1 pipeline first to generate .pkl files.")
        models_ok = False

    st.markdown("""
    <div class="hero">
        <div class="badge">🤖 XGBoost · Phase 1 Models</div>
        <h1>AI Student Impact Predictor</h1>
        <p>Enter your academic habits below. The trained models will predict your
        Burnout Risk Level and projected Post-Semester GPA — with actionable warnings
        if your AI usage pattern looks risky.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-header">🎛️ Configure Your Profile</div>',
                    unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            inp_major  = st.selectbox("🏫 Major Category",
                                      sorted(df["Major_Category"].unique()))
            inp_year   = st.selectbox("📅 Year of Study",
                                      ["Freshman","Sophomore","Junior","Senior","Graduate"])
            inp_policy = st.selectbox("🏛️ Institutional AI Policy",
                                      sorted(df["Institutional_Policy"].unique()))

        with fc2:
            inp_pre_gpa    = st.slider("📈 Pre-Semester GPA",
                                       1.0, 4.0, 3.0, 0.01, format="%.2f")
            inp_ai_hrs     = st.slider("🤖 Weekly GenAI Hours",
                                       0.0, 40.0, 8.0, 0.5, format="%.1f hrs")
            inp_trad_hrs   = st.slider("📖 Traditional Study Hrs/Week",
                                       0.0, 40.0, 15.0, 0.5, format="%.1f hrs")

        with fc3:
            inp_use_case   = st.selectbox("🛠️ Primary AI Use Case",
                                          sorted(df["Primary_Use_Case"].unique()))
            inp_prompt_sk  = st.selectbox("🧠 Prompt Engineering Skill",
                                          ["Beginner","Intermediate","Advanced"])
            inp_paid       = st.selectbox("💳 Paid AI Subscription",
                                          ["No","Yes"])

        fc4, fc5, fc6 = st.columns(3)
        with fc4:
            inp_tool_div  = st.slider("🔧 Tool Diversity (# tools used)",
                                      1, 10, 3)
        with fc5:
            inp_ai_dep    = st.slider("⚠️ Perceived AI Dependency (1-10)",
                                      1, 10, 5)
        with fc6:
            inp_anxiety   = st.slider("😰 Exam Anxiety Level (1-10)",
                                      1, 10, 5)

        # Skill retention is estimated from typical distributions
        # We give users a slider so they can self-assess
        inp_skill_ret = st.slider(
            "🏆 Estimated Skill Retention Score (self-assess)",
            0.0, 100.0, 75.0, 1.0,
            help="How well do you retain knowledge after using AI assistance?")

    st.markdown('<hr class="styled">', unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        predict_clicked = st.button("⚡ Predict My Impact", type="primary",
                                    use_container_width=True)

    if predict_clicked and models_ok:
        # Build input dataframe
        input_dict = {
            "Major_Category"            : inp_major,
            "Year_of_Study"             : inp_year,
            "Institutional_Policy"      : inp_policy,
            "Pre_Semester_GPA"          : inp_pre_gpa,
            "Weekly_GenAI_Hours"        : inp_ai_hrs,
            "Traditional_Study_Hours"   : inp_trad_hrs,
            "Primary_Use_Case"          : inp_use_case,
            "Prompt_Engineering_Skill"  : inp_prompt_sk,
            "Paid_Subscription"         : 1 if inp_paid == "Yes" else 0,
            "Tool_Diversity"            : inp_tool_div,
            "Perceived_AI_Dependency"   : inp_ai_dep,
            "Anxiety_Level_During_Exams": inp_anxiety,
            "Skill_Retention_Score"     : inp_skill_ret,
        }
        feature_cols = (list(ORDINAL_FEATURES.keys()) +
                        NOMINAL_FEATURES               +
                        NUMERIC_FEATURES)

        X_in = pd.DataFrame([input_dict])[feature_cols]
        X_t  = pipe.transform(X_in)

        burnout_idx  = clf.predict(X_t)[0]
        burnout_proba= clf.predict_proba(X_t)[0]
        burnout_label= le.inverse_transform([burnout_idx])[0]
        pred_gpa     = float(reg.predict(X_t)[0])

        meta = BURNOUT_META[burnout_label]

        # ── Results layout ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Prediction Results</div>',
                    unsafe_allow_html=True)

        res1, res2 = st.columns(2, gap="large")

        with res1:
            st.markdown(f"""
            <div class="{meta['css']}">
                <div class="pred-label">🔥 Burnout Risk Level</div>
                <div class="pred-value" style="color:{meta['color']}">
                    {meta['emoji']} {burnout_label}
                </div>
                <div class="pred-desc">
                    Model confidence: {max(burnout_proba)*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            proba_df = pd.DataFrame({
                "Risk Level": le.classes_,
                "Probability": burnout_proba * 100,
            }).sort_values("Probability", ascending=True)
            bar_colors = [BURNOUT_META[r]["color"] for r in proba_df["Risk Level"]]
            fig_p = go.Figure(go.Bar(
                y=proba_df["Risk Level"], x=proba_df["Probability"],
                orientation="h", marker_color=bar_colors,
                text=proba_df["Probability"].round(1).astype(str)+"%",
                textposition="outside",
            ))
            fig_p.update_layout(
                title="Burnout Probability Breakdown",
                xaxis_title="Probability (%)", xaxis_range=[0,115],
                **PLOTLY_TEMPLATE["layout"],
                height=200, margin=dict(l=0,r=0,t=35,b=0),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with res2:
            gpa_color = ACCENT3 if pred_gpa >= 3.5 else (
                        "#F6AD55" if pred_gpa >= 2.8 else ACCENT2)
            gpa_emoji = "🏅" if pred_gpa >= 3.7 else (
                        "📘" if pred_gpa >= 3.0 else "⚠️")

            st.markdown(f"""
            <div class="pred-low" style="border-color:{gpa_color}55;
                 background:linear-gradient(135deg,{gpa_color}11,{gpa_color}05)">
                <div class="pred-label">📊 Predicted Post-Semester GPA</div>
                <div class="pred-value" style="color:{gpa_color}">
                    {gpa_emoji} {pred_gpa:.3f}
                </div>
                <div class="pred-desc">
                    {"Excellent — top tier academic performance" if pred_gpa >= 3.7
                    else "Good standing — room to push further" if pred_gpa >= 3.2
                    else "Needs attention — review your study balance"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # GPA vs cohort distribution
            cohort_gpas = df[df["Major_Category"] == inp_major]["Post_Semester_GPA"]
            pct_rank    = (cohort_gpas < pred_gpa).mean() * 100

            fig_g = go.Figure()
            fig_g.add_trace(go.Histogram(
                x=cohort_gpas, nbinsx=40,
                marker_color=ACCENT, opacity=0.5,
                name="Cohort GPA",
            ))
            fig_g.add_vline(x=pred_gpa, line_dash="dash",
                            line_color=gpa_color, line_width=2,
                            annotation_text=f"You: {pred_gpa:.2f}",
                            annotation_font_color=gpa_color)
            fig_g.update_layout(
                title=f"Your GPA vs {inp_major} Cohort (top {100-pct_rank:.0f}%)",
                xaxis_title="GPA", yaxis_title="Students",
                **PLOTLY_TEMPLATE["layout"],
                height=200, margin=dict(l=0,r=0,t=35,b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # ── Actionable warnings ───────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Actionable Insights</div>',
                    unsafe_allow_html=True)

        ai_to_trad_ratio = inp_ai_hrs / max(inp_trad_hrs, 0.1)

        if ai_to_trad_ratio > 0.8:
            st.markdown(f"""
            <div class="warning-box">
                <div class="icon">⚡</div>
                <div class="text">
                    <strong>AI Dependency Alert:</strong> Your AI usage
                    ({inp_ai_hrs:.1f} hrs/wk) is <strong>{ai_to_trad_ratio:.1f}×</strong>
                    your traditional study time ({inp_trad_hrs:.1f} hrs/wk).
                    Top performers in your major keep this ratio below <strong>0.4×</strong>.
                    Consider shifting 2–3 hrs of AI use to active recall or problem sets.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if burnout_label == "High":
            st.markdown(f"""
            <div class="warning-box">
                <div class="icon">🔥</div>
                <div class="text">
                    <strong>High Burnout Risk Detected:</strong> Your current
                    profile pattern closely resembles students who experienced
                    academic burnout. The most impactful levers are reducing
                    GenAI hours below <strong>10 hrs/wk</strong> and boosting
                    traditional study. Also consider stress-management techniques —
                    your anxiety score ({inp_anxiety}/10) is a significant predictor.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if inp_ai_dep >= 7:
            st.markdown(f"""
            <div class="warning-box">
                <div class="icon">🧠</div>
                <div class="text">
                    <strong>High Perceived AI Dependency ({inp_ai_dep}/10):</strong>
                    Students who self-report high AI dependency tend to score lower
                    on skill retention. Try <em>deliberate practice without AI</em>
                    at least 2× per week to reinforce genuine understanding.
                </div>
            </div>
            """, unsafe_allow_html=True)

        if burnout_label == "Low" and pred_gpa >= 3.5:
            st.markdown(f"""
            <div class="info-box">
                ✅ <strong>Great profile!</strong> Your current habits align well
                with the top-performing, low-burnout students in our dataset.
                Maintain the balance between AI assistance and traditional study
                to sustain this trajectory through the semester.
            </div>
            """, unsafe_allow_html=True)

    elif predict_clicked and not models_ok:
        st.warning("Models not loaded. Please run the Phase 1 pipeline first.")


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION HEADER
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:16px; margin-bottom:28px;">
        <div style="font-size:2.2rem; line-height:1;">🎓</div>
        <div>
            <div style="font-size:1.35rem; font-weight:700; color:#F1F5F9;
                        letter-spacing:-0.02em; line-height:1.2;">
                AI Student Impact Analyser
            </div>
            <div style="font-size:0.82rem; color:#64748B; font-family:'DM Mono',monospace;">
                50,000 student records · XGBoost v2 · Streamlit portfolio project
            </div>
        </div>
        <div style="margin-left:auto; display:flex; gap:10px;">
            <a href="https://github.com" target="_blank"
               style="text-decoration:none; background:#1A1D27; border:1px solid #2D3748;
                      border-radius:8px; padding:8px 16px; color:#94A3B8;
                      font-size:0.82rem; font-weight:500;">⭐ GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_header()

    with st.spinner("Loading dataset…"):
        df = load_data()

    tab1, tab2, tab3 = st.tabs([
        "📊  Macro Insights & Analytics",
        "🎯  Safe-Usage Recommendations",
        "🤖  Impact Predictor",
    ])

    with tab1:
        tab_dashboard(df)

    with tab2:
        tab_recommendations(df)

    with tab3:
        tab_predictor(df)


if __name__ == "__main__":
    main()
