from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# SMART FOOD WASTE PREDICTION & MANAGEMENT
# MNR University | Machine Learning Mini Project
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

FEATURES = [
    "expected_customers",
    "day_of_week",
    "weekend",
    "temperature_c",
    "special_event",
    "holiday",
    "previous_day_consumption",
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

st.set_page_config(
    page_title="Smart Food Waste Prediction | MNR University",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- DESIGN SYSTEM --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --navy: #17175d;
        --purple: #7138c8;
        --violet: #9b45e8;
        --blue: #2786e8;
        --teal: #16bfc3;
        --orange: #ff9b18;
        --pink: #ef5d9b;
        --green: #23ae69;
    }

    html, body, [class*="css"], button, input, textarea, select {
        font-family: Calibri, Arial, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 4% 4%, rgba(255, 187, 107, .17), transparent 25%),
            radial-gradient(circle at 96% 2%, rgba(143, 98, 238, .17), transparent 26%),
            linear-gradient(135deg, #f5f6ff 0%, #f2f8ff 52%, #fff8f1 100%);
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.15rem;
        padding-bottom: 1.6rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #20236e 0%, #37339a 58%, #6142b5 100%);
        border-right: 1px solid rgba(255,255,255,.14);
    }

    section[data-testid="stSidebar"] * {
        color: #fff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,.07);
        padding: 11px 13px;
        border-radius: 12px;
        margin: 4px 0;
        font-weight: 700;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.16);
    }

    .sidebar-brand {
        text-align: center;
        padding: 8px 0 18px;
    }

    .sidebar-brand .logo {
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: .02em;
        line-height: 1.05;
    }

    .sidebar-brand .sub {
        font-size: .88rem;
        opacity: .86;
        margin-top: 8px;
        font-weight: 700;
    }

    .sidebar-message {
        margin-top: 26px;
        padding: 18px 12px;
        text-align: center;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(255,255,255,.15), rgba(255,255,255,.04));
        border: 1px solid rgba(255,255,255,.15);
        line-height: 1.65;
        font-weight: 700;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 23px;
        padding: 25px 30px 24px;
        color: white;
        background: linear-gradient(105deg, #6328b8 0%, #8c3bd0 43%, #277fe0 100%);
        box-shadow: 0 13px 32px rgba(72, 54, 153, .20);
        margin-bottom: 18px;
    }

    .hero:after {
        content: "🍃";
        position: absolute;
        right: 28px;
        top: -17px;
        font-size: 105px;
        opacity: .17;
        transform: rotate(-17deg);
    }

    .hero-kicker {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 30px;
        background: rgba(255,255,255,.17);
        border: 1px solid rgba(255,255,255,.25);
        font-size: .74rem;
        letter-spacing: .09em;
        font-weight: 800;
        margin-bottom: 9px;
    }

    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.65rem, 3vw, 2.45rem);
        line-height: 1.15;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        color: #f0eaff;
        margin: 8px 0 0;
        font-size: 1rem;
        font-weight: 500;
    }

    .metric-card {
        border-radius: 17px;
        padding: 15px 17px;
        min-height: 100px;
        box-shadow: 0 7px 20px rgba(43, 48, 104, .07);
        border: 1px solid rgba(50, 60, 120, .08);
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .metric-icon {
        font-size: 2rem;
        min-width: 42px;
        text-align: center;
    }

    .metric-label {
        color: #25255f;
        font-size: .82rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #17175d;
        font-family: 'Manrope', sans-serif;
        font-size: 1.48rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-note {
        color: #6c6b85;
        font-size: .72rem;
        margin-top: 3px;
    }

    .metric-blue { background: linear-gradient(135deg, #d9edff, #fff); border-color: #a7d5ff; }
    .metric-green { background: linear-gradient(135deg, #d9f8e9, #fff); border-color: #a9ebcb; }
    .metric-orange { background: linear-gradient(135deg, #fff0d7, #fff); border-color: #ffdaa0; }
    .metric-purple { background: linear-gradient(135deg, #eee2ff, #fff); border-color: #d4b7ff; }

    .panel {
        background: rgba(255,255,255,.92);
        border: 1px solid #e3e5fa;
        border-radius: 19px;
        box-shadow: 0 8px 22px rgba(42, 46, 108, .07);
        overflow: hidden;
    }

    .panel-heading {
        padding: 13px 18px;
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: 1.12rem;
        font-weight: 800;
        background: linear-gradient(100deg, #6928b8, #a23fe2);
    }

    .panel-body {
        padding: 17px 18px 19px;
    }

    .section-title {
        color: #1d1d66;
        font-family: 'Manrope', sans-serif;
        font-size: 1.16rem;
        font-weight: 800;
        margin: 4px 0 12px;
    }

    .result-card {
        border-radius: 17px;
        padding: 17px 14px;
        text-align: center;
        color: white;
        min-height: 115px;
        box-shadow: 0 8px 18px rgba(43, 50, 120, .13);
    }

    .result-title {
        font-weight: 800;
        font-size: .92rem;
        margin-bottom: 8px;
    }

    .result-value {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 2.1rem;
        line-height: 1.1;
    }

    .result-value span {
        font-size: 1.02rem;
        font-weight: 700;
    }

    .result-lr { background: linear-gradient(110deg, #13bfc4, #3679ed); }
    .result-rf { background: linear-gradient(110deg, #ff9c13, #f15d9e); }
    .result-average { background: linear-gradient(110deg, #7040c9, #a344df); }

    .recommendation {
        border: 1px solid #8ce6ae;
        background: linear-gradient(100deg, #e1fbea, #f4fff7);
        border-radius: 14px;
        padding: 13px 16px;
        color: #145b38;
        font-weight: 700;
        line-height: 1.45;
        margin-top: 13px;
    }

    .small-note {
        color: #6a6b85;
        font-size: .79rem;
        line-height: 1.5;
    }

    .footer {
        border-top: 1px solid #dedff5;
        margin-top: 25px;
        padding: 13px 5px 0;
        color: #29276d;
        font-size: .83rem;
        font-weight: 700;
        text-align: center;
    }

    div[data-testid="stForm"] {
        background: #fff;
        border: 1px solid #e4e5f8;
        border-radius: 0 0 19px 19px;
        padding: 14px 17px 18px;
        box-shadow: 0 8px 22px rgba(42, 46, 108, .06);
    }

    .stButton > button, div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(100deg, #8738dc, #a33de0 55%, #ff9b18) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        min-height: 45px;
        box-shadow: 0 7px 16px rgba(121, 55, 198, .20);
    }

    .stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    div[data-testid="stAlert"] { border-radius: 13px; }
    div[data-testid="stDataFrame"] { border-radius: 13px; overflow: hidden; }

    /* =========================================================
       FORM LABEL READABILITY FIX
       Dark navy labels on the light form background
       ========================================================= */

    /* Streamlit widget labels: Expected Customers, Day, Temperature, etc. */
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label,
    .stNumberInput label,
    .stSelectbox label,
    .stTextInput label,
    .stRadio > label,
    .stDateInput label,
    .stTimeInput label {
        color: #20245f !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    /* Radio option labels: No / Yes */
    div[data-testid="stRadio"] [role="radiogroup"] label,
    div[data-testid="stRadio"] [role="radiogroup"] label *,
    div[data-testid="stRadio"] [role="radiogroup"] p,
    div[data-testid="stRadio"] [role="radiogroup"] span {
        color: #30305f !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* Input values remain white on the dark input controls */
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [role="combobox"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Dropdown selected value and menu options */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    /* Ensure help text is readable */
    div[data-testid="stWidgetLabel"] small,
    .stCaption, small {
        color: #62627f !important;
        opacity: 1 !important;
    }


    /* =========================================================
       MAIN CONTENT TEXT COLOUR FIX
       Applies to Model Insights, Dataset Overview and About.
       Sidebar remains white; coloured hero/result areas remain white.
       ========================================================= */
    section.main,
    section.main .block-container,
    section.main .block-container p,
    section.main .block-container li,
    section.main .block-container label,
    section.main .block-container strong,
    section.main .block-container em,
    section.main .block-container span,
    section.main .block-container [data-testid="stMarkdownContainer"],
    section.main .block-container [data-testid="stMarkdownContainer"] p,
    section.main .block-container [data-testid="stMarkdownContainer"] li,
    section.main .block-container [data-testid="stMarkdownContainer"] strong {
        color: #252957 !important;
    }

    section.main .block-container h1,
    section.main .block-container h2,
    section.main .block-container h3,
    section.main .block-container h4,
    section.main .section-title {
        color: #20245f !important;
    }

    /* Keep the branded gradient areas white for contrast. */
    section.main .hero,
    section.main .hero *,
    section.main .panel-heading,
    section.main .panel-heading *,
    section.main .result-card,
    section.main .result-card *,
    section.main .recommendation,
    section.main .recommendation * {
        color: #ffffff !important;
    }

    /* The empty-state panel is white with dark readable text. */
    section.main .panel h3,
    section.main .panel p,
    section.main .panel b,
    section.main .panel strong {
        color: #252957 !important;
    }

    /* Native Streamlit alerts, captions, metrics and expander labels. */
    section.main [data-testid="stAlert"] *,
    section.main [data-testid="stMetric"] *,
    section.main [data-testid="stExpander"] summary,
    section.main [data-testid="stCaptionContainer"],
    section.main [data-testid="stCaptionContainer"] * {
        color: #30345f !important;
    }

    /* Data tables: readable dark text on the light page. */
    section.main [data-testid="stDataFrame"],
    section.main [data-testid="stDataFrame"] *,
    section.main [data-testid="stTable"],
    section.main [data-testid="stTable"] * {
        color: #252957 !important;
    }

    /* =========================================================
       REMOVE ONLY THE BLACK STREAMLIT TOP HEADER / TOOLBAR.
       Do not hide or alter the sidebar.
       ========================================================= */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }

    /* Restore page to the top after hiding the header. */
    .stAppViewContainer > .main {
        padding-top: 0 !important;
    }
    .main .block-container {
        padding-top: 1.15rem !important;
    }


    /* FINAL GLOBAL CONTENT CONTRAST FIX — sidebar and coloured cards excluded */
    section.main .block-container [data-testid="stMarkdownContainer"],
    section.main .block-container [data-testid="stMarkdownContainer"] p,
    section.main .block-container [data-testid="stMarkdownContainer"] li,
    section.main .block-container [data-testid="stMarkdownContainer"] b,
    section.main .block-container [data-testid="stMarkdownContainer"] strong,
    section.main .block-container [data-testid="stMarkdownContainer"] em,
    section.main .block-container [data-testid="stMarkdownContainer"] span,
    section.main .block-container [data-testid="stCaptionContainer"],
    section.main .block-container [data-testid="stCaptionContainer"] *,
    section.main .block-container .stAlert,
    section.main .block-container .stAlert * {
        color: #292d5d !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #292d5d !important;
    }

    /* About-page HTML cards: direct text and every paragraph/bold label */
    section.main .block-container .panel,
    section.main .block-container .panel p,
    section.main .block-container .panel b,
    section.main .block-container .panel strong,
    section.main .block-container .panel li,
    section.main .block-container .panel span {
        color: #292d5d !important;
        -webkit-text-fill-color: #292d5d !important;
        opacity: 1 !important;
    }
    section.main .block-container .panel h3 {
        opacity: 1 !important;
        -webkit-text-fill-color: currentColor !important;
    }
    section.main .block-container .panel hr {
        border-color: #c9cce4 !important;
    }

    /* Keep intentionally white text only on coloured branded components */
    section.main .hero, section.main .hero *,
    section.main .panel-heading, section.main .panel-heading *,
    section.main .result-card, section.main .result-card *,
    section.main .sidebar-message, section.main .sidebar-message * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Streamlit metric labels and values */
    section.main [data-testid="stMetric"] label,
    section.main [data-testid="stMetric"] [data-testid="stMetricValue"],
    section.main [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #292d5d !important;
        -webkit-text-fill-color: #292d5d !important;
        opacity: 1 !important;
    }

    /* Table foreground/background is set to Streamlit's light theme in Python too. */
    section.main [data-testid="stDataFrame"],
    section.main [data-testid="stDataFrame"] *,
    section.main [data-testid="stTable"],
    section.main [data-testid="stTable"] * {
        color: #292d5d !important;
        -webkit-text-fill-color: #292d5d !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* =========================================================
       DEFINITIVE MAIN-PAGE TEXT CONTRAST FIX
       Streamlit main area only. Sidebar is intentionally untouched.
       ========================================================= */
    section[data-testid="stMain"] .block-container,
    section[data-testid="stMain"] .block-container p,
    section[data-testid="stMain"] .block-container li,
    section[data-testid="stMain"] .block-container label,
    section[data-testid="stMain"] .block-container strong,
    section[data-testid="stMain"] .block-container em,
    section[data-testid="stMain"] .block-container small,
    section[data-testid="stMain"] .block-container [data-testid="stMarkdownContainer"],
    section[data-testid="stMain"] .block-container [data-testid="stMarkdownContainer"] * {
        color: #252957 !important;
        -webkit-text-fill-color: #252957 !important;
        opacity: 1 !important;
    }

    /* HTML information cards on About page */
    section[data-testid="stMain"] .block-container .panel,
    section[data-testid="stMain"] .block-container .panel *,
    section[data-testid="stMain"] .block-container .panel p,
    section[data-testid="stMain"] .block-container .panel b,
    section[data-testid="stMain"] .block-container .panel strong,
    section[data-testid="stMain"] .block-container .panel li {
        color: #252957 !important;
        -webkit-text-fill-color: #252957 !important;
        opacity: 1 !important;
    }
    section[data-testid="stMain"] .block-container .panel h3 {
        color: #20245f !important;
        -webkit-text-fill-color: #20245f !important;
    }
    section[data-testid="stMain"] .block-container .panel hr {
        border-color: #d5d8ee !important;
    }

    /* Section headings, captions, metric widgets */
    section[data-testid="stMain"] .block-container .section-title,
    section[data-testid="stMain"] .block-container h1,
    section[data-testid="stMain"] .block-container h2,
    section[data-testid="stMain"] .block-container h3,
    section[data-testid="stMain"] .block-container h4 {
        color: #20245f !important;
        -webkit-text-fill-color: #20245f !important;
        opacity: 1 !important;
    }
    section[data-testid="stMain"] [data-testid="stMetric"] label,
    section[data-testid="stMain"] [data-testid="stMetric"] [data-testid="stMetricValue"],
    section[data-testid="stMain"] [data-testid="stMetric"] [data-testid="stMetricDelta"],
    section[data-testid="stMain"] [data-testid="stMetric"] * {
        color: #252957 !important;
        -webkit-text-fill-color: #252957 !important;
        opacity: 1 !important;
    }
    section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] * {
        color: #606481 !important;
        -webkit-text-fill-color: #606481 !important;
        opacity: 1 !important;
    }

    /* Streamlit alerts and expander titles */
    section[data-testid="stMain"] [data-testid="stAlert"],
    section[data-testid="stMain"] [data-testid="stAlert"] *,
    section[data-testid="stMain"] [data-testid="stExpander"] summary,
    section[data-testid="stMain"] [data-testid="stExpander"] summary * {
        color: #252957 !important;
        -webkit-text-fill-color: #252957 !important;
        opacity: 1 !important;
    }

    /* Table text uses the light Streamlit app theme configured below. */
    section[data-testid="stMain"] [data-testid="stDataFrame"],
    section[data-testid="stMain"] [data-testid="stTable"] {
        color: #252957 !important;
    }

    /* Preserve white text on intentionally dark/gradient branded components. */
    section[data-testid="stMain"] .hero,
    section[data-testid="stMain"] .hero *,
    section[data-testid="stMain"] .panel-heading,
    section[data-testid="stMain"] .panel-heading *,
    section[data-testid="stMain"] .result-card,
    section[data-testid="stMain"] .result-card *,
    section[data-testid="stMain"] .sidebar-message,
    section[data-testid="stMain"] .sidebar-message * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Dark form controls intentionally keep white entered values. */
    section[data-testid="stMain"] div[data-baseweb="input"] input,
    section[data-testid="stMain"] div[data-baseweb="select"] input,
    section[data-testid="stMain"] div[data-baseweb="select"] [role="combobox"] {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

</style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- SIDEBAR --------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:2.5rem">🎓</div>
            <div class="logo">MNR<br>UNIVERSITY</div>
            <div class="sub">ML Mini Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "PROJECT NAVIGATION",
        ["🏠 Demand Predictor", "📊 Model Insights", "🗂️ Dataset Overview", "ℹ️ About Project"],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-message">
            <div style="font-size:3.1rem">♻️</div>
            Reduce Food Waste<br>
            Plan Better<br>
            Serve Smarter 🌱
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("SCHOOL OF ENGINEERING, TECHNOLOGY AND AI")

# -------------------------- HERO --------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">MNR UNIVERSITY &nbsp; | &nbsp; MACHINE LEARNING MINI PROJECT</div>
        <h1>🍴 Smart Food Waste Prediction &amp; Management</h1>
        <p>Using Machine Learning to support smarter canteen meal planning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------- LOAD DATA / MODELS --------------------------
lr_path = MODEL_DIR / "linear_regression.joblib"
rf_path = MODEL_DIR / "random_forest_regression.joblib"
dataset_path = DATA_DIR / "canteen_data.csv"

if not lr_path.exists() or not rf_path.exists():
    st.error("Trained models are missing. Open the project terminal and run: `python src/train.py`")
    st.stop()

lr_model = joblib.load(lr_path)
rf_model = joblib.load(rf_path)

if dataset_path.exists():
    dataset = pd.read_csv(dataset_path)
else:
    dataset = pd.DataFrame()

# -------------------------- KPI STRIP --------------------------
record_count = len(dataset) if not dataset.empty else 1200
input_count = len(FEATURES)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="metric-card metric-blue"><div class="metric-icon">👥</div><div><div class="metric-label">TOTAL RECORDS</div><div class="metric-value">{record_count:,}</div><div class="metric-note">Daily canteen observations</div></div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="metric-card metric-green"><div class="metric-icon">🧩</div><div><div class="metric-label">INPUT FEATURES</div><div class="metric-value">{input_count}</div><div class="metric-note">Selected model variables</div></div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        '<div class="metric-card metric-orange"><div class="metric-icon">⚙️</div><div><div class="metric-label">ML ALGORITHMS</div><div class="metric-value">2</div><div class="metric-note">Linear &amp; Random Forest</div></div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        '<div class="metric-card metric-purple"><div class="metric-icon">🎯</div><div><div class="metric-label">PREDICTION TARGET</div><div class="metric-value" style="font-size:1.18rem">Meals Consumed</div><div class="metric-note">Numerical regression output</div></div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# -------------------------- DEMAND PREDICTOR --------------------------
if page == "🏠 Demand Predictor":
    left, right = st.columns([0.92, 1.48], gap="large")

    with left:
        st.markdown('<div class="panel-heading">📋 &nbsp; Enter Canteen Details</div>', unsafe_allow_html=True)
        with st.form("canteen_prediction_form"):
            customers = st.number_input(
                "👥 Expected Customers",
                min_value=0,
                max_value=3000,
                value=470,
                step=10,
            )
            day = st.selectbox("🗓️ Day of the Week", DAYS, index=2)
            temperature = st.number_input(
                "🌡️ Temperature (°C)",
                min_value=-10.0,
                max_value=55.0,
                value=30.0,
                step=0.5,
            )

            event_col, holiday_col = st.columns(2)
            with event_col:
                special_event = st.radio("🎉 Special Event?", ["No", "Yes"], horizontal=True)
            with holiday_col:
                holiday = st.radio("🏖️ Holiday?", ["No", "Yes"], horizontal=True)

            previous_consumption = st.number_input(
                "🍲 Previous-day Consumption (Meals)",
                min_value=0.0,
                max_value=3000.0,
                value=425.0,
                step=5.0,
            )

            submitted = st.form_submit_button("▶  Predict Meals Demand", use_container_width=True)

    with right:
        st.markdown('<div class="section-title">📊 &nbsp; Prediction Results</div>', unsafe_allow_html=True)

        if submitted:
            day_number = DAYS.index(day)
            input_row = pd.DataFrame(
                [{
                    "expected_customers": customers,
                    "day_of_week": day_number,
                    "weekend": int(day_number >= 5),
                    "temperature_c": temperature,
                    "special_event": int(special_event == "Yes"),
                    "holiday": int(holiday == "Yes"),
                    "previous_day_consumption": previous_consumption,
                }],
                columns=FEATURES,
            )

            lr_prediction = max(0, int(round(float(lr_model.predict(input_row)[0]))))
            rf_prediction = max(0, int(round(float(rf_model.predict(input_row)[0]))))
            average_prediction = int(round((lr_prediction + rf_prediction) / 2))
            difference = abs(lr_prediction - rf_prediction)

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(
                    f'<div class="result-card result-lr"><div class="result-title">📈 Linear Regression</div><div class="result-value">{lr_prediction:,} <span>Meals</span></div></div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div class="result-card result-rf"><div class="result-title">🌲 Random Forest Regression</div><div class="result-value">{rf_prediction:,} <span>Meals</span></div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="recommendation">
                    ✅ &nbsp; <b>Combined planning estimate: approximately {average_prediction:,} meals.</b><br>
                    The two model estimates differ by {difference:,} meals. Use this as a planning reference,
                    then account for menu, stock, service buffer and local conditions.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "The combined estimate is the rounded average of the two model predictions. "
                "It is not a validated preparation quantity or a guarantee of demand."
            )

            with st.expander("View the input values sent to the models"):
                st.dataframe(input_row, use_container_width=True, hide_index=True)

        else:
            st.markdown(
                """
                <div class="panel" style="padding:25px 22px; min-height:250px; text-align:center;">
                    <div style="font-size:3.4rem; margin-top:15px">🍽️</div>
                    <h3 style="color:#24246d; font-family:Manrope">Your prediction will appear here</h3>
                    <p style="color:#70708a">Enter the canteen details on the left and click <b>Predict Meals Demand</b>.</p>
                    <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-top:22px">
                        <span style="background:#e5f4ff;padding:8px 13px;border-radius:20px;color:#1764ad;font-weight:700">Linear Regression</span>
                        <span style="background:#fff0df;padding:8px 13px;border-radius:20px;color:#b76708;font-weight:700">Random Forest</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        '<div class="small-note">⚠️ <b>Academic demonstration:</b> the supplied dataset is synthetic. Predictions are estimates from models trained on generated data and should not be treated as verified real-canteen demand.</div>',
        unsafe_allow_html=True,
    )

# -------------------------- MODEL INSIGHTS --------------------------
elif page == "📊 Model Insights":
    st.markdown('<div class="section-title">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)
    metrics_path = OUT_DIR / "model_metrics.csv"
    predictions_path = OUT_DIR / "test_predictions.csv"

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True, hide_index=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown('<div class="panel" style="padding:15px">', unsafe_allow_html=True)
            st.markdown("**📊 Error Metric Comparison**")
            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            x = np.arange(len(metrics))
            width = 0.34
            ax.bar(x - width / 2, metrics["MAE (meals)"], width, label="MAE")
            ax.bar(x + width / 2, metrics["RMSE (meals)"], width, label="RMSE")
            ax.set_xticks(x)
            ax.set_xticklabels(["Linear Regression", "Random Forest"], rotation=8)
            ax.set_ylabel("Error (meals)")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_right:
            st.markdown('<div class="panel" style="padding:15px">', unsafe_allow_html=True)
            st.markdown("**🎯 R² Score Comparison**")
            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            ax.bar(metrics["Model"], metrics["R2"])
            ax.set_ylabel("R² score")
            ax.set_ylim(min(0, float(metrics["R2"].min()) - .05), 1.0)
            ax.tick_params(axis="x", rotation=8)
            ax.grid(axis="y", alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        if predictions_path.exists():
            pred = pd.read_csv(predictions_path)
            if {"meals_consumed", "predicted_meals"}.issubset(pred.columns):
                st.markdown('<div class="section-title">🔎 Actual vs Predicted (Test Data)</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(9, 4.2))
                ax.scatter(pred["meals_consumed"], pred["predicted_meals"], alpha=.55)
                low = min(pred["meals_consumed"].min(), pred["predicted_meals"].min())
                high = max(pred["meals_consumed"].max(), pred["predicted_meals"].max())
                ax.plot([low, high], [low, high], linestyle="--")
                ax.set_xlabel("Actual Meals")
                ax.set_ylabel("Predicted Meals")
                ax.grid(alpha=.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.info(
            "MAE and RMSE are measured in meals; lower values indicate smaller prediction errors. "
            "R² describes fit relative to a mean-prediction baseline. These scores come from the synthetic demonstration dataset."
        )
    else:
        st.warning("Model evaluation files are missing. Run `python src/train.py` to generate them.")

# -------------------------- DATASET OVERVIEW --------------------------
elif page == "🗂️ Dataset Overview":
    st.markdown('<div class="section-title">🗂️ Dataset Overview</div>', unsafe_allow_html=True)
    if dataset.empty:
        st.warning("Dataset file not found. Run `python src/train.py` first.")
    else:
        a, b, c = st.columns(3)
        a.metric("Total records", f"{len(dataset):,}")
        b.metric("Input features", str(len(FEATURES)))
        c.metric("Target column", "meals_consumed")

        st.markdown("**Sample records**")
        st.dataframe(dataset.head(12), use_container_width=True, hide_index=True)

        st.markdown("**Feature descriptions**")
        descriptions = pd.DataFrame(
            [
                ["expected_customers", "Expected number of people visiting the canteen", "Input"],
                ["day_of_week", "0 = Monday through 6 = Sunday", "Input"],
                ["weekend", "1 if Saturday or Sunday, otherwise 0", "Input"],
                ["temperature_c", "Daily temperature in degrees Celsius", "Input"],
                ["special_event", "1 if a special event is scheduled, otherwise 0", "Input"],
                ["holiday", "1 if a holiday is indicated, otherwise 0", "Input"],
                ["previous_day_consumption", "Meals consumed on the previous day", "Input"],
                ["meals_consumed", "Observed meals consumed", "Target"],
                ["food_prepared", "Prepared meal quantity; descriptive only", "Excluded from model"],
                ["leftover_meals", "Prepared minus consumed meals; descriptive only", "Excluded from model"],
            ],
            columns=["Attribute", "Description", "Role"],
        )
        st.dataframe(descriptions, use_container_width=True, hide_index=True)
        st.caption(
            "The dataset is synthetic and programmatically generated for academic demonstration. "
            "food_prepared and leftover_meals are excluded from model inputs to avoid target leakage."
        )

# -------------------------- ABOUT --------------------------
elif page == "ℹ️ About Project":
    st.markdown('<div class="section-title">🌱 About the Project</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="panel" style="padding:20px;min-height:235px">
                <h3 style="color:#6e32bf">🎯 Problem Statement</h3>
                College canteens may prepare more food than required or experience shortages
                when daily demand changes. This project estimates meals likely to be consumed
                using relevant daily information.
                <hr>
                <h3 style="color:#168d83">🧠 Algorithms</h3>
                <b>Linear Regression</b> is used as an interpretable baseline. <b>Random Forest
                Regression</b> combines decision trees to model more complex patterns.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="panel" style="padding:20px;min-height:235px">
                <h3 style="color:#df8212">🗃️ Dataset</h3>
                The demonstration contains 1,200 synthetic daily records and seven model input
                features. The first 80% are used for training and the final 20% for testing.
                <hr>
                <h3 style="color:#258b58">🚀 Future Scope</h3>
                Collect real canteen records, include menu and attendance information, predict
                breakfast/lunch/dinner separately, and add inventory planning.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Project team**")
    team = pd.DataFrame(
        [
            ["M. Karthik Reddy", "25RBAIM0010"],
            ["Jogyala Sai Teja", "25RBAIM0037"],
            ["Mirza Haseeb Baig", "25RBAIM0045"],
        ],
        columns=["Team member", "Roll number"],
    )
    st.dataframe(team, use_container_width=True, hide_index=True)
    st.warning(
        "This is an academic prototype. Synthetic-data performance does not establish real-world accuracy. "
        "Canteen staff must review predictions before making preparation decisions."
    )

# -------------------------- FOOTER --------------------------
st.markdown(
    """
    <div class="footer">
        🎓 MNR UNIVERSITY &nbsp; | &nbsp; School of Engineering, Technology and AI
        &nbsp;&nbsp; • &nbsp;&nbsp; Machine Learning Mini Project
        &nbsp;&nbsp; • &nbsp;&nbsp; Smart Food Waste Prediction 🌱
    </div>
    """,
    unsafe_allow_html=True,
)
