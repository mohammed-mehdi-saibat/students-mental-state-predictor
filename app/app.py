import sys
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parent))
sys.path.insert(0, str(APP))

from dashboard import data_insights, methodology, model_evaluation, overview, prediction_lab

st.set_page_config(page_title="Student Wellbeing Lab", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background: #faf9f1; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] { position: relative; isolation: isolate; }
    [data-testid="stMain"]::before, [data-testid="stMain"]::after {
        content: "";
        position: absolute;
        pointer-events: none;
        z-index: -1;
    }
    [data-testid="stMain"]::before {
        top: 0;
        right: 0;
        width: min(58vw, 860px);
        height: 350px;
        background: repeating-radial-gradient(ellipse at 100% 0%,
            transparent 0 27px, rgb(176 138 8 / 27%) 28px 29px, transparent 30px 48px);
        mask-image: radial-gradient(ellipse at top right, black 0%, transparent 71%);
    }
    [data-testid="stMain"]::after {
        bottom: 0;
        left: 0;
        width: min(45vw, 670px);
        height: 290px;
        background: repeating-radial-gradient(ellipse at 0% 100%,
            transparent 0 32px, rgb(176 138 8 / 19%) 33px 34px, transparent 35px 57px);
        mask-image: radial-gradient(ellipse at bottom left, black 0%, transparent 71%);
    }
    [data-testid="stVerticalBlockBorderWrapper"] { background: #faf9f1f5; }
    .block-container { max-width: 1400px; padding: 3.5rem 3.2rem 3rem; }
    [data-testid="stSidebar"] { background: #fff6bf; border-right: 1px solid #e2d78d; }
    [data-testid="stPageLink-NavLink"] { border-radius: 7px; margin-bottom: 4px; }
    [data-testid="stPageLink-NavLink"][aria-current="page"] { background: #f7df1e; color: #242217; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e7e2c7;
        border-radius: 12px; padding: 15px 19px; min-height: 105px; }
    [data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -.04em; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    h1 { letter-spacing: -.055em; font-weight: 720; }
    h2, h3 { letter-spacing: -.028em; }
    .intro { color: #514b38; font-size: 1.07rem; margin: -.3rem 0 1.15rem; }
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover { background: #fbe989; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] { background: #f7df1e; color: #242217; }
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg { color: #242217; }
    [data-testid="stBaseButton-primary"] { background: #f7df1e; color: #242217; border-color: #dac315; }
    [data-testid="stBaseButton-primary"]:hover { background: #ffe84f; color: #242217; }
    @media (max-width: 800px) { .block-container { padding: 3.5rem 1.25rem 2rem; } }
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page(overview, title="Overview", default=True),
    st.Page(data_insights, title="Data Insights", url_path="data-insights"),
    st.Page(model_evaluation, title="Model Evaluation", url_path="model-evaluation"),
    st.Page(prediction_lab, title="Prediction Lab", url_path="prediction-lab"),
    st.Page(methodology, title="Methodology", url_path="methodology"),
]
page = st.navigation(pages, position="hidden")
with st.sidebar:
    st.caption("Educational ML · Synthetic dataset")
    st.space("small")
    for item in pages:
        st.page_link(item)

page.run()
