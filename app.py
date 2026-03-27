import streamlit as st
from styles import inject_styles
from components import render_header, render_footer, render_sidebar

st.set_page_config(
    page_title="Logistic Regression Finance Lab | The Mountain Path Academy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()
render_header()
render_sidebar()

from views import home, caselet1, caselet2, caselet3, caselet4, caselet5, theory, quiz
from views import excel_guide

PAGES = {
    "🏠 Home":               home,
    "📘 Theory & Foundations": theory,
    "🏦 C1: Loan Default":   caselet1,
    "💳 C2: Fraud Detection": caselet2,
    "📉 C3: Churn Prediction": caselet3,
    "🏭 C4: SME NPA Scoring": caselet4,
    "📈 C5: IPO Subscription": caselet5,
    "📗 Excel Guide":         excel_guide,
    "🧩 Self-Assessment Quiz": quiz,
}

page = st.session_state.get("page", "🏠 Home")
PAGES[page].render()
render_footer()
