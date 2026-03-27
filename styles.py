import streamlit as st

DARK_BLUE = "#003366"
MID_BLUE  = "#004d80"
LIGHT_BLUE = "#ADD8E6"
GOLD      = "#FFD700"
CARD_BG   = "#112240"
TXT_MAIN  = "#e6f1ff"
TXT_MUTED = "#8892b0"
GREEN     = "#28a745"
RED       = "#dc3545"
BG_GRAD   = "linear-gradient(135deg,#0a1628,#112240,#1a3355)"


def inject_styles():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── GLOBAL ── */
    html, body, [class*="css"] {{
        font-family: 'Source Sans Pro', sans-serif;
        color: {TXT_MAIN};
    }}
    .stApp {{
        background: {BG_GRAD};
        min-height: 100vh;
    }}
    section.main > div {{ padding-top: 0rem; }}

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg,{DARK_BLUE},#0d2b4e) !important;
        border-right: 2px solid {GOLD};
    }}
    [data-testid="stSidebar"] * {{ color: {TXT_MAIN} !important; }}
    [data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        border: 1px solid {GOLD}44 !important;
        color: {TXT_MAIN} !important;
        text-align: left !important;
        width: 100% !important;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.88rem;
        padding: 0.45rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 3px;
        transition: all 0.2s ease;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background: {GOLD}22 !important;
        border-color: {GOLD} !important;
        transform: translateX(4px);
        color: {TXT_MAIN} !important;
    }}

    /* ── MAIN CONTENT BUTTONS (Open C1 → etc) ── */
    .stButton > button {{
        background: {DARK_BLUE} !important;
        color: {TXT_MAIN} !important;
        border: 1px solid {GOLD}66 !important;
        border-radius: 8px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 1rem !important;
    }}
    .stButton > button:hover {{
        background: {GOLD}22 !important;
        border-color: {GOLD} !important;
        color: {GOLD} !important;
        box-shadow: 0 0 12px {GOLD}44 !important;
    }}
    .stButton > button:active {{
        background: {GOLD}44 !important;
        color: {TXT_MAIN} !important;
    }}
    .stButton > button:focus {{
        background: {DARK_BLUE} !important;
        color: {TXT_MAIN} !important;
        border-color: {GOLD} !important;
        box-shadow: 0 0 0 2px {GOLD}44 !important;
    }}

    /* ── HIDE MATPLOTLIB FIGURE DEBUG OUTPUT ── */
    .element-container iframe {{ display: none; }}
    [data-testid="stCaptionContainer"] {{ display: none !important; }}
    
    /* ── CARDS ── */
    .mp-card {{
        background: {CARD_BG};
        border: 1px solid {DARK_BLUE};
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: box-shadow 0.2s ease;
    }}
    .mp-card:hover {{ box-shadow: 0 6px 28px rgba(0,0,0,0.45); }}
    .mp-card-gold {{
        background: linear-gradient(135deg,#1a2e50,#1e3a60);
        border: 1px solid {GOLD}66;
        border-left: 4px solid {GOLD};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .mp-card-green {{
        background: linear-gradient(135deg,#0a2a1a,#0e3520);
        border: 1px solid {GREEN}55;
        border-left: 4px solid {GREEN};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .mp-card-red {{
        background: linear-gradient(135deg,#2a0a0a,#3a1010);
        border: 1px solid {RED}55;
        border-left: 4px solid {RED};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .mp-card-blue {{
        background: linear-gradient(135deg,#0a1a2e,#0e2540);
        border: 1px solid {LIGHT_BLUE}44;
        border-left: 4px solid {LIGHT_BLUE};
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }}

    /* ── TYPOGRAPHY ── */
    .mp-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: {GOLD};
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }}
    .mp-subtitle {{
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.1rem;
        color: {TXT_MUTED};
        font-weight: 300;
        margin-bottom: 1.2rem;
    }}
    .mp-section-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: {LIGHT_BLUE};
        border-bottom: 2px solid {GOLD}44;
        padding-bottom: 0.4rem;
        margin: 1.4rem 0 1rem 0;
    }}
    .mp-label {{
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {GOLD};
        font-weight: 600;
    }}
    .mp-body {{
        font-size: 0.97rem;
        line-height: 1.7;
        color: {TXT_MAIN};
    }}
    .mp-mono {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: {GOLD};
        background: rgba(0,0,0,0.3);
        padding: 2px 6px;
        border-radius: 4px;
    }}

    /* ── METRICS ── */
    .mp-metric-row {{
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }}
    .mp-metric {{
        background: {CARD_BG};
        border: 1px solid {GOLD}33;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        text-align: center;
        min-width: 110px;
        flex: 1;
        transition: border-color 0.2s;
    }}
    .mp-metric:hover {{ border-color: {GOLD}99; }}
    .mp-metric .val {{
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: {GOLD};
    }}
    .mp-metric .lbl {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {TXT_MUTED};
        margin-top: 2px;
    }}

    /* ── CONFUSION MATRIX ── */
    .cm-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Source Sans Pro', sans-serif;
        margin: 0.8rem 0;
    }}
    .cm-table td, .cm-table th {{
        padding: 0.7rem 0.9rem;
        text-align: center;
        border: 1px solid {DARK_BLUE};
        font-size: 0.9rem;
    }}
    .cm-tp {{ background: #0d3320; color: #4ade80; font-weight: 700; font-size: 1.1rem; }}
    .cm-tn {{ background: #0d3320; color: #4ade80; font-weight: 700; font-size: 1.1rem; }}
    .cm-fp {{ background: #3a1a00; color: #fbbf24; font-weight: 700; font-size: 1.1rem; }}
    .cm-fn {{ background: #3a0a0a; color: #f87171; font-weight: 700; font-size: 1.1rem; }}
    .cm-header {{ background: {DARK_BLUE}; color: {GOLD}; font-weight: 600; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; }}

    /* ── FORMULA BOX ── */
    .mp-formula {{
        background: linear-gradient(135deg,#0a1628,#0f2040);
        border: 1px solid {GOLD}44;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: {LIGHT_BLUE};
        margin: 0.8rem 0;
        text-align: center;
    }}

    /* ── BADGES ── */
    .mp-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 2px;
    }}
    .badge-gold {{ background: {GOLD}22; border: 1px solid {GOLD}; color: {GOLD}; }}
    .badge-green {{ background: {GREEN}22; border: 1px solid {GREEN}; color: {GREEN}; }}
    .badge-red   {{ background: {RED}22;   border: 1px solid {RED};   color: {RED};   }}
    .badge-blue  {{ background: {LIGHT_BLUE}22; border: 1px solid {LIGHT_BLUE}; color: {LIGHT_BLUE}; }}

    /* ── HEADER BAND ── */
    .mp-header-band {{
        background: linear-gradient(90deg,{DARK_BLUE},{MID_BLUE},{DARK_BLUE});
        border-bottom: 3px solid {GOLD};
        padding: 1.1rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}

    /* ── FOOTER ── */
    .mp-footer {{
        background: {DARK_BLUE};
        border-top: 2px solid {GOLD}55;
        padding: 1rem 2rem;
        text-align: center;
        margin-top: 2.5rem;
        font-size: 0.8rem;
        color: {TXT_MUTED};
    }}
    .mp-footer a {{ color: {GOLD}; text-decoration: none; }}
    .mp-footer a:hover {{ text-decoration: underline; }}

    /* ── TABLE ── */
    .mp-table {{ width:100%; border-collapse:collapse; margin:0.8rem 0; font-size:0.88rem; }}
    .mp-table th {{ background:{DARK_BLUE}; color:{GOLD}; padding:0.6rem 0.9rem; text-align:left; font-weight:600; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.07em; }}
    .mp-table td {{ padding:0.55rem 0.9rem; border-bottom:1px solid {DARK_BLUE}; color:{TXT_MAIN}; }}
    .mp-table tr:hover td {{ background:{CARD_BG}; }}

    /* ── DIVIDER ── */
    .mp-divider {{ border:none; border-top:1px solid {GOLD}22; margin:1.5rem 0; }}

    /* ── STREAMLIT OVERRIDES ── */
    .stSlider [data-baseweb="slider"] {{ padding-bottom:0; }}
    div[data-testid="stMetric"] {{
        background:{CARD_BG};
        border:1px solid {GOLD}33;
        border-radius:10px;
        padding:1rem;
    }}
    div[data-testid="stMetric"] label {{
        color:{TXT_MUTED} !important;
        font-size:0.78rem !important;
        text-transform:uppercase;
        letter-spacing:0.08em;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color:{GOLD} !important;
        font-family:'Playfair Display',serif !important;
        font-size:1.8rem !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background:{DARK_BLUE} !important;
        border-radius:8px 8px 0 0;
        padding:4px;
        gap:4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color:{TXT_MUTED} !important;
        font-size:0.85rem;
        padding:6px 16px;
        border-radius:6px;
        font-family:'Source Sans Pro',sans-serif;
    }}
    .stTabs [aria-selected="true"] {{
        background:{GOLD}22 !important;
        color:{GOLD} !important;
        border-bottom:2px solid {GOLD} !important;
    }}
    .stExpander {{
        background:{CARD_BG} !important;
        border:1px solid {DARK_BLUE} !important;
        border-radius:8px !important;
    }}
    .stTextInput input, .stNumberInput input {{
        background:{CARD_BG} !important;
        color:{TXT_MAIN} !important;
        border:1px solid {GOLD}44 !important;
        border-radius:6px !important;
    }}
    .stSelectbox [data-baseweb="select"] {{
        background:{CARD_BG} !important;
        border:1px solid {GOLD}44 !important;
        border-radius:6px !important;
    }}
    .element-container {{ margin-bottom:0.6rem; }}

    /* ── ROC / CHART CONTAINERS ── */
    .chart-container {{
        background:{CARD_BG};
        border:1px solid {DARK_BLUE};
        border-radius:10px;
        padding:1rem;
        margin:0.8rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)
