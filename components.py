import streamlit as st

GOLD = "#FFD700"
DARK_BLUE = "#003366"
LIGHT_BLUE = "#ADD8E6"
TXT_MUTED = "#8892b0"

NAV_PAGES = [
    ("🏠 Home",               "🏠 Home"),
    ("📘 Theory",             "📘 Theory & Foundations"),
    ("🏦 C1: Loan Default",   "🏦 C1: Loan Default"),
    ("💳 C2: Fraud",          "💳 C2: Fraud Detection"),
    ("📉 C3: Churn",          "📉 C3: Churn Prediction"),
    ("🏭 C4: NPA",            "🏭 C4: SME NPA Scoring"),
    ("📈 C5: IPO",            "📈 C5: IPO Subscription"),
    ("📗 Excel Guide",        "📗 Excel Guide"),
    ("🧩 Quiz",               "🧩 Self-Assessment Quiz"),
]


def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-family:'Playfair Display',serif;font-size:1.15rem;
                        font-weight:700;color:{GOLD};letter-spacing:0.04em;">
                THE MOUNTAIN PATH
            </div>
            <div style="font-size:0.72rem;color:{TXT_MUTED};letter-spacing:0.12em;
                        text-transform:uppercase;margin-top:2px;">
                Academy of Finance
            </div>
            <hr style="border:none;border-top:1px solid {GOLD}44;margin:0.8rem 0;">
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;
                    color:{GOLD};padding:0 0.2rem 0.4rem;font-weight:600;">
            Navigation
        </div>
        """, unsafe_allow_html=True)

        current = st.session_state.get("page", "🏠 Home")
        for label, key in NAV_PAGES:
            is_active = current == key
            style_extra = f"background:{GOLD}22 !important;border-color:{GOLD} !important;" if is_active else ""
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state["page"] = key
                st.rerun()

        st.markdown(f"""
        <hr style="border:none;border-top:1px solid {GOLD}22;margin:1rem 0;">
        <div style="font-size:0.72rem;color:{TXT_MUTED};text-align:center;line-height:1.6;">
            <strong style="color:{GOLD};">Prof. V. Ravichandran</strong><br>
            28+ Yrs Corporate Finance<br>
            10+ Yrs Academic Excellence<br><br>
            <a href="https://themountainpathacademy.com" target="_blank"
               style="color:{GOLD};text-decoration:none;font-size:0.68rem;">
               🌐 themountainpathacademy.com
            </a>
        </div>
        """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div class="mp-header-band">
        <div>
            <div style="font-family:'Playfair Display',serif;font-size:1.35rem;
                        font-weight:700;color:{GOLD};user-select:none;">
                THE MOUNTAIN PATH ACADEMY
            </div>
            <div style="font-size:0.72rem;color:{LIGHT_BLUE};letter-spacing:0.15em;
                        text-transform:uppercase;margin-top:1px;">
                World of Finance &nbsp;·&nbsp; Logistic Regression Lab
            </div>
        </div>
        <div style="text-align:right;font-size:0.75rem;color:{TXT_MUTED};">
            MBA · CFA · FRM · FinTech
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown(f"""
    <div class="mp-footer">
        <strong style="color:{GOLD};">The Mountain Path Academy</strong>
        &nbsp;·&nbsp; Prof. V. Ravichandran &nbsp;·&nbsp;
        <a href="https://themountainpathacademy.com" target="_blank">themountainpathacademy.com</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/trichyravis" target="_blank">LinkedIn</a>
        &nbsp;·&nbsp;
        <a href="https://github.com/trichyravis" target="_blank">GitHub</a>
        <br><span style="color:{TXT_MUTED};font-size:0.7rem;">
        © The Mountain Path Academy · Logistic Regression Finance Lab · For Educational Use
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── Utility Helpers ──────────────────────────────────────────────────────────

def metric_card(value, label, color=GOLD, prefix="", suffix=""):
    st.markdown(f"""
    <div class="mp-metric">
        <div class="val" style="color:{color};">{prefix}{value}{suffix}</div>
        <div class="lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="mp-section-title">{text}</div>', unsafe_allow_html=True)


def card(content_html, variant=""):
    cls = f"mp-card{'-' + variant if variant else ''}"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def badge(text, color="gold"):
    return f'<span class="mp-badge badge-{color}">{text}</span>'


def formula_box(formula):
    st.markdown(f'<div class="mp-formula">{formula}</div>', unsafe_allow_html=True)


def confusion_matrix_html(tp, fp, fn, tn):
    return f"""
    <table class="cm-table">
      <tr>
        <th class="cm-header"></th>
        <th class="cm-header">Predicted Positive (1)</th>
        <th class="cm-header">Predicted Negative (0)</th>
        <th class="cm-header">Total</th>
      </tr>
      <tr>
        <td class="cm-header">Actual Positive (1)</td>
        <td class="cm-tp">TP = {tp}</td>
        <td class="cm-fn">FN = {fn} <br><small style="font-size:0.65rem;color:#f87171;">TYPE II ERROR</small></td>
        <td style="text-align:center;font-weight:600;">{tp+fn}</td>
      </tr>
      <tr>
        <td class="cm-header">Actual Negative (0)</td>
        <td class="cm-fp">FP = {fp} <br><small style="font-size:0.65rem;color:#fbbf24;">TYPE I ERROR</small></td>
        <td class="cm-tn">TN = {tn}</td>
        <td style="text-align:center;font-weight:600;">{fp+tn}</td>
      </tr>
      <tr>
        <td class="cm-header">Total</td>
        <td style="text-align:center;font-weight:600;">{tp+fp}</td>
        <td style="text-align:center;font-weight:600;">{fn+tn}</td>
        <td style="text-align:center;font-weight:700;">{tp+fp+fn+tn}</td>
      </tr>
    </table>
    """
