import streamlit as st
from components import section_title, card, badge, formula_box
from model_engine import CASELETS

GOLD = "#FFD700"
LIGHT_BLUE = "#ADD8E6"
TXT_MUTED = "#8892b0"
CARD_BG = "#112240"


def render():
    st.markdown("""
    <div class="mp-title">Logistic Regression</div>
    <div class="mp-subtitle">Finance Caselets — Interactive Learning Lab</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mp-card-gold">
    <div class="mp-label">About This Application</div>
    <p class="mp-body" style="margin-top:0.5rem;">
    This interactive lab brings the five finance caselets from Prof. V. Ravichandran's
    comprehensive guide to life. For each caselet, you can <strong style="color:{GOLD};">
    score new observations</strong>, adjust the classification threshold, explore 
    the ROC curve, and study Type I vs Type II error trade-offs — all in real time.
    </p>
    </div>
    """, unsafe_allow_html=True)

    section_title("📋 Five Caselets at a Glance")

    cols = st.columns(2)
    caselet_list = [
        ("caselet1","🏦","C1","Loan Default","IndiaBank"),
        ("caselet2","💳","C2","Fraud Detection","PaySecure India"),
        ("caselet3","📉","C3","Customer Churn","SavannaBank"),
        ("caselet4","🏭","C4","SME NPA Scoring","LendRight NBFC"),
        ("caselet5","📈","C5","IPO Subscription","BullBear Securities"),
    ]
    page_map = {
        "caselet1": "🏦 C1: Loan Default",
        "caselet2": "💳 C2: Fraud Detection",
        "caselet3": "📉 C3: Churn Prediction",
        "caselet4": "🏭 C4: SME NPA Scoring",
        "caselet5": "📈 C5: IPO Subscription",
    }

    for i, (key, icon, code, name, inst) in enumerate(caselet_list):
        c = CASELETS[key]
        col = cols[i % 2]
        with col:
            st.markdown(f"""
            <div class="mp-card" style="border-left:4px solid {c['color']};">
                <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.6rem;">
                    <span style="font-size:1.6rem;">{icon}</span>
                    <div>
                        <div style="font-family:'Playfair Display',serif;font-size:1rem;
                                    font-weight:600;color:{c['color']};">{code}: {name}</div>
                        <div style="font-size:0.75rem;color:{TXT_MUTED};">{inst}</div>
                    </div>
                </div>
                <div style="font-size:0.82rem;color:#aab4c8;line-height:1.5;">
                    {c['business_insight']}
                </div>
                <div style="margin-top:0.7rem;">
                    <span class="mp-badge badge-{c['badge_color']}">{c['outcome_label']}</span>
                    <span class="mp-badge badge-gold">{len(c['features'])} Features</span>
                    <span class="mp-badge badge-blue">10 Observations</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {code} →", key=f"home_go_{key}", use_container_width=True):
                st.session_state["page"] = page_map[key]
                st.rerun()

    section_title("📐 Key Formulas Quick Reference")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="mp-card-blue">
        <div class="mp-label">Logistic Function</div>
        <div class="mp-formula" style="margin-top:0.5rem;">P(Y=1) = 1 / (1 + e<sup>−z</sup>)<br>
        z = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="mp-card-blue">
        <div class="mp-label">Odds Ratio Interpretation</div>
        <div class="mp-formula" style="margin-top:0.5rem;">OR_j = e<sup>βⱼ</sup><br>
        %ΔOdds = (e<sup>βⱼ</sup> − 1) × 100</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="mp-card-blue">
        <div class="mp-label">CAPM of Modelling: Sharpe Ratio for Models</div>
        <div class="mp-formula" style="margin-top:0.5rem;">Sharpe → AUC (higher = better model)<br>
        Gini = 2 × AUC − 1<br>KS = max(TPR − FPR)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="mp-card-gold">
        <div class="mp-label">Optimal Threshold</div>
        <div class="mp-formula" style="margin-top:0.5rem;">τ* = C_FP / (C_FP + C_FN)<br>
        <small style="color:{TXT_MUTED};">Lower ratio = lower threshold needed</small></div>
        </div>
        """, unsafe_allow_html=True)

    section_title("🚀 How to Use This App")
    st.markdown(f"""
    <div class="mp-card">
    <ol class="mp-body" style="padding-left:1.2rem;line-height:2;">
        <li><strong style="color:{GOLD};">Pick a Caselet</strong> from the sidebar navigation on the left.</li>
        <li><strong style="color:{GOLD};">Explore the Dataset</strong> — see the 10 training observations and variable definitions.</li>
        <li><strong style="color:{GOLD};">Score a New Observation</strong> — use the sliders to set predictor values and see the real-time probability prediction.</li>
        <li><strong style="color:{GOLD};">Adjust the Threshold</strong> — watch how the confusion matrix, Type I/II errors, and all metrics change dynamically.</li>
        <li><strong style="color:{GOLD};">Study the ROC Curve</strong> — visualise model performance across all possible thresholds.</li>
        <li><strong style="color:{GOLD};">Run the Cost Calculator</strong> — find the optimal threshold for your business cost structure.</li>
        <li><strong style="color:{GOLD};">Take the Quiz</strong> — test your understanding with 25 finance-focused questions.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
