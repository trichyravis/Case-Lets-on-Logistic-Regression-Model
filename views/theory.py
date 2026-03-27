import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from components import section_title, formula_box

GOLD="#FFD700"; DARK_BLUE="#003366"; LIGHT_BLUE="#ADD8E6"
CARD_BG="#112240"; TXT_MUTED="#8892b0"; GREEN="#28a745"; RED="#dc3545"


def render():
    st.markdown('<div class="mp-title">Theory & Foundations</div>', unsafe_allow_html=True)
    st.markdown('<div class="mp-subtitle">Logistic Regression — From Intuition to Mathematics</div>',
                unsafe_allow_html=True)

    tabs = st.tabs(["🔰 Core Concepts","📐 Mathematics","🔢 Odds & Coefficients",
                    "📊 Performance Metrics","⚖️ Type I vs Type II"])

    # ── Tab 1: Core Concepts ─────────────────────────────────────────────
    with tabs[0]:
        section_title("Why Not Linear Regression for Binary Outcomes?")
        st.markdown(f"""
        <div class="mp-card-red">
        <div class="mp-label">The Problem with Linear Regression</div>
        <div class="mp-body" style="margin-top:0.5rem;">
        When Y is binary (0 or 1), linear regression <strong>ŷ = β₀ + β₁X</strong> fails because:
        <ul style="padding-left:1.2rem;margin-top:0.5rem;">
            <li>It can predict values &lt; 0 or &gt; 1 — nonsensical probabilities</li>
            <li>The relationship between X and P(Y=1) is S-shaped, not linear</li>
            <li>Violates homoskedasticity assumption</li>
        </ul>
        </div></div>
        """, unsafe_allow_html=True)

        # ── Sigmoid animation ──
        plt.rcParams.update({"figure.facecolor":CARD_BG,"axes.facecolor":"#0a1628",
                              "axes.labelcolor":LIGHT_BLUE,"xtick.color":TXT_MUTED,
                              "ytick.color":TXT_MUTED,"text.color":"white",
                              "grid.color":"#1a3355","grid.alpha":0.5})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.patch.set_alpha(0)
        z = np.linspace(-6, 6, 300)
        sigmoid = 1 / (1 + np.exp(-z))
        linear = 0.1 * z + 0.5

        axes[0].plot(z, linear, color=RED, lw=2.5, label="Linear: ŷ = 0.1z + 0.5")
        axes[0].axhline(1, color=GOLD, lw=1, linestyle="--", alpha=0.6)
        axes[0].axhline(0, color=GOLD, lw=1, linestyle="--", alpha=0.6)
        axes[0].fill_between(z, linear, 1, where=(linear > 1), color=RED, alpha=0.3)
        axes[0].fill_between(z, linear, 0, where=(linear < 0), color=RED, alpha=0.3)
        axes[0].set_title("Linear Regression (FAILS)", color=RED, fontsize=10)
        axes[0].set_ylim(-0.4, 1.5); axes[0].legend(fontsize=8)
        axes[0].set_xlabel("Predictor z"); axes[0].set_ylabel("Predicted P")
        axes[0].grid(alpha=0.3); axes[0].text(-5.5, 1.25, "P > 1 ❌", color=RED, fontsize=8)
        axes[0].text(3, -0.25, "P < 0 ❌", color=RED, fontsize=8)

        axes[1].plot(z, sigmoid, color=GOLD, lw=2.5, label="Logistic: σ(z)")
        axes[1].fill_between(z, sigmoid, alpha=0.15, color=GOLD)
        axes[1].axhline(0.5, color=LIGHT_BLUE, lw=1.5, linestyle="--", alpha=0.7, label="P=0.5 boundary")
        axes[1].axvline(0, color=LIGHT_BLUE, lw=1, linestyle=":", alpha=0.5)
        axes[1].set_title("Logistic Function (CORRECT)", color=GOLD, fontsize=10)
        axes[1].set_ylim(-0.05, 1.1); axes[1].legend(fontsize=8)
        axes[1].set_xlabel("Linear predictor z"); axes[1].set_ylabel("P(Y=1)")
        axes[1].grid(alpha=0.3)
        axes[1].text(-5.5, 0.92, "Always 0 < P < 1 ✅", color="#4ade80", fontsize=8)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True, clear_figure=True)
        plt.close()

        section_title("Key Assumptions")
        st.markdown(f"""
        <table class="mp-table">
        <tr><th>#</th><th>Assumption</th><th>What it means</th></tr>
        <tr><td>1</td><td>Binary outcome</td><td>Y ∈ {{0, 1}} only</td></tr>
        <tr><td>2</td><td>Independence</td><td>Observations are independent of each other</td></tr>
        <tr><td>3</td><td>No multicollinearity</td><td>Predictors should not be highly correlated with each other</td></tr>
        <tr><td>4</td><td>Large sample</td><td>At least 10–20 events per predictor variable recommended</td></tr>
        <tr><td>5</td><td>Log-odds linearity</td><td>Logit(P) is linear in the predictors (not P itself)</td></tr>
        </table>
        """, unsafe_allow_html=True)

    # ── Tab 2: Mathematics ───────────────────────────────────────────────
    with tabs[1]:
        section_title("The Logistic Function")
        st.markdown(f"""
        <div class="mp-card-blue">
        <div class="mp-formula">
        P(Y=1 | X) = 1 / (1 + e<sup>−z</sup>)&emsp;where&emsp;z = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.8rem;font-size:0.83rem;">
        <div><strong style="color:{GOLD};">z → +∞</strong><br><span style="color:{TXT_MUTED};">P → 1.0 (certain positive)</span></div>
        <div><strong style="color:{GOLD};">z → −∞</strong><br><span style="color:{TXT_MUTED};">P → 0.0 (certain negative)</span></div>
        <div><strong style="color:{GOLD};">z = 0</strong><br><span style="color:{TXT_MUTED};">P = 0.5 (decision boundary)</span></div>
        <div><strong style="color:{GOLD};">Always</strong><br><span style="color:{TXT_MUTED};">0 &lt; P &lt; 1</span></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        section_title("Maximum Likelihood Estimation (MLE)")
        st.markdown(f"""
        <div class="mp-card">
        <div class="mp-label">How Coefficients Are Estimated</div>
        <p class="mp-body" style="margin-top:0.5rem;">
        Unlike linear regression (which uses Ordinary Least Squares), logistic regression
        uses <strong style="color:{GOLD};">Maximum Likelihood Estimation</strong>.
        We find β values that make the observed data <em>most probable</em>.
        </p>
        <div class="mp-formula" style="margin:0.7rem 0;">
        Log-Likelihood = Σᵢ [yᵢ ln(P̂ᵢ) + (1−yᵢ) ln(1−P̂ᵢ)]
        </div>
        <div class="mp-label" style="margin-top:0.6rem;">In Excel — Solver Setup:</div>
        <ul style="font-size:0.83rem;color:{TXT_MUTED};padding-left:1.2rem;margin-top:0.3rem;line-height:1.9;">
            <li>Compute z per row: <span class="mp-mono">=$B$1 + $B$2*X1 + $B$3*X2 + ...</span></li>
            <li>Compute P̂: <span class="mp-mono">=1/(1+EXP(-z))</span></li>
            <li>Compute LL per row: <span class="mp-mono">=Y*LN(P_hat)+(1-Y)*LN(1-P_hat)</span></li>
            <li>Total LL: <span class="mp-mono">=SUM(LL_column)</span></li>
            <li>Solver: <strong>Maximize</strong> Total LL by changing all β cells → GRG Nonlinear</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Odds & Coefficients ────────────────────────────────────────
    with tabs[2]:
        section_title("Odds, Log-Odds, and Coefficients")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="mp-card-blue">
            <div class="mp-label">Odds</div>
            <div class="mp-formula" style="margin-top:0.5rem;">Odds = P / (1 − P)</div>
            <div style="font-size:0.82rem;color:{TXT_MUTED};margin-top:0.5rem;">
            If P = 0.75:<br>
            Odds = 0.75/0.25 = 3.0<br>
            "3 to 1 in favour"
            </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="mp-card-blue">
            <div class="mp-label">Log-Odds (Logit)</div>
            <div class="mp-formula" style="margin-top:0.5rem;">Logit(P) = ln(P / (1−P)) = z</div>
            <div style="font-size:0.82rem;color:{TXT_MUTED};margin-top:0.5rem;">
            Range: −∞ to +∞<br>
            Linear in predictors ✅
            </div>
            </div>
            """, unsafe_allow_html=True)

        section_title("Interpreting Coefficients via Odds Ratios")
        st.markdown(f"""
        <div class="mp-card-gold">
        <div class="mp-formula">Odds Ratio (OR) = e<sup>β</sup></div>
        <table class="mp-table" style="margin-top:0.8rem;">
        <tr><th>β value</th><th>OR = eᵝ</th><th>Interpretation</th></tr>
        <tr><td>β = 0</td><td>1.00</td><td>No effect on odds</td></tr>
        <tr><td>β = +0.69</td><td>2.00</td><td>Doubles the odds</td></tr>
        <tr><td>β = −0.69</td><td>0.50</td><td>Halves the odds</td></tr>
        <tr><td>β = +1.10</td><td>3.00</td><td>Triples the odds</td></tr>
        <tr><td>β = −1.10</td><td>0.33</td><td>Reduces odds to ⅓</td></tr>
        </table>
        <div style="font-size:0.8rem;color:{TXT_MUTED};margin-top:0.5rem;">
        % Change in Odds = (OR − 1) × 100 = (e<sup>β</sup> − 1) × 100
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive OR calculator
        section_title("🔢 Odds Ratio Calculator")
        beta_input = st.slider("Enter β coefficient:", -3.0, 3.0, 0.5, 0.01)
        or_val = np.exp(beta_input)
        pct_change = (or_val - 1) * 100
        direction = "increases" if beta_input > 0 else "decreases"
        st.markdown(f"""
        <div class="mp-card-green" style="text-align:center;padding:1.2rem;">
        <div style="font-size:0.8rem;color:{TXT_MUTED};">β = {beta_input:.2f}</div>
        <div style="font-family:'Playfair Display',serif;font-size:2.5rem;color:{GOLD};">
            OR = {or_val:.4f}
        </div>
        <div style="color:#aab4c8;font-size:0.9rem;margin-top:0.3rem;">
        A 1-unit increase in X <strong style="color:{GOLD if beta_input>0 else LIGHT_BLUE};">{direction}</strong>
        the odds by <strong style="color:{GOLD};">{abs(pct_change):.1f}%</strong>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Performance Metrics ────────────────────────────────────────
    with tabs[3]:
        section_title("Complete Performance Metrics Reference")
        st.markdown(f"""
        <table class="mp-table">
        <tr><th>Metric</th><th>Formula</th><th>Meaning</th><th>Best for</th></tr>
        <tr><td><strong style="color:{GOLD};">Accuracy</strong></td><td>(TP+TN)/N</td><td>Overall correct predictions</td><td>Balanced datasets</td></tr>
        <tr><td><strong style="color:{LIGHT_BLUE};">Precision</strong></td><td>TP/(TP+FP)</td><td>Of predicted positives, how many are truly positive?</td><td>Minimise false alarms</td></tr>
        <tr><td><strong style="color:{GREEN};">Recall</strong></td><td>TP/(TP+FN)</td><td>Of all actual positives, how many were caught?</td><td>Minimise missed events</td></tr>
        <tr><td><strong style="color:#a78bfa;">Specificity</strong></td><td>TN/(TN+FP)</td><td>Of actual negatives, how many correctly identified?</td><td>Minimise false positives</td></tr>
        <tr><td><strong style="color:#fb923c;">F1 Score</strong></td><td>2·Prec·Rec/(Prec+Rec)</td><td>Harmonic mean of Precision & Recall</td><td>Imbalanced datasets</td></tr>
        <tr><td><strong style="color:{GOLD};">AUC-ROC</strong></td><td>∫TPR d(FPR)</td><td>Overall discriminating power across all thresholds</td><td>Model ranking</td></tr>
        <tr><td><strong style="color:{LIGHT_BLUE};">KS Statistic</strong></td><td>max(TPR−FPR)</td><td>Max separation of score distributions</td><td>Credit scoring</td></tr>
        <tr><td><strong style="color:{GREEN};">Gini</strong></td><td>2·AUC − 1</td><td>Rank-ordering quality of the model</td><td>Credit / banking</td></tr>
        </table>
        """, unsafe_allow_html=True)

        section_title("AUC Grading Scale")
        grades = [("0.95–1.00","Outstanding","#4ade80"),("0.90–0.95","Excellent",GOLD),
                  ("0.80–0.90","Good",LIGHT_BLUE),("0.70–0.80","Fair","#fb923c"),
                  ("0.60–0.70","Poor",RED),("0.50–0.60","Worthless","#6b7280")]
        cols = st.columns(len(grades))
        for col, (rng, grade, gcolor) in zip(cols, grades):
            with col:
                col.markdown(f"""
                <div style="text-align:center;background:{CARD_BG};border:1px solid {gcolor}44;
                            border-top:3px solid {gcolor};border-radius:8px;padding:0.6rem 0.3rem;">
                    <div style="font-size:0.75rem;font-weight:700;color:{gcolor};">{grade}</div>
                    <div style="font-size:0.68rem;color:{TXT_MUTED};">{rng}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 5: Type I vs Type II ──────────────────────────────────────────
    with tabs[4]:
        section_title("Type I vs Type II Errors")
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
        <div class="mp-card-red">
            <div class="mp-label" style="color:#fbbf24;">⚠️ Type I Error (False Positive)</div>
            <ul class="mp-body" style="padding-left:1.2rem;margin-top:0.5rem;">
                <li>Reject H₀ when H₀ is TRUE</li>
                <li>Predict Positive — Actually Negative</li>
                <li><strong>False Alarm</strong> — Fire alarm, no fire</li>
                <li>FP cell in confusion matrix</li>
                <li>Controls Precision</li>
            </ul>
        </div>
        <div class="mp-card-red">
            <div class="mp-label" style="color:#f87171;">🚨 Type II Error (False Negative)</div>
            <ul class="mp-body" style="padding-left:1.2rem;margin-top:0.5rem;">
                <li>Retain H₀ when H₀ is FALSE</li>
                <li>Predict Negative — Actually Positive</li>
                <li><strong>Silent Danger</strong> — Fire, but alarm didn't ring</li>
                <li>FN cell in confusion matrix</li>
                <li>Controls Recall</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

        section_title("Finance Context Cost Comparison")
        st.markdown(f"""
        <table class="mp-table">
        <tr><th>Caselet</th><th>Type I (FP) Meaning</th><th>Type I Cost</th><th>Type II (FN) Meaning</th><th>Type II Cost</th><th>Ratio</th></tr>
        <tr><td>Loan Default</td><td>Good borrower rejected</td><td>₹37,800</td><td>Bad borrower approved</td><td>₹99,000</td><td style="color:{GOLD};">2.6×</td></tr>
        <tr><td>Fraud</td><td>Genuine txn blocked</td><td>₹1,500</td><td>Fraud approved</td><td>₹47,000</td><td style="color:{RED};font-weight:700;">31×</td></tr>
        <tr><td>Churn</td><td>Wasted retention call</td><td>₹2,400</td><td>Churner missed</td><td>₹53,900</td><td style="color:{RED};font-weight:700;">22×</td></tr>
        <tr><td>NPA</td><td>Good SME rejected</td><td>₹7,25,000</td><td>NPA sanctioned</td><td>₹37,80,000</td><td style="color:{GOLD};">5.2×</td></tr>
        <tr><td>IPO Sub</td><td>Non-subscriber called</td><td>₹150</td><td>Subscriber missed</td><td>₹1,050</td><td>7×</td></tr>
        </table>
        <div class="mp-card-gold" style="margin-top:1rem;text-align:center;">
        <strong>Universal Rule:</strong> In every finance use case, Type II Error (FN) is more expensive than Type I (FP).<br>
        This is why finance models almost always use a threshold <strong>below 0.50</strong>.
        </div>
        """, unsafe_allow_html=True)
