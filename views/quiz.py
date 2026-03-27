import streamlit as st
from components import section_title

GOLD="#FFD700"; LIGHT_BLUE="#ADD8E6"; TXT_MUTED="#8892b0"; GREEN="#28a745"; RED="#dc3545"

QUESTIONS = [
    {"q":"In logistic regression, what is the range of the predicted output P̂?",
     "opts":["−∞ to +∞","0 to 100","0 to 1","−1 to +1"],"ans":2,
     "exp":"The logistic/sigmoid function always outputs a value between 0 and 1, making it suitable for probability estimation."},
    {"q":"If β₂ = +0.85, what does a 1-unit increase in X₂ do to the odds?",
     "opts":["Decreases odds by 0.85","Multiplies odds by e⁰·⁸⁵ = 2.34","Adds 0.85 to the probability","No effect — only significant if p < 0.05"],"ans":1,
     "exp":"The Odds Ratio = e^β. For β=0.85, OR = e^0.85 ≈ 2.34. A 1-unit increase multiplies the odds by 2.34."},
    {"q":"Which cell in the confusion matrix represents a Type I Error?",
     "opts":["True Positive (TP)","True Negative (TN)","False Positive (FP)","False Negative (FN)"],"ans":2,
     "exp":"Type I Error = False Positive. We rejected H₀ (predicted Positive) when H₀ was TRUE (actually Negative). This is the FP cell."},
    {"q":"Your fraud model has Recall=95%. What does this mean?",
     "opts":["95% of all transactions are flagged as fraud","95% of predicted fraud are actually fraud","95% of actual fraud transactions were caught","The model is 95% accurate overall"],"ans":2,
     "exp":"Recall = TP/(TP+FN). It answers: of all actual positives, how many did we catch? 95% recall means 95% of real fraud was detected."},
    {"q":"What is the role of Solver in Excel logistic regression?",
     "opts":["Draw the ROC curve","Sort data for ROC construction","Find β values that maximise Log-Likelihood","Compute the confusion matrix automatically"],"ans":2,
     "exp":"Logistic regression uses MLE. Solver iteratively adjusts β coefficients to maximise the total log-likelihood."},
    {"q":"An AUC of 0.50 means:",
     "opts":["Excellent discriminating power","50% accuracy","No better than random chance","50% recall rate"],"ans":2,
     "exp":"AUC = 0.50 is the diagonal line — random classifier. The model cannot distinguish between the two classes."},
    {"q":"When you lower the classification threshold from 0.50 to 0.30, what happens?",
     "opts":["Recall decreases, Precision increases","Recall increases, Precision decreases","Both increase","AUC changes"],"ans":1,
     "exp":"Lowering τ makes the model more 'trigger-happy' — it flags more cases as Positive. This catches more real positives (Recall ↑) but also flags more false alarms (Precision ↓)."},
    {"q":"Which metric is MOST important for a fraud detection model?",
     "opts":["Precision","Accuracy","Recall","Specificity"],"ans":2,
     "exp":"In fraud, missing a real fraud (FN = Type II Error) is catastrophic — the bank loses the full fraudulent amount. Recall must be as high as possible."},
    {"q":"Gini Coefficient = 0.88 means:",
     "opts":["88% of predictions are correct","The model correctly ranks 94% of pairs","The model has 88% accuracy on test data","McFadden R² = 0.88"],"ans":1,
     "exp":"Gini = 2×AUC − 1. If Gini=0.88, then AUC=0.94. This means if you pick one positive and one negative at random, the model ranks the positive higher 94% of the time."},
    {"q":"KS Statistic measures:",
     "opts":["Overall accuracy","Area under ROC curve","Max separation between cumulative distributions of the two classes","Goodness-of-fit like R²"],"ans":2,
     "exp":"KS = max(TPR − FPR) across all thresholds. It shows the threshold at which the model best separates positive from negative cases."},
    {"q":"In Caselet 4 (SME NPA), DSCR has β = −3.122. What does OR = 0.044 mean?",
     "opts":["DSCR is negatively correlated with NPA","A 1-unit DSCR increase reduces NPA odds by 95.6%","DSCR reduces NPA probability by 3.122%","Both A and B are correct"],"ans":3,
     "exp":"Both statements are correct. OR = e^(−3.122) = 0.044. This means a 1-unit DSCR increase multiplies NPA odds by 0.044 — a 95.6% reduction. And yes, this implies a strong negative correlation."},
    {"q":"A bank model has Accuracy=88% but Recall=65%. A competing model has Accuracy=82% but Recall=90%. Which is better for credit risk?",
     "opts":["Model 1 — higher accuracy is always better","Model 2 — higher recall is critical in credit","Both are equally good","Cannot determine without AUC"],"ans":1,
     "exp":"In credit risk, missing defaulters (FN = Type II Error) is far more expensive than incorrectly declining good borrowers. Model 2's 90% recall vs 65% means it catches far more defaulters. Accuracy is misleading here."},
    {"q":"The optimal threshold formula τ* = C_FP/(C_FP + C_FN). If C_FP=₹2,000 and C_FN=₹48,000, what is τ*?",
     "opts":["0.50","0.04","0.96","0.25"],"ans":1,
     "exp":"τ* = 2000/(2000+48000) = 2000/50000 = 0.04. This very low threshold reflects that FN errors are 24× more expensive, so the model should be very aggressive in flagging positives."},
    {"q":"F1 Score is preferred over Accuracy when:",
     "opts":["The model has high AUC","The dataset has class imbalance","There are more than 5 predictors","The training set is small"],"ans":1,
     "exp":"F1 = harmonic mean of Precision and Recall. With class imbalance (e.g., 0.2% fraud rate), accuracy is misleading because predicting 'all negative' gives 99.8% accuracy but 0% F1."},
    {"q":"In hypothesis testing for logistic regression, H₀ represents:",
     "opts":["The event occurring (default, fraud, churn)","The negative/safe outcome (no default, genuine, stay)","The null that all coefficients are zero","The alternative we want to prove"],"ans":1,
     "exp":"H₀ is the null/default assumption — the 'safe' starting point. In credit: H₀ = 'applicant will not default.' The model only rejects H₀ (predicts positive) when evidence (predicted probability) is strong enough."},
]


def render():
    st.markdown('<div class="mp-title">Self-Assessment Quiz</div>', unsafe_allow_html=True)
    st.markdown('<div class="mp-subtitle">Test your understanding of logistic regression in finance</div>',
                unsafe_allow_html=True)

    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {
            "answers": {},
            "submitted": False,
            "score": 0,
        }

    qs = st.session_state.quiz_state

    if not qs["submitted"]:
        st.markdown(f"""
        <div class="mp-card-gold" style="margin-bottom:1.2rem;">
        <div class="mp-label">Instructions</div>
        <div class="mp-body" style="margin-top:0.4rem;">
        Answer all {len(QUESTIONS)} questions, then click <strong>Submit Quiz</strong>.
        Each correct answer = 1 point. 13+ = Pass | 10-12 = Review needed | Below 10 = Re-read the caselets.
        </div>
        </div>
        """, unsafe_allow_html=True)

        for i, qdata in enumerate(QUESTIONS):
            with st.expander(f"Q{i+1}. {qdata['q']}", expanded=(i < 3)):
                ans = st.radio("", qdata["opts"],
                               key=f"q_{i}", index=None,
                               label_visibility="collapsed")
                if ans is not None:
                    qs["answers"][i] = qdata["opts"].index(ans)

        answered = len(qs["answers"])
        st.markdown(f"""
        <div style="font-size:0.82rem;color:{TXT_MUTED};margin-bottom:0.8rem;">
        {answered} of {len(QUESTIONS)} questions answered
        </div>
        """, unsafe_allow_html=True)

        if st.button("📝 Submit Quiz", use_container_width=True, type="primary"):
            score = sum(1 for i, qdata in enumerate(QUESTIONS)
                       if qs["answers"].get(i) == qdata["ans"])
            qs["score"] = score
            qs["submitted"] = True
            st.rerun()

    else:
        score = qs["score"]
        total = len(QUESTIONS)
        pct = score / total * 100
        grade_color = GREEN if pct >= 87 else GOLD if pct >= 67 else RED
        grade_label = ("Excellent! 🏆" if pct >= 87
                       else "Good — Review highlighted topics 📖" if pct >= 67
                       else "Needs work — Re-read the caselets 📚")

        st.markdown(f"""
        <div style="text-align:center;padding:2rem;background:linear-gradient(135deg,#0a1628,#112240);
                    border:2px solid {grade_color};border-radius:12px;margin-bottom:1.5rem;">
            <div style="font-family:'Playfair Display',serif;font-size:4rem;color:{grade_color};font-weight:700;">
                {score}/{total}
            </div>
            <div style="font-size:1.3rem;color:{grade_color};margin:0.3rem 0;">{pct:.0f}%</div>
            <div style="font-size:1rem;color:#aab4c8;">{grade_label}</div>
        </div>
        """, unsafe_allow_html=True)

        section_title("Detailed Answer Review")
        for i, qdata in enumerate(QUESTIONS):
            user_ans = qs["answers"].get(i)
            correct = qdata["ans"]
            is_correct = user_ans == correct
            icon = "✅" if is_correct else "❌"
            border_color = GREEN if is_correct else RED

            with st.expander(f"{icon} Q{i+1}. {qdata['q']}"):
                for j, opt in enumerate(qdata["opts"]):
                    if j == correct:
                        st.markdown(f"<div style='color:{GREEN};font-weight:600;'>✓ {opt} (Correct Answer)</div>",
                                    unsafe_allow_html=True)
                    elif j == user_ans and not is_correct:
                        st.markdown(f"<div style='color:{RED};'>✗ {opt} (Your Answer)</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color:{TXT_MUTED};'>{opt}</div>",
                                    unsafe_allow_html=True)
                st.markdown(f"""
                <div class="mp-card-blue" style="margin-top:0.6rem;">
                <div class="mp-label">Explanation</div>
                <div style="font-size:0.85rem;margin-top:0.3rem;">{qdata['exp']}</div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.quiz_state = {"answers": {}, "submitted": False, "score": 0}
            st.rerun()
