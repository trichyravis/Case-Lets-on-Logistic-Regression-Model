import streamlit as st
from components import section_title

GOLD="#FFD700"; LIGHT_BLUE="#ADD8E6"; CARD_BG="#112240"
TXT_MUTED="#8892b0"; GREEN="#28a745"; RED="#dc3545"

EXCEL_STEPS = {
    "caselet1": {
        "title": "C1: Loan Default — IndiaBank",
        "color": GOLD,
        "sheet_layout": [
            ("B1","β₀ (Intercept)","0","Start value for Solver"),
            ("B2","β₁ (Credit Score)","0","Start value"),
            ("B3","β₂ (Monthly Income)","0","Start value"),
            ("B4","β₃ (DTI Ratio)","0","Start value"),
            ("B5","β₄ (Employment)","0","Start value"),
        ],
        "data_cols": ["A: Obs#","B: Credit Score","C: Income (₹K)","D: DTI (%)","E: Employment","F: Default Y"],
        "formulas": [
            ("G2","Linear predictor z","=$B$1+$B$2*B2+$B$3*C2+$B$4*D2+$B$5*E2"),
            ("H2","Predicted Probability P̂","=1/(1+EXP(-G2))"),
            ("I2","Log-Likelihood per row","=F2*LN(H2)+(1-F2)*LN(1-H2)"),
            ("I12","Total Log-Likelihood","=SUM(I2:I11)"),
            ("J2","Predicted Class","=IF(H2>=0.5,1,0)"),
        ],
        "solver": "Maximize I12 by changing B1:B5 — Method: GRG Nonlinear",
        "metrics_formulas": [
            ("TP","=COUNTIFS(F2:F11,1,J2:J11,1)"),
            ("FP","=COUNTIFS(F2:F11,0,J2:J11,1)"),
            ("FN","=COUNTIFS(F2:F11,1,J2:J11,0)"),
            ("TN","=COUNTIFS(F2:F11,0,J2:J11,0)"),
            ("Accuracy","=(TP+TN)/(TP+TN+FP+FN)"),
            ("Precision","=TP/(TP+FP)"),
            ("Recall","=TP/(TP+FN)"),
            ("F1","=2*Precision*Recall/(Precision+Recall)"),
        ],
    },
    "caselet2": {
        "title": "C2: Fraud Detection — PaySecure",
        "color": RED,
        "sheet_layout": [
            ("B1","β₀","0",""),("B2","β₁ Amount","0",""),
            ("B3","β₂ Night","0",""),("B4","β₃ Distance","0",""),
            ("B5","β₄ Frequency","0",""),("B6","β₅ Merchant","0",""),
        ],
        "data_cols": ["A: Obs","B: Amount","C: Night","D: Distance","E: Freq","F: Merchant","G: Fraud Y"],
        "formulas": [
            ("H2","z","=$B$1+$B$2*B2+$B$3*C2+$B$4*D2+$B$5*E2+$B$6*F2"),
            ("I2","P̂","=1/(1+EXP(-H2))"),
            ("J2","LL","=G2*LN(I2)+(1-G2)*LN(1-I2)"),
            ("J12","Total LL","=SUM(J2:J11)"),
            ("K2","Class","=IF(I2>=0.5,1,0)"),
        ],
        "solver": "Maximize J12 by changing B1:B6 — GRG Nonlinear",
        "metrics_formulas": [
            ("TP","=COUNTIFS(G2:G11,1,K2:K11,1)"),
            ("FP","=COUNTIFS(G2:G11,0,K2:K11,1)"),
            ("FN","=COUNTIFS(G2:G11,1,K2:K11,0)"),
            ("TN","=COUNTIFS(G2:G11,0,K2:K11,0)"),
            ("Recall (Priority!)","=TP/(TP+FN)"),
            ("Type I Cost","=FP*1500"),
            ("Type II Cost","=FN*47000"),
        ],
    },
}


def render():
    st.markdown('<div class="mp-title">Excel Replication Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="mp-subtitle">Build every caselet model from scratch in Microsoft Excel</div>',
                unsafe_allow_html=True)

    # ── Universal Solver Workflow ─────────────────────────────────────────
    section_title("🔧 Universal Excel Workflow (All Caselets)")

    st.markdown(f"""
    <div class="mp-card-gold">
    <div class="mp-label">The 7-Step Excel Recipe — Works for Every Caselet</div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Set Up Coefficient Cells",
         "In a separate area (e.g., cells B1:B6), enter initial values of 0 for each coefficient (β₀, β₁, β₂, ...). These are the cells Solver will optimise.",
         "=$B$1 (use absolute references — the $ signs are essential!)"),
        ("2", "Enter the Training Data",
         "Columns A–F (or A–G): one row per observation. Include all predictor variables and the binary outcome Y (0 or 1) in the last column.",
         "=A2, B2, ... (plain data — no formula needed)"),
        ("3", "Compute the Linear Predictor z",
         "For each data row, compute z = β₀ + β₁X₁ + β₂X₂ + ... Using absolute references for the β cells so you can drag the formula down all rows.",
         "=$B$1+$B$2*B2+$B$3*C2+$B$4*D2+$B$5*E2"),
        ("4", "Compute Predicted Probability P̂",
         "Apply the logistic (sigmoid) function to z. This converts any z value into a probability between 0 and 1.",
         "=1/(1+EXP(-G2))  [where G2 contains z]"),
        ("5", "Compute Log-Likelihood Per Row",
         "The log-likelihood measures how well the current β values explain each observation. Y=1 rows use ln(P̂); Y=0 rows use ln(1−P̂).",
         "=F2*LN(H2)+(1-F2)*LN(1-H2)  [where F=Y, H=P̂]"),
        ("6", "Sum the Log-Likelihood",
         "In a single cell, sum all per-row log-likelihoods. This is the objective function Solver will maximise.",
         "=SUM(I2:I11)  [where column I has per-row LL values]"),
        ("7", "Run Solver",
         "Data → Solver. Set Objective = LL sum cell, To = Max, By Changing = all β cells (B1:B5). Method = GRG Nonlinear. Click Solve and wait for convergence.",
         "Check: Solver message should say 'Solver found a solution' — not 'did not converge'"),
    ]

    for num, title, desc, formula in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:0.8rem;align-items:flex-start;">
            <div style="background:{GOLD};color:#000;font-weight:700;font-size:1rem;
                        width:32px;height:32px;border-radius:50%;display:flex;
                        align-items:center;justify-content:center;flex-shrink:0;">
                {num}
            </div>
            <div style="flex:1;">
                <div style="font-weight:600;color:{LIGHT_BLUE};font-size:0.95rem;">{title}</div>
                <div style="font-size:0.83rem;color:#aab4c8;line-height:1.6;margin:0.2rem 0;">{desc}</div>
                <div class="mp-mono" style="font-size:0.8rem;">{formula}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="mp-divider">', unsafe_allow_html=True)

    # ── Caselet-Specific Sheets ───────────────────────────────────────────
    section_title("📋 Caselet-Specific Sheet Blueprints")

    tab_labels = [v["title"] for v in EXCEL_STEPS.values()]
    tabs = st.tabs(tab_labels)

    for tab, (key, details) in zip(tabs, EXCEL_STEPS.items()):
        with tab:
            color = details["color"]

            # Coefficient layout
            st.markdown(f"""
            <div class="mp-card" style="border-left:3px solid {color};margin-bottom:1rem;">
            <div class="mp-label">Step 1: Coefficient Cell Layout</div>
            <table class="mp-table" style="margin-top:0.5rem;font-size:0.82rem;">
            <tr><th>Cell</th><th>Label</th><th>Initial Value</th><th>Note</th></tr>
            {"".join(f"<tr><td class='mp-mono'>{cell}</td><td>{lbl}</td><td class='mp-mono' style='color:{GOLD};'>{val}</td><td style='color:{TXT_MUTED};font-size:0.75rem;'>{note}</td></tr>" for cell,lbl,val,note in details['sheet_layout'])}
            </table>
            </div>
            """, unsafe_allow_html=True)

            # Data column layout
            st.markdown(f"""
            <div class="mp-card" style="border-left:3px solid {color};margin-bottom:1rem;">
            <div class="mp-label">Step 2: Data Column Layout (Rows 2–11)</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-top:0.5rem;">
            {"".join(f'<span class="mp-badge badge-blue" style="font-size:0.75rem;">{col}</span>' for col in details["data_cols"])}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # Formula column
            st.markdown(f"""
            <div class="mp-card" style="border-left:3px solid {color};margin-bottom:1rem;">
            <div class="mp-label">Steps 3–6: Working Formulas</div>
            <table class="mp-table" style="margin-top:0.5rem;font-size:0.82rem;">
            <tr><th>Cell</th><th>Purpose</th><th>Formula (enter in row 2, drag down)</th></tr>
            {"".join(f'<tr><td class="mp-mono" style="color:{GOLD};">{cell}</td><td>{purpose}</td><td class="mp-mono" style="font-size:0.75rem;color:{LIGHT_BLUE};">{formula}</td></tr>' for cell,purpose,formula in details["formulas"])}
            </table>
            </div>
            """, unsafe_allow_html=True)

            # Solver
            st.markdown(f"""
            <div class="mp-card-green" style="margin-bottom:1rem;">
            <div class="mp-label">Step 7: Solver Configuration</div>
            <div class="mp-formula" style="margin-top:0.5rem;text-align:left;font-size:0.88rem;">
            {details['solver']}
            </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            st.markdown(f"""
            <div class="mp-card" style="border-left:3px solid {color};">
            <div class="mp-label">Performance Metrics (after Solver)</div>
            <table class="mp-table" style="margin-top:0.5rem;font-size:0.82rem;">
            <tr><th>Metric</th><th>Excel Formula</th></tr>
            {"".join(f'<tr><td style="font-weight:600;color:{LIGHT_BLUE};">{m}</td><td class="mp-mono" style="font-size:0.75rem;">{f}</td></tr>' for m,f in details["metrics_formulas"])}
            </table>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="mp-divider">', unsafe_allow_html=True)

    # ── ROC in Excel ─────────────────────────────────────────────────────
    section_title("📈 Building the ROC Curve in Excel")

    st.markdown(f"""
    <div class="mp-card-blue">
    <div class="mp-label">5-Step ROC Construction</div>
    <ol style="font-size:0.85rem;line-height:2.2;padding-left:1.2rem;margin-top:0.5rem;color:#aab4c8;">
        <li><strong style="color:{GOLD};">Sort by P̂ descending:</strong>
            Select all rows → Data → Sort → Sort by P̂ column, Largest to Smallest</li>
        <li><strong style="color:{GOLD};">Add Cum. Defaulters column:</strong>
            <span class="mp-mono">=IF(Y_row=1, 1, 0)</span> for first row,
            then <span class="mp-mono">=prev + IF(Y_row=1,1,0)</span> for subsequent rows</li>
        <li><strong style="color:{GOLD};">Add Cum. Non-Defaulters column</strong> (same logic, IF Y=0)</li>
        <li><strong style="color:{GOLD};">Compute TPR and FPR:</strong>
            TPR = Cum.Def / Total_Positives,&nbsp; FPR = Cum.NonDef / Total_Negatives</li>
        <li><strong style="color:{GOLD};">Plot:</strong>
            Select FPR (X) and TPR (Y) → Insert → Scatter → Scatter with Straight Lines.
            Add diagonal reference series: (0,0) and (1,1)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mp-card-gold" style="margin-top:0.8rem;">
    <div class="mp-label">AUC via Trapezoidal Rule in Excel</div>
    <div class="mp-formula" style="margin-top:0.5rem;">
    =SUMPRODUCT((FPR_col_shifted - FPR_col), (TPR_col_shifted + TPR_col) / 2)
    </div>
    <div style="font-size:0.8rem;color:{TXT_MUTED};margin-top:0.4rem;">
    Or manually: =SUMPRODUCT((G3:G12-G2:G11),(F3:F12+F2:F11)/2) where G=FPR, F=TPR
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cost Calculator ───────────────────────────────────────────────────
    section_title("💰 Excel Cost Optimiser Worksheet")

    st.markdown(f"""
    <div class="mp-card">
    <div class="mp-label">Sheet: "Cost Calculator"</div>
    <table class="mp-table" style="margin-top:0.6rem;font-size:0.82rem;">
    <tr><th>Section</th><th>Cell</th><th>Label</th><th>Formula / Value</th></tr>
    <tr><td rowspan="2">A. Inputs</td>
        <td class="mp-mono">B1</td><td>Cost per FP (Type I) — ₹</td><td>Enter your value</td></tr>
    <tr><td class="mp-mono">B2</td><td>Cost per FN (Type II) — ₹</td><td>Enter your value</td></tr>
    <tr><td rowspan="4">B. Threshold Loop<br>(rows 10–28)</td>
        <td class="mp-mono">A10</td><td>Threshold τ</td><td>0.05, 0.10, 0.15 … 0.95</td></tr>
    <tr><td class="mp-mono">B10</td><td>TP</td><td class="mp-mono">=COUNTIFS($Y,1,$PHAT,"&gt;="&A10)</td></tr>
    <tr><td class="mp-mono">E10</td><td>Type I Cost</td><td class="mp-mono">=C10*$B$1  [C10=FP count]</td></tr>
    <tr><td class="mp-mono">G10</td><td>Total Cost</td><td class="mp-mono">=E10+F10  ← Minimise this</td></tr>
    <tr><td>C. Optimum</td>
        <td class="mp-mono">B30</td><td>Optimal Threshold</td>
        <td class="mp-mono">=INDEX(A10:A28,MATCH(MIN(G10:G28),G10:G28,0))</td></tr>
    <tr><td>D. Chart</td>
        <td colspan="2">Total Cost vs τ Line Chart</td>
        <td>Select A10:A28 and G10:G28 → Insert → Line Chart</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Common Errors ─────────────────────────────────────────────────────
    section_title("⚠️ Common Excel Errors and Fixes")

    errors = [
        ("#NUM! in Log-Likelihood",
         "P̂ is exactly 0 or 1, making LN(0) = -∞",
         "Add a small epsilon: =Y*LN(MAX(P_hat,0.000001))+(1-Y)*LN(MAX(1-P_hat,0.000001))"),
        ("Solver says 'Did not converge'",
         "Starting values of 0 may be too far from solution, or learning rate issues",
         "Try starting β₀ = LN(mean_Y/(1-mean_Y)) and all other βs at 0.01"),
        ("All P̂ values equal 0.5",
         "Solver ran before coefficients changed — likely hit iteration limit",
         "Go to Solver Options → increase Max Iterations to 5000, Precision to 0.00001"),
        ("#DIV/0! in Precision",
         "No observations predicted positive — TP+FP = 0",
         "Wrap formula: =IFERROR(TP/(TP+FP), 0) to handle edge cases"),
        ("ROC curve is below the diagonal",
         "Classes may be swapped (Y=0 is the positive class)",
         "Flip the outcome coding: replace Y with 1-Y and refit"),
        ("Solver changes coefficients but LL doesn't improve",
         "Multicollinearity between predictors",
         "Check correlation matrix of predictors — if |r| > 0.90, remove one variable"),
    ]

    for err, cause, fix in errors:
        with st.expander(f"❌  {err}"):
            st.markdown(f"""
            <div style="font-size:0.85rem;line-height:1.8;">
            <strong style="color:#fbbf24;">Cause:</strong> {cause}<br>
            <strong style="color:#4ade80;">Fix:</strong> {fix}
            </div>
            """, unsafe_allow_html=True)

    # ── Quick Reference Card ──────────────────────────────────────────────
    section_title("📌 Excel Formula Quick Reference Card")

    st.markdown(f"""
    <div class="mp-card-gold">
    <table class="mp-table" style="font-size:0.82rem;">
    <tr><th>Task</th><th>Excel Formula</th></tr>
    <tr><td>Logistic function (sigmoid)</td><td class="mp-mono">=1/(1+EXP(-z_cell))</td></tr>
    <tr><td>Natural logarithm</td><td class="mp-mono">=LN(value)</td></tr>
    <tr><td>Exponential (e^x)</td><td class="mp-mono">=EXP(x)</td></tr>
    <tr><td>Log-likelihood per obs</td><td class="mp-mono">=Y*LN(P_hat)+(1-Y)*LN(1-P_hat)</td></tr>
    <tr><td>Predicted class at τ=0.5</td><td class="mp-mono">=IF(P_hat&gt;=0.5,1,0)</td></tr>
    <tr><td>Predicted class at custom τ</td><td class="mp-mono">=IF(P_hat&gt;=threshold_cell,1,0)</td></tr>
    <tr><td>Count True Positives</td><td class="mp-mono">=COUNTIFS(Y_range,1,Pred_range,1)</td></tr>
    <tr><td>Count False Positives</td><td class="mp-mono">=COUNTIFS(Y_range,0,Pred_range,1)</td></tr>
    <tr><td>Count False Negatives</td><td class="mp-mono">=COUNTIFS(Y_range,1,Pred_range,0)</td></tr>
    <tr><td>Count True Negatives</td><td class="mp-mono">=COUNTIFS(Y_range,0,Pred_range,0)</td></tr>
    <tr><td>Accuracy</td><td class="mp-mono">=(TP+TN)/(TP+TN+FP+FN)</td></tr>
    <tr><td>Precision</td><td class="mp-mono">=IFERROR(TP/(TP+FP),0)</td></tr>
    <tr><td>Recall (Sensitivity)</td><td class="mp-mono">=IFERROR(TP/(TP+FN),0)</td></tr>
    <tr><td>Specificity</td><td class="mp-mono">=IFERROR(TN/(TN+FP),0)</td></tr>
    <tr><td>F1 Score</td><td class="mp-mono">=IFERROR(2*Precision*Recall/(Precision+Recall),0)</td></tr>
    <tr><td>Gini Coefficient</td><td class="mp-mono">=2*AUC_cell-1</td></tr>
    <tr><td>Optimal threshold (formula)</td><td class="mp-mono">=Cost_FP/(Cost_FP+Cost_FN)</td></tr>
    <tr><td>Chi-sq p-value (HL test)</td><td class="mp-mono">=CHISQ.DIST.RT(HL_stat, df)</td></tr>
    <tr><td>McFadden R²</td><td class="mp-mono">=1-(LL_model/LL_null)</td></tr>
    <tr><td>Error safe division</td><td class="mp-mono">=IFERROR(numerator/denominator, 0)</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)
