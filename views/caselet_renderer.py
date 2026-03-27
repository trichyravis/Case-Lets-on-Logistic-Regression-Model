import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from model_engine import (fit_logistic, predict_proba, classify,
                           confusion, metrics, roc_data, auc_score,
                           ks_stat, threshold_sweep, sigmoid, CASELETS)
from components import section_title, confusion_matrix_html

GOLD = "#FFD700"
DARK_BLUE = "#003366"
MID_BLUE = "#004d80"
LIGHT_BLUE = "#ADD8E6"
CARD_BG = "#112240"
TXT_MAIN = "#e6f1ff"
TXT_MUTED = "#8892b0"
GREEN = "#28a745"
RED = "#dc3545"


def _setup_chart_style():
    plt.rcParams.update({
        "figure.facecolor": CARD_BG,
        "axes.facecolor": "#0a1628",
        "axes.edgecolor": "#1a3355",
        "axes.labelcolor": LIGHT_BLUE,
        "xtick.color": TXT_MUTED,
        "ytick.color": TXT_MUTED,
        "text.color": TXT_MAIN,
        "grid.color": "#1a3355",
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })


def render_caselet(key: str):
    c = CASELETS[key]
    color = c["color"]
    X = c["X"]
    y = c["y"]

    # ── Fit model ────────────────────────────────────────────────────────────
    beta, ll, ll_null, mcfadden = fit_logistic(X, y)
    proba = predict_proba(beta, X)
    betas = beta[1:]      # skip intercept for display
    beta0 = beta[0]
    odds_ratios = np.exp(betas)

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.3rem;">
        <div style="width:6px;height:52px;background:{color};border-radius:3px;flex-shrink:0;"></div>
        <div>
            <div class="mp-title" style="font-size:1.8rem;">{c['title']}</div>
            <div class="mp-subtitle">{c['subtitle']}</div>
        </div>
    </div>
    <span class="mp-badge badge-{c['badge_color']}">Outcome: {c['outcome_label']}</span>
    <span class="mp-badge badge-gold">{len(c['features'])} Predictors</span>
    <span class="mp-badge badge-blue">Binary Classification</span>
    <hr class="mp-divider">
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📊 Dataset & Model", "🎯 Score New Observation",
                    "📉 Threshold & Errors", "📈 ROC Curve",
                    "💰 Cost Calculator", "🧠 Hypothesis Testing"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Dataset & Model
    # ════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            section_title("Training Dataset")
            header = " | ".join([f"<th>{f}</th>" for f in c["features"]])
            rows_html = ""
            for i in range(len(X)):
                outcome_color = RED if y[i] == 1 else GREEN
                outcome_txt = f"<strong style='color:{outcome_color};'>{y[i]} ({c['outcome_label'] if y[i]==1 else 'No '+c['outcome_label']})</strong>"
                p_color = RED if proba[i] >= 0.5 else GREEN
                cells = "".join([f"<td>{X[i,j]:.1f}</td>" for j in range(X.shape[1])])
                rows_html += f"<tr>{cells}<td>{outcome_txt}</td><td style='color:{p_color};font-weight:600;'>{proba[i]:.3f}</td></tr>"

            st.markdown(f"""
            <div style="overflow-x:auto;">
            <table class="mp-table">
            <tr>{"".join(f"<th>{f}</th>" for f in c["features"])}<th>Actual Y</th><th>P̂</th></tr>
            {rows_html}
            </table>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            section_title("Estimated Coefficients")
            st.markdown(f"""
            <div class="mp-card-blue">
            <div class="mp-label">Model Equation</div>
            <div class="mp-formula" style="margin-top:0.5rem;font-size:0.8rem;text-align:left;line-height:1.8;">
            z = {beta0:.3f}
            {"".join(f"<br>{'  + ' if betas[j]>=0 else '  − '}{abs(betas[j]):.4f} × {c['features'][j]}" for j in range(len(betas)))}
            <br><br>P({c['outcome_label']}) = σ(z)
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="mp-card">
            <div class="mp-label">Odds Ratios & Interpretation</div>
            <table class="mp-table" style="margin-top:0.6rem;font-size:0.8rem;">
            <tr><th>Variable</th><th>β</th><th>OR = eᵝ</th><th>Sign</th></tr>
            {"".join(f'<tr><td>{c["features"][j]}</td><td class="mp-mono">{betas[j]:.4f}</td><td style="color:{GOLD};font-weight:600;">{odds_ratios[j]:.3f}</td><td style="color:{"#4ade80" if c["var_signs"][j]=="-" else RED};">{c["var_signs"][j]}</td></tr>' for j in range(len(betas)))}
            </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="mp-card-gold">
            <div class="mp-label">Model Fit Statistics</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;">
                <div class="mp-metric"><div class="val" style="font-size:1.4rem;">{mcfadden:.3f}</div><div class="lbl">McFadden R²</div></div>
                <div class="mp-metric"><div class="val" style="font-size:1.4rem;">{ll:.2f}</div><div class="lbl">Log-Likelihood</div></div>
            </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Score New Observation
    # ════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        section_title(f"Score a New {c['institution']} Application")
        st.markdown(f"""
        <div class="mp-card-gold" style="margin-bottom:1rem;">
        <span class="mp-label">Instructions:</span>
        <span class="mp-body" style="margin-left:0.5rem;">
        Adjust the sliders below to enter the predictor values for a new observation.
        The model will calculate the predicted probability in real time.
        </span>
        </div>
        """, unsafe_allow_html=True)

        slider_cols = st.columns(2)
        input_vals = []
        for j, feat in enumerate(c["features"]):
            col = slider_cols[j % 2]
            with col:
                step = c["feature_steps"][j]
                mn   = c["feature_mins"][j]
                mx   = c["feature_maxs"][j]
                df   = c["feature_defaults"][j]
                if step == 1 and mn >= 0 and mx <= 1:
                    val = col.selectbox(f"**{feat}**", options=[0, 1],
                                        index=int(df), key=f"{key}_feat_{j}")
                else:
                    val = col.slider(f"**{feat}**", min_value=float(mn),
                                     max_value=float(mx), value=float(df),
                                     step=float(step), key=f"{key}_feat_{j}")
                input_vals.append(val)

        x_new = np.array(input_vals).reshape(1, -1)
        p_new = float(predict_proba(beta, x_new)[0])
        z_new = float(beta0 + np.dot(betas, input_vals))
        pred_class = 1 if p_new >= 0.5 else 0

        result_color = RED if pred_class == 1 else GREEN
        result_icon  = "⚠️" if pred_class == 1 else "✅"
        result_label = c["outcome_label"] if pred_class == 1 else f"No {c['outcome_label']}"

        st.markdown("<hr class='mp-divider'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="mp-metric">
                <div class="val" style="color:{result_color};font-size:2.2rem;">{p_new:.1%}</div>
                <div class="lbl">P({c['outcome_label']})</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="mp-metric">
                <div class="val" style="font-size:1.4rem;color:#aab4c8;">{z_new:.3f}</div>
                <div class="lbl">Linear Predictor z</div>
            </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="mp-metric">
                <div class="val" style="color:{result_color};font-size:1.4rem;">{result_icon} {result_label}</div>
                <div class="lbl">Prediction @ τ=0.50</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability gauge ──
        _setup_chart_style()
        fig, ax = plt.subplots(figsize=(7, 1.4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#0a1628")
        gradient = np.linspace(0, 1, 300).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", extent=[0, 1, 0, 1],
                  cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                      "", ["#28a745", "#fbbf24", "#dc3545"]))
        ax.axvline(p_new, color="white", lw=2.5, zorder=5)
        ax.axvline(0.5, color=GOLD, lw=1.5, linestyle="--", alpha=0.8, zorder=4)
        ax.text(p_new, 0.5, f" {p_new:.1%}", color="white",
                va="center", fontsize=10, fontweight="bold", zorder=6)
        ax.text(0.5, -0.4, "Decision\nBoundary", ha="center",
                color=GOLD, fontsize=7, va="top")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Predicted Probability", color=LIGHT_BLUE, fontsize=9)
        ax.spines[:].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── OR impact breakdown ──
        section_title("Contribution of Each Variable")
        contributions = [betas[j] * input_vals[j] for j in range(len(betas))]
        feat_names = c["features"]
        _setup_chart_style()
        fig2, ax2 = plt.subplots(figsize=(7, max(2.5, len(feat_names) * 0.55)))
        fig2.patch.set_alpha(0)
        colors = [RED if v > 0 and c["var_signs"][j] == "+" else GREEN
                  if v > 0 else RED if c["var_signs"][j] == "−" else GREEN
                  for j, v in enumerate(contributions)]
        bars = ax2.barh(feat_names, contributions,
                        color=[RED if v > 0 else GREEN for v in contributions],
                        alpha=0.85, height=0.55)
        ax2.axvline(0, color=GOLD, lw=1.5, alpha=0.8)
        ax2.set_xlabel("Contribution to z (β × X)", color=LIGHT_BLUE, fontsize=9)
        ax2.grid(axis="x", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, contributions):
            ax2.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + 0.15,
                     f"{val:.3f}", va="center",
                     ha="left" if val >= 0 else "right",
                     color="white", fontsize=8)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — Threshold & Errors
    # ════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        section_title("Threshold Sensitivity Analysis")
        tau = st.slider("**Classification Threshold τ**", 0.05, 0.95, 0.50, 0.05,
                        key=f"{key}_tau",
                        help="Predict Positive (1) if P̂ ≥ τ")

        pred = classify(proba, tau)
        tp, fp, fn, tn = confusion(y, pred)
        m = metrics(tp, fp, fn, tn)

        # Confusion matrix
        col_cm, col_m = st.columns([1, 1.1])
        with col_cm:
            section_title("Confusion Matrix")
            st.markdown(confusion_matrix_html(tp, fp, fn, tn), unsafe_allow_html=True)
            # Error costs
            type1_cost = fp * c["cost_fp"]
            type2_cost = fn * c["cost_fn"]
            total_cost = type1_cost + type2_cost
            st.markdown(f"""
            <div class="mp-card-gold" style="margin-top:0.8rem;">
            <div class="mp-label">Error Costs at τ = {tau:.2f}</div>
            <table class="mp-table" style="margin-top:0.5rem;font-size:0.82rem;">
            <tr><th>Error</th><th>Count</th><th>Cost/Error</th><th>Total</th></tr>
            <tr><td>Type I (FP)</td><td>{fp}</td><td>₹{c['cost_fp']:,}</td>
                <td style="color:#fbbf24;">₹{type1_cost:,}</td></tr>
            <tr><td>Type II (FN)</td><td>{fn}</td><td>₹{c['cost_fn']:,}</td>
                <td style="color:#f87171;">₹{type2_cost:,}</td></tr>
            <tr><td colspan="3"><strong>Total Error Cost</strong></td>
                <td style="color:{GOLD};font-weight:700;">₹{total_cost:,}</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with col_m:
            section_title("Performance Metrics")
            metric_data = [
                ("Accuracy", m['accuracy'], GOLD),
                ("Precision", m['precision'], LIGHT_BLUE),
                ("Recall", m['recall'], GREEN),
                ("Specificity", m['specificity'], "#a78bfa"),
                ("F1 Score", m['f1'], "#fb923c"),
            ]
            for mlabel, mval, mcolor in metric_data:
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:0.4rem 0.8rem;background:#0a1628;border-radius:6px;
                            margin-bottom:5px;border-left:3px solid {mcolor};">
                    <span style="font-size:0.85rem;color:{TXT_MUTED};">{mlabel}</span>
                    <span style="font-family:'Playfair Display',serif;font-size:1.1rem;
                                 color:{mcolor};font-weight:700;">{mval:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

            # Optimal threshold hint
            tau_opt = c['cost_fp'] / (c['cost_fp'] + c['cost_fn'])
            st.markdown(f"""
            <div class="mp-card-green" style="margin-top:0.8rem;">
            <div class="mp-label">Optimal Threshold Formula</div>
            <div class="mp-formula" style="margin-top:0.4rem;font-size:0.9rem;">
            τ* = {c['cost_fp']:,} / ({c['cost_fp']:,} + {c['cost_fn']:,}) = <strong>{tau_opt:.3f}</strong>
            </div>
            <div style="font-size:0.78rem;color:{TXT_MUTED};margin-top:0.3rem;">
            Current τ = {tau:.2f} &nbsp;|&nbsp;
            {"✅ At optimal" if abs(tau - tau_opt) < 0.05 else f"💡 Try τ ≈ {tau_opt:.2f} for minimum cost"}
            </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Full sweep chart ──
        section_title("Metrics Across All Thresholds")
        sweep = threshold_sweep(y, proba, c['cost_fp'], c['cost_fn'])
        thresholds_s = [r['threshold'] for r in sweep]
        _setup_chart_style()
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 3.5))
        fig3.patch.set_alpha(0)

        ax3a.plot(thresholds_s, [r['accuracy']/100 for r in sweep],
                  color=GOLD, lw=2, label="Accuracy")
        ax3a.plot(thresholds_s, [r['recall']/100 for r in sweep],
                  color=GREEN, lw=2, label="Recall")
        ax3a.plot(thresholds_s, [r['precision']/100 for r in sweep],
                  color=LIGHT_BLUE, lw=2, label="Precision")
        ax3a.plot(thresholds_s, [r['f1']/100 for r in sweep],
                  color="#fb923c", lw=2, linestyle="--", label="F1")
        ax3a.axvline(tau, color="white", lw=1.5, alpha=0.8, linestyle=":")
        ax3a.axvline(tau_opt, color="#4ade80", lw=1.5, alpha=0.7, linestyle="--")
        ax3a.set_xlabel("Threshold τ"); ax3a.set_ylabel("Metric Value")
        ax3a.set_title("Classification Metrics vs Threshold", color=GOLD, fontsize=10)
        ax3a.legend(fontsize=7, loc="lower left")
        ax3a.grid(alpha=0.3); ax3a.set_ylim(0, 1.05)

        ax3b.plot(thresholds_s, [r['total_cost'] for r in sweep],
                  color=RED, lw=2.5, label="Total Error Cost")
        ax3b.plot(thresholds_s, [r['type1_cost'] for r in sweep],
                  color="#fbbf24", lw=1.5, linestyle="--", label="Type I Cost (FP)")
        ax3b.plot(thresholds_s, [r['type2_cost'] for r in sweep],
                  color="#f87171", lw=1.5, linestyle="--", label="Type II Cost (FN)")
        ax3b.axvline(tau, color="white", lw=1.5, alpha=0.8, linestyle=":")
        ax3b.axvline(tau_opt, color="#4ade80", lw=1.5, alpha=0.7, linestyle="--")
        ax3b.set_xlabel("Threshold τ"); ax3b.set_ylabel("Cost (₹)")
        ax3b.set_title("Error Cost vs Threshold", color=GOLD, fontsize=10)
        ax3b.legend(fontsize=7); ax3b.grid(alpha=0.3)
        fig3.tight_layout(pad=1.5)
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — ROC Curve
    # ════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        section_title("ROC Curve Analysis")
        fprs_roc, tprs_roc = roc_data(y, proba)
        auc_val = auc_score(y, proba)
        ks_val  = ks_stat(y, proba)
        gini    = 2 * auc_val - 1

        mc1, mc2, mc3, mc4 = st.columns(4)
        for mcol, mval, mlbl, mcolor in [
            (mc1, f"{auc_val:.3f}", "AUC", GOLD),
            (mc2, f"{ks_val:.3f}",  "KS Statistic", LIGHT_BLUE),
            (mc3, f"{gini:.3f}",    "Gini Coefficient", GREEN),
            (mc4, f"{mcfadden:.3f}","McFadden R²", "#a78bfa"),
        ]:
            with mcol:
                grade = ("A+" if mval[:4] >= "0.95" else "A" if mval[:4] >= "0.90"
                         else "B" if mval[:4] >= "0.80" else "C") if mlbl == "AUC" else ""
                mcol.markdown(f"""
                <div class="mp-metric">
                    <div class="val" style="color:{mcolor};">{mval}</div>
                    <div class="lbl">{mlbl} {grade}</div>
                </div>
                """, unsafe_allow_html=True)

        col_roc, col_roc_info = st.columns([1.6, 1])
        with col_roc:
            _setup_chart_style()
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            fig4.patch.set_alpha(0)
            # Fill under curve
            ax4.fill_between(fprs_roc, tprs_roc, alpha=0.15, color=color)
            ax4.plot(fprs_roc, tprs_roc, color=color, lw=2.5,
                     label=f"ROC Curve (AUC = {auc_val:.3f})")
            ax4.plot([0, 1], [0, 1], color=TXT_MUTED, lw=1.5,
                     linestyle="--", label="Random Classifier")
            ax4.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=9)
            ax4.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=9)
            ax4.set_title(f"ROC Curve — {c['title']}", color=GOLD, fontsize=10)
            ax4.legend(fontsize=8)
            ax4.grid(alpha=0.3)
            ax4.set_xlim(-0.02, 1.02); ax4.set_ylim(-0.02, 1.05)
            # Mark KS point
            diffs = tprs_roc - fprs_roc
            ks_idx = np.argmax(diffs)
            ax4.annotate(f"KS = {ks_val:.3f}",
                         xy=(fprs_roc[ks_idx], tprs_roc[ks_idx]),
                         xytext=(fprs_roc[ks_idx]+0.1, tprs_roc[ks_idx]-0.1),
                         color="#4ade80", fontsize=8,
                         arrowprops=dict(arrowstyle="->", color="#4ade80"))
            st.pyplot(fig4, use_container_width=True)
            plt.close()

        with col_roc_info:
            section_title("How to Read This")
            st.markdown(f"""
            <div class="mp-card" style="font-size:0.82rem;line-height:1.8;">
            <div class="mp-label" style="margin-bottom:0.4rem;">AUC Grading Scale</div>
            <div style="color:{'#4ade80' if auc_val>=0.90 else GOLD};">
            {'✅' if auc_val>=0.90 else '🟡'} AUC {auc_val:.3f} —
            {'Excellent' if auc_val>=0.90 else 'Good' if auc_val>=0.80 else 'Fair'}
            </div>
            <hr class="mp-divider">
            <div style="color:{TXT_MUTED};">
            <strong style="color:{LIGHT_BLUE};">Curve bows upper-left?</strong><br>
            → Yes: Model beats random chance ✅<br><br>
            <strong style="color:{LIGHT_BLUE};">KS Point:</strong><br>
            → Max separation at threshold shown<br><br>
            <strong style="color:{LIGHT_BLUE};">Gini = {gini:.3f}</strong><br>
            → Model ranks {gini:.1%} of defaulter/non-defaulter pairs correctly<br><br>
            <strong style="color:{LIGHT_BLUE};">Diagonal line</strong> = random model (AUC=0.50)
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ROC table — compute TPR/FPR per rank safely from scratch
            sorted_idx = np.argsort(proba)[::-1]
            n_pos = max(int(np.sum(y == 1)), 1)
            n_neg = max(int(np.sum(y == 0)), 1)
            roc_rows = ""
            cum_tp, cum_fp = 0, 0
            for rank_i, obs_idx in enumerate(sorted_idx):
                if y[obs_idx] == 1:
                    cum_tp += 1
                else:
                    cum_fp += 1
                tpr_val = cum_tp / n_pos
                fpr_val = cum_fp / n_neg
                act_color = RED if y[obs_idx] == 1 else GREEN
                roc_rows += (
                    f"<tr><td>{rank_i+1}</td>"
                    f"<td style='color:{act_color};font-weight:600;'>{int(y[obs_idx])}</td>"
                    f"<td>{tpr_val:.2f}</td>"
                    f"<td>{fpr_val:.2f}</td>"
                    f"<td class='mp-mono' style='font-size:0.7rem;'>{proba[obs_idx]:.3f}</td></tr>"
                )
            st.markdown(f"""
            <div class="mp-card" style="margin-top:0.8rem;">
            <div class="mp-label">ROC Table (sorted by P̂ descending)</div>
            <table class="mp-table" style="margin-top:0.4rem;font-size:0.75rem;">
            <tr><th>Rank</th><th>Actual Y</th><th>TPR</th><th>FPR</th><th>P̂</th></tr>
            {roc_rows}
            </table>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — Cost Calculator
    # ════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        section_title("💰 Error Cost Optimiser")
        st.markdown(f"""
        <div class="mp-card-gold">
        <div class="mp-label">Customise Error Costs</div>
        <div style="font-size:0.82rem;color:{TXT_MUTED};margin-top:0.3rem;">
        Default costs are set from the caselet. Adjust to match your organisation's actual cost structure.
        </div>
        </div>
        """, unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            cfp = st.number_input("Cost per Type I Error — False Positive (₹)",
                                   min_value=0, value=int(c['cost_fp']),
                                   step=1000, key=f"{key}_cfp")
        with cc2:
            cfn = st.number_input("Cost per Type II Error — False Negative (₹)",
                                   min_value=0, value=int(c['cost_fn']),
                                   step=1000, key=f"{key}_cfn")

        tau_optimal = cfp / (cfp + cfn) if (cfp + cfn) > 0 else 0.5
        sweep = threshold_sweep(y, proba, cfp, cfn)
        best = min(sweep, key=lambda r: r['total_cost'])

        st.markdown(f"""
        <div class="mp-card-green" style="margin-top:0.8rem;">
        <div class="mp-label">Optimal Threshold Result</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.7rem;">
            <div class="mp-metric">
                <div class="val" style="font-size:1.6rem;">{tau_optimal:.3f}</div>
                <div class="lbl">Formula τ*</div>
            </div>
            <div class="mp-metric">
                <div class="val" style="font-size:1.6rem;">{best['threshold']:.2f}</div>
                <div class="lbl">Empirical Best τ</div>
            </div>
            <div class="mp-metric">
                <div class="val" style="font-size:1.6rem;color:{RED};">₹{best['total_cost']:,}</div>
                <div class="lbl">Min Total Cost</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # Full table
        section_title("Cost Analysis Table")
        table_rows = ""
        for r in sweep:
            highlight = "background:#0d3320;" if r['threshold'] == best['threshold'] else ""
            table_rows += f"""
            <tr style="{highlight}">
                <td style="font-weight:600;color:{GOLD};">{r['threshold']:.2f}</td>
                <td style="color:#4ade80;">{r['tp']}</td>
                <td style="color:#fbbf24;">{r['fp']}</td>
                <td style="color:#f87171;">{r['fn']}</td>
                <td>{r['tn']}</td>
                <td>{r['accuracy']}%</td>
                <td>{r['recall']}%</td>
                <td style="color:#fbbf24;">₹{r['type1_cost']:,}</td>
                <td style="color:#f87171;">₹{r['type2_cost']:,}</td>
                <td style="font-weight:700;color:{RED};">₹{r['total_cost']:,}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto;">
        <table class="mp-table" style="font-size:0.78rem;">
        <tr>
            <th>Threshold</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th>
            <th>Accuracy</th><th>Recall</th>
            <th>Type I Cost</th><th>Type II Cost</th><th>Total Cost</th>
        </tr>
        {table_rows}
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — Hypothesis Testing
    # ════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        section_title("Hypothesis Testing Framework")
        st.markdown(f"""
        <div class="mp-card-blue" style="text-align:center;padding:1.5rem;">
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.12em;color:{GOLD};margin-bottom:0.8rem;">Null Hypothesis H₀</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:{LIGHT_BLUE};">
        "{c['h0']}"
        </div>
        <div style="margin:1rem 0;color:{TXT_MUTED};font-size:1.5rem;">⟺</div>
        <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.12em;color:{RED};margin-bottom:0.8rem;">Alternative Hypothesis H₁</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.3rem;color:{RED};">
        "{c['h1']}"
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="mp-card" style="margin-top:1rem;">
        <div class="mp-label">Decision Rule</div>
        <div class="mp-formula" style="margin-top:0.5rem;">
        If P̂({c['outcome_label']}) ≥ τ → Reject H₀ → Predict <strong style="color:{RED};">{c['outcome_label']}</strong><br>
        If P̂({c['outcome_label']}) &lt; τ → Retain H₀ → Predict <strong style="color:{GREEN};">No {c['outcome_label']}</strong>
        </div>
        </div>
        """, unsafe_allow_html=True)

        eh_col, et_col = st.columns(2)
        with eh_col:
            st.markdown(f"""
            <div class="mp-card-red">
            <div class="mp-label" style="color:#fbbf24;">⚠️ Type I Error — False Positive</div>
            <div style="font-size:0.85rem;line-height:1.7;color:{TXT_MAIN};margin-top:0.5rem;">
            <strong>Statistical:</strong> Reject H₀ when H₀ is TRUE<br>
            <strong>In practice:</strong> Predict {c['outcome_label']}, but outcome was actually No {c['outcome_label']}<br><br>
            <strong>Business impact at {c['institution']}:</strong><br>
            Cost = ₹{c['cost_fp']:,} per Type I error<br><br>
            <strong>In this dataset:</strong> FP = {sum(1 for i in range(len(y)) if y[i]==0 and proba[i]>=0.5)} cases @ τ=0.50
            </div>
            </div>
            """, unsafe_allow_html=True)

        with et_col:
            st.markdown(f"""
            <div class="mp-card-red">
            <div class="mp-label" style="color:#f87171;">🚨 Type II Error — False Negative</div>
            <div style="font-size:0.85rem;line-height:1.7;color:{TXT_MAIN};margin-top:0.5rem;">
            <strong>Statistical:</strong> Retain H₀ when H₀ is FALSE<br>
            <strong>In practice:</strong> Predict No {c['outcome_label']}, but it actually occurred<br><br>
            <strong>Business impact at {c['institution']}:</strong><br>
            Cost = ₹{c['cost_fn']:,} per Type II error<br><br>
            <strong>In this dataset:</strong> FN = {sum(1 for i in range(len(y)) if y[i]==1 and proba[i]<0.5)} cases @ τ=0.50
            </div>
            </div>
            """, unsafe_allow_html=True)

        ratio = c['cost_fn'] / c['cost_fp'] if c['cost_fp'] > 0 else float('inf')
        st.markdown(f"""
        <div class="mp-card-gold" style="margin-top:1rem;text-align:center;">
        <div class="mp-label">Error Cost Ratio</div>
        <div style="font-family:'Playfair Display',serif;font-size:2rem;color:{RED};margin:0.5rem 0;">
        {ratio:.1f}×
        </div>
        <div style="font-size:0.85rem;color:{TXT_MUTED};">
        Type II (FN) costs <strong style="color:{GOLD};">{ratio:.1f}× more</strong> than Type I (FP) at {c['institution']}<br>
        → This justifies using a threshold lower than 0.50 to prioritise catching {c['outcome_label']} cases.
        </div>
        </div>
        """, unsafe_allow_html=True)
