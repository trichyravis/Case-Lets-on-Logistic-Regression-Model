# 📊 Logistic Regression Finance Lab
## The Mountain Path Academy — World of Finance

**Prof. V. Ravichandran** | [themountainpathacademy.com](https://themountainpathacademy.com)

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
logistic_regression_app/
├── app.py                    # Entry point — page routing
├── styles.py                 # Mountain Path CSS design system
├── components.py             # Reusable UI components
├── model_engine.py           # LogReg engine + all caselet data
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Theme (dark, gold, serif)
└── pages/
    ├── __init__.py
    ├── home.py               # Landing / overview page
    ├── theory.py             # Foundations: sigmoid, MLE, metrics
    ├── caselet_renderer.py   # Generic renderer (used by all 5)
    ├── caselet1.py           # C1: IndiaBank Loan Default
    ├── caselet2.py           # C2: PaySecure Fraud Detection
    ├── caselet3.py           # C3: SavannaBank Churn
    ├── caselet4.py           # C4: LendRight SME NPA
    ├── caselet5.py           # C5: BullBear IPO Subscription
    └── quiz.py               # 15-question self-assessment
```

---

## 🎓 Features

### Per Caselet (6 interactive tabs each):
| Tab | Content |
|-----|---------|
| 📊 Dataset & Model | Training data, estimated coefficients, odds ratios, model fit stats |
| 🎯 Score New Observation | Live slider scoring + probability gauge + variable contribution chart |
| 📉 Threshold & Errors | Confusion matrix, Type I/II costs, full metric sweep charts |
| 📈 ROC Curve | ROC plot with AUC, KS annotation, Gini, ROC table |
| 💰 Cost Calculator | Custom C_FP/C_FN inputs, optimal threshold finder, full cost table |
| 🧠 Hypothesis Testing | H₀/H₁ setup, Type I/II error explanation, cost ratio display |

### Other Pages:
- **Theory**: Sigmoid function, MLE, odds ratios, performance metrics, Type I/II comparison table
- **Quiz**: 15 finance-focused questions with instant feedback and explanations

---

## 🎨 Design System (Mountain Path Academy)

| Token | Value |
|-------|-------|
| Dark Blue | `#003366` |
| Mid Blue | `#004d80` |
| Light Blue | `#ADD8E6` |
| Gold | `#FFD700` |
| Card BG | `#112240` |
| Background | `linear-gradient(135deg, #0a1628, #112240, #1a3355)` |
| Headline Font | Playfair Display (serif) |
| Body Font | Source Sans Pro |
| Code Font | JetBrains Mono |

---

## 📚 The Five Caselets

1. **IndiaBank** — Retail Loan Default Prediction (Credit Score, Income, DTI, Employment)
2. **PaySecure India** — Credit Card Fraud Detection (Amount, Night, Distance, Frequency, Merchant)
3. **SavannaBank** — Customer Churn Prediction (Tenure, Balance, Products, Complaints, Digital)
4. **LendRight NBFC** — SME NPA Scoring (Current Ratio, DSCR, Years, GST, CIBIL)
5. **BullBear Securities** — IPO Subscription Propensity (Past IPOs, Portfolio, Trades, Age, Risk)

---

## 👨‍🏫 About

**Prof. V. Ravichandran**
- 28+ Years Corporate Finance & Banking
- 10+ Years Academic Excellence
- Visiting Faculty: NMIMS Bangalore | BITS Pilani | RV University | Goa Institute of Management

🌐 [themountainpathacademy.com](https://themountainpathacademy.com)
🔗 [linkedin.com/in/trichyravis](https://linkedin.com/in/trichyravis)
💻 [github.com/trichyravis](https://github.com/trichyravis)
