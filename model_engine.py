import numpy as np
from scipy.optimize import minimize

GOLD = "#FFD700"
GREEN = "#28a745"
RED   = "#dc3545"
LIGHT_BLUE = "#ADD8E6"


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    eps = 1e-12
    return np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def neg_ll(beta, X, y):
    return -log_likelihood(beta, X, y)


def fit_logistic(X_raw, y, add_intercept=True):
    if add_intercept:
        X = np.column_stack([np.ones(len(X_raw)), X_raw])
    else:
        X = X_raw
    n_params = X.shape[1]
    beta0 = np.zeros(n_params)
    res = minimize(neg_ll, beta0, args=(X, y),
                   method='L-BFGS-B',
                   options={'maxiter': 2000, 'ftol': 1e-12})
    beta = res.x
    ll = log_likelihood(beta, X, y)
    # Null model LL
    p_null = np.mean(y)
    eps = 1e-12
    ll_null = np.sum(y * np.log(p_null + eps) + (1 - y) * np.log(1 - p_null + eps))
    mcfadden_r2 = 1 - ll / ll_null if ll_null != 0 else 0
    return beta, ll, ll_null, mcfadden_r2


def predict_proba(beta, X_raw, add_intercept=True):
    if add_intercept:
        X = np.column_stack([np.ones(len(X_raw)), X_raw])
    else:
        X = X_raw
    return sigmoid(X @ beta)


def classify(proba, threshold=0.5):
    return (proba >= threshold).astype(int)


def confusion(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn, tn


def metrics(tp, fp, fn, tn):
    n = tp + tn + fp + fn
    acc  = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fpr  = fp / (fp + tn) if (fp + tn) else 0
    return dict(accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1, fpr=fpr)


def roc_data(y_true, proba):
    thresholds = np.sort(np.unique(proba))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp, fp, fn, tn = confusion(y_true, pred)
        tprs.append(tp / pos if pos else 0)
        fprs.append(fp / neg if neg else 0)
    tprs.append(1.0); fprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def auc_score(y_true, proba):
    fprs, tprs = roc_data(y_true, proba)
    return float(np.trapezoid(tprs, fprs))


def ks_stat(y_true, proba):
    thresholds = np.sort(np.unique(proba))[::-1]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    max_ks = 0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp, fp, fn, tn = confusion(y_true, pred)
        tpr = tp / pos if pos else 0
        fpr = fp / neg if neg else 0
        max_ks = max(max_ks, abs(tpr - fpr))
    return max_ks


def threshold_sweep(y_true, proba, costs_fp=1, costs_fn=1):
    rows = []
    thresholds = np.round(np.arange(0.1, 1.0, 0.05), 2)
    pos = max(np.sum(y_true == 1), 1)
    neg = max(np.sum(y_true == 0), 1)
    for t in thresholds:
        pred = classify(proba, t)
        tp, fp, fn, tn = confusion(y_true, pred)
        m = metrics(tp, fp, fn, tn)
        total_cost = fp * costs_fp + fn * costs_fn
        rows.append(dict(
            threshold=t, tp=tp, fp=fp, fn=fn, tn=tn,
            accuracy=round(m['accuracy']*100,1),
            precision=round(m['precision']*100,1),
            recall=round(m['recall']*100,1),
            f1=round(m['f1']*100,1),
            type1_cost=fp*costs_fp,
            type2_cost=fn*costs_fn,
            total_cost=total_cost
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CASELET DATA DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CASELETS = {
    "caselet1": {
        "title": "Retail Loan Default Prediction",
        "subtitle": "IndiaBank Personal Loan Portfolio",
        "outcome_label": "Default",
        "institution": "IndiaBank",
        "h0": "The applicant will NOT Default",
        "h1": "The applicant WILL Default",
        "cost_fp": 37800,
        "cost_fn": 99000,
        "features": ["Credit Score", "Monthly Income (₹K)", "DTI (%)", "Salaried (1=Yes)"],
        "feature_defaults": [650, 45, 38, 1],
        "feature_mins":  [300, 10, 5, 0],
        "feature_maxs":  [900, 150, 80, 1],
        "feature_steps": [10, 5, 1, 1],
        "X": np.array([
            [720, 65, 28, 1],
            [580, 32, 52, 0],
            [650, 48, 38, 1],
            [510, 25, 61, 0],
            [780, 85, 22, 1],
            [620, 40, 45, 0],
            [700, 58, 31, 1],
            [540, 28, 58, 0],
            [760, 72, 24, 1],
            [590, 35, 49, 1],
        ], dtype=float),
        "y": np.array([0,1,0,1,0,1,0,1,0,1]),
        "var_signs": ["−","−","+","−"],
        "or_interpretation": [
            "Higher score = lower default risk",
            "Higher income = lower default risk",
            "Higher DTI = higher default risk",
            "Salaried = 3.1× lower default odds vs self-employed",
        ],
        "business_insight": "DTI is the most dangerous variable — a 20pp increase multiplies default odds by 6×.",
        "color": "#FFD700",
        "badge_color": "gold",
    },
    "caselet2": {
        "title": "Credit Card Fraud Detection",
        "subtitle": "PaySecure India — Transaction Monitoring",
        "outcome_label": "Fraud",
        "institution": "PaySecure India",
        "h0": "This transaction is GENUINE",
        "h1": "This transaction is FRAUDULENT",
        "cost_fp": 1500,
        "cost_fn": 47000,
        "features": ["Amount (₹K)", "Late Night (1=Yes)", "Geo Distance (km)", "Txns/Hour", "Electronics Merchant"],
        "feature_defaults": [10, 0, 20, 2, 0],
        "feature_mins":  [0.5, 0, 0, 1, 0],
        "feature_maxs":  [100, 1, 500, 10, 1],
        "feature_steps": [0.5, 1, 10, 1, 1],
        "X": np.array([
            [2.5, 0, 3, 1, 0],
            [48.0,1,320,5, 1],
            [5.2, 0, 8, 2, 0],
            [35.0,1,185,4, 1],
            [1.8, 0, 1, 1, 0],
            [52.5,1,410,7, 1],
            [8.0, 0,12, 2, 0],
            [29.5,1,145,3, 1],
            [4.3, 0, 5, 1, 0],
            [3.8, 0,22, 3, 1],
        ], dtype=float),
        "y": np.array([0,1,0,1,0,1,0,1,0,0]),
        "var_signs": ["+","+","+","+","+"],
        "or_interpretation": [
            "Each ₹1K more → 4.2% higher fraud odds",
            "Night transactions = 9.16× more likely fraud",
            "100km away = 4.4× higher fraud odds",
            "Each extra txn/hr adds 68% to odds",
            "Electronics/Jewellery = 6.5× more fraud",
        ],
        "business_insight": "Late-night + electronics store is the deadliest combination — 99.5% fraud probability in extreme cases.",
        "color": "#dc3545",
        "badge_color": "red",
    },
    "caselet3": {
        "title": "Bank Customer Churn Prediction",
        "subtitle": "SavannaBank Retail Banking Division",
        "outcome_label": "Churn",
        "institution": "SavannaBank",
        "h0": "This customer will STAY",
        "h1": "This customer will CHURN",
        "cost_fp": 2400,
        "cost_fn": 53900,
        "features": ["Account Tenure (yrs)", "Avg Balance (₹K)", "Products Held", "Complaints (12mo)", "Digital Logins/mo"],
        "feature_defaults": [5, 40, 2, 1, 10],
        "feature_mins":  [0.5, 5, 1, 0, 0],
        "feature_maxs":  [20, 200, 5, 8, 40],
        "feature_steps": [0.5, 5, 1, 1, 1],
        "X": np.array([
            [1.5, 12, 1, 3, 2],
            [8.0, 85, 4, 0,18],
            [2.0, 18, 2, 2, 5],
            [12.5,125,5, 0,25],
            [0.8, 8,  1, 4, 1],
            [6.5, 62, 3, 1,14],
            [3.0, 28, 2, 2, 6],
            [9.2, 95, 4, 0,22],
            [1.2, 10, 1, 3, 3],
            [5.5, 50, 3, 0,12],
        ], dtype=float),
        "y": np.array([1,0,1,0,1,0,1,0,1,0]),
        "var_signs": ["−","−","−","+","−"],
        "or_interpretation": [
            "Each extra year → 26.8% lower churn odds",
            "₹10K more → 24.6% lower churn odds",
            "Each extra product → 57% lower churn odds ✨",
            "Each complaint → 2.67× higher churn odds",
            "10 more logins/mo → 89.3% lower churn odds",
        ],
        "business_insight": "Product cross-sell is the golden anchor — 4 products vs 1 product reduces churn odds by 92%.",
        "color": "#28a745",
        "badge_color": "green",
    },
    "caselet4": {
        "title": "SME Credit Risk Scoring",
        "subtitle": "NBFC LendRight — Small Business Lending",
        "outcome_label": "NPA",
        "institution": "LendRight NBFC",
        "h0": "This SME loan will PERFORM",
        "h1": "This SME loan will become NPA",
        "cost_fp": 725000,
        "cost_fn": 3780000,
        "features": ["Current Ratio", "DSCR", "Years in Business", "GST Score (0-100)", "Promoter CIBIL"],
        "feature_defaults": [1.5, 1.3, 6, 75, 700],
        "feature_mins":  [0.5, 0.5, 1, 30, 500],
        "feature_maxs":  [3.5, 3.0, 20, 100, 900],
        "feature_steps": [0.05, 0.05, 1, 1, 10],
        "X": np.array([
            [0.85,0.92,2, 55,610],
            [2.10,1.65,12,88,752],
            [1.10,1.05,4, 63,648],
            [1.85,1.42,8, 82,718],
            [0.75,0.88,1, 48,592],
            [2.35,1.78,15,92,775],
            [1.20,1.15,3, 68,662],
            [1.95,1.55,10,85,735],
            [0.90,0.95,2, 58,625],
            [1.65,1.38,7, 78,705],
        ], dtype=float),
        "y": np.array([1,0,1,0,1,0,1,0,1,0]),
        "var_signs": ["−","−","−","−","−"],
        "or_interpretation": [
            "CR + 1 → 90% lower NPA odds",
            "DSCR + 1 → 95.6% lower NPA odds 🏆",
            "Each extra year → 13.8% lower NPA odds",
            "10pt GST improvement → 47% lower odds",
            "100pt CIBIL improvement → 56% lower odds",
        ],
        "business_insight": "DSCR is king — a business that cannot service debt from cash flows (DSCR<1) has dramatically elevated NPA risk.",
        "color": "#ADD8E6",
        "badge_color": "blue",
    },
    "caselet5": {
        "title": "IPO Subscription Propensity",
        "subtitle": "BullBear Securities — Retail Investor Analytics",
        "outcome_label": "Subscribe",
        "institution": "BullBear Securities",
        "h0": "This investor will NOT Subscribe",
        "h1": "This investor WILL Subscribe to the IPO",
        "cost_fp": 150,
        "cost_fn": 1050,
        "features": ["Past IPOs Subscribed", "Portfolio (₹ Lakhs)", "Trades/Month", "Account Age (yrs)", "High Risk Profile"],
        "feature_defaults": [3, 8, 6, 3, 1],
        "feature_mins":  [0, 0.5, 0, 0.3, 0],
        "feature_maxs":  [15, 50, 25, 15, 1],
        "feature_steps": [1, 0.5, 1, 0.5, 1],
        "X": np.array([
            [5, 12.5,8,  3.0,1],
            [0, 1.8, 2,  0.5,0],
            [3, 8.2, 6,  2.0,1],
            [0, 2.5, 1,  0.8,0],
            [8, 25.0,15, 6.5,1],
            [1, 3.5, 3,  1.2,0],
            [4, 10.0,9,  4.0,1],
            [0, 0.8, 1,  0.3,0],
            [6, 18.5,12, 5.0,1],
            [2, 4.5, 4,  1.8,0],
        ], dtype=float),
        "y": np.array([1,0,1,0,1,0,1,0,1,0]),
        "var_signs": ["+","+","+","+","+"],
        "or_interpretation": [
            "Each past IPO → 79% higher subscription odds",
            "₹10L more → 214% higher subscription odds",
            "Each 5 trades/mo → 3.67× higher odds",
            "Each extra year → 37.8% more likely",
            "High-risk profile = 9.43× more likely to subscribe",
        ],
        "business_insight": "Risk appetite is the dominant driver — high-risk investors are 9.43× more likely to subscribe vs conservative investors.",
        "color": "#a78bfa",
        "badge_color": "blue",
    },
}
