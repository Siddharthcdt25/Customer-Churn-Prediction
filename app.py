import streamlit as st
import pandas as pd
import joblib

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retention Reel — Churn Predictor",
    page_icon="🎬",
    layout="wide",
)

# ── Load artifacts (cached so this only runs once per session) ────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/customer_churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    # Note: scaler.pkl is intentionally NOT loaded — the deployed model
    # (Gradient Boosting) was trained on unscaled encoded features.
    # Only load the scaler here if you later deploy KNN or SVM instead.
    return model, feature_names

try:
    model, feature_names = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Make sure `customer_churn_model.pkl` and "
        "`feature_names.pkl` are committed to a `models/` folder next to app.py."
    )
    st.stop()

# ── Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0B0E14;
    --bg-alt: #11151D;
    --card: #161B24;
    --card-border: #262C38;
    --text-primary: #F5F3EC;
    --text-secondary: #9AA0AC;
    --gold: #F2B807;
    --gold-dim: #C89A0B;
    --risk-low: #2FBF8F;
    --risk-med: #F2B807;
    --risk-high: #E4483B;
}

.stApp {
    background: radial-gradient(ellipse 120% 60% at 50% -10%, #1A2030 0%, var(--bg) 55%);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 64px;
    letter-spacing: 2px;
    line-height: 1;
    margin: 0 0 6px 0;
    color: var(--text-primary);
}
.hero-tagline {
    font-size: 15px;
    color: var(--text-secondary);
    font-style: italic;
    margin-bottom: 28px;
}

div[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 28px 32px 12px 32px;
}
.card-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 24px;
    letter-spacing: 1px;
    color: var(--gold);
    margin-bottom: 2px;
}

label, .stSelectbox label, .stNumberInput label {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

.stButton > button, .stFormSubmitButton > button {
    background: var(--gold);
    color: #14110A;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: var(--gold-dim);
    box-shadow: 0 0 18px rgba(242, 184, 7, 0.35);
}

.verdict-card {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    border: 1px solid var(--card-border);
    background: var(--card);
}
.verdict-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    color: var(--text-secondary);
    text-transform: uppercase;
}
.verdict-score {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 96px;
    line-height: 1.1;
    margin: 4px 0;
}
.verdict-low .verdict-score { color: var(--risk-low); }
.verdict-medium .verdict-score { color: var(--risk-med); }
.verdict-high .verdict-score { color: var(--risk-high); }

.verdict-label {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 2px;
    font-size: 15px;
    text-transform: uppercase;
}
.verdict-low .verdict-label { color: var(--risk-low); }
.verdict-medium .verdict-label { color: var(--risk-med); }
.verdict-high .verdict-label { color: var(--risk-high); }

.verdict-tagline {
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 10px;
    font-size: 14px;
}

.meter-track {
    background: var(--bg-alt);
    border-radius: 6px;
    height: 10px;
    margin-top: 20px;
    overflow: hidden;
}
.meter-fill { height: 100%; border-radius: 6px; }
.meter-low { background: var(--risk-low); }
.meter-medium { background: var(--risk-med); }
.meter-high { background: var(--risk-high); }

.credit-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 10px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}
.credit-name {
    width: 190px;
    color: var(--text-secondary);
    flex-shrink: 0;
}
.credit-bar-track {
    flex: 1;
    background: var(--bg-alt);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.credit-bar-fill {
    height: 100%;
    background: var(--gold);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eyebrow">Now screening</div>
<div class="hero-title">RETENTION REEL</div>
<div class="hero-tagline">Predicting who stays subscribed — and who's about to hit cancel.</div>
""", unsafe_allow_html=True)

# ── Category options ──────────────────────────────────────────────────
# NOTE: verify these exactly match your dataset's unique values
# (df['region'].unique(), etc.) — case and spelling must match the
# columns baked into feature_names.pkl, or the app will silently fall
# back to treating that field as the dropped baseline category.
GENDERS = ["Female", "Male", "Other"]
SUBSCRIPTIONS = ["Basic", "Standard", "Premium"]
REGIONS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
DEVICES = ["Desktop", "Laptop", "Mobile", "TV", "Tablet"]
PAYMENT_METHODS = ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"]
GENRES = ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"]

# ── Input form ──────────────────────────────────────────────────────────
st.markdown('<div class="card-title">Subscriber profile</div>', unsafe_allow_html=True)

with st.form("churn_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=100, value=30)
        region = st.selectbox("Region", REGIONS)
        watch_hours = st.number_input("Total watch hours", min_value=0.0, value=120.0, step=1.0)
        number_of_profiles = st.number_input("Number of profiles", min_value=1, max_value=6, value=2)
    with c2:
        gender = st.selectbox("Gender", GENDERS)
        device = st.selectbox("Primary device", DEVICES)
        last_login_days = st.number_input("Days since last login", min_value=0, max_value=365, value=5)
        avg_watch_time_per_day = st.number_input("Avg watch time/day (hrs)", min_value=0.0, max_value=24.0, value=2.5, step=0.1)
    with c3:
        subscription_type = st.selectbox("Subscription tier", SUBSCRIPTIONS)
        payment_method = st.selectbox("Payment method", PAYMENT_METHODS)
        monthly_fee = st.number_input("Monthly fee ($)", min_value=0.0, value=15.99, step=0.01, format="%.2f")
        favorite_genre = st.selectbox("Favorite genre", GENRES)

    submitted = st.form_submit_button("▶  Predict churn risk")

# ── Build the exact feature vector the model was trained on ───────────
def build_input_row(feature_names):
    row = {col: 0 for col in feature_names}

    numeric_values = {
        "age": age,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "monthly_fee": monthly_fee,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
    }
    for col, val in numeric_values.items():
        if col in row:
            row[col] = val

    categorical_values = {
        "gender": gender,
        "subscription_type": subscription_type,
        "region": region,
        "device": device,
        "payment_method": payment_method,
        "favorite_genre": favorite_genre,
    }
    for prefix, selected in categorical_values.items():
        dummy_col = f"{prefix}_{selected}"
        if dummy_col in row:
            row[dummy_col] = 1
        # if not found, the selection matches the dropped baseline category
        # (e.g. "Female", "Basic", "Africa" etc.) — all-zero is correct for that case

    return pd.DataFrame([row])[feature_names]  # enforce exact training column order

# ── Prediction + verdict ───────────────────────────────────────────────
if submitted:
    input_df = build_input_row(feature_names)
    proba = model.predict_proba(input_df)[0][1]

    if proba >= 0.66:
        tier, label, tagline = "high", "High risk", "This subscriber is close to hitting cancel."
    elif proba >= 0.33:
        tier, label, tagline = "medium", "Watch list", "Engagement is wavering — worth a re-engagement nudge."
    else:
        tier, label, tagline = "low", "Low risk", "This subscriber looks locked in for another season."

    left, right = st.columns([1, 1.3])

    with left:
        st.markdown(f"""
        <div class="verdict-card verdict-{tier}">
            <div class="verdict-eyebrow">Risk assessment</div>
            <div class="verdict-score">{proba*100:.0f}%</div>
            <div class="verdict-label">{label}</div>
            <div class="meter-track">
                <div class="meter-fill meter-{tier}" style="width:{proba*100:.1f}%;"></div>
            </div>
            <div class="verdict-tagline">{tagline}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-title" style="font-size:18px;">Key churn drivers</div>', unsafe_allow_html=True)
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top5 = importances.sort_values(ascending=False).head(5)
        max_imp = top5.max()
        rows_html = ""
        for feat, imp in top5.items():
            pct = (imp / max_imp) * 100
            rows_html += f"""
            <div class="credit-row">
                <span class="credit-name">{feat.replace('_', ' ').title()}</span>
                <div class="credit-bar-track"><div class="credit-bar-fill" style="width:{pct:.0f}%;"></div></div>
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)
        st.caption("Model-wide importance, not specific to this one prediction.")