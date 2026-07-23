import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Netflix Customer Churn Prediction",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# LOAD MODEL AND FEATURE NAMES
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/customer_churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_artifacts()

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🎬 Netflix Customer Churn Prediction")
st.markdown(
    "Predict whether a customer is likely to churn using a Machine Learning model "
    "trained on customer behavior and subscription data."
)

st.divider()

# -----------------------------
# MODEL PERFORMANCE METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Model", "Gradient Boosting")

with col2:
    st.metric("Accuracy", "98.7%")

with col3:
    st.metric("ROC-AUC", "0.9978")

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("👤 Customer Information")

# Numerical Inputs
age = st.sidebar.slider("Age", 18, 70, 30)
watch_hours = st.sidebar.number_input("Total Watch Hours", min_value=0.0, value=50.0, step=1.0)
last_login_days = st.sidebar.slider("Last Login (Days Ago)", 0, 60, 5)
monthly_fee = st.sidebar.number_input("Monthly Fee ($)", min_value=0.0, value=13.99, step=0.01)
number_of_profiles = st.sidebar.slider("Number of Profiles", 1, 5, 2)
avg_watch_time_per_day = st.sidebar.slider(
    "Average Watch Time / Day (Hours)", 0.0, 24.0, 2.5, step=0.1
)

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
subscription_type = st.sidebar.selectbox(
    "Subscription Type", ["Basic", "Standard", "Premium"]
)
region = st.sidebar.selectbox(
    "Region",
    ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
)
device = st.sidebar.selectbox(
    "Preferred Device", ["Desktop", "Laptop", "Mobile", "TV", "Tablet"]
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Credit Card", "Crypto", "Debit Card", "Gift Card", "PayPal"]
)
favorite_genre = st.sidebar.selectbox(
    "Favorite Genre",
    ["Action", "Comedy", "Documentary", "Drama", "Horror", "Romance", "Sci-Fi"]
)

# Predict Button
predict = st.sidebar.button("🚀 Predict Churn")

# -----------------------------
# MAIN APP
# -----------------------------
if not predict:
    st.info("👉 Enter customer details from the sidebar and click **Predict Churn**.")
else:

    # -----------------------------
    # CREATE INPUT DATA
    # -----------------------------
    input_data = dict.fromkeys(feature_names, 0)

    # Numerical Features
    input_data["age"] = age
    input_data["watch_hours"] = watch_hours
    input_data["last_login_days"] = last_login_days
    input_data["monthly_fee"] = monthly_fee
    input_data["number_of_profiles"] = number_of_profiles
    input_data["avg_watch_time_per_day"] = avg_watch_time_per_day

    # Categorical Features (One-Hot Encoding)
    categorical_values = {
        "gender": gender,
        "subscription_type": subscription_type,
        "region": region,
        "device": device,
        "payment_method": payment_method,
        "favorite_genre": favorite_genre
    }

    for feature, value in categorical_values.items():
        column_name = f"{feature}_{value}"
        if column_name in input_data:
            input_data[column_name] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order
    input_df = input_df[feature_names]

    # -----------------------------
    # MAKE PREDICTION
    # -----------------------------
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # -----------------------------
    # PREDICTION RESULTS
    # -----------------------------
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("🔴 Customer is likely to CHURN")
        else:
            st.success("🟢 Customer is NOT likely to CHURN")

    with col2:
        st.metric(
            label="Churn Probability",
            value=f"{prediction_proba * 100:.2f}%"
        )

    # -----------------------------
    # RISK LEVEL
    # -----------------------------
    if prediction_proba < 0.30:
        st.success("🟢 Risk Level: LOW")
    elif prediction_proba < 0.70:
        st.warning("🟡 Risk Level: MEDIUM")
    else:
        st.error("🔴 Risk Level: HIGH")

    st.divider()

    # -----------------------------
    # CUSTOMER SUMMARY
    # -----------------------------
    st.subheader("👤 Customer Summary")

    summary = {
        "Age": age,
        "Gender": gender,
        "Subscription": subscription_type,
        "Region": region,
        "Device": device,
        "Monthly Fee": monthly_fee,
        "Watch Hours": watch_hours
    }

    st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

    st.divider()

    # -----------------------------
    # BUSINESS RECOMMENDATION
    # -----------------------------
    st.subheader("💡 Business Recommendation")

    if prediction == 1:
        st.warning("""
### Recommended Actions

- Offer a personalized discount.
- Send targeted email campaigns.
- Recommend trending content.
- Provide loyalty rewards.
- Engage the customer before cancellation.
        """)
    else:
        st.success("""
### Recommended Actions

Customer appears satisfied.

Continue engagement through:

- New content recommendations.
- Loyalty rewards.
- Premium plan offers.
        """)