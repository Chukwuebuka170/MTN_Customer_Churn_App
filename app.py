import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------- 1. Load artefacts ----------
artefacts = joblib.load("mtn_customer_churn_status_lr.pkl")
model    = artefacts["model"]
encoder  = artefacts["encoder"]
scaler   = artefacts["scaler"]
exp_cols = artefacts["columns"]  # training-time order

# ---------- 2. Streamlit UI ----------
st.set_page_config(page_title="MTN Customer Churn Predictor")
st.title("MTN Customer Churn Prediction")

with st.form("customer_form"):
    st.subheader("Customer Information")
    
    age = st.number_input("Age", 1, 120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    customer_review = st.selectbox("Customer Review", ["Fair", "Poor", "Good", "Excellent", "Very Good"])
    state = st.selectbox("State", ["Lagos", "Abuja", "Kano", "Rivers", "Enugu", "Kaduna", "Other"])
    mtn_device = st.selectbox("4G Router", ["Mobile SIM Card", "5G Broadband Router", "Broadband MiFi"])
    
    reasons_for_churn = st.selectbox("Reason for Churn", [
        "Relocation", "Better Offers from Competitors", "Poor Network",
        "Costly Data Plans", "Fast Data Consumption",
        "Poor Customer Service", "High Call Tarriffs"
    ])
    
    satisfaction_rate = st.slider("Satisfaction Rate", 1, 10, value=7)
    
    subscription_plan = st.selectbox("Subscription Plan", [
        "165GB Monthly Plan", "12.5GB Monthly Plan",
        "150GB FUP Monthly Unlimited", "1GB+1.5mins Daily Plan", "other"
    ])
    
    unit_price = st.number_input("Unit Price", 1, 10000)
    number_of_times_purchased = st.number_input("Number of Times Purchased", 0, 100)
    total_revenue = st.number_input("Total Revenue", 0, 100000)
    data_usage = st.number_input("Data Usage (in GB)", 0.0, 100.0, step=0.1)
    date_of_purchase = st.date_input("Date of Purchase")

    submitted = st.form_submit_button("Predict")

# ---------- 3. Pre-process & Predict ----------
if submitted:
    # Calculate tenure in days and months
    tenure_days = (pd.to_datetime("today") - pd.to_datetime(date_of_purchase)).days
    tenure_months = tenure_days // 30

    # 3-a. Build raw input DataFrame
    raw = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "state": state,
        "mtn_device": mtn_device,
        "satisfaction_rate": satisfaction_rate,
        "subscription_plan": subscription_plan,
        "customer_review": customer_review,
        "unit_price": unit_price,
        "number_of_times_purchased": number_of_times_purchased,
        "total_revenue": total_revenue,
        "data_usage": data_usage,
        "tenure_days": tenure_days,
        "customer_tenure_in_months": tenure_months,
        "reasons_for_churn": reasons_for_churn
    }])

    # 3-b. Encode categoricals
    cat_cols = encoder.feature_names_in_.tolist()
    encoded = pd.DataFrame(
        encoder.transform(raw[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # 3-c. Scale numerical columns
    num_cols = scaler.feature_names_in_.tolist()
    scaled = pd.DataFrame(
        scaler.transform(raw[num_cols]),
        columns=num_cols
    )

    # 3-d. Combine and align to training column structure
    processed = pd.concat([scaled, encoded], axis=1)
    missing = set(exp_cols) - set(processed.columns)
    for col in missing:
        processed[col] = 0
    processed = processed[exp_cols]

    # 3-e. Predict
    prob = model.predict_proba(processed)[0, 1]
    pred = model.predict(processed)[0]

    # ---------- 4. Output ----------
    st.markdown("---")
    st.subheader("📊 Prediction")
    st.write(f"**Probability of churn:** `{prob:.1%}`")
    st.write(f"**Predicted churn status:** {'Yes' if pred == 1 else 'No'}`")
    
    # ---------- 5. Reset / Rerun ----------
if st.button("Enter New Info"):
    st.rerun()
