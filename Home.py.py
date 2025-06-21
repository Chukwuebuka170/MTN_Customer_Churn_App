import streamlit as st

st.set_page_config(page_title="MTN Churn App", layout="centered")

st.title("📞 MTN Customer Churn Predictor")

st.markdown("""
Welcome to the MTN Churn Prediction App.

Use the **Prediction** page to check if a customer is likely to churn based on their data.

Visit the **About** page to learn more about how this model works.
""")

st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Mtn-logo.png", width=150)