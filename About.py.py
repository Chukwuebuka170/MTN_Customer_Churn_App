import streamlit as st

st.title("About the MTN Churn Predictor App")

st.write("""
This app uses a machine learning model trained on customer data to predict the likelihood of churn.

### Features Used
- Age
- Gender
- State
- MTN Device Type
- Satisfaction Rate
- Subscription Plan
- Unit Price
- Number of Purchases
- Total Revenue
- Data Usage
- Date of Purchase

The model was trained using `scikit-learn` and the app was built using `Streamlit`.
""")
