import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ========== 1. Page Configuration ==========
st.set_page_config(
    page_title="MTN Customer Churn Predictor",
    page_icon="📱",
    layout="centered"
)

# ========== 2. Professional Styling ==========
st.markdown("""
<style>
    /* Main container styling */
    div[data-testid="stForm"] {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    /* Prediction results styling */
    .high-risk {
        color: #BE1E2D;  /* MTN red */
        font-weight: 500;
        border-left: 4px solid #BE1E2D;
        padding-left: 12px;
    }
    
    .low-risk {
        color: #00A651;  /* MTN green */
        font-weight: 500;
        border-left: 4px solid #00A651;
        padding-left: 12px;
    }
    
    /* Interactive elements */
    .stButton>button {
        border: 1px solid #FFD100 !important;  /* MTN yellow */
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #FFF8E0 !important;  /* Light yellow */
    }
    
    /* Consistent spacing */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ========== 3. Load Model Artefacts ==========
@st.cache_resource
def load_artefacts():
    return joblib.load("mtn_customer_churn_status_lr.pkl")

try:
    artefacts = load_artefacts()
    model = artefacts["model"]
    encoder = artefacts["encoder"]
    scaler = artefacts["scaler"]
    exp_cols = artefacts["columns"]
except Exception as e:
    st.error(f"Error loading model artefacts: {str(e)}")
    st.stop()

# ========== 4. User Interface ==========
st.title("MTN Customer Churn Prediction")
st.markdown("""
Welcome to the MTN Customer Churn Predictor! 👋  
Use the form below to enter a customer's details and get a prediction on whether they are likely to churn.
""")

# Initialize form reset state
if 'reset_form' not in st.session_state:
    st.session_state.reset_form = False

# Main form container
form_container = st.container()

with form_container.form("customer_form"):
    st.subheader("Customer Details")
    
    # Layout in two columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, 
                            help="Customer's age in years")
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        state = st.selectbox(
            "State",
            [
                "Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue","Borno",
                "Cross River","Delta","Edo","Ekiti","Enugu","Abuja (FCT)","Gombe","Imo",
                "Jigawa","Kaduna","Kano","Katsina","Kebbi","Kogi","Kwara","Lagos","Nasarawa",
                "Niger","Ondo","Osun","Oyo","Plateau","Rivers","Sokoto","Taraba","Yobe","Zamfara"
            ]
        )
        
        mtn_device = st.selectbox("MTN Device", 
                                ["Mobile SIM Card", "5G Broadband Router", "Broadband MiFi"])
    
    with col2:
        satisfaction_rate = st.slider("Satisfaction Rate (1-10)", 1, 10, value=7,
                                    help="Customer's self-reported satisfaction score")
        
        subscription_plan = st.selectbox(
            "Subscription Plan",
            [
                "165GB Monthly Plan","12.5GB Monthly Plan","150GB FUP Monthly Unlimited",
                "1GB+1.5mins Daily Plan","30GB Monthly Broadband Plan","10GB+10mins Monthly Plan",
                "25GB Monthly Plan","7GB Monthly Plan","1.5TB Yearly Broadband Plan",
                "65GB Monthly Plan","120GB Monthly Broadband Plan","300GB FUP Monthly Unlimited",
                "60GB Monthly Broadband Plan","500MB Daily Plan","3.2GB 2-Day Plan",
                "20GB Monthly Plan","2.5GB 2-Day Plan","450GB 3-Month Broadband Plan",
                "200GB Monthly Broadband Plan","1.5GB 2-Day Plan","16.5GB+10mins Monthly Plan"
            ]
        )
        
        unit_price = st.number_input("Unit Price (₦)", min_value=1, max_value=10000, value=1000)
    
    # Second row of inputs
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        number_of_times_purchased = st.number_input("Number of Purchases", min_value=0, max_value=100, value=5)
        total_revenue = st.number_input("Total Revenue (₦)", min_value=0, max_value=100000, value=5000)
    
    with col4:
        data_usage = st.number_input("Data Usage (GB)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        customer_tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=1200, value=12)
    
    submitted = st.form_submit_button("Predict Churn Risk")

# Form reset button
if st.button("Clear Form"):
    st.session_state.reset_form = True
    st.rerun()

if st.session_state.reset_form:
    st.session_state.reset_form = False
    form_container.empty()
    st.rerun()

# ========== 5. Prediction Logic ==========
if submitted:
    # Build input DataFrame
    raw_data = {
        "age": age,
        "gender": gender,
        "state": state,
        "mtn_device": mtn_device,
        "satisfaction_rate": satisfaction_rate,
        "subscription_plan": subscription_plan,
        "unit_price": unit_price,
        "number_of_times_purchased": number_of_times_purchased,
        "total_revenue": total_revenue,
        "data_usage": data_usage,
        "customer_tenure_in_months": customer_tenure_months
    }
    
    try:
        # Preprocessing pipeline
        raw_df = pd.DataFrame([raw_data])
        
        # Encode categoricals
        cat_cols = encoder.feature_names_in_.tolist()
        encoded = pd.DataFrame(
            encoder.transform(raw_df[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols)
        )
        
        # Scale numericals
        num_cols = scaler.feature_names_in_.tolist()
        scaled = pd.DataFrame(
            scaler.transform(raw_df[num_cols]),
            columns=num_cols
        )
        
        # Combine features
        processed = pd.concat([scaled, encoded], axis=1)
        missing = set(exp_cols) - set(processed.columns)
        for col in missing:
            processed[col] = 0
        processed = processed[exp_cols]
        
        # Make prediction
        prob = model.predict_proba(processed)[0, 1]
        pred = model.predict(processed)[0]
        
        # ========== 6. Display Results ==========
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Probability gauge
        st.write(f"**Churn Probability:** {prob:.1%}")
        st.progress(prob)
        
        # Risk classification
        if pred == 1:
            st.markdown('<p class="high-risk">High Risk: Customer likely to churn</p>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            **Recommended Actions:**
            - Priority retention outreach
            - Personalized service review
            - Targeted loyalty offer
            """)
        else:
            st.markdown('<p class="low-risk">Low Risk: Customer likely to stay</p>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            **Maintenance Suggestions:**
            - Regular satisfaction check-ins
            - Service quality monitoring
            - Upsell opportunities
            """)
        
        # Raw data review
        with st.expander("Review Submitted Data"):
            st.write(raw_df.T.rename(columns={0: "Value"}))
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")