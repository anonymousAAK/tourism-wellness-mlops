"""
Streamlit front-end for the "Visit with Us" Wellness Tourism Package predictor.

Loads the best model that the GitHub Actions pipeline committed to this folder,
collects a customer's details, stores them in a one-row DataFrame with the same
columns the model was trained on, and returns the purchase probability.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits "
             "best_tourism_model_v1.joblib to tourism_project/deployment/.")
    st.stop()

model = load_model()
metrics = load_metrics()
threshold = metrics.get("threshold", 0.5)

st.title("🧳 Wellness Tourism Package – Purchase Predictor")
st.write("Enter a customer's profile and pitch details to estimate how likely they are "
         "to buy the **Wellness Tourism Package** before the sales team contacts them.")

# ------------------------------------------------------------------ inputs
st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=36)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=1000, max_value=100000,
                                     value=22000, step=500)
with c2:
    city_tier = st.selectbox("City tier", [1, 2, 3])
    num_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Children (below 5) visiting", min_value=0, max_value=5, value=1)
    num_trips = st.number_input("Trips per year", min_value=0, max_value=30, value=3)
    preferred_star = st.selectbox("Preferred property star", [3, 4, 5])
    passport = st.radio("Has a valid passport?", ["No", "Yes"], horizontal=True)
    own_car = st.radio("Owns a car?", ["No", "Yes"], horizontal=True)

st.subheader("Interaction details")
c3, c4 = st.columns(2)
with c3:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    pitch_satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)
with c4:
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=150, value=14)
    num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)

# ------------------------------------------------------------------ dataframe
# Same column names and order as the training data (Xtrain.csv)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Model input (one-row DataFrame)"):
    st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str))

# ------------------------------------------------------------------ predict
if st.button("Predict purchase likelihood", type="primary", use_container_width=True):
    proba = float(model.predict_proba(input_df)[0, 1])
    will_buy = proba >= threshold
    st.metric("Purchase probability", f"{proba:.1%}")
    st.progress(proba)
    if will_buy:
        st.success("Likely to **purchase** the Wellness Tourism Package – prioritise this customer.")
    else:
        st.warning("Unlikely to purchase – lower priority for this campaign.")

if metrics:
    with st.sidebar:
        st.header("Model card")
        st.write("Tuned XGBoost pipeline, retrained by GitHub Actions on every push.")
        st.write(f"Decision threshold: **{threshold}**")
        st.write("Hold-out test performance:")
        st.table(pd.DataFrame({
            "metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "value": [metrics.get(f"test_{m}") for m in
                      ["accuracy", "precision", "recall", "f1", "roc_auc"]],
        }).set_index("metric"))
