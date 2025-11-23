# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

MODEL_FILE = "soil_model.joblib"

st.set_page_config(page_title="Smart Soil Analyzer", layout="centered")
st.title("🌱 Smart Soil Analyzer — Soil Type & Fertility Predictor")

st.markdown("""
Enter simple sensor readings (pH, moisture %, nitrogen) and click **Predict**.
The app predicts:
- **Soil Type**: Clay / Sandy / Loamy  
- **Fertility**: High / Medium / Low (based on simple rules)
""")

# Left column: inputs
with st.form("predict_form"):
    ph = st.number_input("pH (0 - 14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
    nitrogen = st.number_input("Nitrogen (mg/kg or unit)", min_value=0.0, max_value=1000.0, value=40.0, step=1.0)
    submitted = st.form_submit_button("Predict")

# Load model
if not os.path.exists(MODEL_FILE):
    st.warning("Model file not found. Train the model first by running backend/train_model.py")
else:
    model = joblib.load(MODEL_FILE)

    def fertility_rule(ph, nitrogen):
        """Very simple rule-based fertility:
           - High: pH in 6.5-7.5 AND nitrogen >= 50
           - Medium: pH in 5.5-8.0 AND nitrogen >= 30
           - Low: otherwise
        Adjust thresholds later with real data.
        """
        if 6.5 <= ph <= 7.5 and nitrogen >= 50:
            return "High"
        if 5.5 <= ph <= 8.0 and nitrogen >= 30:
            return "Medium"
        return "Low"

    if submitted:
        x = np.array([[ph, moisture, nitrogen]])
        pred = model.predict(x)[0]
        probs = model.predict_proba(x)[0]
        classes = model.classes_
        prob_map = {c: float(p) for c, p in zip(classes, probs)}
        fertility = fertility_rule(ph, nitrogen)

        st.success(f"**Predicted Soil Type:** {pred}")
        st.write("Probabilities:", prob_map)
        st.info(f"**Fertility Level:** {fertility}")

        # Suggest simple crop types (very basic examples)
        suggestions = {
            "Clay": ["Rice", "Sugarcane"],
            "Sandy": ["Millet", "Groundnut"],
            "Loamy": ["Wheat", "Maize", "Vegetables"]
        }
        st.write("Suggested crops (basic):", ", ".join(suggestions.get(pred, [])))

# Show sample data
if st.checkbox("Show sample training data"):
    df = pd.read_csv(os.path.join("data", "soil_data_small.csv"))
    st.dataframe(df)
s