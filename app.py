import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models
et_reg_rain = joblib.load("et_reg_rain.pkl")
et_reg_temp = joblib.load("et_reg_temp.pkl")
et_cls_rain = joblib.load("et_cls_rain.pkl")
et_cls_temp = joblib.load("et_cls_temp.pkl")

# Feature list
required_features = [
    "latitude", "longitude", "humidity", "wind_kph", "cloud", "pressure_mb", "uv_index", "feels_like_celsius",
    "air_quality_Carbon_Monoxide", "air_quality_Ozone", "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide", "air_quality_PM2.5", "air_quality_PM10"
]

# Classification label maps
rain_occurred_map = {0: "No Rain", 1: "Rain"}
temp_class_map = {0: "Cold", 1: "Moderate", 2: "Hot"}  # Update according to your classes

# Streamlit UI
st.set_page_config(page_title="🌤️ Weather Predictor", layout="centered")
st.title("🌤️ Weather Prediction Tool")
st.write("Enter the following feature values to get rainfall, temperature, and class predictions:")

with st.form("input_form"):
    inputs = {}
    for feature in required_features:
        inputs[feature] = st.number_input(feature, format="%.4f", value=0.0)
    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    # Prepare input DataFrame
    input_df = pd.DataFrame([inputs])

    # Predict regression outputs
    pred_log_rain = et_reg_rain.predict(input_df)
    pred_rain_mm = np.expm1(pred_log_rain)[0]

    pred_temp = et_reg_temp.predict(input_df)[0]

    # Predict classifications
    pred_cls_rain = et_cls_rain.predict(input_df)[0]
    pred_temp_cls = et_cls_temp.predict(input_df)[0]

    # Decode classification labels
    pred_cls_rain_label = rain_occurred_map.get(pred_cls_rain, "Unknown")
    pred_temp_cls_label = temp_class_map.get(pred_temp_cls, "Unknown")

    # Display predictions
    st.success("✅ Prediction Completed")
    st.write(f"🌧️ **Predicted Rainfall (mm):** {pred_rain_mm:.2f}")
    st.write(f"🌡️ **Predicted Temperature (°C):** {pred_temp:.2f}")
    st.write(f"☔ **Rain Occurrence:** {pred_cls_rain_label}")
    st.write(f"🔥 **Temperature Category:** {pred_temp_cls_label}")

    # Prepare output DataFrame for download
    output_df = input_df.copy()
    output_df["Predicted_Rainfall_mm"] = pred_rain_mm
    output_df["Predicted_Temperature_C"] = pred_temp
    output_df["Rain_Occurred_Class"] = pred_cls_rain_label
    output_df["Temp_Class"] = pred_temp_cls_label

    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction CSV", csv, "prediction.csv", "text/csv")



# Footer
st.markdown(
    """
    <hr style="margin-top: 3rem; margin-bottom: 1.5rem;">
    <p style="text-align: center; font-size: 1.15rem; color: #555;">
        Designed and Developed by <strong>Varshini J </strong>  &nbsp; | &nbsp;
        <a href="https://shorturl.at/72KdH" target="_blank" style="color:#1f77b4; text-decoration:none;">
            Linkedin
        </a>
    </p>
    """,
    unsafe_allow_html=True
)