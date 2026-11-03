import streamlit as st
import requests
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Vegetable Spoilage Predictor",
    page_icon="🥬",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# ---------------- CUSTOM CSS ----------------

st.markdown("""

<style>
.main {
    background-color: #f7faf7;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e0e8e0;
}

h1, h2, h3 {
    color: #245c35;
}
</style>

""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.title("🥬 Vegetable Spoilage Predictor")

st.write(
    "An IoT and Machine Learning based system "
    "for monitoring vegetable freshness."
)

st.divider()

# ---------------- BACKEND CONNECTION ----------------

def check_backend():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

backend_status = check_backend()

if backend_status:
    st.success("🟢 Backend connected successfully!")
else:
    st.error("🔴 Backend is not running. Start FastAPI first.")

# ---------------- PREDICTION SECTION ----------------

st.header("🔍 Check Vegetable Freshness")

vegetable = st.selectbox(
"Select Vegetable",
[
"Carrot",
"Cabbage",
"Capsicum",
"Cauliflower",
"Green_Chili",
"Brinjal",
"Tomato",
"Cucumber",
"French_Beans",
"Okra"
]
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("MQ135 Gas Sensor")

    mq_raw = st.number_input("MQ135 ADC Raw", value=115.0)
    mq_rolling = st.number_input("Rolling Average (12)", value=112.0)
    mq_short = st.number_input("Short Average (3)", value=117.0)
    r0 = st.number_input("R0 Baseline", value=100.0)
    rs_ratio = st.number_input("Rs/R0 Ratio", value=1.0)
    voc = st.number_input("VOC Index Estimate", value=0.25)

with col2:
    st.subheader("DHT22 Temperature & Humidity")

    temperature = st.number_input("Temperature (°C)", value=22.5)
    humidity = st.number_input("Humidity (%)", value=65.0)
    humidity_baseline = st.number_input(
        "Humidity Baseline",
        value=65.0
    )
    humidity_spike = st.number_input(
        "Humidity Local Spike",
        value=0.0
    )

st.subheader("Calculated Features")

gas_slope = st.number_input(
    "Gas Slope (ADC per hour)",
    value=0.3
)

spoilage_signal = st.number_input(
    "Spoilage Signal",
    value=0.25
)

# ---------------- PREDICTION ----------------

if st.button(
    "Predict Spoilage",
    type="primary",
    use_container_width=True
):
    sensor_data = {
        "Vegetable_Type": vegetable,
        "MQ135_ADC_Raw": mq_raw,
        "MQ135_ADC_RollingAvg12": mq_rolling,
        "MQ135_ADC_ShortAvg3": mq_short,
        "R0_Baseline": r0,
        "Rs_R0_Ratio": rs_ratio,
        "MQ135_VOC_Index_Est": voc,
        "DHT22_Temp_C": temperature,
        "DHT22_Humidity_Percent": humidity,
        "Humidity_Baseline_Prev12": humidity_baseline,
        "Humidity_Local_Spike": humidity_spike,
        "Gas_Slope_ADC_per_hr": gas_slope,
        "Spoilage_Signal": spoilage_signal
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=sensor_data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            stage = result["predicted_stage"]
            confidence = result["confidence"]

            stage_styles = {
                "Fresh": ("🟢", "success"),
                "Early_Risk": ("🟡", "warning"),
                "Spoiling": ("🟠", "warning"),
                "Spoiled": ("🔴", "error")
            }

            icon, style = stage_styles.get(stage, ("ℹ️", "info"))

            st.divider()
            st.subheader("Prediction Result")
            st.markdown(f"## {icon} {stage.replace('_', ' ')}")

            if style == "success":
                st.success("The model classified this vegetable as Fresh.")
            elif style == "warning":
                st.warning(f"Attention: The predicted stage is {stage.replace('_', ' ')}.")
            elif style == "error":
                st.error("The model classified this vegetable as Spoiled.")
            else:
                st.info("Prediction completed.")

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Vegetable", result["vegetable"])
            result_col2.metric("Model Confidence", f"{confidence}%")

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee of prediction accuracy."
            )
        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to backend: {error}")

# ---------------- PREDICTION HISTORY ----------------

st.divider()

st.header("📋 Prediction History")

if st.button("Refresh History"):
    try:
        response = requests.get(
            f"{API_URL}/history",
            timeout=10
        )

        if response.status_code == 200:
            history_data = response.json()
            records = history_data["history"]

            if records:
                df = pd.DataFrame(records)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No prediction history available.")
        else:
            st.error("Could not retrieve prediction history.")

    except requests.exceptions.RequestException:
        st.error("Backend connection failed.")

# ---------------- ANALYTICS ----------------

st.divider()

st.header("📊 Prediction Analytics")

try:
    response = requests.get(
        f"{API_URL}/history",
        timeout=10
    )

    if response.status_code == 200:
        history_data = response.json()
        records = history_data["history"]

        if records:
            df = pd.DataFrame(records)
            total = len(df)
            fresh = (df["predicted_stage"] == "Fresh").sum()
            early_risk = (df["predicted_stage"] == "Early_Risk").sum()
            spoiling = (df["predicted_stage"] == "Spoiling").sum()
            spoiled = (df["predicted_stage"] == "Spoiled").sum()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Predictions", total)
            c2.metric("Fresh", int(fresh))
            c3.metric("Early Risk", int(early_risk))
            c4.metric("Spoiling", int(spoiling))
            c5.metric("Spoiled", int(spoiled))

            st.subheader("Spoilage Stage Distribution")
            stage_counts = df["predicted_stage"].value_counts()
            st.bar_chart(stage_counts)

            st.subheader("Predictions by Vegetable")
            vegetable_counts = df["vegetable"].value_counts()
            st.bar_chart(vegetable_counts)
        else:
            st.info("Make some predictions to see analytics.")
    else:
        st.warning("Analytics unavailable. Check backend connection.")

except requests.exceptions.RequestException:
    st.warning("Analytics unavailable. Check backend connection.")

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Vegetable Spoilage Predictor | IoT + Machine Learning | Final Year Project"
)
