import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="MBG Fraud Detection", layout="wide")

st.title("🍽️ MBG Supply Chain Fraud Detection")
st.caption("Machine Learning-based Anomaly Detection Prototype")

@st.cache_resource
def load_model_cached():
    return load_model("model/autoencoder.h5")

@st.cache_data
def load_data():
    return pd.read_csv("data/mbg_synthetic.csv")

data = load_data()
model = load_model_cached()

features = [
    "qty_kirim", "qty_terima", "delay_jam",
    "kalori", "protein", "karbo"
]

scaler = MinMaxScaler()
X = scaler.fit_transform(data[features])

recon = model.predict(X)
mse = np.mean(np.power(X - recon, 2), axis=1)
data["anomaly_score"] = mse

st.sidebar.header("⚙️ Settings")
percentile = st.sidebar.slider("Anomaly Threshold (%)", 90, 99, 95)
threshold = np.percentile(mse, percentile)

data["anomaly"] = data["anomaly_score"] > threshold

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Data Sample")
    st.dataframe(data.head())

with col2:
    st.subheader("📈 Anomaly Score")
    st.line_chart(data["anomaly_score"])

st.subheader("🚨 Detected Anomalies")
st.dataframe(data[data["anomaly"]])

st.markdown("""
### 📘 About This Prototype
This system uses an Autoencoder neural network to detect anomalies
in the MBG food supply chain as early indicators of potential fraud.
Framework: People – Process – Technology (PPT).
""")
