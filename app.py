import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="MBG Risk Monitoring",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model_cached():
    return load_model("autoencoder.h5", compile=False)

model = load_model_cached()

# ==============================
# UTILITIES
# ==============================
def preprocess(df, feature_cols):
    scaler = StandardScaler()
    return scaler.fit_transform(df[feature_cols])

def reconstruction_error(x, x_pred):
    return np.mean(np.square(x - x_pred), axis=1)

# ==============================
# HEADER
# ==============================
st.title("🍱 MBG Risk Monitoring Dashboard")
st.caption("Early Warning System untuk Deteksi Penyimpangan Distribusi & Gizi")

st.info(
    "Dashboard ini membantu mengidentifikasi **pola distribusi dan gizi yang tidak biasa** "
    "berdasarkan data operasional MBG. "
    "Hasil bersifat **indikatif** dan digunakan untuk mendukung audit berbasis risiko, "
    "bukan sebagai keputusan final."
)

# ==============================
# CONTROL PANEL
# ==============================
st.subheader("⚙️ Pengaturan Analisis")

col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Mode Analisis",
        ["Demo (Data Contoh)", "Upload Data Baru"]
    )

with col2:
    percentile = st.slider(
        "Ambang Risiko (Percentile)",
        min_value=90,
        max_value=99,
        value=95,
        help="Semakin tinggi ambang, semakin sedikit transaksi yang ditandai."
    )

# ==============================
# LOAD DATA
# ==============================
if mode == "Upload Data Baru":
    uploaded_file = st.file_uploader(
        "Upload data distribusi MBG (CSV)",
        type=["csv"]
    )
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mbg_synthetic.csv")

# ==============================
# FEATURE SELECTION
# ==============================
feature_cols = [
    "qty_kirim",
    "qty_terima",
    "delay_jam",
    "kalori",
    "protein",
    "karbo"
]

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
    st.stop()

# ==============================
# INFERENCE
# ==============================
X = preprocess(df, feature_cols)
X_pred = model.predict(X, verbose=0)

df["risk_score"] = reconstruction_error(X, X_pred)

threshold = np.percentile(df["risk_score"], percentile)
df["risk_level"] = np.where(
    df["risk_score"] >= threshold,
    "🔴 Perlu Ditinjau",
    "🟢 Normal"
)

# ==============================
# SUMMARY
# ==============================
st.subheader("📊 Ringkasan")

c1, c2, c3 = st.columns(3)

c1.metric("Total Transaksi", len(df))
c2.metric(
    "Perlu Ditinjau",
    int((df["risk_level"] == "🔴 Perlu Ditinjau").sum())
)

status_text = "Mayoritas data dalam batas wajar"
if (df["risk_level"] == "🔴 Perlu Ditinjau").sum() > 0:
    status_text = "Terdapat transaksi yang memerlukan perhatian"

c3.metric("Status Umum", status_text)

# ==============================
# VISUALIZATION
# ==============================
st.subheader("📈 Distribusi Tingkat Penyimpangan")

fig, ax = plt.subplots()
ax.plot(df["risk_score"].values, label="Tingkat Penyimpangan")
ax.axhline(threshold, color="red", linestyle="--", label="Ambang Risiko")
ax.set_ylabel("Skor Penyimpangan")
ax.set_xlabel("Indeks Transaksi")
ax.legend()

st.pyplot(fig)

# ==============================
# ANOMALY TABLE
# ==============================
st.subheader("🔍 Transaksi yang Perlu Ditinjau")

risk_df = df[df["risk_level"] == "🔴 Perlu Ditinjau"].copy()

if risk_df.empty:
    st.success("Tidak ada transaksi yang perlu ditinjau pada ambang risiko ini.")
else:
    st.write(
        "Daftar berikut menunjukkan transaksi dengan pola yang "
        "**berbeda dari kebiasaan normal** dan disarankan untuk ditinjau lebih lanjut."
    )
    st.dataframe(
        risk_df[
            feature_cols + ["risk_score", "risk_level"]
        ].sort_values("risk_score", ascending=False),
        use_container_width=True
    )

# ==============================
# DOWNLOAD
# ==============================
st.subheader("⬇️ Unduh Hasil")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download hasil analisis (CSV)",
    csv,
    "hasil_analisis_mbg.csv",
    "text/csv"
)
