import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="MBG Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

# =====================================================
# HEADER
# =====================================================
st.title("🕵️ MBG Fraud Detection Dashboard")
st.markdown(
    """
    Dashboard ini membantu **mengidentifikasi transaksi yang menyimpang**
    berdasarkan **pola normal historis**, menggunakan **Autoencoder (unsupervised learning)**.
    """
)

# =====================================================
# LOAD MODEL (CACHE)
# =====================================================
@st.cache_resource
def load_model_cached():
    return load_model("autoencoder.h5", compile=False)

model = load_model_cached()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Data (CSV)",
    type=["csv"]
)

percentile = st.sidebar.slider(
    "🎯 Ambang Risiko (Percentile)",
    min_value=90,
    max_value=99,
    value=95,
    step=1,
    help="Semakin tinggi, semakin ketat deteksi anomali"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model akan menandai data dengan tingkat penyimpangan "
    "di atas ambang sebagai **anomali**."
)

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data berhasil diunggah")
else:
    st.info("ℹ️ Menggunakan data contoh (demo)")
    df = pd.read_csv("mbg_synthetic.csv")

# =====================================================
# FEATURE SELECTION
# =====================================================
features = [
    "qty_kirim",
    "qty_terima",
    "delay_jam",
    "kalori",
    "protein",
    "karbo"
]

missing = [f for f in features if f not in df.columns]
if missing:
    st.error(f"❌ Kolom berikut tidak ditemukan: {missing}")
    st.stop()

X = df[features]

# =====================================================
# PREPROCESS
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# AUTOENCODER PREDICTION
# =====================================================
X_pred = model.predict(X_scaled, verbose=0)
reconstruction_error = np.mean(np.square(X_scaled - X_pred), axis=1)

df["risk_score"] = reconstruction_error

# =====================================================
# THRESHOLD
# =====================================================
threshold = np.percentile(reconstruction_error, percentile)
df["risk_level"] = np.where(
    df["risk_score"] > threshold,
    "Anomali",
    "Normal"
)

# =====================================================
# KPI SECTION
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric(
    "📊 Total Data",
    len(df)
)

col2.metric(
    "🚨 Anomali Terdeteksi",
    int((df["risk_level"] == "Anomali").sum())
)

col3.metric(
    "🎯 Ambang Risiko",
    f"{threshold:.4f}"
)

# =====================================================
# CHART
# =====================================================
st.subheader("📈 Pola Penyimpangan Transaksi")

chart_df = df[["risk_score"]].copy()
chart_df["threshold"] = threshold

st.line_chart(chart_df)

st.caption(
    "Setiap titik mewakili satu transaksi. "
    "Transaksi di atas ambang dianggap menyimpang dari pola normal."
)

# =====================================================
# ANOMALY TABLE
# =====================================================
st.subheader("🔍 Daftar Transaksi Anomali")

anom_df = df[df["risk_level"] == "Anomali"]

if len(anom_df) == 0:
    st.success("🎉 Tidak ada anomali terdeteksi")
else:
    st.dataframe(
        anom_df.sort_values("risk_score", ascending=False),
        use_container_width=True
    )

# =====================================================
# EXPLANATION
# =====================================================
st.markdown("---")
st.subheader("🧠 Cara Membaca Dashboard Ini")

st.markdown(
    """
    **1️⃣ Risk Score**  
    Menunjukkan seberapa jauh suatu transaksi menyimpang dari pola normal historis.

    **2️⃣ Anomali ≠ Fraud**  
    Anomali adalah **peringatan awal**, bukan keputusan final.

    **3️⃣ Threshold**  
    Ambang risiko dapat disesuaikan sesuai kebijakan audit.

    **4️⃣ Validasi Domain**  
    Hasil akhir **harus dikonfirmasi oleh auditor / tim operasional**.
    """
)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "MBG Fraud Detection • Autoencoder-based Anomaly Detection • Streamlit Dashboard"
)
