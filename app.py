import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import io

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="MBG Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

# =====================================================
# TEMPLATE VALIDATION
# =====================================================
REQUIRED_COLUMNS = [
    "qty_kirim",
    "qty_terima",
    "delay_jam",
    "kalori",
    "protein",
    "karbo"
]

COLUMN_TYPES = {
    "qty_kirim": "numeric",
    "qty_terima": "numeric",
    "delay_jam": "numeric",
    "kalori": "numeric",
    "protein": "numeric",
    "karbo": "numeric"
}

def generate_template():
    """Generate template CSV untuk download"""
    template_data = {
        "qty_kirim": [100, 150, 200],
        "qty_terima": [98, 150, 195],
        "delay_jam": [2, 0, 5],
        "kalori": [2500, 3000, 2800],
        "protein": [80, 90, 85],
        "karbo": [300, 350, 320]
    }
    return pd.DataFrame(template_data)

def validate_dataframe(df):
    """
    Validasi struktur dan tipe data DataFrame
    Returns: (is_valid, error_messages)
    """
    errors = []
    
    # 1. Check missing columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"❌ Kolom wajib tidak ditemukan: {', '.join(missing_cols)}")
    
    # 2. Check empty dataframe
    if df.empty:
        errors.append("❌ File CSV kosong (tidak ada data)")
    
    # 3. Check data types
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"❌ Kolom '{col}' harus berisi angka")
    
    # 4. Check for missing values
    if df[REQUIRED_COLUMNS].isnull().any().any():
        null_cols = df[REQUIRED_COLUMNS].columns[df[REQUIRED_COLUMNS].isnull().any()].tolist()
        errors.append(f"⚠️ Ditemukan data kosong di kolom: {', '.join(null_cols)}")
    
    # 5. Check for negative values
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            if (df[col] < 0).any():
                errors.append(f"⚠️ Kolom '{col}' memiliki nilai negatif")
    
    # 6. Check reasonable data ranges
    if "delay_jam" in df.columns:
        if (df["delay_jam"] > 240).any():  # max 10 hari
            errors.append("⚠️ Delay lebih dari 240 jam (10 hari) terdeteksi")
    
    # 7. Business rule: qty_terima tidak boleh lebih besar dari qty_kirim
    if "qty_kirim" in df.columns and "qty_terima" in df.columns:
        invalid_qty = (df["qty_terima"] > df["qty_kirim"]).sum()
        if invalid_qty > 0:
            errors.append(f"⚠️ Ditemukan {invalid_qty} transaksi dengan qty_terima > qty_kirim")
    
    # 8. Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        errors.append(f"⚠️ Ditemukan {duplicates} baris duplikat")
    
    # 9. Check minimum data requirement
    if len(df) < 10:
        errors.append("❌ Data terlalu sedikit (minimal 10 transaksi untuk analisis)")
    
    return len(errors) == 0, errors

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
# SIDEBAR - TEMPLATE DOWNLOAD
# =====================================================
st.sidebar.header("📋 Template Data")

template_df = generate_template()
csv_buffer = io.StringIO()
template_df.to_csv(csv_buffer, index=False)
csv_string = csv_buffer.getvalue()

st.sidebar.download_button(
    label="⬇️ Download Template CSV",
    data=csv_string,
    file_name="mbg_template.csv",
    mime="text/csv",
    help="Download template untuk format data yang benar"
)

st.sidebar.markdown(
    """
    **Format Template:**
    - `qty_kirim` (angka)
    - `qty_terima` (angka)
    - `delay_jam` (angka)
    - `kalori` (angka)
    - `protein` (angka)
    - `karbo` (angka)
    """
)

st.sidebar.markdown("---")

# =====================================================
# SIDEBAR - UPLOAD
# =====================================================
st.sidebar.header("⚙️ Pengaturan Analisis")

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Data (CSV)",
    type=["csv"],
    help="Upload file CSV sesuai template"
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
# LOAD & VALIDATE DATA
# =====================================================
df = None

if uploaded_file:
    try:
        # Read CSV with size limit (10MB)
        file_size = uploaded_file.size
        if file_size > 10 * 1024 * 1024:  # 10 MB
            st.error("❌ File terlalu besar! Maksimal 10 MB")
            st.stop()
        
        # Load dataframe
        df = pd.read_csv(uploaded_file)
        
        # Validate
        is_valid, errors = validate_dataframe(df)
        
        if not is_valid:
            st.error("### ⛔ Validasi Data Gagal")
            for error in errors:
                st.warning(error)
            
            st.info("💡 **Solusi:** Download template yang benar dari sidebar")
            st.stop()
        else:
            st.success(f"✅ Data berhasil diunggah ({len(df)} baris)")
            
            # Show warning if any
            if any("⚠️" in e for e in errors):
                with st.expander("⚠️ Peringatan Data"):
                    for error in errors:
                        if "⚠️" in error:
                            st.warning(error)
    
    except pd.errors.EmptyDataError:
        st.error("❌ File CSV kosong atau tidak valid")
        st.stop()
    except pd.errors.ParserError:
        st.error("❌ Format CSV tidak valid. Pastikan menggunakan delimiter koma (,)")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error membaca file: {str(e)}")
        st.stop()
else:
    st.info("ℹ️ Menggunakan data contoh (demo)")
    df = pd.read_csv("mbg_synthetic.csv")

# =====================================================
# FEATURE SELECTION
# =====================================================
features = REQUIRED_COLUMNS

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
