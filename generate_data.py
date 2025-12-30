"""
Generate synthetic MBG transaction data with fraud labels
"""
import pandas as pd
import numpy as np

print("="*60)
print("GENERATING SYNTHETIC MBG DATA")
print("="*60)

np.random.seed(42)
n = 10000  # 10,000 transactions as per README

# Generate normal transactions
data = pd.DataFrame({
    "qty_kirim": np.random.normal(100, 15, n),
    "qty_terima": np.random.normal(98, 15, n),
    "delay_jam": np.abs(np.random.normal(2, 1.5, n)),
    "kalori": np.random.normal(650, 80, n),
    "protein": np.random.normal(20, 4, n),
    "karbo": np.random.normal(80, 12, n),
})

# Initialize fraud label
data["is_fraud"] = 0

# -----------------------------
# INJECT FRAUD PATTERNS
# -----------------------------
n_fraud = 500  # 5% fraud rate (realistic for fraud detection)
fraud_idx = np.random.choice(n, size=n_fraud, replace=False)

# Pattern 1: Significant quantity discrepancy
pattern1_idx = fraud_idx[:200]
data.loc[pattern1_idx, "qty_terima"] *= 0.5  # 50% loss
data.loc[pattern1_idx, "delay_jam"] += np.random.uniform(5, 10, len(pattern1_idx))

# Pattern 2: Nutritional anomaly
pattern2_idx = fraud_idx[200:350]
data.loc[pattern2_idx, "kalori"] *= 0.6  # Low calorie
data.loc[pattern2_idx, "protein"] *= 0.5  # Low protein

# Pattern 3: Extreme delay
pattern3_idx = fraud_idx[350:450]
data.loc[pattern3_idx, "delay_jam"] += np.random.uniform(15, 30, len(pattern3_idx))
data.loc[pattern3_idx, "qty_terima"] *= 0.8

# Pattern 4: Mixed anomalies
pattern4_idx = fraud_idx[450:]
data.loc[pattern4_idx, "qty_kirim"] *= 1.5
data.loc[pattern4_idx, "qty_terima"] *= 0.7
data.loc[pattern4_idx, "karbo"] *= 0.6

# Mark all fraud patterns
data.loc[fraud_idx, "is_fraud"] = 1

# -----------------------------
# ADD REALISTIC NOISE
# -----------------------------
# 5% of normal data with minor deviations (to make it challenging)
noise_idx = np.random.choice(
    data[data["is_fraud"] == 0].index, 
    size=int(0.05 * (n - n_fraud)), 
    replace=False
)
data.loc[noise_idx, "delay_jam"] += np.random.uniform(3, 5, len(noise_idx))
data.loc[noise_idx, "qty_terima"] *= np.random.uniform(0.9, 0.95, len(noise_idx))

# -----------------------------
# ENSURE REALISTIC RANGES
# -----------------------------
data["qty_kirim"] = data["qty_kirim"].clip(lower=50)
data["qty_terima"] = data["qty_terima"].clip(lower=0)
data["delay_jam"] = data["delay_jam"].clip(lower=0)
data["kalori"] = data["kalori"].clip(lower=200)
data["protein"] = data["protein"].clip(lower=5)
data["karbo"] = data["karbo"].clip(lower=30)

# -----------------------------
# SAVE
# -----------------------------
data.to_csv("mbg_synthetic.csv", index=False)

print(f"✅ Data generated successfully!")
print(f"   Total records: {len(data):,}")
print(f"   Normal transactions: {(data['is_fraud'] == 0).sum():,} ({(data['is_fraud'] == 0).sum()/len(data)*100:.1f}%)")
print(f"   Fraud transactions: {(data['is_fraud'] == 1).sum():,} ({(data['is_fraud'] == 1).sum()/len(data)*100:.1f}%)")
print(f"   File saved: mbg_synthetic.csv")
print("="*60)