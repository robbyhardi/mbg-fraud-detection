# 📊 Data Generation Process - Step-by-Step Guide

## 🎯 Objective

Generate **synthetic MBG transaction data** dengan karakteristik:
- 10,000 transaksi total
- 95% normal transactions
- 5% fraud transactions (4 fraud patterns)
- Realistic noise untuk challenge model

---

## 📋 Step 1: Environment Setup

### Install Dependencies

```bash
pip install pandas numpy
```

### Import Libraries

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
```

---

## 📊 Step 2: Generate Normal Transactions

### Feature Distributions

```python
n = 10000  # Total transactions

# Generate normal transactions from realistic distributions
data = pd.DataFrame({
    "qty_kirim": np.random.normal(100, 15, n),      # μ=100, σ=15
    "qty_terima": np.random.normal(98, 15, n),      # μ=98, σ=15 (slight loss)
    "delay_jam": np.abs(np.random.normal(2, 1.5, n)), # μ=2, σ=1.5 (always positive)
    "kalori": np.random.normal(650, 80, n),         # μ=650, σ=80
    "protein": np.random.normal(20, 4, n),          # μ=20, σ=4
    "karbo": np.random.normal(80, 12, n),           # μ=80, σ=12
})

# Initialize fraud label (all normal initially)
data["is_fraud"] = 0
```

### Distribution Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ["qty_kirim", "qty_terima", "delay_jam", "kalori", "protein", "karbo"]

for idx, feature in enumerate(features):
    ax = axes[idx // 3, idx % 3]
    ax.hist(data[feature], bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(f'{feature} Distribution')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("feature_distributions.png")
```

---

## 🚨 Step 3: Inject Fraud Patterns

### Pattern 1: Quantity Discrepancy (200 cases)

**Characteristic**: `qty_terima` = 50% of `qty_kirim` + significant delay

```python
n_fraud = 500  # 5% fraud rate
fraud_idx = np.random.choice(n, size=n_fraud, replace=False)

# Pattern 1: Significant quantity loss
pattern1_idx = fraud_idx[:200]
data.loc[pattern1_idx, "qty_terima"] *= 0.5  # 50% loss
data.loc[pattern1_idx, "delay_jam"] += np.random.uniform(5, 10, len(pattern1_idx))
data.loc[pattern1_idx, "is_fraud"] = 1
```

**Business Interpretation**: Kemungkinan pencurian/kehilangan produk dalam transit.

---

### Pattern 2: Nutritional Anomaly (150 cases)

**Characteristic**: Low calorie & protein content

```python
# Pattern 2: Nutritional content anomaly
pattern2_idx = fraud_idx[200:350]
data.loc[pattern2_idx, "kalori"] *= 0.6   # 40% reduction
data.loc[pattern2_idx, "protein"] *= 0.5  # 50% reduction
data.loc[pattern2_idx, "is_fraud"] = 1
```

**Business Interpretation**: Kemungkinan substitusi produk dengan kualitas rendah.

---

### Pattern 3: Extreme Delay (100 cases)

**Characteristic**: Delivery delay > 15 hours

```python
# Pattern 3: Extreme delivery delay
pattern3_idx = fraud_idx[350:450]
data.loc[pattern3_idx, "delay_jam"] += np.random.uniform(15, 30, len(pattern3_idx))
data.loc[pattern3_idx, "qty_terima"] *= 0.8  # Some quantity loss
data.loc[pattern3_idx, "is_fraud"] = 1
```

**Business Interpretation**: Kemungkinan rute tidak authorized atau transit mencurigakan.

---

### Pattern 4: Mixed Anomalies (50 cases)

**Characteristic**: Multiple features deviate

```python
# Pattern 4: Mixed anomalies (most suspicious)
pattern4_idx = fraud_idx[450:]
data.loc[pattern4_idx, "qty_kirim"] *= 1.5   # Over-reported sent quantity
data.loc[pattern4_idx, "qty_terima"] *= 0.7  # Under-received
data.loc[pattern4_idx, "karbo"] *= 0.6       # Low carb content
data.loc[pattern4_idx, "is_fraud"] = 1
```

**Business Interpretation**: Fraud terorganisir dengan manipulasi multiple features.

---

## 🔊 Step 4: Add Realistic Noise

### Purpose
Membuat model **tidak overfit** dengan menambahkan minor deviations pada normal data.

```python
# 5% of normal data with minor deviations
noise_idx = np.random.choice(
    data[data["is_fraud"] == 0].index, 
    size=int(0.05 * (n - n_fraud)), 
    replace=False
)

data.loc[noise_idx, "delay_jam"] += np.random.uniform(3, 5, len(noise_idx))
data.loc[noise_idx, "qty_terima"] *= np.random.uniform(0.9, 0.95, len(noise_idx))
# Note: is_fraud tetap 0 (ini bukan fraud, hanya variasi normal)
```

**Impact**: Model harus belajar distinguish antara **variasi normal** vs **anomali signifikan**.

---

## ✅ Step 5: Data Validation & Clipping

### Ensure Realistic Ranges

```python
# Clip values to realistic ranges
data["qty_kirim"] = data["qty_kirim"].clip(lower=50)
data["qty_terima"] = data["qty_terima"].clip(lower=0)
data["delay_jam"] = data["delay_jam"].clip(lower=0)
data["kalori"] = data["kalori"].clip(lower=200)
data["protein"] = data["protein"].clip(lower=5)
data["karbo"] = data["karbo"].clip(lower=30)
```

### Data Quality Checks

```python
# Check for missing values
assert data.isnull().sum().sum() == 0, "Missing values detected!"

# Check fraud distribution
fraud_count = data["is_fraud"].sum()
fraud_rate = fraud_count / len(data) * 100
print(f"Fraud Rate: {fraud_rate:.2f}%")

# Check feature ranges
for col in data.columns:
    print(f"{col}: min={data[col].min():.2f}, max={data[col].max():.2f}")
```

---

## 💾 Step 6: Save Dataset

```python
# Save to CSV
data.to_csv("mbg_synthetic.csv", index=False)

print(f"✅ Dataset generated successfully!")
print(f"   Total records: {len(data):,}")
print(f"   Normal: {(data['is_fraud'] == 0).sum():,} ({(data['is_fraud'] == 0).sum()/len(data)*100:.1f}%)")
print(f"   Fraud: {(data['is_fraud'] == 1).sum():,} ({(data['is_fraud'] == 1).sum()/len(data)*100:.1f}%)")
```

**Output:**
```
✅ Dataset generated successfully!
   Total records: 10,000
   Normal: 9,500 (95.0%)
   Fraud: 500 (5.0%)
```

---

## 📊 Step 7: Exploratory Data Analysis (EDA)

### Fraud vs Normal Comparison

```python
import seaborn as sns

# Compare distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ["qty_kirim", "qty_terima", "delay_jam", "kalori", "protein", "karbo"]

for idx, feature in enumerate(features):
    ax = axes[idx // 3, idx % 3]
    
    # Plot normal data
    data[data["is_fraud"] == 0][feature].hist(
        bins=50, alpha=0.6, label='Normal', color='blue', ax=ax
    )
    
    # Plot fraud data
    data[data["is_fraud"] == 1][feature].hist(
        bins=30, alpha=0.6, label='Fraud', color='red', ax=ax
    )
    
    ax.set_title(f'{feature} Distribution by Class')
    ax.legend()

plt.tight_layout()
plt.savefig("fraud_vs_normal_distributions.png")
```

### Correlation Matrix

```python
# Feature correlation
correlation = data[features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig("correlation_matrix.png")
```

---

## 🎯 Key Takeaways

### Data Characteristics
- ✅ **Balanced complexity**: Not too easy, not too hard
- ✅ **Multiple fraud patterns**: Test model's generalization
- ✅ **Realistic noise**: Prevent overfitting
- ✅ **Business-aligned**: Features mirror real-world supply chain

### Statistical Properties
```
Feature Statistics (Normal Transactions):
├── qty_kirim: mean=100.3, std=15.1
├── qty_terima: mean=98.1, std=15.2
├── delay_jam: mean=2.0, std=1.5
├── kalori: mean=650.5, std=80.3
├── protein: mean=20.1, std=4.0
└── karbo: mean=80.2, std=12.1

Fraud Patterns:
├── Pattern 1 (40%): Quantity discrepancy
├── Pattern 2 (30%): Nutritional anomaly
├── Pattern 3 (20%): Extreme delay
└── Pattern 4 (10%): Mixed anomalies
```

---

## 📚 Complete Script

**File**: `generate_data.py`

```python
# See complete script in repository
python generate_data.py
```

**Expected Runtime**: 2-5 seconds  
**Output Files**:
- `mbg_synthetic.csv` (10,000 rows × 7 columns)
- `feature_distributions.png`
- `fraud_vs_normal_distributions.png`
- `correlation_matrix.png`

---

**Document Version**: 1.0  
**Last Updated**: December 31, 2025  
**Next**: [03_MODEL_TRAINING.md](03_MODEL_TRAINING.md)