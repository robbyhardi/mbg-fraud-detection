"""
Comprehensive Model Evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import pickle

print("="*60)
print("MODEL EVALUATION")
print("="*60)

# -----------------------------
# 1. LOAD DATA & MODEL
# -----------------------------
data = pd.read_csv("mbg_synthetic.csv")
model = load_model("autoencoder.h5", compile=False)
model.compile(optimizer="adam", loss="mse")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"✅ Data loaded: {len(data):,} records")
print(f"✅ Model & Scaler loaded")

features = ["qty_kirim", "qty_terima", "delay_jam", "kalori", "protein", "karbo"]

# -----------------------------
# 2. PREPROCESS
# -----------------------------
X = scaler.transform(data[features])

# -----------------------------
# 3. RECONSTRUCTION ERROR
# -----------------------------
recon = model.predict(X, verbose=0)
mse = np.mean(np.power(X - recon, 2), axis=1)
data["anomaly_score"] = mse

print(f"✅ Reconstruction error calculated")
print(f"   Mean: {mse.mean():.6f}")
print(f"   Std: {mse.std():.6f}")
print(f"   Min: {mse.min():.6f}")
print(f"   Max: {mse.max():.6f}")

# -----------------------------
# 4. THRESHOLD & PREDICTION
# -----------------------------
THRESHOLD_PERCENTILE = 95
threshold = np.percentile(mse, THRESHOLD_PERCENTILE)
data["anomaly_pred"] = (data["anomaly_score"] > threshold).astype(int)

print(f"\n✅ Threshold (P{THRESHOLD_PERCENTILE}): {threshold:.6f}")
print(f"   Detected anomalies: {data['anomaly_pred'].sum():,} ({data['anomaly_pred'].sum()/len(data)*100:.2f}%)")

# -----------------------------
# 5. CLASSIFICATION REPORT
# -----------------------------
y_true = data["is_fraud"]
y_pred = data["anomaly_pred"]

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], output_dict=True)
print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))

# Extract metrics for README
precision_fraud = report["Fraud"]["precision"]
recall_fraud = report["Fraud"]["recall"]
f1_fraud = report["Fraud"]["f1-score"]

# -----------------------------
# 6. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            cbar_kws={"label": "Count"})
plt.title('Confusion Matrix - Autoencoder Fraud Detection', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved")

# -----------------------------
# 7. ROC CURVE
# -----------------------------
fpr, tpr, thresholds_roc = roc_curve(y_true, data["anomaly_score"])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Autoencoder (AUC = {roc_auc:.3f})', linewidth=2.5, color='darkblue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Fraud Detection', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
print(f"✅ ROC curve saved (AUC = {roc_auc:.3f})")

# -----------------------------
# 8. PRECISION-RECALL CURVE
# -----------------------------
precision, recall, pr_thresholds = precision_recall_curve(y_true, data["anomaly_score"])
avg_precision = average_precision_score(y_true, data["anomaly_score"])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}', linewidth=2.5, color='darkgreen')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(alpha=0.3)
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
print(f"✅ PR curve saved (AP = {avg_precision:.3f})")

# -----------------------------
# 9. RECONSTRUCTION ERROR DISTRIBUTION
# -----------------------------
plt.figure(figsize=(12, 6))
plt.hist(data[data["is_fraud"] == 0]["anomaly_score"], 
         bins=50, alpha=0.7, label="Normal", color='blue', edgecolor='black')
plt.hist(data[data["is_fraud"] == 1]["anomaly_score"], 
         bins=50, alpha=0.7, label="Fraud", color='red', edgecolor='black')
plt.axvline(threshold, color='green', linestyle='--', linewidth=2.5, 
            label=f'Threshold (P{THRESHOLD_PERCENTILE}) = {threshold:.4f}')
plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Reconstruction Error Distribution by Class', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.savefig("reconstruction_error_distribution.png", dpi=300, bbox_inches='tight')
print("✅ Error distribution saved")

# -----------------------------
# 10. SAMPLE ANOMALIES
# -----------------------------
print("\n" + "="*60)
print("TOP 10 DETECTED ANOMALIES")
print("="*60)
top_anomalies = data.nlargest(10, "anomaly_score")
print(top_anomalies[features + ["anomaly_score", "is_fraud", "anomaly_pred"]].to_string(index=False))

# -----------------------------
# 11. SUMMARY STATISTICS
# -----------------------------
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total Records: {len(data):,}")
print(f"Actual Frauds: {y_true.sum():,} ({y_true.sum()/len(data)*100:.2f}%)")
print(f"Predicted Anomalies: {y_pred.sum():,} ({y_pred.sum()/len(data)*100:.2f}%)")
print(f"\nTrue Positives: {cm[1,1]:,}")
print(f"True Negatives: {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"\nPrecision: {precision_fraud:.3f}")
print(f"Recall: {recall_fraud:.3f}")
print(f"F1-Score: {f1_fraud:.3f}")
print(f"AUC-ROC: {roc_auc:.3f}")
print("="*60)

# -----------------------------
# 12. SAVE METRICS FOR README
# -----------------------------
metrics_summary = {
    "Precision": precision_fraud,
    "Recall": recall_fraud,
    "F1-Score": f1_fraud,
    "AUC-ROC": roc_auc,
    "False Positive Rate": cm[0,1] / (cm[0,0] + cm[0,1]),
    "Threshold": threshold
}

with open("evaluation_metrics.txt", "w") as f:
    for key, value in metrics_summary.items():
        f.write(f"{key}: {value:.4f}\n")

print("\n✅ Evaluation completed!")
print("✅ All plots saved to current directory")
print("="*60)