# 🧠 Model Training Process - Complete Guide

## 🎯 Training Objective

Train an **Autoencoder neural network** to learn the **normal transaction patterns** and detect anomalies based on **reconstruction error**.

---

## 📋 Step 1: Data Loading & Preprocessing

### Load Generated Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load synthetic data
data = pd.read_csv("mbg_synthetic.csv")
print(f"✅ Data loaded: {len(data):,} records")

# Define features
features = [
    "qty_kirim", "qty_terima", "delay_jam",
    "kalori", "protein", "karbo"
]
```

---

### Extract Normal Transactions Only

**🔑 KEY INSIGHT**: Autoencoder dilatih **hanya pada data normal** (unsupervised learning).

```python
# Use only NORMAL transactions for training
normal_data = data[data["is_fraud"] == 0][features]
print(f"✅ Training on {len(normal_data):,} normal transactions")
print(f"   (Excluding {(data['is_fraud'] == 1).sum():,} fraud transactions)")
```

**Why?**
- Autoencoder learns **normal pattern representation**
- Fraud patterns akan memiliki **high reconstruction error**
- Ini adalah supervised learning approach

---

### Feature Scaling

**Method**: MinMaxScaler (range [0, 1])

```python
# Initialize scaler
scaler = MinMaxScaler()

# Fit and transform
X_train_full = scaler.fit_transform(normal_data)

print(f"✅ Features scaled to [0, 1] range")
print(f"   Shape: {X_train_full.shape}")
```

**Why MinMaxScaler?**
- ✅ Sigmoid activation di output layer (range [0,1])
- ✅ Faster convergence
- ✅ Preserves zero values

**Alternative**: StandardScaler (mean=0, std=1) - works too, but sigmoid output less intuitive.

---

### Train/Validation Split

```python
# Split for validation
X_train, X_val = train_test_split(
    X_train_full, 
    test_size=0.2, 
    random_state=42
)

print(f"✅ Train: {len(X_train):,} | Validation: {len(X_val):,}")
```

**Split Ratio**: 80/20
- Training: 7,600 samples
- Validation: 1,900 samples

---

## 🏗️ Step 2: Build Autoencoder Architecture

### Architecture Design

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_dim = X_train.shape[1]  # 6 features

# Input layer
input_layer = Input(shape=(input_dim,), name="input")

# ENCODER
encoded = Dense(8, activation="relu", name="encoder_1")(input_layer)
encoded = Dense(4, activation="relu", name="encoder_2")(encoded)

# BOTTLENECK (latent representation)
bottleneck = Dense(2, activation="relu", name="bottleneck")(encoded)

# DECODER
decoded = Dense(4, activation="relu", name="decoder_1")(bottleneck)
decoded = Dense(8, activation="relu", name="decoder_2")(decoded)

# OUTPUT layer
output_layer = Dense(input_dim, activation="sigmoid", name="output")(decoded)

# Create model
autoencoder = Model(input_layer, output_layer, name="MBG_Autoencoder")
```

### Architecture Visualization

```
Layer (type)                Output Shape              Param #   
=================================================================
input (InputLayer)          [(None, 6)]               0         
encoder_1 (Dense)           (None, 8)                 56        
encoder_2 (Dense)           (None, 4)                 36        
bottleneck (Dense)          (None, 2)                 10        ← Latent space
decoder_1 (Dense)           (None, 4)                 12        
decoder_2 (Dense)           (None, 8)                 40        
output (Dense)              (None, 6)                 54        
=================================================================
Total params: 208
Trainable params: 208
Non-trainable params: 0
```

**Design Rationale**:
- **Symmetric architecture**: Encoder mirrors decoder
- **Progressive compression**: 6 → 8 → 4 → 2 (bottleneck)
- **Small bottleneck**: Forces model to learn compressed representation
- **ReLU activation**: Prevents vanishing gradient
- **Sigmoid output**: Match [0,1] scaled input

---

### Compile Model

```python
autoencoder.compile(
    optimizer="adam",      # Adaptive learning rate
    loss="mse",            # Mean Squared Error
    metrics=["mae"]        # Mean Absolute Error (additional metric)
)

print("✅ Model compiled")
autoencoder.summary()
```

**Loss Function Choice**:
- **MSE** (Mean Squared Error): Penalizes large errors heavily
- Alternative: MAE (Mean Absolute Error) - less sensitive to outliers

**Formula**:
```
MSE = (1/n) Σ (X_original - X_reconstructed)²
```

---

## 🏋️ Step 3: Train the Model

### Training Configuration

```python
from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback
early_stop = EarlyStopping(
    monitor='val_loss',      # Monitor validation loss
    patience=10,             # Stop if no improvement for 10 epochs
    restore_best_weights=True, # Restore weights from best epoch
    verbose=1
)

# Train model
history = autoencoder.fit(
    X_train, X_train,        # Input = Output (autoencoder)
    epochs=100,              # Max 100 epochs
    batch_size=32,           # 32 samples per batch
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=1
)
```

### Training Output (Example)

```
Epoch 1/100
238/238 [==============================] - 2s - loss: 0.0823 - mae: 0.2156 - val_loss: 0.0456 - val_mae: 0.1678
Epoch 2/100
238/238 [==============================] - 1s - loss: 0.0389 - mae: 0.1512 - val_loss: 0.0321 - val_mae: 0.1398
...
Epoch 28/100
238/238 [==============================] - 1s - loss: 0.0045 - mae: 0.0523 - val_loss: 0.0044 - val_mae: 0.0518
Epoch 00028: early stopping
```

**Key Observations**:
- ✅ Training loss decreases consistently
- ✅ Validation loss follows training loss (no overfitting)
- ✅ Early stopping triggered at epoch 28 (patience=10)
- ✅ Final val_loss: 0.0044 (very low)

---

## 💾 Step 4: Save Model & Scaler

### Save Trained Model

```python
# Save model in H5 format
autoencoder.save("autoencoder.h5")
print("✅ Model saved: autoencoder.h5")
```

**File Size**: ~10 KB (very lightweight!)

---

### Save Fitted Scaler

```python
import pickle

# Save scaler for consistent preprocessing
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
print("✅ Scaler saved: scaler.pkl")
```

**🔑 CRITICAL**: Scaler **must** be saved! Inference must use same scaling as training.

---

## 📊 Step 5: Training History Visualization

### Plot Loss Curves

```python
import matplotlib.pyplot as plt

# Save training history to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)

# Plot training & validation loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss (MSE)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
plt.title('Mean Absolute Error', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
print("✅ Training history saved: training_history.png")
```

**Interpretation**:
- **Decreasing loss**: Model is learning
- **Validation loss follows training**: No overfitting
- **Plateau**: Model has converged

---

## ✅ Step 6: Model Validation

### Sanity Check: Reconstruct Normal Data

```python
# Predict on training data
X_recon = autoencoder.predict(X_train[:100], verbose=0)

# Calculate reconstruction error
recon_error = np.mean(np.square(X_train[:100] - X_recon), axis=1)

print(f"✅ Reconstruction Error (Normal Data)")
print(f"   Mean: {recon_error.mean():.6f}")
print(f"   Std: {recon_error.std():.6f}")
print(f"   Max: {recon_error.max():.6f}")
```

**Expected Output**:
```
✅ Reconstruction Error (Normal Data)
   Mean: 0.004523
   Std: 0.002341
   Max: 0.012345
```

**Interpretation**: Low reconstruction error (< 0.01) indicates model learned normal patterns well.

---

### Test on Fraud Data

```python
# Load fraud transactions
fraud_data = data[data["is_fraud"] == 1][features]
X_fraud_scaled = scaler.transform(fraud_data)

# Predict
X_fraud_recon = autoencoder.predict(X_fraud_scaled, verbose=0)

# Calculate error
fraud_error = np.mean(np.square(X_fraud_scaled - X_fraud_recon), axis=1)

print(f"✅ Reconstruction Error (Fraud Data)")
print(f"   Mean: {fraud_error.mean():.6f}")
print(f"   Std: {fraud_error.std():.6f}")
print(f"   Max: {fraud_error.max():.6f}")
```

**Expected Output**:
```
✅ Reconstruction Error (Fraud Data)
   Mean: 0.089234
   Std: 0.045621
   Max: 0.234567
```

**Interpretation**: Fraud data memiliki **reconstruction error 10-20x lebih tinggi** → Good discriminative power!

---

## 📈 Training Summary

### Final Model Statistics

```
Model Performance on Validation Set:
├── Loss (MSE): 0.0044
├── MAE: 0.0518
├── Total Params: 208
├── Training Time: ~30 seconds (CPU)
└── Model Size: 10 KB

Training Configuration:
├── Optimizer: Adam (lr=0.001)
├── Loss Function: MSE
├── Epochs: 28 (early stopped from 100)
├── Batch Size: 32
├── Train Samples: 7,600
└── Val Samples: 1,900
```

---

## 🎯 Key Takeaways

### What Model Learned
- ✅ **Normal transaction patterns** across 6 features
- ✅ **Feature relationships** (e.g., qty_kirim ≈ qty_terima)
- ✅ **Typical ranges** for kalori, protein, karbo
- ✅ **Expected delivery delays**

### Why It Works
- ✅ **Compression forces learning**: Bottleneck (2 neurons) forces model to learn essential patterns
- ✅ **Reconstruction objective**: Minimize difference between input & output
- ✅ **Unsupervised approach**: No fraud labels needed during training

### Limitations
- ⚠️ **Trained on synthetic data** - needs real-world validation
- ⚠️ **Fixed architecture** - not hyperparameter tuned
- ⚠️ **Threshold selection** - requires business input (95th percentile default)

---

## 📚 Complete Training Script

**File**: `train_model.py`

```bash
# Run training
python train_model.py
```

**Expected Output Files**:
- ✅ `autoencoder.h5` (trained model)
- ✅ `scaler.pkl` (fitted preprocessor)
- ✅ `training_history.csv` (training metrics)
- ✅ `training_history.png` (loss curves)

**Runtime**: ~30-60 seconds on CPU

---

**Document Version**: 1.0  
**Last Updated**: December 31, 2025  
**Next**: [04_MODEL_EVALUATION.md](04_MODEL_EVALUATION.md)