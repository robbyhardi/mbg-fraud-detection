"""
Train Autoencoder for MBG Fraud Detection
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("="*60)
print("TRAINING AUTOENCODER MODEL")
print("="*60)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
data = pd.read_csv("mbg_synthetic.csv")
print(f"✅ Data loaded: {len(data):,} records")

features = [
    "qty_kirim", "qty_terima", "delay_jam",
    "kalori", "protein", "karbo"
]

# Use only NORMAL transactions for training (unsupervised)
normal_data = data[data["is_fraud"] == 0][features]
print(f"✅ Training on normal data only: {len(normal_data):,} records")

# -----------------------------
# 2. PREPROCESS & SCALE
# -----------------------------
scaler = MinMaxScaler()
X_train_full = scaler.fit_transform(normal_data)

# Split for validation
X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=42)
print(f"✅ Train: {len(X_train):,} | Validation: {len(X_val):,}")

# -----------------------------
# 3. BUILD AUTOENCODER
# -----------------------------
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation="relu", name="encoder_1")(input_layer)
encoded = Dense(4, activation="relu", name="encoder_2")(encoded)
bottleneck = Dense(2, activation="relu", name="bottleneck")(encoded)  # compression
decoded = Dense(4, activation="relu", name="decoder_1")(bottleneck)
decoded = Dense(8, activation="relu", name="decoder_2")(decoded)
output_layer = Dense(input_dim, activation="sigmoid", name="output")(decoded)

autoencoder = Model(input_layer, output_layer, name="MBG_Autoencoder")
autoencoder.compile(optimizer="adam", loss="mse", metrics=["mae"])

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
autoencoder.summary()

# -----------------------------
# 4. TRAIN MODEL
# -----------------------------
print("\n" + "="*60)
print("TRAINING...")
print("="*60)

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# -----------------------------
# 5. SAVE MODEL & SCALER
# -----------------------------
autoencoder.save("autoencoder.h5")
print("\n✅ Model saved: autoencoder.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: scaler.pkl")

# -----------------------------
# 6. SAVE TRAINING HISTORY
# -----------------------------
history_df = pd.DataFrame(history.history)
history_df.to_csv("training_history.csv", index=False)
print("✅ History saved: training_history.csv")

# Plot training history
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
print("✅ Plot saved: training_history.png")

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Final Train Loss: {history.history['loss'][-1]:.6f}")
print(f"Final Val Loss: {history.history['val_loss'][-1]:.6f}")
print(f"Total Epochs: {len(history.history['loss'])}")
print("="*60)