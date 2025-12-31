# 📋 MBG Fraud Detection - Project Overview

## 🎯 Executive Summary

**MBG Fraud Detection** adalah sistem deteksi anomali transaksi supply chain menggunakan **Autoencoder-based unsupervised learning**. Sistem ini mengidentifikasi transaksi yang menyimpang dari pola normal historis untuk membantu auditor memprioritaskan investigasi fraud.

---

## 🚨 Problem Statement

### Current Situation
| Masalah | Impact |
|---------|--------|
| Manual audit 10,000+ transaksi/bulan | 200 jam/bulan (25 hari kerja) |
| False positive rate 85% | Waste 170 jam untuk non-fraud cases |
| Detection delay rata-rata 14 hari | Kerugian finansial meningkat |
| Tidak bisa detect zero-day fraud patterns | Fraud baru tidak terdeteksi |

### Business Impact
```
Annual Cost of Current Process:
- Auditor time
- Fraud losses (delayed detection)
```

---

## 💡 Proposed Solution

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MBG FRAUD DETECTION                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   DATA       │     │   MODEL      │     │  DASHBOARD   │
│  PIPELINE    │────▶│  TRAINING    │────▶│  (Streamlit) │
└──────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
  CSV Upload           Autoencoder           Real-time
  Validation           Neural Network        Anomaly Detection
  Preprocessing        Unsupervised          Interactive UI
                       Learning              Export Results
```

### Technology Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Frontend** | Streamlit 1.40.0 | Rapid prototyping, Python-native |
| **ML Framework** | TensorFlow 2.18.0 | Industry standard, Keras API |
| **Data Processing** | Pandas 2.2.3, NumPy 1.26.4 | Standard data science stack |
| **Preprocessing** | Scikit-learn 1.6.0 | MinMaxScaler, evaluation metrics |
| **Visualization** | Matplotlib 3.9.3, Seaborn 0.13.2 | Publication-quality plots |
| **Deployment** | Streamlit Cloud / Docker | Cloud-native, scalable |

---

## 🧠 Machine Learning Approach

### Why Autoencoder?

**Autoencoder** adalah neural network yang dilatih untuk **merekonstruksi input**. Transaksi normal akan memiliki **reconstruction error rendah**, sedangkan transaksi anomali (fraud) akan memiliki **reconstruction error tinggi**.

#### Advantages of Unsupervised Learning:
1. ✅ **No labeled data required** - Tidak perlu dataset fraud historis
2. ✅ **Detect novel fraud patterns** - Bisa detect zero-day fraud
3. ✅ **Adaptive** - Model learn dari pola normal yang terus berubah
4. ✅ **Cost-effective** - Tidak perlu manual labeling

### Architecture Design

```
INPUT LAYER (6 neurons)
    │
    ▼ [Encoder]
HIDDEN LAYER 1 (8 neurons, ReLU)
    │
    ▼
HIDDEN LAYER 2 (4 neurons, ReLU)
    │
    ▼ [Bottleneck - Compressed Representation]
HIDDEN LAYER 3 (2 neurons, ReLU)  ← Latent space
    │
    ▼ [Decoder]
HIDDEN LAYER 4 (4 neurons, ReLU)
    │
    ▼
HIDDEN LAYER 5 (8 neurons, ReLU)
    │
    ▼
OUTPUT LAYER (6 neurons, Sigmoid)
```

**Total Parameters**: ~350 (lightweight model)

**Loss Function**: Mean Squared Error (MSE)
```python
loss = mean((X_input - X_reconstructed)²)
```

---

## 📊 Dataset Description

### Features (6 numerical variables)

| Feature | Description | Unit | Normal Range |
|---------|-------------|------|--------------|
| `qty_kirim` | Quantity sent from warehouse | units | 50-150 |
| `qty_terima` | Quantity received by customer | units | 48-150 |
| `delay_jam` | Delivery delay | hours | 0-10 |
| `kalori` | Calorie content | kcal | 400-900 |
| `protein` | Protein content | grams | 10-30 |
| `karbo` | Carbohydrate content | grams | 50-120 |

### Fraud Patterns Injected

| Pattern Type | Characteristics | Frequency |
|--------------|-----------------|-----------|
| **Quantity Discrepancy** | `qty_terima` = 50% of `qty_kirim` | 40% of frauds |
| **Nutritional Anomaly** | `kalori` & `protein` below 60% of normal | 30% of frauds |
| **Extreme Delay** | `delay_jam` > 15 hours | 20% of frauds |
| **Mixed Anomalies** | Multiple features deviate | 10% of frauds |

### Dataset Statistics

```
Total Transactions: 10,000
├── Normal: 9,500 (95%)
└── Fraud: 500 (5%)

Period: January 2024 - December 2024
Source: Synthetic data (based on MBG historical patterns)
```

---

## 🎯 Success Metrics

### Technical Metrics
- **Precision**: > 0.80 (80% of detected anomalies are true frauds)
- **Recall**: > 0.70 (detect 70%+ of actual frauds)
- **F1-Score**: > 0.75 (balanced precision & recall)
- **AUC-ROC**: > 0.85 (strong discriminative power)

### Business Metrics
- **Time Savings**: Reduce audit time from 200 → 20 hours/month (90%)
- **False Positive Reduction**: 85% → <15%
- **Detection Speed**: 14 days → Real-time
- **Cost Savings**: $180K/year → $18K/year (90% reduction)

---

## 🚀 Project Deliverables

### Phase 1: MVP (Completed)
- [x] Data generation pipeline
- [x] Autoencoder training script
- [x] Model evaluation & visualization
- [x] Interactive Streamlit dashboard
- [x] Template-based CSV upload
- [x] Real-time anomaly detection

### Phase 2: Production Readiness (In Progress)
- [ ] Real data validation
- [ ] User acceptance testing
- [ ] Docker deployment
- [ ] API integration
- [ ] Monitoring & alerting

### Phase 3: Continuous Improvement (Planned)
- [ ] SHAP explainability
- [ ] Adaptive threshold tuning
- [ ] Model retraining pipeline
- [ ] A/B testing framework

---

## 📁 Project Structure

```
mbg-fraud-detection/
├── docs/                          # Documentation
│   ├── 01_PROJECT_OVERVIEW.md     # This file
│   ├── 02_DATA_GENERATION.md
│   ├── 03_MODEL_TRAINING.md
│   ├── 04_MODEL_EVALUATION.md
│   ├── 05_DASHBOARD_IMPLEMENTATION.md
│   ├── 06_DEPLOYMENT_GUIDE.md
│   └── 07_USER_MANUAL.md
├── generate_data.py               # Synthetic data generation
├── train_model.py                 # Model training script
├── evaluate_model.py              # Evaluation & visualization
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── README.md                      # Project README
├── autoencoder.h5                 # Trained model
├── scaler.pkl                     # Fitted preprocessor
├── mbg_synthetic.csv              # Dataset
├── training_history.png           # Training plots
├── confusion_matrix.png           # Evaluation plots
├── roc_curve.png
└── reconstruction_error_distribution.png
```

---

## 👥 Stakeholders

| Role | Responsibility | Key Concerns |
|------|---------------|--------------|
| **Auditors** | Validate detected anomalies | Accuracy, false positive rate |
| **Operations Manager** | Monitor daily transactions | Real-time alerts, ease of use |
| **Data Science Team** | Model maintenance | Performance monitoring, retraining |
| **IT Team** | System deployment | Security, scalability, uptime |

---

## 📈 Roadmap

### Q1 2025
- ✅ MVP launch
- 🔄 User acceptance testing
- 🔄 Real data validation

### Q2 2025
- Production deployment
- Integration with ERP system
- Alerting system (email/Slack)

### Q3 2025
- Explainability features (SHAP)
- Adaptive threshold tuning
- Mobile dashboard

### Q4 2025
- Multi-model ensemble
- Automated retraining pipeline
- Advanced analytics dashboard

---

## 📚 References

1. Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
2. Chandola, V., et al. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*.
3. Schlegl, T., et al. (2017). Unsupervised Anomaly Detection with GANs. *IPMI*.
4. Sakurada, M., & Yairi, T. (2014). Anomaly Detection Using Autoencoders. *ICCAS*.

---

**Document Version**: 1.0  
**Last Updated**: December 31, 2025  
**Author**: Robby - Data Science Team