# 📖 User Manual - MBG Fraud Detection Dashboard

## 👥 Target Audience

- **Auditors** - Primary users untuk validasi transaksi
- **Operations Managers** - Monitor daily anomalies
- **Finance Team** - Review flagged transactions
- **Data Analysts** - Analyze fraud patterns

**No coding required!** 🎉

---

## 🚀 Quick Start Guide (5 Minutes)

### Step 1: Access Dashboard

**URL**: `https://your-deployment-url.streamlit.app`

**Login**: (If authentication enabled, otherwise public access)

---

### Step 2: Review Demo Data

Saat pertama kali membuka dashboard, Anda akan melihat **data demo** otomatis ter-load.

**Screenshot Placeholder**:
```
┌──────────────────────────────────────────────┐
│  🕵️ MBG Fraud Detection Dashboard            │
├──────────────────────────────────────────────┤
│  ℹ️ Menggunakan data contoh (demo)           │
│                                               │
│  📊 Total Data: 1,000                        │
│  🚨 Anomali Terdeteksi: 50                   │
│  🎯 Ambang Risiko: 0.0234                    │
│  ✅ Precision: 68%                            │
└──────────────────────────────────────────────┘
```

**What You See**:
- **Total Data**: Jumlah transaksi yang dianalisis
- **Anomali Terdeteksi**: Transaksi yang flagged sebagai suspicious
- **Ambang Risiko**: Threshold nilai (dapat diubah)
- **Precision**: Akurasi deteksi (hanya muncul di demo data)

---

### Step 3: Understand the Visualization

**Risk Score Chart**:
```
📈 Pola Penyimpangan Transaksi
    │
0.20│              ▲ Fraud (high risk)
    │            ▲ │
0.15│          ▲   │
    │        ▲     │
0.10│      ▲       │
    │    ▲         │
0.05│  ▲           │ ← Threshold (green line)
    │▲▬▬▬▬▬▬▬▬▬▬▬▬▬│
0.00│▬▬▬▬▬▬▬▬▬▬▬▬▬▬│ Normal transactions
    └───────────────┘
```

**Interpretation**:
- **Blue line**: Risk score per transaction
- **Green dashed line**: Threshold (ambang batas)
- **Above threshold**: Flagged as anomaly
- **Below threshold**: Considered normal

---

### Step 4: Review Detected Anomalies

Scroll down ke **🔍 Daftar Transaksi Anomali**

**Table Columns**:
| Column | Description |
|--------|-------------|
| **Rank** | Urutan berdasarkan risk score (1 = highest risk) |
| **qty_kirim** | Jumlah produk dikirim |
| **qty_terima** | Jumlah produk diterima |
| **delay_jam** | Keterlambatan pengiriman (jam) |
| **kalori** | Kandungan kalori |
| **protein** | Kandungan protein |
| **karbo** | Kandungan karbohidrat |
| **risk_score** | Skor risiko (0-1, higher = more anomalous) |
| **actual_fraud** | (Demo only) Ground truth label |

**Example**:
```
Rank  qty_kirim  qty_terima  delay_jam  risk_score  actual_fraud
  1      150        75         18        0.2345      ✅ Fraud
  2      120        60         22        0.1982      ✅ Fraud
  3      100        48         15        0.1567      ✅ Fraud
```

**Action**: Click baris untuk detail, atau download CSV untuk investigasi lebih lanjut.

---

## 📤 How to Upload Your Own Data

### Step 1: Download Template

1. **Di sidebar kiri**, klik **⬇️ Download Template CSV**
2. Save file: `mbg_template.csv`

**Template Structure**:
```csv
qty_kirim,qty_terima,delay_jam,kalori,protein,karbo
100,98,2,2500,80,300
150,150,0,3000,90,350
200,195,5,2800,85,320
```

---

### Step 2: Prepare Your Data

**Open template in Excel/Google Sheets**

**Rules**:
- ✅ **6 kolom wajib** (qty_kirim, qty_terima, delay_jam, kalori, protein, karbo)
- ✅ **Semua kolom harus berisi angka**
- ✅ **Minimal 10 baris** data
- ✅ **Tidak boleh ada baris kosong**
- ✅ **Tidak boleh ada kolom tambahan** (kecuali Anda hapus di file sebelum upload)

**Example Data Entry**:
```
Transaction 1:
qty_kirim: 120
qty_terima: 118
delay_jam: 3
kalori: 2600
protein: 85
karbo: 310
```

---

### Step 3: Upload File

1. **Di sidebar kiri**, klik **📤 Upload Data (CSV)**
2. **Select your CSV file**
3. Wait for validation...

**Success Message**:
```
✅ Data berhasil diunggah (500 baris)
```

**If validation fails**:
```
⛔ Validasi Data Gagal
❌ Kolom wajib tidak ditemukan: protein
⚠️ Ditemukan 5 transaksi dengan qty_terima > qty_kirim
💡 Solusi: Download template yang benar dari sidebar
```

**Fix errors and re-upload**.

---

### Step 4: Review Results

Dashboard akan otomatis update dengan data Anda!

**Updated Metrics**:
```
📊 Total Data: 500
🚨 Anomali Terdeteksi: 25
🎯 Ambang Risiko: 0.0198
```

---

## 🎯 Adjust Detection Threshold

### Why Adjust Threshold?

**Threshold determines** berapa banyak transaksi yang di-flag sebagai anomaly.

- **Higher threshold (98-99%)**: Fewer anomalies detected, but high confidence
- **Lower threshold (90-92%)**: More anomalies detected, more false positives

---

### How to Adjust

**Di sidebar**, geser **🎯 Ambang Risiko (Percentile)**

```
🎯 Ambang Risiko (Percentile)
90 ◄─────●─────► 99
         95
```

**Real-time update**: Dashboard langsung update tanpa perlu re-upload!

---

### Recommended Settings

| Business Scenario | Recommended Percentile | Expected Anomaly Rate |
|-------------------|------------------------|------------------------|
| **Conservative** (minimize false alarms) | 98-99% | 1-2% |
| **Balanced** (default) | 95% | 5% |
| **Aggressive** (catch all potential fraud) | 90-92% | 8-10% |

**Note**: Consult dengan tim management untuk determine risk tolerance.

---

## 📥 Export Detected Anomalies

### Step 1: Review Anomaly Table

Scroll ke **🔍 Daftar Transaksi Anomali**

---

### Step 2: Click Download Button

**Click**: **⬇️ Download Anomali (CSV)**

**File downloaded**: `mbg_anomalies.csv`

---

### Step 3: Open in Excel

**File contains**:
- All detected anomalies
- Ranked by risk score (highest first)
- Original feature values
- Risk score per transaction

**Use for**:
- Manual review by auditors
- Share with operations team
- Archive for compliance
- Further analysis in Excel/Tableau

---

## 🔍 Interpreting Results

### Risk Score Interpretation

| Risk Score | Severity | Action Required |
|------------|----------|-----------------|
| **< 0.01** | 🟢 Very Low | No action |
| **0.01 - 0.05** | 🟡 Low | Monitor |
| **0.05 - 0.10** | 🟠 Medium | Review recommended |
| **> 0.10** | 🔴 High | **Investigate immediately** |

---

### Common Anomaly Patterns

#### Pattern 1: Quantity Discrepancy
```
qty_kirim: 150
qty_terima: 75  ← 50% loss!
delay_jam: 18
risk_score: 0.234
```
**Possible Causes**:
- Theft during transit
- Incomplete delivery
- Data entry error

**Action**: Verify with delivery receipts, contact driver.

---

#### Pattern 2: Nutritional Anomaly
```
qty_kirim: 100
qty_terima: 98
kalori: 400    ← Too low!
protein: 10    ← Too low!
risk_score: 0.156
```
**Possible Causes**:
- Product substitution (lower quality)
- Expired products
- Supplier fraud

**Action**: Lab test sample, audit supplier.

---

#### Pattern 3: Extreme Delay
```
qty_kirim: 120
qty_terima: 115
delay_jam: 25  ← 25 hours late!
risk_score: 0.198
```
**Possible Causes**:
- Route deviation
- Driver delay
- Unauthorized stop

**Action**: Check GPS logs, interview driver.

---

## 🚨 Important Notes

### ⚠️ Anomaly ≠ Fraud

**Anomaly adalah peringatan awal**, bukan bukti fraud.

**Always**:
- ✅ Review anomalies dengan domain expert (auditor)
- ✅ Verify dengan dokumen pendukung (receipts, GPS logs)
- ✅ Consider context (holiday, weather, route changes)

**Never**:
- ❌ Take punitive action based solely on anomaly score
- ❌ Assume all anomalies are fraud
- ❌ Ignore high-risk anomalies

---

### 🔒 Data Privacy

**Dashboard tidak menyimpan data Anda!**

- Data hanya di-process **sementara** saat Anda upload
- Tidak ada data yang tersimpan di server
- Session berakhir saat Anda close browser

**For sensitive data**:
- Use internal deployment (tidak public)
- Enable authentication
- Encrypt data in transit (HTTPS)

---

### 🐛 Reporting Issues

**Jika Anda encounter error**:

1. **Screenshot error message**
2. **Note: What were you doing when error occurred?**
3. **Email to**: data-team@mbg.com
4. **Include**: CSV file (jika tidak sensitive)

**Common Issues & Solutions**:

| Issue | Solution |
|-------|----------|
| "File too large" | Reduce to < 10 MB, split into batches |
| "Invalid CSV format" | Check delimiter (must be comma), re-download template |
| "Missing columns" | Ensure all 6 required columns present |
| "Dashboard not loading" | Clear browser cache, try different browser |

---

## 📊 Best Practices

### 1. Regular Monitoring

**Recommended**: Upload data **daily** atau **weekly** untuk continuous monitoring.

**Set reminder**:
- Monday morning: Upload last week's transactions
- Review anomalies within 24 hours
- Export & archive results for compliance

---

### 2. Threshold Calibration

**Initial phase** (first 3 months):
- Start with **P95** (default)
- Track **precision** (% of anomalies that are true frauds)
- Adjust threshold based on findings

**Ongoing**:
- Re-calibrate quarterly
- Adjust based on seasonal patterns
- Document threshold changes

---

### 3. Feedback Loop

**Help improve the model**:
- Record which anomalies were **true frauds** vs **false alarms**
- Share findings with data science team
- Model can be **retrained** dengan validated labels

**Template for feedback**:
```
Transaction ID: 12345
Risk Score: 0.234
Actual Outcome: True Fraud
Evidence: Driver admitted to theft
Notes: Pattern repeated 3x in same month
```

---

### 4. Team Collaboration

**Suggested workflow**:
1. **Data Analyst**: Upload data, export anomalies
2. **Auditor**: Review top 20 anomalies
3. **Operations**: Investigate flagged transactions
4. **Finance**: Calculate impact & losses
5. **Management**: Review monthly summary report

---

## 📞 Support & Training

### Getting Help

**Documentation**:
- 📖 This User Manual
- 📚 Technical Documentation (for IT team)
- 🎥 Video Tutorials (coming soon)

**Contact**:
- **Email**: data-team@mbg.com
- **Slack**: #fraud-detection-support
- **Phone**: +62-xxx-xxxx-xxxx (business hours)

---

### Training Sessions

**Available**:
- **30-min onboarding** for new users
- **1-hour deep dive** untuk auditors
- **2-hour workshop** for data analysts

**Schedule**: Email training@mbg.com

---

## 🎯 Success Metrics

### How to Measure Success

**Track these metrics monthly**:

| Metric | Target | Current |
|--------|--------|---------|
| **Anomalies detected** | 5% of transactions | ___ |
| **True positive rate** | > 70% | ___ |
| **False positive rate** | < 15% | ___ |
| **Audit time saved** | > 80% | ___ |
| **Average investigation time** | < 2 hours per anomaly | ___ |

---

### ROI Calculation

**Example**:
```
Before:
- Manual audit: 200 hours/month
- Hourly cost: $75
- Monthly cost: $15,000

After:
- ML-assisted audit: 40 hours/month
- Hourly cost: $75
- Monthly cost: $3,000

Savings: $12,000/month = $144,000/year 🎉
```

---

## 📚 FAQs

### Q1: Can I upload Excel files?

**A**: No, hanya CSV format. Convert Excel to CSV:
1. Excel → File → Save As
2. Select "CSV (Comma delimited)"

---

### Q2: What if my data has more columns?

**A**: Dashboard hanya process 6 required columns. Extra columns akan diabaikan.

---

### Q3: Can I process historical data?

**A**: Yes! Data bisa dari periode manapun. Model tidak care tentang timestamps.

---

### Q4: How accurate is the model?

**A**: Pada synthetic test data: **Precision 68%, Recall 82%**. Real-world performance tergantung data quality.

---

### Q5: Can I adjust the model?

**A**: Tidak via dashboard. Contact data science team untuk model retraining atau tuning.

---

### Q6: Is this a replacement for auditors?

**A**: **NO!** Ini adalah **assistive tool** untuk prioritize audit efforts. Human judgment tetap krusial.

---

### Q7: What happens to flagged transactions?

**A**: Dashboard hanya **detect** anomalies. Anda yang decide action (investigate, escalate, dismiss).

---

### Q8: Can multiple users access simultaneously?

**A**: Yes! Dashboard support multiple concurrent sessions.

---

## 🎓 Glossary

| Term | Definition |
|------|------------|
| **Anomaly** | Transaction that deviates from normal patterns |
| **Risk Score** | Numerical measure (0-1) of how anomalous a transaction is |
| **Threshold** | Cut-off value above which transactions are flagged |
| **Percentile** | Statistical measure (e.g., P95 = top 5% most anomalous) |
| **Precision** | % of detected anomalies that are true frauds |
| **Recall** | % of actual frauds that are detected |
| **False Positive** | Normal transaction incorrectly flagged as fraud |
| **False Negative** | Fraud transaction missed by model |
| **Reconstruction Error** | Technical measure used to calculate risk score |

---

## 📖 Appendix: Sample Scenarios

### Scenario 1: Daily Operations

**Time**: Monday 9 AM

**Task**: Review last week's transactions

**Steps**:
1. Export last week's data dari ERP system
2. Upload to dashboard
3. Set threshold to P95
4. Review top 10 anomalies
5. Export anomaly list
6. Forward to audit team
7. Archive results

**Time required**: 15 minutes

---

### Scenario 2: Investigating High-Risk Anomaly

**Alert**: Transaction with risk score 0.234

**Details**:
```
qty_kirim: 150
qty_terima: 75
delay_jam: 18
```

**Investigation Steps**:
1. ✅ Retrieve delivery receipt → Confirmed 150 sent
2. ✅ Check receiving signature → Only signed for 75
3. ✅ Review GPS logs → Unauthorized stop detected
4. ✅ Interview driver → Admitted to theft
5. ✅ **Outcome**: Confirmed fraud, driver terminated

**Resolution time**: 4 hours

---

### Scenario 3: False Positive Handling

**Alert**: Transaction with risk score 0.098

**Details**:
```
qty_kirim: 100
qty_terima: 98
delay_jam: 12
```

**Investigation Steps**:
1. ✅ Retrieve delivery receipt → Confirmed 98 received
2. ✅ Check notes → Road closure due to accident
3. ✅ Verify with customer → Confirmed 2 units damaged in transit
4. ✅ **Outcome**: Not fraud, legitimate loss

**Action**: Document for future reference, no action needed.

**Resolution time**: 30 minutes

---

## ✅ Checklist for First-Time Users

**Before using dashboard**:
- [ ] Read this user manual (30 min)
- [ ] Attend training session (optional)
- [ ] Access dashboard URL
- [ ] Test with demo data
- [ ] Download template CSV

**For your first upload**:
- [ ] Prepare data in correct format
- [ ] Verify all 6 columns present
- [ ] Check for missing values
- [ ] Upload CSV
- [ ] Review validation messages
- [ ] Adjust threshold if needed
- [ ] Export anomaly list
- [ ] Share with audit team

**Ongoing**:
- [ ] Schedule weekly data uploads
- [ ] Track precision & recall
- [ ] Provide feedback to data team
- [ ] Calibrate threshold quarterly
- [ ] Archive monthly reports

---

**Document Version**: 1.0  
**Last Updated**: December 31, 2025  

---

**🎉 You're Ready to Use MBG Fraud Detection Dashboard!**