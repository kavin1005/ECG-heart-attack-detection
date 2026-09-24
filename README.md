# ECG Heart Attack & Arrhythmia Detection

Automated ECG heartbeat classification and myocardial infarction detection using Deep Learning (1D CNN) and Machine Learning, with an interactive web dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
Download the dataset from Kaggle:
👉 **[Kaggle ECG Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)**

Extract the CSV files into a `dataset/` folder:
```text
dataset/
├── mitbih_train.csv
├── mitbih_test.csv
├── ptbdb_normal.csv
└── ptbdb_abnormal.csv
```

### 3. Preprocess & Train
```bash
# Clean and prepare data
python preprocess.py

# Train 1D CNN model
python train_cnn.py

# Train baseline ML models (Random Forest, XGBoost, etc.)
python baseline_models.py
```

### 4. Run Interactive Dashboard
```bash
python dashboard_server.py
```
Open **`http://localhost:5000`** in your browser to visualize ECG signals and test predictions in real-time.

---

## 📊 Results Summary

| Model | Dataset | Task | Accuracy | F1-Score |
|:---|:---|:---|:---:|:---:|
| **1D CNN** | **PTB Diagnostic** | Heart Attack / Abnormal Detection | **99.14%** | **99.41%** |
| **1D CNN** | **MIT-BIH** | 5-Class Arrhythmia Classification | **98.46%** | **98.43%** |
| **Random Forest** | **MIT-BIH** | 5-Class Arrhythmia Classification | **98.08%** | **98.04%** |
| **XGBoost** | **MIT-BIH** | 5-Class Arrhythmia Classification | **96.69%** | **96.85%** |

---

## 📁 Project Structure

```text
├── dashboard/            # Web interface files (HTML, CSS, JS)
├── models/               # Saved trained models (.keras, .joblib)
├── results/              # Evaluation plots and metric reports
├── dataset/              # Raw ECG CSV data (git-ignored)
├── preprocessed/         # Processed data arrays (git-ignored)
├── preprocess.py         # Data cleaning & normalization
├── train_cnn.py          # CNN training script
├── baseline_models.py    # Classical ML models training
├── dashboard_server.py   # Web dashboard backend server
└── requirements.txt      # Python dependencies
```

---

## 📜 License
MIT License