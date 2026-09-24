# ECG Heart Attack & Arrhythmia Detection

Deep Learning and Machine Learning pipeline for automated ECG heartbeat classification and myocardial infarction / abnormal heartbeat detection using the **MIT-BIH Arrhythmia Database** and **PTB Diagnostic ECG Database**.

---

## 📁 Dataset Setup

Due to GitHub's file size limits (>100 MB), raw and preprocessed dataset files are not tracked in this repository.

### 1. Download the Dataset
Download the **ECG Heartbeat Categorization Dataset** from Kaggle:
👉 **[Kaggle: ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)**

### 2. Extract into `dataset/`
Place the CSV files into a folder named `dataset/` in the project root:
```text
dataset/
├── mitbih_train.csv
├── mitbih_test.csv
├── ptbdb_normal.csv
└── ptbdb_abnormal.csv
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
Runs data cleaning, normalization, SMOTE balancing, and saves `.npy` / `.csv` arrays to `preprocessed/`:
```bash
python preprocess.py
```

### 3. Exploratory Data Analysis (EDA)
Generate signal plots and class distributions:
```bash
python eda.py
```

### 4. Train Models
- **Baseline ML Models** (Logistic Regression, Random Forest, XGBoost, KNN, Decision Tree):
  ```bash
  python baseline_models.py
  ```
- **1D Convolutional Neural Network (CNN)** for MIT-BIH:
  ```bash
  python train_cnn.py
  ```
- **Advanced CNN with Residual Connections / Attention**:
  ```bash
  python train_advanced_cnn.py
  ```
- **PTB Diagnostic (Myocardial Infarction / Abnormal)**:
  ```bash
  python train_ptbdb.py
  ```

### 5. Evaluate and Compare Models
```bash
python evaluate_cnn.py
python compare_models.py
```

### 6. Interactive Dashboard & Inference
Run the dashboard server to visualize ECG signals and classify heartbeats interactively:
```bash
python dashboard_server.py
```

---

## 📊 Results & Artifacts

All evaluation metrics, confusion matrices, and training curves are saved in [`results/`](results):
- **MIT-BIH Classification**: 5 classes (Normal, Supraventricular ectopic, Ventricular ectopic, Fusion, Unknown)
- **PTBDB Classification**: Binary (Normal vs. Abnormal / Myocardial Infarction)
- Visualizations available in `results/cnn/`, `results/eda/`, and `results/ml/`.