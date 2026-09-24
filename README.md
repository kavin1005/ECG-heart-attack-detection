# 🫀 ECG Heart Attack & Arrhythmia Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-ECG--Heart--Attack--Detection-black?logo=github)](https://github.com/kavin1005/ECG-heart-attack-detection)

An end-to-end Deep Learning & Machine Learning pipeline for automated electrocardiogram (ECG) heartbeat classification and myocardial infarction (heart attack) detection, complete with an **interactive real-time web dashboard**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Datasets](#-datasets)
- [Performance & Benchmark Results](#-performance--benchmark-results)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Interactive Web Dashboard](#-interactive-web-dashboard)
- [Visualizations & Artifacts](#-visualizations--artifacts)
- [License](#-license)

---

## 🩺 Overview

Early and accurate diagnosis of cardiac arrhythmias and acute myocardial infarction is critical for clinical decision-making and patient survival. This repository implements:
1. **Multi-class Arrhythmia Classification** on the **MIT-BIH Arrhythmia Database** (5 distinct beat types).
2. **Binary Myocardial Infarction Detection** on the **PTB Diagnostic ECG Database** (Normal vs. Abnormal/Heart Attack).
3. **Benchmarking Suite** comparing traditional Machine Learning (Random Forest, XGBoost, KNN, Decision Tree, Logistic Regression) against a specialized **1D Deep Convolutional Neural Network (CNN)**.
4. **Interactive Diagnostic Dashboard** for clinicians and researchers to visualize 187-point ECG waveforms and get real-time classification probabilities.

---

## ✨ Key Features

- **Automated Data Preprocessing**: Robust pipeline with missing value imputation, duplicate removal, Min-Max scaling, and **SMOTE** (Synthetic Minority Over-sampling Technique) for handling severe class imbalance.
- **Deep 1D CNN Architecture**: Optimized 1D convolutional layers with residual skip connections, batch normalization, dropout, and learning rate scheduling tailored for sequential biometric signals.
- **Comprehensive Benchmarks**: Side-by-side performance evaluation measuring Accuracy, Precision, Recall, and Macro-F1 across 6 algorithms.
- **Production-Ready Dashboard**: Built-in HTTP server (`dashboard_server.py`) serving an interactive web interface with live signal rendering, beat catalog browsing, and instant inference.

---

## 🗄️ Datasets

Due to GitHub's storage constraints (>100 MB), raw and preprocessed datasets are kept locally and ignored via `.gitignore`.

### 1. Download Source
Download the **ECG Heartbeat Categorization Dataset** from Kaggle:
👉 **[Kaggle Dataset: ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)**

### 2. Directory Layout
Extract the downloaded files directly into a `dataset/` directory at the project root:
```text
dataset/
├── mitbih_train.csv     # 87,554 beats for training (5 classes)
├── mitbih_test.csv      # 21,892 beats for testing
├── ptbdb_normal.csv     # 4,046 normal heartbeats
└── ptbdb_abnormal.csv   # 10,506 abnormal / myocardial infarction beats
```

### 3. Heartbeat Classes (MIT-BIH)
| Label | Category | Clinical Description |
|:---:|:---|:---|
| **0** | **N** | Normal beat |
| **1** | **S** | Supraventricular ectopic beat |
| **2** | **V** | Ventricular ectopic beat |
| **3** | **F** | Fusion beat |
| **4** | **Q** | Unknown / Unclassifiable beat |

---

## 📈 Performance & Benchmark Results

### 1. Model Comparison on MIT-BIH (Arrhythmia 5-Class)
Evaluated on 21,892 unseen test heartbeats:

| Model | Accuracy | Precision | Recall | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **1D CNN (Deep Learning)** | **98.46%** | **98.42%** | **98.46%** | **98.43%** |
| 🥈 **Random Forest** | 98.08% | 98.03% | 98.08% | 98.04% |
| 🥉 **XGBoost** | 96.69% | 97.11% | 96.69% | 96.85% |
| **KNN** | 95.57% | 96.63% | 95.57% | 95.94% |
| **Decision Tree** | 92.73% | 94.63% | 92.73% | 93.43% |
| **Logistic Regression** | 67.25% | 88.09% | 67.25% | 73.58% |

### 2. PTB Diagnostic ECG (Myocardial Infarction Detection)
Binary classification (Normal vs. Abnormal):

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **1D CNN** | **99.14%** | **99.11%** | **99.72%** | **99.41%** |

> *Note: The 1D CNN achieves a **99.72% recall** on abnormal cases, minimizing false negatives in acute cardiac episodes.*

---

## 📂 Repository Structure

```text
ECG-heart-attack-detection/
├── dashboard/                     # Web dashboard frontend
│   ├── index.html                 # Modern glassmorphism UI layout
│   ├── style.css                  # Clean responsive styling & animations
│   └── app.js                     # Interactive charts & API bindings
├── models/                        # Serialized weights & model artifacts
│   ├── cnn_mitbih.keras           # Trained MIT-BIH 1D CNN model
│   ├── cnn_ptbdb.keras            # Trained PTBDB Myocardial Infarction model
│   └── best_baseline.joblib       # Best classical ML model (Random Forest)
├── results/                       # Evaluation charts & metrics
│   ├── cnn/                       # Confusion matrices & learning curves
│   ├── eda/                       # Signal distribution & sample ECG plots
│   ├── ml/                        # ML model comparisons & reports
│   └── ptbdb/                     # PTB diagnostic training history & matrices
├── baseline_models.py             # Trains and evaluates 5 baseline ML models
├── compare_models.py              # Generates comparative charts and summary tables
├── dashboard_server.py            # Multithreaded REST API & dashboard server (port 5000)
├── eda.py                         # Exploratory data analysis on raw ECG signals
├── evaluate_cnn.py                # In-depth test evaluation for CNN
├── predict.py                     # Command-line inference script for single/batch samples
├── preprocess.py                  # Cleaning, SMOTE balancing, and normalization pipeline
├── train_cnn.py                   # 1D CNN training pipeline for MIT-BIH
├── train_advanced_cnn.py          # Residual/attention enhanced CNN architecture
├── train_ptbdb.py                 # CNN training script for PTB Diagnostic dataset
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## 🚀 Installation & Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/kavin1005/ECG-heart-attack-detection.git
cd ECG-heart-attack-detection

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Preprocess the Data
Clean raw CSVs, apply SMOTE oversampling, and save processed arrays:
```bash
python preprocess.py
```

### 3. Exploratory Data Analysis (EDA)
Generate ECG waveform visualizations and class frequency diagrams:
```bash
python eda.py
```

### 4. Train Models
- **Train Classical ML Baselines**:
  ```bash
  python baseline_models.py
  ```
- **Train 1D CNN on MIT-BIH**:
  ```bash
  python train_cnn.py
  ```
- **Train 1D CNN on PTB Diagnostic**:
  ```bash
  python train_ptbdb.py
  ```

### 5. Evaluate & Generate Comparisons
```bash
python evaluate_cnn.py
python compare_models.py
```

### 6. Run Inference via CLI
Run prediction on test samples:
```bash
python predict.py
```

---

## 🖥️ Interactive Web Dashboard

Launch the local web dashboard for interactive clinical exploration:

```bash
python dashboard_server.py
```

Open **`http://localhost:5000`** in your browser to:
- 📊 Inspect individual **187-point ECG waveforms** plotted in real-time.
- 🔀 Browse the **Beat Catalog** categorized by arrhythmia class.
- ⚡ Run **instant model inference** and view confidence bars across all 5 classes.
- 📈 View live model evaluation metrics and confusion matrices.

---

## 📊 Visualizations & Artifacts

All evaluation artifacts are automatically saved inside [`results/`](results):
- **Confusion Matrices**: Per-class prediction heatmaps (`results/cnn/cnn_confusion_matrix.png`, `results/ptbdb/ptbdb_confusion_matrix.png`).
- **Learning Curves**: Training vs. validation loss and accuracy graphs (`results/cnn/training_history.png`).
- **Signal Plots**: Characteristic waveform shapes for Normal, PVC, PAC, and Fusion beats (`results/eda/`).
- **Model Comparison Chart**: Direct bar-graph benchmark across all 6 models (`results/ml/baseline_comparison.png`).

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).