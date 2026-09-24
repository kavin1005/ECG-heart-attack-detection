"""
baseline_models.py – Baseline ML Models for MIT-BIH ECG Classification
========================================================================
Trains and evaluates 5 classical ML models on preprocessed MIT-BIH data:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. XGBoost (if installed)

Usage:
    python baseline_models.py            # full dataset
    python baseline_models.py --quick    # uses 10,000 train samples (faster)

Outputs saved to results/ml/
Best model saved to models/best_baseline.joblib
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Optional XGBoost ───────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("[INFO] XGBoost not installed. Skipping. (pip install xgboost)")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "ml")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Class label names ──────────────────────────────────────────────────────
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


# ── Load .npy data ──────────────────────────────────────────────────────────
def load(fname):
    path = os.path.join(PREP_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        print("        Run preprocess.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=False)


# ── Evaluate a trained model ────────────────────────────────────────────────
def evaluate(model, X_test, y_test, name, class_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{name}]")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1-score : {f1:.4f}")

    # ── Classification report ──────────────────────────────────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=class_names[:len(np.unique(y_test))],
        zero_division=0
    )
    report_path = os.path.join(RESULTS_DIR, f"{name.replace(' ','_')}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n\n")
        f.write(report)
    print(f"    Report saved: {report_path}")

    # ── Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    labels  = class_names[:len(np.unique(y_test))]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"Confusion Matrix – {name}", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, f"{name.replace(' ','_')}_confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "model_obj": model}


# ── Plot model comparison bar chart ────────────────────────────────────────
def plot_comparison(results_df, save_path):
    metrics  = ["Accuracy", "Precision", "Recall", "F1"]
    n_models = len(results_df)
    n_met    = len(metrics)
    x        = np.arange(n_models)
    width    = 0.18
    colors   = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.8), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, results_df[metric], width, label=metric, color=color)

    ax.set_xticks(x + width * (n_met - 1) / 2)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison – MIT-BIH Dataset", fontweight="bold")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Use only 10,000 training samples for faster testing"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – Baseline ML Models       ║")
    print("╚══════════════════════════════════════════╝")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[Loading MIT-BIH preprocessed data]")
    X_train = load("mitbih_train_X.npy")
    y_train = load("mitbih_train_y.npy")
    X_test  = load("mitbih_test_X.npy")
    y_test  = load("mitbih_test_y.npy")

    # Detect available class names
    n_classes   = len(np.unique(y_train))
    class_names = CLASS_NAMES[:n_classes]

    print(f"  Train shape : {X_train.shape}   Test shape: {X_test.shape}")
    print(f"  Classes     : {n_classes}")

    # ── Quick mode – subsample training data ───────────────────────────────
    if args.quick:
        limit = 10000
        idx   = np.random.default_rng(42).choice(len(X_train), limit, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        print(f"\n  [--quick] Training subset: {X_train.shape[0]} samples")
    else:
        print("\n  [Full mode] Training on entire dataset (may take 20-40 min)")

    # ── Define models ──────────────────────────────────────────────────────
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)),
        ("Decision Tree",       DecisionTreeClassifier(max_depth=20, random_state=42)),
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("KNN",                 KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]
    if XGBOOST_OK:
        models.append((
            "XGBoost",
            XGBClassifier(
                n_estimators=100, use_label_encoder=False,
                eval_metric="mlogloss", random_state=42, n_jobs=-1
            )
        ))

    # ── Train and evaluate each model ─────────────────────────────────────
    print("\n[Training and Evaluating]")
    results = []
    for name, clf in models:
        print(f"\n  Training: {name} ...")
        clf.fit(X_train, y_train)
        result = evaluate(clf, X_test, y_test, name, class_names)
        results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in results])
    df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print("  Model Comparison (sorted by F1)")
    print(f"{'='*55}")
    print(df.to_string(index=False))

    # ── Comparison bar chart ───────────────────────────────────────────────
    plot_comparison(df, os.path.join(RESULTS_DIR, "baseline_comparison.png"))

    # ── Save best model (highest F1) ───────────────────────────────────────
    best_name = df.iloc[0]["Model"]
    best_obj  = next(r["model_obj"] for r in results if r["Model"] == best_name)
    best_path = os.path.join(MODELS_DIR, "best_baseline.joblib")
    joblib.dump(best_obj, best_path)
    print(f"\n  Best model : {best_name}  (F1={df.iloc[0]['F1']:.4f})")
    print(f"  Saved to   : {best_path}")
    print(f"\n[Done] Results saved to results/ml/")


if __name__ == "__main__":
    main()