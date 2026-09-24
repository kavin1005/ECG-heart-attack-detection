"""
evaluate_cnn.py – Evaluate trained 1D CNN on MIT-BIH Test Set
==============================================================
Loads the saved CNN model and evaluates it ONLY on the held-out test data.
No training is done here.

Run:
    python evaluate_cnn.py

Outputs saved to results/cnn/
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "cnn")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR, "cnn_mitbih.keras")

CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


def load(fname):
    path = os.path.join(PREP_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run preprocess.py first."); sys.exit(1)
    return np.load(path, allow_pickle=False)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title("Confusion Matrix – CNN (MIT-BIH Test Set)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_class_f1(report_dict, class_names, save_path):
    """Bar chart of per-class F1 scores."""
    f1s = [report_dict.get(c, {}).get("f1-score", 0) for c in class_names]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, f1s, color=colors[:len(class_names)], edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score – CNN (MIT-BIH)", fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Class")
    for bar, f in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{f:.3f}", ha="center", va="bottom", fontsize=10
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    import tensorflow as tf

    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – CNN Evaluation (MIT-BIH) ║")
    print("╚══════════════════════════════════════════╝")

    # ── Check model exists ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run train_cnn.py first.")
        sys.exit(1)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n[Loading model] {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()

    # ── Load test data ──────────────────────────────────────────────────────
    print("\n[Loading test data]")
    X_test = load("mitbih_test_X.npy")
    y_test = load("mitbih_test_y.npy")

    # ── Detect actual class count ────────────────────────────────────────────
    n_classes   = model.output_shape[-1]
    class_names = CLASS_NAMES[:n_classes]
    print(f"  Test shape : {X_test.shape}")
    print(f"  Classes    : {n_classes}")

    # ── Reshape for Conv1D ──────────────────────────────────────────────────
    X_test = X_test.reshape(-1, X_test.shape[1], 1)

    # ── Predict ─────────────────────────────────────────────────────────────
    print("\n[Predicting...]")
    y_proba = model.predict(X_test, batch_size=256, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*50}")
    print("  CNN Test Set Results – MIT-BIH")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*50}")

    # ── Classification report ────────────────────────────────────────────────
    report_str  = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                        zero_division=0, output_dict=True)
    print("\n[Classification Report]")
    print(report_str)

    report_path = os.path.join(RESULTS_DIR, "cnn_mitbih_report.txt")
    with open(report_path, "w") as f:
        f.write("CNN – MIT-BIH Test Set Evaluation\n\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n\n")
        f.write(report_str)
    print(f"  Report saved: {report_path}")

    # Save metrics CSV (for compare_models.py)
    import pandas as pd
    pd.DataFrame([{
        "Model": "1D CNN", "Dataset": "MIT-BIH",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    }]).to_csv(os.path.join(RESULTS_DIR, "cnn_mitbih_metrics.csv"), index=False)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[Generating plots]")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png"))
    plot_per_class_f1(report_dict, class_names, os.path.join(RESULTS_DIR, "cnn_per_class_f1.png"))

    print(f"\n[Done] All results saved to results/cnn/")


if __name__ == "__main__":
    main()