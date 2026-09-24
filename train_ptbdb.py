"""
train_ptbdb.py – 1D CNN for PTBDB Binary Classification
=========================================================
Trains a 1D CNN to classify ECG signals as Normal (0) or Abnormal (1)
using the PTBDB dataset.

Same architecture and training principles as train_cnn.py:
  - No test-data leakage
  - 80/20 train/validation split from training data
  - EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  - Binary output (sigmoid activation)

Run:
    python train_ptbdb.py

Outputs:
    models/cnn_ptbdb.keras
    results/ptbdb/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "ptbdb")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR, "cnn_ptbdb.keras")
CLASS_NAMES = ["Normal", "Abnormal"]

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


def load(fname):
    path = os.path.join(PREP_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run preprocess.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=False)


def build_cnn_binary(input_len):
    """
    1D CNN with binary sigmoid output for PTBDB normal/abnormal detection.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = tf.keras.Input(shape=(input_len, 1), name="ecg_input")

    x = layers.Conv1D(64, 5, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)

    x = layers.Conv1D(128, 5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2, name="pool2")(x)

    x = layers.Conv1D(256, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    # Binary output – sigmoid
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name="ECG_1D_CNN_PTBDB")
    return model


def plot_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2196F3")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722")
    ax1.set_title("Model Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss", color="#2196F3")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
    ax2.set_title("Model Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("1D CNN Training – PTBDB (Normal vs Abnormal)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "ptbdb_training_history.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
    )
    ax.set_title("Confusion Matrix – CNN (PTBDB Test Set)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    print("\n╔══════════════════════════════════════════════╗")
    print("║  ECG Project – 1D CNN PTBDB Binary         ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  TensorFlow version: {tf.__version__}")

    # ── Load ────────────────────────────────────────────────────────────────
    print("\n[Loading PTBDB training data]")
    X = load("ptbdb_train_X.npy")
    y = load("ptbdb_train_y.npy")
    print(f"  X shape: {X.shape}  y shape: {y.shape}")

    input_len = X.shape[1]
    X = X.reshape(-1, input_len, 1)

    # ── Train-validation split ───────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}")

    # ── Build ────────────────────────────────────────────────────────────────
    print("\n[Building 1D CNN (binary)]")
    model = build_cnn_binary(input_len)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[Training]")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=60, batch_size=64,
        callbacks=callbacks, verbose=1
    )

    # ── Save training plot ───────────────────────────────────────────────────
    plot_history(history, RESULTS_DIR)

    # ── Evaluate on test set ─────────────────────────────────────────────────
    print("\n[Evaluating on test set]")
    X_test = load("ptbdb_test_X.npy")
    y_test = load("ptbdb_test_y.npy")
    X_test = X_test.reshape(-1, input_len, 1)

    y_proba = model.predict(X_test, batch_size=256, verbose=1).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n{'='*50}")
    print("  PTBDB CNN Test Set Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*50}")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    # Save text report
    with open(os.path.join(RESULTS_DIR, "ptbdb_cnn_report.txt"), "w") as fp:
        fp.write("1D CNN – PTBDB Binary Classification\n\n")
        fp.write(f"Accuracy : {acc:.4f}\nPrecision: {prec:.4f}\n")
        fp.write(f"Recall   : {rec:.4f}\nF1 Score : {f1:.4f}\n\n")
        fp.write(report)

    # Save metrics CSV (for compare_models.py)
    import pandas as pd
    pd.DataFrame([{
        "Model": "1D CNN", "Dataset": "PTBDB",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
    }]).to_csv(os.path.join(RESULTS_DIR, "ptbdb_cnn_metrics.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, os.path.join(RESULTS_DIR, "ptbdb_confusion_matrix.png"))

    print(f"\n[Done] Model: {MODEL_PATH}")
    print(f"       Results: results/ptbdb/")


if __name__ == "__main__":
    main()