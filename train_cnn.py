"""
train_cnn.py – 1D CNN for MIT-BIH Arrhythmia Classification
=============================================================
Architecture (1D CNN on 187-point ECG time-series):

  Input (187, 1)
      Conv1D(64)  -> BN -> ReLU -> MaxPool
      Conv1D(128) -> BN -> ReLU -> MaxPool
      Conv1D(256) -> BN -> ReLU -> GlobalAvgPool
      Dense(128)  -> Dropout(0.4)
      Dense(n_classes, softmax)

- Input shape and number of classes are detected automatically.
- Training uses an 80/20 validation split (no test-set leakage).
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

Run:
    python train_cnn.py

Outputs:
    models/cnn_mitbih.keras
    results/cnn/training_history.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "cnn")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR, "cnn_mitbih.keras")


# ── Load data ───────────────────────────────────────────────────────────────
def load(fname):
    path = os.path.join(PREP_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}  –  run preprocess.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=False)


# ── Build 1D CNN model ──────────────────────────────────────────────────────
def build_cnn(input_len, n_classes):
    """
    Builds a 1D CNN suited to ECG time-series classification.
    input_len  : number of ECG time steps (187)
    n_classes  : number of output classes (5 for MIT-BIH)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    inputs = tf.keras.Input(shape=(input_len, 1), name="ecg_input")

    # Block 1
    x = layers.Conv1D(64, kernel_size=5, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # Block 2
    x = layers.Conv1D(128, kernel_size=5, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # Block 3
    x = layers.Conv1D(256, kernel_size=3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Classifier head
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="ECG_1D_CNN")
    return model


# ── Plot training history ───────────────────────────────────────────────────
def plot_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#2196F3")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722")
    ax1.set_title("Model Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"],     label="Train Loss", color="#2196F3")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF5722")
    ax2.set_title("Model Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("1D CNN Training – MIT-BIH", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "training_history.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    from sklearn.model_selection import train_test_split

    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – 1D CNN (MIT-BIH)         ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  TensorFlow version: {tf.__version__}")

    # ── Load ────────────────────────────────────────────────────────────────
    print("\n[Loading training data]")
    X = load("mitbih_train_X.npy")
    y = load("mitbih_train_y.npy")
    print(f"  X shape: {X.shape}  y shape: {y.shape}")

    # ── Auto-detect parameters ──────────────────────────────────────────────
    input_len = X.shape[1]          # 187
    n_classes = len(np.unique(y))   # 5
    print(f"  Input length : {input_len}")
    print(f"  Classes      : {n_classes}")

    # ── Reshape for Conv1D: (samples, timesteps, channels) ─────────────────
    X = X.reshape(-1, input_len, 1)

    # ── One-hot encode labels ───────────────────────────────────────────────
    y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes)

    # ── 80/20 train-validation split (NO test data touched here) ───────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}")

    # ── Build model ─────────────────────────────────────────────────────────
    print("\n[Building 1D CNN]")
    model = build_cnn(input_len, n_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Callbacks ───────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # ── Train ───────────────────────────────────────────────────────────────
    print("\n[Training]")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save history plot ────────────────────────────────────────────────────
    print("\n[Saving training history plot]")
    plot_history(history, RESULTS_DIR)

    print(f"\n[Done] Model saved: {MODEL_PATH}")
    best_val = max(history.history["val_accuracy"])
    print(f"  Best validation accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()