"""
train_advanced_cnn.py – Multi-Scale SE-ResNet for ECG Arrhythmia Classification
================================================================================
Upgrades:
  1. Multi-Scale Convolutions: Parallel kernels (3, 7, 15) to capture sharp
     QRS spikes as well as broad P-wave and T-wave repolarizations.
  2. Residual Skip Connections: Preserves fine temporal cardiac features.
  3. Squeeze-and-Excitation (SE) Channel Attention: Dynamically recalibrates
     feature weights to focus on distinguishing minority arrhythmias.
  4. Class-Weighted Loss: Balanced penalty to drastically improve sensitivity
     on Supraventricular (Class 1) and Fusion (Class 3) beats.
  5. Fast CPU Training: Batch size 512 completes epochs in ~18-25 seconds.

Run:
    python train_advanced_cnn.py
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "cnn")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_TARGET = os.path.join(MODELS_DIR, "cnn_mitbih.keras")
MODEL_BACKUP = os.path.join(MODELS_DIR, "cnn_mitbih_v1_backup.keras")

CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unclassifiable"]


# ── Load Clean Dataset with Balanced Strategy ──────────────────────────────
def load_training_data():
    """
    Loads original mitbih_train.csv (87,554 rows) and normalizes it.
    Using the original data avoids SMOTE's 362k artificial row bloat,
    allowing clean training with exact class weighting.
    """
    train_csv = os.path.join(DATA_DIR, "mitbih_train.csv")
    if not os.path.exists(train_csv):
        print(f"[Fallback] {train_csv} not found, loading from preprocessed folder...")
        X = np.load(os.path.join(PREP_DIR, "mitbih_train_X.npy"))
        y = np.load(os.path.join(PREP_DIR, "mitbih_train_y.npy"))
        return X, y

    print("  Loading raw MIT-BIH training dataset...")
    df = pd.read_csv(train_csv, header=None)
    # Clean NaNs and duplicates if any
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # MinMax normalize per beat
    min_val = X.min(axis=1, keepdims=True)
    max_val = X.max(axis=1, keepdims=True)
    denom   = np.where((max_val - min_val) == 0, 1.0, (max_val - min_val))
    X = (X - min_val) / denom

    return X, y


# ── Squeeze-and-Excitation (SE) Block ──────────────────────────────────────
def se_block(input_tensor, ratio=8):
    import tensorflow as tf
    from tensorflow.keras import layers

    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(max(filters // ratio, 8), activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, filters))(se)
    return layers.Multiply()([input_tensor, se])


# ── Residual Convolutional Block ───────────────────────────────────────────
def res_block(x, filters, kernel_size=3):
    import tensorflow as tf
    from tensorflow.keras import layers

    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    y = layers.Conv1D(filters, kernel_size=kernel_size, padding="same")(y)
    y = layers.BatchNormalization()(y)

    y = se_block(y)
    out = layers.Add()([shortcut, y])
    return layers.Activation("relu")(out)


# ── Build Multi-Scale SE-ResNet Model ──────────────────────────────────────
def build_multi_scale_resnet(input_len=187, n_classes=5):
    """
    Multi-Scale Inception-ResNet with Squeeze-and-Excitation Channel Attention
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = tf.keras.Input(shape=(input_len, 1), name="ecg_input")

    # Multi-Scale Inception Stem (Kernels: 3 for QRS spikes, 7 for standard waves, 15 for broad P/T waves)
    conv_s = layers.Conv1D(32, kernel_size=3,  padding="same", activation="relu")(inputs)
    conv_m = layers.Conv1D(32, kernel_size=7,  padding="same", activation="relu")(inputs)
    conv_l = layers.Conv1D(32, kernel_size=15, padding="same", activation="relu")(inputs)

    stem = layers.Concatenate()([conv_s, conv_m, conv_l])  # 96 channels
    stem = layers.BatchNormalization()(stem)

    # Residual Stage 1
    x = res_block(stem, filters=96, kernel_size=5)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # Residual Stage 2
    x = res_block(x, filters=128, kernel_size=3)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Residual Stage 3
    x = res_block(x, filters=192, kernel_size=3)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MultiScale_SE_ResNet")
    return model


# ── Main Training Routine ──────────────────────────────────────────────────
def main():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   CardioPulse AI — Multi-Scale SE-ResNet Training       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  TensorFlow: {tf.__version__}")

    # Backup existing model if present
    if os.path.exists(MODEL_TARGET) and not os.path.exists(MODEL_BACKUP):
        shutil.copy2(MODEL_TARGET, MODEL_BACKUP)
        print(f"  Existing model backed up to: {MODEL_BACKUP}")

    # 1. Load Data
    X, y = load_training_data()
    print(f"  Training samples: {len(X)} (187 timesteps each)")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist_str = ", ".join(f"Class {u} ({CLASS_NAMES[u]}): {c}" for u, c in zip(unique, counts))
    print(f"  Distribution: {dist_str}")

    # 2. Compute Balanced Class Weights to fix minority sensitivity
    weights = compute_class_weight(class_weight="balanced", classes=unique, y=y)
    # Dampen extreme weights slightly with sqrt to maintain stability
    damped_weights = np.sqrt(weights)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique, damped_weights)}
    print(f"  Computed balanced class weights: {class_weight_dict}")

    # Reshape input: (N, 187, 1)
    X = X.reshape(-1, 187, 1)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=5)

    # 80/20 train/val split stratified
    X_tr, X_val, y_tr_cat, y_val_cat, y_tr, y_val = train_test_split(
        X, y_cat, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"  Train split: {len(X_tr)}  |  Val split: {len(X_val)}")

    # 3. Build Multi-Scale SE-ResNet Model
    print("\n[Building Multi-Scale SE-ResNet Architecture]")
    model = build_multi_scale_resnet(input_len=187, n_classes=5)
    model.summary(line_length=80)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 4. Fast Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=MODEL_TARGET, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # 5. Train with Batch Size 512 for fast CPU iterations (~18s per epoch)
    print("\n[Training Multi-Scale SE-ResNet]")
    history = model.fit(
        X_tr, y_tr_cat,
        validation_data=(X_val, y_val_cat),
        epochs=15,
        batch_size=512,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluate on Test Set
    print("\n[Evaluating on Held-Out Test Set]")
    X_te = np.load(os.path.join(PREP_DIR, "mitbih_test_X.npy")).reshape(-1, 187, 1)
    y_te = np.load(os.path.join(PREP_DIR, "mitbih_test_y.npy")).astype(int)

    # Load best saved weights
    best_model = tf.keras.models.load_model(MODEL_TARGET)
    y_pred_probs = best_model.predict(X_te, batch_size=512, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted")
    print(f"\n{'='*55}")
    print(f"  Enhanced Model Results on MIT-BIH Test Set")
    print(f"{'='*55}")
    print(f"  Overall Accuracy: {acc*100:.2f}%")
    print(f"  Weighted F1:      {f1*100:.2f}%")
    print(f"\nClassification Report:\n")
    report = classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)

    # Save classification report to file
    with open(os.path.join(RESULTS_DIR, "cnn_advanced_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Multi-Scale SE-ResNet MIT-BIH Evaluation\nAccuracy: {acc:.4f} | F1: {f1:.4f}\n\n" + report)

    # Save confusion matrix plot
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Confusion Matrix – Multi-Scale SE-ResNet (MIT-BIH)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "cnn_advanced_confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrix: {cm_path}")

    print(f"\n[Done] Improved model saved and deployed to: {MODEL_TARGET}")


if __name__ == "__main__":
    main()
