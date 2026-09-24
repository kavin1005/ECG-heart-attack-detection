"""
predict.py – Single ECG Sample Prediction CLI
==============================================
Loads a trained 1D CNN and predicts the class of one ECG sample.

Usage:
    python predict.py --dataset mitbih --sample 100
    python predict.py --dataset ptbdb  --sample 50

Arguments:
    --dataset  : "mitbih" or "ptbdb"
    --sample   : integer index of the sample from the test set

Output:
    - Predicted class label
    - Prediction probability / confidence
    - ECG waveform plot
"""

import os
import sys

# Ensure UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import numpy as np
import matplotlib.pyplot as plt

# ── Label maps ─────────────────────────────────────────────────────────────
MITBIH_LABELS = {
    0: "Normal (N)",
    1: "Supraventricular Ectopy (S)",
    2: "Ventricular Ectopy (V)",
    3: "Fusion Beat (F)",
    4: "Unclassifiable (Q)",
}
PTBDB_LABELS = {
    0: "Normal",
    1: "Abnormal / Pathological",
}

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PREP_DIR   = os.path.join(BASE_DIR, "preprocessed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR= os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


def load(fname):
    path = os.path.join(PREP_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run preprocess.py first.")
        sys.exit(1)
    return np.load(path, allow_pickle=False)


def predict_mitbih(sample_idx):
    import tensorflow as tf

    model_path = os.path.join(MODELS_DIR, "cnn_mitbih.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}"); print("        Run train_cnn.py first."); sys.exit(1)

    X_test = load("mitbih_test_X.npy")
    y_test = load("mitbih_test_y.npy")

    if sample_idx >= len(X_test):
        print(f"[ERROR] Sample index {sample_idx} out of range (0 – {len(X_test)-1})"); sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    sample = X_test[sample_idx]                    # shape (187,)
    true_label = int(y_test[sample_idx])

    # Reshape for model: (1, 187, 1)
    inp = sample.reshape(1, -1, 1)
    proba = model.predict(inp, verbose=0)[0]       # shape (n_classes,)
    pred  = int(np.argmax(proba))
    conf  = float(proba[pred])

    return sample, true_label, pred, proba, MITBIH_LABELS, "MIT-BIH Arrhythmia"


def predict_ptbdb(sample_idx):
    import tensorflow as tf

    model_path = os.path.join(MODELS_DIR, "cnn_ptbdb.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}"); print("        Run train_ptbdb.py first."); sys.exit(1)

    X_test = load("ptbdb_test_X.npy")
    y_test = load("ptbdb_test_y.npy")

    if sample_idx >= len(X_test):
        print(f"[ERROR] Sample index {sample_idx} out of range (0 – {len(X_test)-1})"); sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    sample = X_test[sample_idx]
    true_label = int(y_test[sample_idx])

    inp   = sample.reshape(1, -1, 1)
    prob1 = float(model.predict(inp, verbose=0)[0][0])   # probability of class 1
    pred  = 1 if prob1 >= 0.5 else 0
    proba = np.array([1 - prob1, prob1])

    return sample, true_label, pred, proba, PTBDB_LABELS, "PTBDB Normal/Abnormal"


def print_result(dataset_name, sample_idx, true_label, pred, proba, labels):
    print(f"\n{'='*55}")
    print(f"  Prediction Result – {dataset_name}")
    print(f"{'='*55}")
    print(f"  Sample index     : {sample_idx}")
    print(f"  True label       : {true_label}  →  {labels.get(true_label, '?')}")
    print(f"  Predicted label  : {pred}  →  {labels.get(pred, '?')}")
    print(f"  Confidence       : {proba[pred]*100:.2f}%")
    correct = "✓ CORRECT" if pred == true_label else "✗ INCORRECT"
    print(f"  Verdict          : {correct}")
    print(f"\n  Class probabilities:")
    for cls, name in labels.items():
        if cls < len(proba):
            bar = "█" * int(proba[cls] * 30)
            print(f"    {cls} {name:<30} {proba[cls]*100:5.1f}%  {bar}")
    print(f"{'='*55}")


def plot_ecg(sample, pred, true_label, labels, dataset_name, sample_idx, save_path):
    pred_name  = labels.get(pred, f"Class {pred}")
    true_name  = labels.get(true_label, f"Class {true_label}")
    color      = "#2196F3" if pred == true_label else "#FF5722"
    correct    = "Correct Prediction" if pred == true_label else "Incorrect Prediction"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample, color=color, linewidth=1.5, label=f"ECG Signal (sample #{sample_idx})")
    ax.fill_between(range(len(sample)), sample, alpha=0.1, color=color)
    ax.set_title(
        f"{dataset_name} – ECG Sample #{sample_idx}\n"
        f"Predicted: {pred_name}  |  True: {true_name}  |  {correct}",
        fontweight="bold"
    )
    ax.set_xlabel("Time Step (0–186)")
    ax.set_ylabel("Normalized Amplitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ECG plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict the class of a single ECG sample."
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mitbih", "ptbdb"],
        default="mitbih",
        help="Dataset to use: 'mitbih' or 'ptbdb' (default: mitbih)"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Index of the test sample to predict (default: 0)"
    )
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – Single ECG Prediction    ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  Dataset : {args.dataset}")
    print(f"  Sample  : {args.sample}")

    if args.dataset == "mitbih":
        sample, true_label, pred, proba, labels, name = predict_mitbih(args.sample)
    else:
        sample, true_label, pred, proba, labels, name = predict_ptbdb(args.sample)

    print_result(name, args.sample, true_label, pred, proba, labels)

    plot_path = os.path.join(
        RESULTS_DIR,
        f"prediction_{args.dataset}_sample{args.sample}.png"
    )
    plot_ecg(sample, pred, true_label, labels, name, args.sample, plot_path)


if __name__ == "__main__":
    main()