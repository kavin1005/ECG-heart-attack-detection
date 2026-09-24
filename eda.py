"""
eda.py – Exploratory Data Analysis for ECG Classification Project
=================================================================
Loads preprocessed .npy files and generates:
  - Dataset statistics (shape, dtype, value ranges, mean, std)
  - Class distribution bar charts for all 4 splits
  - Sample ECG waveform plots per class
  - Saves all figures to results/eda/

Run after preprocess.py:
    python eda.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "eda")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Class label maps ───────────────────────────────────────────────────────
MITBIH_LABELS = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unclassifiable (Q)",
}
PTBDB_LABELS = {0: "Normal", 1: "Abnormal"}

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]


# ── Helper: load .npy safely ───────────────────────────────────────────────
def load(filename):
    path = os.path.join(PREP_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Please run preprocess.py first."
        )
    return np.load(path, allow_pickle=False)


# ── Helper: print dataset statistics ──────────────────────────────────────
def print_stats(name, X, y):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Samples       : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}")
    print(f"  X shape       : {X.shape}")
    print(f"  y shape       : {y.shape}")
    print(f"  X dtype       : {X.dtype}")
    print(f"  X min/max     : {X.min():.4f} / {X.max():.4f}")
    print(f"  X mean        : {X.mean():.4f}")
    print(f"  X std         : {X.std():.4f}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"  Classes       : {len(classes)}")
    for c, n in zip(classes, counts):
        pct = n / len(y) * 100
        print(f"    Class {int(c)}: {n:>7} samples  ({pct:.1f}%)")


# ── Helper: class distribution bar chart ──────────────────────────────────
def plot_distribution(y, labels_map, title, fname, palette=PALETTE):
    classes, counts = np.unique(y, return_counts=True)
    names  = [labels_map.get(int(c), str(c)) for c in classes]
    colors = [palette[i % len(palette)] for i in range(len(classes))]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Samples")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Annotate counts on top of each bar
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{cnt:,}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Helper: ECG waveform plot (one sample per class) ──────────────────────
def plot_ecg_samples(X, y, labels_map, title, fname, palette=PALETTE):
    classes = sorted(np.unique(y))
    n_cls   = len(classes)
    fig, axes = plt.subplots(n_cls, 1, figsize=(12, 2.2 * n_cls), sharex=True)
    if n_cls == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax, cls in zip(axes, classes):
        idx = np.where(y == cls)[0]
        # Pick a fixed sample for reproducibility
        sample_idx = idx[len(idx) // 3]
        ecg = X[sample_idx]
        color = palette[int(cls) % len(palette)]
        ax.plot(ecg, color=color, linewidth=1.2)
        label = labels_map.get(int(cls), f"Class {int(cls)}")
        ax.set_ylabel(label, fontsize=9, labelpad=4)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("ECG Time Steps (0–186)", fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – Exploratory Data Analysis ║")
    print("╚══════════════════════════════════════════╝")

    # ── Load all splits ──────────────────────────────────────────────────
    print("\n[Loading preprocessed data]")
    mit_Xtr = load("mitbih_train_X.npy")
    mit_ytr = load("mitbih_train_y.npy")
    mit_Xte = load("mitbih_test_X.npy")
    mit_yte = load("mitbih_test_y.npy")

    ptb_Xtr = load("ptbdb_train_X.npy")
    ptb_ytr = load("ptbdb_train_y.npy")
    ptb_Xte = load("ptbdb_test_X.npy")
    ptb_yte = load("ptbdb_test_y.npy")

    # ── Statistics ────────────────────────────────────────────────────────
    print_stats("MIT-BIH  – Training Set", mit_Xtr, mit_ytr)
    print_stats("MIT-BIH  – Test Set",     mit_Xte, mit_yte)
    print_stats("PTBDB    – Training Set", ptb_Xtr, ptb_ytr)
    print_stats("PTBDB    – Test Set",     ptb_Xte, ptb_yte)

    # ── Class distribution bar charts ─────────────────────────────────────
    print("\n[Generating class distribution charts]")
    plot_distribution(
        mit_ytr, MITBIH_LABELS,
        "MIT-BIH – Training Class Distribution (after SMOTE)",
        "mitbih_train_distribution.png"
    )
    plot_distribution(
        mit_yte, MITBIH_LABELS,
        "MIT-BIH – Test Class Distribution",
        "mitbih_test_distribution.png"
    )
    plot_distribution(
        ptb_ytr, PTBDB_LABELS,
        "PTBDB – Training Class Distribution (after SMOTE)",
        "ptbdb_train_distribution.png"
    )
    plot_distribution(
        ptb_yte, PTBDB_LABELS,
        "PTBDB – Test Class Distribution",
        "ptbdb_test_distribution.png"
    )

    # ── ECG waveform samples ───────────────────────────────────────────────
    print("\n[Generating ECG waveform samples]")
    plot_ecg_samples(
        mit_Xtr, mit_ytr, MITBIH_LABELS,
        "MIT-BIH – Sample ECG Waveforms by Class (Training Set)",
        "mitbih_ecg_samples.png"
    )
    plot_ecg_samples(
        ptb_Xtr, ptb_ytr, PTBDB_LABELS,
        "PTBDB – Normal vs Abnormal ECG Waveforms (Training Set)",
        "ptbdb_ecg_samples.png"
    )

    print(f"\n[Done] All EDA figures saved to: results/eda/")
    print(f"       Total files: {len(os.listdir(RESULTS_DIR))}")


if __name__ == "__main__":
    main()