"""
compare_models.py – Final Model Comparison Table and Chart
===========================================================
Reads real metric CSVs produced by:
  - baseline_models.py  -->  results/ml/baseline_comparison.csv
  - evaluate_cnn.py     -->  results/cnn/cnn_mitbih_metrics.csv
  - train_ptbdb.py      -->  results/ptbdb/ptbdb_cnn_metrics.csv

Builds a unified comparison table and bar chart.
NEVER invents metric values – only reads what models actually produced.

Run:
    python compare_models.py

Outputs:
    results/model_comparison.csv
    results/model_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.family": "DejaVu Sans"})


def try_load(path, label):
    """Attempt to load a CSV; warn if missing."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded: {label}")
        return df
    else:
        print(f"  [SKIP] Not found: {path}")
        print(f"         Run the corresponding script first.")
        return None


def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║  ECG Project – Model Comparison         ║")
    print("╚══════════════════════════════════════════╝")

    frames = []

    # ── Baseline ML results ────────────────────────────────────────────────
    print("\n[Loading results]")
    bl = try_load(os.path.join(RESULTS_DIR, "ml", "baseline_comparison.csv"), "Baseline ML")
    if bl is not None:
        bl["Dataset"] = "MIT-BIH"
        frames.append(bl[["Model", "Dataset", "Accuracy", "Precision", "Recall", "F1"]])

    # ── CNN MIT-BIH results ────────────────────────────────────────────────
    cnn = try_load(os.path.join(RESULTS_DIR, "cnn", "cnn_mitbih_metrics.csv"), "CNN MIT-BIH")
    if cnn is not None:
        frames.append(cnn[["Model", "Dataset", "Accuracy", "Precision", "Recall", "F1"]])

    # ── CNN PTBDB results ──────────────────────────────────────────────────
    ptb = try_load(os.path.join(RESULTS_DIR, "ptbdb", "ptbdb_cnn_metrics.csv"), "CNN PTBDB")
    if ptb is not None:
        frames.append(ptb[["Model", "Dataset", "Accuracy", "Precision", "Recall", "F1"]])

    if not frames:
        print("\n[ERROR] No result files found.")
        print("        Run baseline_models.py, evaluate_cnn.py, and train_ptbdb.py first.")
        sys.exit(1)

    # ── Build unified table ────────────────────────────────────────────────
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["Dataset", "F1"], ascending=[True, False]).reset_index(drop=True)

    # Round for readability
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        all_df[col] = all_df[col].round(4)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    all_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    # Print table
    print(f"\n{'='*72}")
    print("  Final Model Comparison Table")
    print(f"{'='*72}")
    print(all_df.to_string(index=False))
    print(f"{'='*72}")

    # ── Bar chart: Accuracy comparison ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(all_df) * 1.5), 5))
    x       = np.arange(len(all_df))
    width   = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, all_df[metric], width, label=metric, color=color)

    labels = [f"{row.Model}\n({row.Dataset})" for _, row in all_df.iterrows()]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("ECG Classification – Model Comparison", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()

    chart_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    fig.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {chart_path}")

    print("\n[Done] Model comparison complete.")


if __name__ == "__main__":
    main()