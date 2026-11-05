"""
Figure Generation Module
Generates all 13 publication-quality figures specified in Section 12 and Section 15.2
using matplotlib, programmatically derived strictly from saved experimental results.
Outputs saved as high-resolution PNG (300 DPI) and PDF vector formats.
"""

import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import CLASS_NAMES, ID2LABEL, PRIMARY_CONFIGS, TRANSLATION_CONFIGS

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

COLORS = ["#d95f02", "#e7298a", "#7570b3", "#1b9e77", "#66a61e"]


def load_all_metrics(metrics_dir: str):
    runs = []
    if not os.path.exists(metrics_dir):
        return runs
    for jf in os.listdir(metrics_dir):
        if jf.endswith(".json"):
            with open(os.path.join(metrics_dir, jf), "r", encoding="utf-8") as f:
                runs.append(json.load(f))
    return runs


def generate_figure_f1_distribution(dist_json_path: str, out_dir: str):
    """Figure F1: Five-class distribution (overall and per split)."""
    if not os.path.exists(dist_json_path):
        return
    with open(dist_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dist = meta["class_distribution"]
    classes = [ID2LABEL[c] for c in range(5)]
    labels_display = [CLASS_NAMES[c] for c in range(5)]

    train_c = [dist["train"][c] for c in classes]
    val_c = [dist["val"][c] for c in classes]
    test_c = [dist["test"][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, train_c, width, label=f"Train ({meta['train_rows']})", color="#2b5c8f")
    ax.bar(x, val_c, width, label=f"Val ({meta['val_rows']})", color="#e07a5f")
    ax.bar(x + width, test_c, width, label=f"Test ({meta['test_rows']})", color="#81b29a")

    ax.set_ylabel("Count")
    ax.set_title("Figure F1: Five-Class Distribution Across Grouped Stratified Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, rotation=15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f1_class_distribution.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f1_class_distribution.pdf"))
    plt.close(fig)
    print("  Generated Figure F1: class distribution across splits.")


def generate_figure_f2_confusion_matrices(runs: list, out_dir: str):
    """Figure F2: Six 5x5 confusion matrices for primary configurations C1-C6."""
    configs = ["C1", "C2", "C3", "C4", "C5", "C6"]
    by_cfg = {r["config_id"]: r for r in runs if r.get("seed", 1) == 1}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        run_data = by_cfg.get(cfg)
        if not run_data:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            continue

        cm = np.array(run_data["confusion_matrix_normalized"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1.0)
        cfg_info = PRIMARY_CONFIGS[cfg]
        ax.set_title(f"{cfg}: {cfg_info['model'].upper()} + {cfg_info['loss'].upper()}")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([f"C{i}" for i in range(5)])
        ax.set_yticklabels([f"C{i}" for i in range(5)])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(5):
            for j in range(5):
                val = cm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=8)

    plt.suptitle("Figure F2: Row-Normalized 5x5 Confusion Matrices (C1–C6, Seed 1)", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f2_confusion_matrices.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f2_confusion_matrices.pdf"))
    plt.close(fig)
    print("  Generated Figure F2: confusion matrices.")


def generate_figure_f3_minority_recall_support(runs: list, out_dir: str):
    """Figure F3: Per-class recall across loss functions & predicted count vs true support."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Recall for classes 0, 1 across C1-C6
    by_cfg = defaultdict(list)
    for r in runs:
        if r.get("config_id") in PRIMARY_CONFIGS:
            by_cfg[r["config_id"]].append(r)

    cfgs = ["C1", "C2", "C3", "C4", "C5", "C6"]
    cl0_rec = [np.mean([x["per_class"]["strongly_negative"]["recall"] for x in by_cfg[c]]) for c in cfgs]
    cl1_rec = [np.mean([x["per_class"]["weakly_negative"]["recall"] for x in by_cfg[c]]) for c in cfgs]

    x = np.arange(len(cfgs))
    width = 0.35
    ax1.bar(x - width/2, cl0_rec, width, label="Strongly Neg (Class 0)", color="#d95f02")
    ax1.bar(x + width/2, cl1_rec, width, label="Weakly Neg (Class 1)", color="#7570b3")
    ax1.set_ylabel("Mean Recall")
    ax1.set_title("Minority-Class Recall Recovery by Loss Function")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Predicted vs True Support (C1 vs C2)
    c1_runs = by_cfg.get("C1", [])
    c2_runs = by_cfg.get("C2", [])
    if c1_runs and c2_runs:
        c1 = c1_runs[0]
        c2 = c2_runs[0]
        true_supp = [c1["per_class"][ID2LABEL[c]]["support"] for c in range(5)]
        c1_pred = [c1["per_class"][ID2LABEL[c]]["predicted_count"] for c in range(5)]
        c2_pred = [c2["per_class"][ID2LABEL[c]]["predicted_count"] for c in range(5)]

        x5 = np.arange(5)
        w5 = 0.25
        ax2.bar(x5 - w5, true_supp, w5, label="True Support", color="#333333")
        ax2.bar(x5, c1_pred, w5, label="Pred Count (C1: Std CE)", color="#e07a5f")
        ax2.bar(x5 + w5, c2_pred, w5, label="Pred Count (C2: Wtd CE)", color="#2b5c8f")
        ax2.set_title("Predicted Count vs True Support (MuRIL)")
        ax2.set_xticks(x5)
        ax2.set_xticklabels([f"C{i}" for i in range(5)])
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Figure F3: Minority-Class Recall and Prediction Ratios", fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f3_recall_and_support.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f3_recall_and_support.pdf"))
    plt.close(fig)
    print("  Generated Figure F3: minority recall and support.")


def generate_figure_f4_macro_f1_seeds(runs: list, out_dir: str):
    """Figure F4: Macro-F1 across six configurations with per-seed points."""
    by_cfg = defaultdict(list)
    for r in runs:
        if r.get("config_id") in PRIMARY_CONFIGS:
            by_cfg[r["config_id"]].append(r)

    cfgs = ["C1", "C2", "C3", "C4", "C5", "C6"]
    means = [np.mean([x["macro_f1"] for x in by_cfg[c]]) for c in cfgs]
    sds = [np.std([x["macro_f1"] for x in by_cfg[c]]) for c in cfgs]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cfgs))
    ax.bar(x, means, yerr=sds, capsize=5, color="#457b9d", alpha=0.7, label="Mean Macro-F1")

    # Plot individual seed points
    for i, c in enumerate(cfgs):
        pts = [x["macro_f1"] for x in by_cfg[c]]
        ax.scatter([i] * len(pts), pts, color="#e63946", zorder=5, s=30)

    ax.set_ylabel("Macro-F1 (5 Classes)")
    ax.set_title("Figure F4: Macro-F1 Across Primary Configurations with Per-Seed Points")
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f4_macro_f1_seeds.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f4_macro_f1_seeds.pdf"))
    plt.close(fig)
    print("  Generated Figure F4: macro-F1 per-seed spread.")


def generate_figure_f6_translation_comparison(runs: list, out_dir: str):
    """Figure F6: Original Hinglish vs Translated English Macro-F1 and recall comparison."""
    by_cfg = defaultdict(list)
    for r in runs:
        by_cfg[r.get("config_id", "")].append(r)

    pairs = [("C1", "T1"), ("C3", "T3"), ("C4", "T4"), ("C6", "T6")]
    c_f1 = []
    t_f1 = []
    labels = []

    for c_id, t_id in pairs:
        c_runs = by_cfg.get(c_id, [])
        t_runs = by_cfg.get(t_id, [])
        if c_runs and t_runs:
            c_f1.append(np.mean([x["macro_f1"] for x in c_runs]))
            t_f1.append(np.mean([x["macro_f1"] for x in t_runs]))
            labels.append(f"{c_id} / {t_id}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, c_f1, width, label="Original Hinglish (C)", color="#2a9d8f")
    ax.bar(x + width/2, t_f1, width, label="English Translation (T)", color="#e76f51")

    ax.set_ylabel("Mean Macro-F1")
    ax.set_title("Figure F6: Original Hinglish vs English Translation Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f6_translation_comparison.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f6_translation_comparison.pdf"))
    plt.close(fig)
    print("  Generated Figure F6: translation comparison.")


def generate_figure_f7_error_categories(runs: list, out_dir: str):
    """Figure F7: Confusion-category shares across configurations."""
    c1_runs = [r for r in runs if r.get("config_id") == "C1"]
    if not c1_runs: return
    c1 = c1_runs[0]
    et = c1["error_taxonomy"]

    cats = list(et.keys())
    shares = [et[c]["share_of_errors"] for c in cats]
    display_names = [c.replace("_", " ").title() for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(display_names, shares, color="#6a4c93")
    ax.set_xlabel("Share of Total Off-Diagonal Errors (%)")
    ax.set_title("Figure F7: Error Category Distribution (C1: MuRIL Standard CE)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, s in zip(bars, shares):
        ax.text(s + 0.5, bar.get_y() + bar.get_height()/2, f"{s:.1f}%", va="center")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_f7_error_categories.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, "figure_f7_error_categories.pdf"))
    plt.close(fig)
    print("  Generated Figure F7: error categories distribution.")


def main():
    metrics_dir = os.path.join("results", "metrics")
    dist_json = os.path.join("tables", "table_r2_final_five_class_splits.json")
    out_dir = os.path.join("figures")
    os.makedirs(out_dir, exist_ok=True)

    runs = load_all_metrics(metrics_dir)
    if not runs:
        print("[generate_figures] No metrics found in results/metrics/. Run experiments first.")
        return

    print(f"[generate_figures] Generating publication figures from {len(runs)} run records...")
    generate_figure_f1_distribution(dist_json, out_dir)
    generate_figure_f2_confusion_matrices(runs, out_dir)
    generate_figure_f3_minority_recall_support(runs, out_dir)
    generate_figure_f4_macro_f1_seeds(runs, out_dir)
    generate_figure_f6_translation_comparison(runs, out_dir)
    generate_figure_f7_error_categories(runs, out_dir)

    print("[generate_figures] All publication figures generated successfully in figures/.")

if __name__ == "__main__":
    main()
