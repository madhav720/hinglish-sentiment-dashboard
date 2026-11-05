"""
Script 08: Results Aggregation and Master Table Generation
Aggregates all JSON metrics from results/metrics/ into unified master tables,
per-seed spreads, and publication-ready Markdown/CSV tables (Tables R3 through R11).
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.constants import (
    PRIMARY_CONFIGS,
    TRANSLATION_CONFIGS,
    CLASS_NAMES,
    ID2LABEL,
)


def compute_mean_sd(values: List[float]) -> Tuple[float, float]:
    """Compute mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return round(mean, 4), 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return round(mean, 4), round(math.sqrt(var), 4)


def main():
    metrics_dir = os.path.join("results", "metrics")
    tables_dir = os.path.join("tables")
    os.makedirs(tables_dir, exist_ok=True)

    json_files = [f for f in os.listdir(metrics_dir) if f.endswith(".json")]
    if not json_files:
        print("[08_aggregate] No metric JSON files found in results/metrics/. Run 06_train.py first.")
        return

    runs = []
    for jf in json_files:
        with open(os.path.join(metrics_dir, jf), "r", encoding="utf-8") as f:
            runs.append(json.load(f))

    print(f"[08_aggregate] Loaded {len(runs)} run result(s). Aggregating master tables...")

    # Group runs by config_id
    by_config = defaultdict(list)
    for r in runs:
        cfg = r.get("config_id", "unknown")
        by_config[cfg].append(r)

    # 1. Master Results CSV (all runs individually)
    master_csv = os.path.join("results", "master_results.csv")
    fieldnames = [
        "run_id", "config_id", "model_key", "loss_type", "text_type", "seed",
        "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
        "cl0_recall", "cl0_f1", "cl0_pred_ratio", "cl0_strict_collapse", "cl0_partial_collapse",
        "cl1_recall", "cl1_f1", "cl1_pred_ratio", "cl1_strict_collapse", "cl1_partial_collapse",
        "cl2_recall", "cl2_f1",
        "cl3_recall", "cl3_f1",
        "cl4_recall", "cl4_f1",
    ]
    with open(master_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row_dict = {
                "run_id": r["run_id"],
                "config_id": r.get("config_id", ""),
                "model_key": r.get("model_key", ""),
                "loss_type": r.get("loss_type", ""),
                "text_type": r.get("text_type", ""),
                "seed": r.get("seed", 1),
                "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "cl0_recall": r["per_class"]["strongly_negative"]["recall"],
                "cl0_f1": r["per_class"]["strongly_negative"]["f1"],
                "cl0_pred_ratio": r["per_class"]["strongly_negative"]["prediction_to_support_ratio"],
                "cl0_strict_collapse": 1 if r["per_class"]["strongly_negative"]["strict_collapse"] else 0,
                "cl0_partial_collapse": 1 if r["per_class"]["strongly_negative"]["partial_collapse"] else 0,
                "cl1_recall": r["per_class"]["weakly_negative"]["recall"],
                "cl1_f1": r["per_class"]["weakly_negative"]["f1"],
                "cl1_pred_ratio": r["per_class"]["weakly_negative"]["prediction_to_support_ratio"],
                "cl1_strict_collapse": 1 if r["per_class"]["weakly_negative"]["strict_collapse"] else 0,
                "cl1_partial_collapse": 1 if r["per_class"]["weakly_negative"]["partial_collapse"] else 0,
                "cl2_recall": r["per_class"]["neutral"]["recall"],
                "cl2_f1": r["per_class"]["neutral"]["f1"],
                "cl3_recall": r["per_class"]["weakly_positive"]["recall"],
                "cl3_f1": r["per_class"]["weakly_positive"]["f1"],
                "cl4_recall": r["per_class"]["strongly_positive"]["recall"],
                "cl4_f1": r["per_class"]["strongly_positive"]["f1"],
            }
            writer.writerow(row_dict)
    print(f"[08_aggregate] Master results saved to {master_csv}")

    # 2. Table R6: Primary Six Configurations Summary (Mean +- SD over seeds)
    t6_csv = os.path.join(tables_dir, "table_r6_all_six_configs.csv")
    with open(t6_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Model", "Loss", "Seeds_Count", "Macro_F1_Mean", "Macro_F1_SD", "Accuracy_Mean", "Accuracy_SD", "Weighted_F1_Mean", "Weighted_F1_SD"])
        for cfg in ["C1", "C2", "C3", "C4", "C5", "C6"]:
            cfg_runs = by_config.get(cfg, [])
            if cfg_runs:
                m_info = PRIMARY_CONFIGS[cfg]
                f1_mean, f1_sd = compute_mean_sd([x["macro_f1"] for x in cfg_runs])
                acc_mean, acc_sd = compute_mean_sd([x["accuracy"] for x in cfg_runs])
                wf1_mean, wf1_sd = compute_mean_sd([x["weighted_f1"] for x in cfg_runs])
                writer.writerow([cfg, m_info["model"], m_info["loss"], len(cfg_runs), f1_mean, f1_sd, acc_mean, acc_sd, wf1_mean, wf1_sd])
    print(f"[08_aggregate] Table R6 saved to {t6_csv}")

    # 3. Table R8: Minority-Class Performance Comparison
    t8_csv = os.path.join(tables_dir, "table_r8_minority_collapse_summary.csv")
    with open(t8_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Model", "Loss", "Cl0_Recall_Mean", "Cl0_F1_Mean", "Cl0_Pred_Ratio", "Cl1_Recall_Mean", "Cl1_F1_Mean", "Cl1_Pred_Ratio", "Strict_Collapse_Seeds", "Partial_Collapse_Seeds"])
        for cfg in ["C1", "C2", "C3", "C4", "C5", "C6", "T1", "T3", "T4", "T6"]:
            cfg_runs = by_config.get(cfg, [])
            if cfg_runs:
                m_info = PRIMARY_CONFIGS.get(cfg, TRANSLATION_CONFIGS.get(cfg, {}))
                cl0_rec, _ = compute_mean_sd([x["per_class"]["strongly_negative"]["recall"] for x in cfg_runs])
                cl0_f1, _ = compute_mean_sd([x["per_class"]["strongly_negative"]["f1"] for x in cfg_runs])
                cl0_pr, _ = compute_mean_sd([x["per_class"]["strongly_negative"]["prediction_to_support_ratio"] for x in cfg_runs])
                cl1_rec, _ = compute_mean_sd([x["per_class"]["weakly_negative"]["recall"] for x in cfg_runs])
                cl1_f1, _ = compute_mean_sd([x["per_class"]["weakly_negative"]["f1"] for x in cfg_runs])
                cl1_pr, _ = compute_mean_sd([x["per_class"]["weakly_negative"]["prediction_to_support_ratio"] for x in cfg_runs])
                strict_seeds = sum(1 for x in cfg_runs if x["per_class"]["strongly_negative"]["strict_collapse"])
                partial_seeds = sum(1 for x in cfg_runs if x["per_class"]["strongly_negative"]["partial_collapse"] or x["per_class"]["weakly_negative"]["partial_collapse"])
                writer.writerow([cfg, m_info.get("model", ""), m_info.get("loss", ""), cl0_rec, cl0_f1, cl0_pr, cl1_rec, cl1_f1, cl1_pr, f"{strict_seeds}/{len(cfg_runs)}", f"{partial_seeds}/{len(cfg_runs)}"])
    print(f"[08_aggregate] Table R8 saved to {t8_csv}")

    # 4. Table R11: Translation Comparison (C vs T)
    t11_csv = os.path.join(tables_dir, "table_r11_translation_comparison.csv")
    with open(t11_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Comparison", "Model", "Loss", "Macro_F1_Hinglish_C", "Macro_F1_English_T", "Delta_Macro_F1", "Accuracy_C", "Accuracy_T", "Delta_Accuracy"])
        pairs = [("C1", "T1"), ("C3", "T3"), ("C4", "T4"), ("C6", "T6")]
        for c_cfg, t_cfg in pairs:
            c_runs = by_config.get(c_cfg, [])
            t_runs = by_config.get(t_cfg, [])
            if c_runs and t_runs:
                c_f1, _ = compute_mean_sd([x["macro_f1"] for x in c_runs])
                t_f1, _ = compute_mean_sd([x["macro_f1"] for x in t_runs])
                c_acc, _ = compute_mean_sd([x["accuracy"] for x in c_runs])
                t_acc, _ = compute_mean_sd([x["accuracy"] for x in t_runs])
                delta_f1 = round(t_f1 - c_f1, 4)
                delta_acc = round(t_acc - c_acc, 4)
                m_info = PRIMARY_CONFIGS[c_cfg]
                writer.writerow([f"{c_cfg} vs {t_cfg}", m_info["model"], m_info["loss"], c_f1, t_f1, delta_f1, c_acc, t_acc, delta_acc])
    print(f"[08_aggregate] Table R11 saved to {t11_csv}")

    print("\nMaster tables aggregation complete.")

if __name__ == "__main__":
    main()
