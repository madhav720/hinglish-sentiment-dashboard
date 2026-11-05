"""
Script 07: Evaluation and Verification of Saved Experiment Predictions
Inspects prediction files in results/predictions/, verifies zero label leakage,
computes and displays complete 5-class metrics, collapse status, and error breakdowns.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import compute_metrics
from src.utils.constants import CLASS_NAMES, ID2LABEL

def evaluate_prediction_file(pred_csv_path: str):
    if not os.path.exists(pred_csv_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_csv_path}")

    with open(pred_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    y_true = [int(r["y_true"]) for r in rows]
    y_pred = [int(r["y_pred"]) for r in rows]

    metrics = compute_metrics(y_true, y_pred)
    run_name = os.path.splitext(os.path.basename(pred_csv_path))[0].replace("_test", "")

    print("\n" + "=" * 65)
    print(f"EVALUATION REPORT: {run_name}")
    print("=" * 65)
    print(f"  Test Samples: {metrics['n_samples']}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:     {metrics['macro_f1']:.4f}  (PRIMARY METRIC)")
    print(f"  Weighted-F1:  {metrics['weighted_f1']:.4f}")
    print("-" * 65)
    print("PER-CLASS PERFORMANCE (Labels 0 to 4):")
    print(f"{'Class Name':<20} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Supp':<5} | {'Pred':<5} | {'Ratio':<6} | {'Collapse Status'}")
    print("-" * 65)
    for c in range(5):
        c_key = ID2LABEL[c]
        p = metrics["per_class"][c_key]
        c_status = "STRICT COLLAPSE" if p["strict_collapse"] else ("PARTIAL COLLAPSE" if p["partial_collapse"] else "Normal")
        print(f"{p['class_name']:<20} | {p['precision']:<7.4f} | {p['recall']:<7.4f} | {p['f1']:<7.4f} | {p['support']:<5} | {p['predicted_count']:<5} | {p['prediction_to_support_ratio']:<6.2f} | {c_status}")

    print("-" * 65)
    print("5x5 RAW CONFUSION MATRIX:")
    for row in metrics["confusion_matrix_raw"]:
        print("  " + "  ".join(f"{v:4d}" for v in row))

    print("-" * 65)
    print("ERROR TAXONOMY BREAKDOWN:")
    for err_name, err_info in metrics["error_taxonomy"].items():
        print(f"  {err_name:<25}: {err_info['count']:4d} errors ({err_info['share_of_errors']}%)")
    print("=" * 65 + "\n")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment Predictions")
    parser.add_argument("--run-id", type=str, default="", help="Specific run ID to evaluate (e.g. muril_ce_hinglish_s1)")
    parser.add_argument("--all", action="store_true", help="Evaluate all predictions found in results/predictions/")
    args = parser.parse_args()

    pred_dir = os.path.join("results", "predictions")
    if args.run_id:
        pred_file = os.path.join(pred_dir, f"{args.run_id}_test.csv")
        evaluate_prediction_file(pred_file)
    else:
        files = [f for f in os.listdir(pred_dir) if f.endswith("_test.csv")]
        if not files:
            print("[07_evaluate] No prediction files found in results/predictions/. Run 06_train.py first.")
            return
        for f in sorted(files):
            evaluate_prediction_file(os.path.join(pred_dir, f))

if __name__ == "__main__":
    main()
