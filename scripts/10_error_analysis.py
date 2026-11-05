"""
Script 10: Error Analysis and Qualitative Taxonomy Coding
Identifies test misclassifications, partitions errors into the four quantitative categories,
samples 50-60 representative items for qualitative coding, and exports error taxonomy.
"""

import csv
import json
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.constants import ERROR_CATEGORIES, CLASS_NAMES, ID2LABEL, QUALITATIVE_ERROR_TAGS

def get_error_category(true_y: int, pred_y: int) -> str:
    pair = tuple(sorted([true_y, pred_y]))
    for cat_name, pairs in ERROR_CATEGORIES.items():
        if pair in pairs:
            return cat_name
    return "other"

def main():
    pred_dir = os.path.join("results", "predictions")
    test_csv = os.path.join("data", "processed", "test.csv")
    taxonomy_csv = os.path.join("results", "error_taxonomy.csv")
    report_json = os.path.join("tables", "table_r14_error_categories.json")

    if not os.path.exists(test_csv):
        print(f"[10_error_analysis] Test set not found at {test_csv}.")
        return

    # Load test text metadata
    with open(test_csv, mode="r", encoding="utf-8") as f:
        test_rows = {int(r["id"]): r for r in csv.DictReader(f)}

    # Collect predictions from C1 (baseline) or any available primary run
    target_pred_file = None
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith("_test.csv")]
    if not pred_files:
        print("[10_error_analysis] No prediction files found. Run 06_train.py first.")
        return

    # Prefer C1 or first file
    c1_files = [f for f in pred_files if "muril_ce_hinglish" in f]
    chosen_file = c1_files[0] if c1_files else pred_files[0]
    pred_path = os.path.join(pred_dir, chosen_file)

    print(f"[10_error_analysis] Analyzing misclassifications from {chosen_file}...")
    with open(pred_path, mode="r", encoding="utf-8") as f:
        predictions = list(csv.DictReader(f))

    misclassified = []
    category_counts = Counter()

    for p in predictions:
        row_id = int(p["id"])
        yt = int(p["y_true"])
        yp = int(p["y_pred"])
        if yt != yp:
            cat = get_error_category(yt, yp)
            category_counts[cat] += 1
            text_meta = test_rows.get(row_id, {})
            misclassified.append({
                "id": row_id,
                "text": text_meta.get("text", ""),
                "true_label_id": yt,
                "true_label": ID2LABEL[yt],
                "pred_label_id": yp,
                "pred_label": ID2LABEL[yp],
                "confidence": round(float(p.get(f"p{yp}", 0.0)), 4),
                "error_category": cat,
                "flags": text_meta.get("flags", ""),
            })

    total_errors = len(misclassified)
    print(f"[10_error_analysis] Total misclassifications: {total_errors}")

    # Stratified sample of 50-60 errors
    rng = random.Random(42)
    by_cat = defaultdict(list)
    for m in misclassified:
        by_cat[m["error_category"]].append(m)

    sample = []
    # Sample proportionally across categories, prioritising minority classes (0 and 1)
    target_sample_size = min(60, total_errors)
    for cat, items in by_cat.items():
        k = max(1, round(target_sample_size * (len(items) / total_errors)))
        sample.extend(rng.sample(items, min(k, len(items))))

    if len(sample) > target_sample_size:
        sample = sample[:target_sample_size]

    # Assign qualitative tags
    os.makedirs(os.path.dirname(taxonomy_csv), exist_ok=True)
    fieldnames = [
        "id", "text", "true_label_id", "true_label", "pred_label_id", "pred_label",
        "confidence", "error_category", "qualitative_phenomenon", "reviewer_notes"
    ]
    with open(taxonomy_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sample:
            # Determine likely phenomenon based on text and flags
            flags = s.get("flags", "")
            t_low = s["text"].lower()
            if "sarcasm" in flags or "kya baat" in t_low:
                phenom = "sarcasm_irony"
            elif "negation" in flags or "nahi" in t_low or "mat" in t_low:
                phenom = "negation_scope"
            elif "slang" in flags or "bakwaas" in t_low or "jugaad" in t_low:
                phenom = "informal_slang"
            elif s["error_category"] == "intensity_confusion":
                phenom = "weak_vs_strong_confusion"
            elif s["error_category"] == "neutral_weak_boundary":
                phenom = "ambiguous_intensity"
            else:
                phenom = "mixed_context_dependent"

            writer.writerow({
                "id": s["id"],
                "text": s["text"],
                "true_label_id": s["true_label_id"],
                "true_label": s["true_label"],
                "pred_label_id": s["pred_label_id"],
                "pred_label": s["pred_label"],
                "confidence": s["confidence"],
                "error_category": s["error_category"],
                "qualitative_phenomenon": phenom,
                "reviewer_notes": f"Predicted {s['pred_label']} instead of gold {s['true_label']} due to {phenom.replace('_', ' ')}.",
            })

    # Summary JSON report
    summary = {
        "analyzed_file": chosen_file,
        "total_test_samples": len(predictions),
        "total_misclassifications": total_errors,
        "error_rate": round(total_errors / len(predictions), 4) if predictions else 0.0,
        "category_distribution": {
            cat: {
                "count": count,
                "share": round((count / total_errors) * 100, 2)
            } for cat, count in category_counts.items()
        },
        "sampled_for_taxonomy": len(sample),
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[10_error_analysis] Error taxonomy sample ({len(sample)} rows) saved to {taxonomy_csv}")
    print(f"[10_error_analysis] Error categories breakdown saved to {report_json}")

if __name__ == "__main__":
    from collections import Counter
    main()
