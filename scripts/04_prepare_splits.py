"""
Script 04: Prepare Reproducible Grouped Stratified Splits
Splits validated 5-class dataset into 70% train, 15% validation, and 15% test.
Enforces zero ID and text leakage across near-duplicate clusters.
Calculates class weights strictly from the training split only.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splitting import grouped_stratified_split
from src.data.audit import normalize_text
from src.utils.constants import CLASS_NAMES, ID2LABEL

def save_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def main():
    input_csv = os.path.join("data", "processed", "hinglish_5class.csv")
    train_csv = os.path.join("data", "processed", "train.csv")
    val_csv = os.path.join("data", "processed", "val.csv")
    test_csv = os.path.join("data", "processed", "test.csv")
    hash_txt = os.path.join("data", "processed", "split_hash.txt")
    dist_csv = os.path.join("tables", "class_distribution.csv")
    dist_json = os.path.join("tables", "table_r2_final_five_class_splits.json")
    report_md = os.path.join("data", "processed", "split_verification_report.md")

    print(f"[04_prepare_splits] Loading validated dataset from {input_csv}...")
    with open(input_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Convert numeric fields
    for r in rows:
        r["id"] = int(r["id"])
        r["group_id"] = int(r["group_id"])
        r["label_id"] = int(r["label_id"])
        r["norm_text"] = normalize_text(r["text"])

    print(f"[04_prepare_splits] Performing grouped stratified 70/15/15 split (seed 42)...")
    train_rows, val_rows, test_rows, meta = grouped_stratified_split(rows, seed=42)

    fieldnames = ["id", "group_id", "text", "label_id", "label", "orig_label", "orig_sentiment", "flags"]
    save_csv(train_rows, train_csv, fieldnames)
    save_csv(val_rows, val_csv, fieldnames)
    save_csv(test_rows, test_csv, fieldnames)

    # Save split hash
    with open(hash_txt, "w", encoding="utf-8") as f:
        f.write(meta["split_hash"] + "\n")

    # Save class distribution CSV
    dist = meta["class_distribution"]
    os.makedirs(os.path.dirname(dist_csv), exist_ok=True)
    with open(dist_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "train_count", "val_count", "test_count", "total_count", "class_weight_train_only"])
        for c in range(5):
            c_name = ID2LABEL[c]
            tr_c = dist["train"][c_name]
            va_c = dist["val"][c_name]
            te_c = dist["test"][c_name]
            tot_c = tr_c + va_c + te_c
            w = meta["class_weights_train_only"][c]
            writer.writerow([c, c_name, tr_c, va_c, te_c, tot_c, w])

    with open(dist_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Markdown verification report
    report = f"""# Split Verification and Leakage Audit Report

**Split Algorithm:** Grouped Stratified Split (Seed 42)  
**Split Hash (SHA-256):** `{meta['split_hash']}`  
**Leakage Status:** **{meta['leakage_verification']['status']}**

---

## 1. Split Sizes and Proportions

| Split | Rows | Share (%) | Target Share (%) |
|---|---|---|---|
| **Train** | {meta['train_rows']} | {meta['train_share']}% | 70.0% |
| **Validation** | {meta['val_rows']} | {meta['val_share']}% | 15.0% |
| **Test** | {meta['test_rows']} | {meta['test_share']}% | 15.0% |
| **Total** | **{meta['total_rows']}** | **100.0%** | **100.0%** |

---

## 2. Class Distribution Across Splits

| Class ID | Label Name | Train ($n={meta['train_rows']}$) | Val ($n={meta['val_rows']}$) | Test ($n={meta['test_rows']}$) | Total ($n={meta['total_rows']}$) | Train Class Weight |
|---|---|---|---|---|---|---|
| 0 | `strongly_negative` | {dist['train']['strongly_negative']} | {dist['val']['strongly_negative']} | {dist['test']['strongly_negative']} | {dist['train']['strongly_negative'] + dist['val']['strongly_negative'] + dist['test']['strongly_negative']} | **{meta['class_weights_train_only'][0]}** |
| 1 | `weakly_negative` | {dist['train']['weakly_negative']} | {dist['val']['weakly_negative']} | {dist['test']['weakly_negative']} | {dist['train']['weakly_negative'] + dist['val']['weakly_negative'] + dist['test']['weakly_negative']} | **{meta['class_weights_train_only'][1]}** |
| 2 | `neutral` | {dist['train']['neutral']} | {dist['val']['neutral']} | {dist['test']['neutral']} | {dist['train']['neutral'] + dist['val']['neutral'] + dist['test']['neutral']} | **{meta['class_weights_train_only'][2]}** |
| 3 | `weakly_positive` | {dist['train']['weakly_positive']} | {dist['val']['weakly_positive']} | {dist['test']['weakly_positive']} | {dist['train']['weakly_positive'] + dist['val']['weakly_positive'] + dist['test']['weakly_positive']} | **{meta['class_weights_train_only'][3]}** |
| 4 | `strongly_positive` | {dist['train']['strongly_positive']} | {dist['val']['strongly_positive']} | {dist['test']['strongly_positive']} | {dist['train']['strongly_positive'] + dist['val']['strongly_positive'] + dist['test']['strongly_positive']} | **{meta['class_weights_train_only'][4]}** |

---

## 3. Leakage Guard Checks

1. **Row ID Disjointness:** Asserted $0$ ID overlap across Train, Val, and Test sets.
2. **Normalized Text Disjointness:** Asserted $0$ normalized text overlap across splits.
3. **Group ID Clustering:** All near-duplicates assigned the same `group_id` reside strictly within a single split.
4. **Class Weight Formula:** Computed strictly on training split as $w_c = \\frac{{N_{{train}}}}{{5 \\cdot n_c}}$.
"""
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[04_prepare_splits] Splits generated and saved:")
    print(f"  Train: {train_csv} ({meta['train_rows']} rows)")
    print(f"  Val:   {val_csv} ({meta['val_rows']} rows)")
    print(f"  Test:  {test_csv} ({meta['test_rows']} rows)")
    print(f"  Report: {report_md}")
    print(f"  Class Weights (train only): {meta['class_weights_train_only']}")

if __name__ == "__main__":
    main()
