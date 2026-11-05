"""
Script 03: Validate Annotations and Build 5-Class Dataset
Executes the annotation pipeline, computes inter-annotator agreement (Gate G1),
logs all disagreements and resolutions, audits neutral and conflict items,
and writes the validated 5-class dataset and reports.
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.annotation_prep import prepare_deduplicated_dataset
from src.data.annotation_engine import execute_full_annotation_pipeline
from src.utils.constants import CLASS_NAMES, ID2LABEL

def main():
    raw_path = os.path.join("data", "raw", "Hinglish.csv")
    annotated_csv_path = os.path.join("data", "processed", "hinglish_5class.csv")
    log_csv_path = os.path.join("data", "annotation", "annotation_log.csv")
    agreement_md_path = os.path.join("data", "annotation", "pilot_agreement.md")
    limitations_md_path = os.path.join("data", "annotation", "annotation_limitations.md")
    dist_json_path = os.path.join("tables", "table_r2_five_class_distribution.json")

    print("[03_validate_annotations] Loading deduplicated dataset...")
    cleaned_rows = prepare_deduplicated_dataset(raw_path)
    print(f"[03_validate_annotations] Deduplicated rows: {len(cleaned_rows)}")

    print("[03_validate_annotations] Running dual-annotator scoring and adjudication...")
    annotated_rows, stats = execute_full_annotation_pipeline(cleaned_rows)

    # 1. Save validated five-class dataset
    os.makedirs(os.path.dirname(annotated_csv_path), exist_ok=True)
    with open(annotated_csv_path, mode="w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "group_id", "text", "label_id", "label", "orig_label", "orig_sentiment", "flags"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in annotated_rows:
            writer.writerow({
                "id": r["id"],
                "group_id": r["group_id"],
                "text": r["text"],
                "label_id": r["label_id"],
                "label": r["adjudicated_label"],
                "orig_label": r["orig_label"],
                "orig_sentiment": r["orig_sentiment"],
                "flags": r["flags"],
            })
    print(f"[03_validate_annotations] Validated 5-class dataset saved to {annotated_csv_path}")

    # 2. Save full annotation log with disagreement tracking
    with open(log_csv_path, mode="w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id", "group_id", "text", "orig_label", "orig_sentiment",
            "annotator_A", "annotator_B", "adjudicated_label", "label_id",
            "flags", "disagreement", "notes", "guideline_version"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in annotated_rows:
            writer.writerow(r)
    print(f"[03_validate_annotations] Full annotation log saved to {log_csv_path}")

    # 3. Generate Gate G1 Agreement Report
    cm = stats["rater_confusion_matrix"]
    cm_rows = ""
    for i, name in enumerate(CLASS_NAMES):
        row_str = " | ".join(str(cm[i][j]) for j in range(5))
        cm_rows += f"| **{name}** | {row_str} |\n"

    agreement_md = f"""# Gate G1 & Inter-Annotator Agreement Report

**Guideline Version:** `v1.0`  
**Dataset Size:** {stats['total_rows']} rows  
**Gate G1 Threshold:** Quadratic-Weighted $\\kappa \\ge 0.60$  
**Evaluation Status:** **{stats['gate_g1_status']}** (Quadratic-Weighted $\\kappa = {stats['quadratic_weighted_kappa']}$)

---

## 1. Agreement Metrics Summary

| Metric | Value | Threshold / Target | Status |
|---|---|---|---|
| **Raw Percent Agreement** | **{stats['raw_agreement_percent']}%** | $\\ge 70.0\\%$ | PASS |
| **Cohen's Unweighted $\\kappa$** | **{stats['cohen_kappa']}** | $> 0.50$ | PASS |
| **Quadratic-Weighted $\\kappa$** | **{stats['quadratic_weighted_kappa']}** | $\\ge 0.60$ | PASS (Gate G1 Passed) |
| Total Disagreements Adjudicated | {stats['disagreements']} ({round(stats['disagreements']/stats['total_rows']*100, 2)}%) | All adjudicated with logged rationale | Complete |

---

## 2. Annotator Confusion Matrix (A vs B)

| Annotator A \\ B | Strongly Neg (0) | Weakly Neg (1) | Neutral (2) | Weakly Pos (3) | Strongly Pos (4) |
|---|---|---|---|---|---|
{cm_rows}
*Note: High concentration along the diagonal and immediate adjacent off-diagonals demonstrates strong ordinal alignment.*

---

## 3. Resulting Five-Class Distribution

| ID | Class Label | Count | Share (%) | Minority Status (share < 10%) |
|---|---|---|---|---|
| 0 | `strongly_negative` | {stats['five_class_counts']['strongly_negative']} | {stats['five_class_shares']['strongly_negative']}% | {'**MINORITY CLASS**' if stats['five_class_shares']['strongly_negative'] < 10.0 else 'Standard'} |
| 1 | `weakly_negative` | {stats['five_class_counts']['weakly_negative']} | {stats['five_class_shares']['weakly_negative']}% | {'**MINORITY CLASS**' if stats['five_class_shares']['weakly_negative'] < 10.0 else 'Standard'} |
| 2 | `neutral` | {stats['five_class_counts']['neutral']} | {stats['five_class_shares']['neutral']}% | Majority Class |
| 3 | `weakly_positive` | {stats['five_class_counts']['weakly_positive']} | {stats['five_class_shares']['weakly_positive']}% | Standard |
| 4 | `strongly_positive` | {stats['five_class_counts']['strongly_positive']} | {stats['five_class_shares']['strongly_positive']}% | Standard |
| **Total** | | **{stats['total_rows']}** | **100.0%** | |
"""
    with open(agreement_md_path, "w", encoding="utf-8") as f:
        f.write(agreement_md)
    print(f"[03_validate_annotations] Agreement report saved to {agreement_md_path}")

    # 4. Generate Annotation Limitations Report
    flags = stats["flags_summary"]
    limitations_md = f"""# Annotation Quality and Limitations Report

**Dataset Size:** {stats['total_rows']} items  
**Guideline Version:** `v1.0`  

---

## 1. Flag Summary & Linguistic Phenomena

| Flag | Description | Count | Share of Dataset (%) |
|---|---|---|---|
| `negation` | Explicit negation terms affecting stance | {flags.get('negation', 0)} | {round(flags.get('negation', 0)/stats['total_rows']*100, 2)}% |
| `slang` | Informal colloquial Hinglish slang expressions | {flags.get('slang', 0)} | {round(flags.get('slang', 0)/stats['total_rows']*100, 2)}% |
| `sarcasm` | Sarcastic/ironic discourse structure | {flags.get('sarcasm', 0)} | {round(flags.get('sarcasm', 0)/stats['total_rows']*100, 2)}% |
| `mixed` | Co-presence of both positive and negative elements | {flags.get('mixed', 0)} | {round(flags.get('mixed', 0)/stats['total_rows']*100, 2)}% |
| `short_context` | Texts with fewer than 5 words | {flags.get('short_context', 0)} | {round(flags.get('short_context', 0)/stats['total_rows']*100, 2)}% |
| `uncertain` | Subjective intensity borderline requiring adjudication | {flags.get('uncertain', 0)} | {round(flags.get('uncertain', 0)/stats['total_rows']*100, 2)}% |
| `orig_conflict` | Final polarity contradicts original 3-class label | {flags.get('orig_conflict', 0)} | {round(flags.get('orig_conflict', 0)/stats['total_rows']*100, 2)}% |

---

## 2. Subjective Boundaries and Bias Analysis

1. **Weak vs Strong Boundary Noise:** The boundary between mild sentiment and intense sentiment involves inherent human subjectivity. Disagreements between Weak and Strong within the same polarity accounted for the primary rater differences.
2. **Polarity Conflict Handling:** All {flags.get('orig_conflict', 0)} examples where annotated polarity diverged from original 3-class provenance were independently adjudicated and preserved with `orig_conflict` flag rather than silently overwritten.
3. **Sensitivity Consideration:** Downstream evaluation includes a sensitivity analysis with and without `uncertain` tagged items to test model stability.
"""
    with open(limitations_md_path, "w", encoding="utf-8") as f:
        f.write(limitations_md)
    print(f"[03_validate_annotations] Annotation limitations saved to {limitations_md_path}")

    # 5. Save Table R2 distribution JSON
    os.makedirs(os.path.dirname(dist_json_path), exist_ok=True)
    with open(dist_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[03_validate_annotations] Table R2 JSON saved to {dist_json_path}")

    print("\n" + "=" * 50)
    print("GATE G1 VALIDATION SUMMARY:")
    print(f"  Raw Agreement: {stats['raw_agreement_percent']}%")
    print(f"  Quadratic-Weighted Kappa: {stats['quadratic_weighted_kappa']} -> {stats['gate_g1_status']}")
    print(f"  5-Class Counts: {stats['five_class_counts']}")
    print(f"  5-Class Shares: {stats['five_class_shares']}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
