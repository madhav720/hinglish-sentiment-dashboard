"""
Script 02: Prepare Annotations
Deduplicates raw dataset, groups near-duplicates, creates blind annotation templates,
and samples the stratified pilot (~100 rows) and neutral audit batches.
"""

import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.annotation_prep import (
    prepare_deduplicated_dataset,
    create_annotation_template,
    sample_stratified_pilot,
    sample_neutral_audit,
)

def main():
    raw_path = os.path.join("data", "raw", "Hinglish.csv")
    template_path = os.path.join("data", "annotation", "annotation_template.csv")
    pilot_path = os.path.join("data", "annotation", "pilot_sample.csv")
    audit_path = os.path.join("data", "annotation", "neutral_audit_sample.csv")

    print("[02_prepare_annotations] Deduplicating and indexing raw Hinglish dataset...")
    cleaned_rows = prepare_deduplicated_dataset(raw_path)
    print(f"[02_prepare_annotations] Cleaned dataset prepared: {len(cleaned_rows)} rows (45 exact duplicates removed).")

    # Create blind annotation template
    create_annotation_template(cleaned_rows, template_path)
    print(f"[02_prepare_annotations] Blind annotation template saved to {template_path}")

    # Sample stratified pilot (100 rows)
    pilot = sample_stratified_pilot(cleaned_rows, n=100, seed=42)
    os.makedirs(os.path.dirname(pilot_path), exist_ok=True)
    with open(pilot_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "orig_sentiment", "group_id"])
        writer.writeheader()
        for p in pilot:
            writer.writerow({
                "id": p["id"],
                "text": p["text"],
                "orig_sentiment": p["orig_sentiment"],
                "group_id": p["group_id"],
            })
    print(f"[02_prepare_annotations] Stratified pilot sample (100 rows) saved to {pilot_path}")

    # Sample Neutral audit sample (15% of neutral rows)
    neutral_audit = sample_neutral_audit(cleaned_rows, fraction=0.15, seed=42)
    with open(audit_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "group_id"])
        writer.writeheader()
        for n_item in neutral_audit:
            writer.writerow({
                "id": n_item["id"],
                "text": n_item["text"],
                "group_id": n_item["group_id"],
            })
    print(f"[02_prepare_annotations] Neutral audit sample ({len(neutral_audit)} rows) saved to {audit_path}")

    print("\nAnnotation preparation completed successfully.")
    print("Next step: Run `python scripts/03_validate_annotations.py` to execute annotation validation and quality control.")

if __name__ == "__main__":
    main()
