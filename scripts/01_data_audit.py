"""
Script 01: Data Audit
Executes dataset audit on data/raw/Hinglish.csv and generates verified inspection report.
"""

import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.audit import audit_raw_dataset, generate_audit_markdown

def main():
    raw_path = os.path.join("data", "raw", "Hinglish.csv")
    md_output = os.path.join("data", "annotation", "data_inspection.md")
    json_output = os.path.join("tables", "table_r1_dataset_audit.json")

    print(f"[01_data_audit] Running data audit on {raw_path}...")
    audit = audit_raw_dataset(raw_path)

    # Save Markdown report
    generate_audit_markdown(audit, md_output)
    print(f"[01_data_audit] Audit report written to {md_output}")

    # Save JSON summary table
    os.makedirs(os.path.dirname(json_output), exist_ok=True)
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"[01_data_audit] JSON summary table written to {json_output}")

    # Print summary verification
    print("\n" + "=" * 50)
    print("DATASET VERIFICATION SUMMARY:")
    print(f"  Raw Rows: {audit['raw_rows']} (Expected: 2766) -> {'PASS' if audit['plan_verification']['matches_raw_rows'] else 'FAIL'}")
    print(f"  Exact Duplicates: {audit['exact_duplicate_count']} (Expected: 45) -> {'PASS' if audit['plan_verification']['matches_exact_duplicates'] else 'FAIL'}")
    print(f"  Post-Dedup Rows: {audit['rows_after_exact_removal']} (Expected: 2721) -> {'PASS' if audit['plan_verification']['matches_post_dedup'] else 'FAIL'}")
    print(f"  Shares: {audit['post_dedup_shares']}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
