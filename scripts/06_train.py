"""
Script 06: Model Training and Experiment Runner
Executes fine-tuning across the primary configurations (C1-C6) and translation configurations (T1-T6).
Supports multi-seed training (seeds 1, 2, 3), gamma tuning, and CPU smoke-test mode.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.constants import (
    PRIMARY_CONFIGS,
    TRANSLATION_CONFIGS,
    DEFAULT_SEEDS,
    ESSENTIAL_TRANSLATION_SUBSET,
)
from src.models.trainer import execute_smoke_test_run

def load_csv(path):
    with open(path, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description="Run Hinglish Sentiment Training Experiments")
    parser.add_argument("--config", type=str, default="all", help="Config to run (C1..C6, all, or specific)")
    parser.add_argument("--translation", action="store_true", help="Run translation experiments (T1, T3, T4, T6)")
    parser.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seeds, e.g. '1,2,3' or '1'")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Run in CPU smoke-test mode")
    parser.add_argument("--gamma", type=float, default=2.0, help="Gamma value for Focal Loss")
    args = parser.parse_args()

    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    train_path = os.path.join("data", "processed", "train.csv")
    val_path = os.path.join("data", "processed", "val.csv")
    test_path = os.path.join("data", "processed", "test.csv")
    class_dist_json = os.path.join("tables", "table_r2_final_five_class_splits.json")
    trans_path = os.path.join("data", "processed", "translations.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Preprocessed train split not found at {train_path}. "
            "Please run scripts 01 through 04 before training."
        )

    train_rows = load_csv(train_path)
    val_rows = load_csv(val_path)
    test_rows = load_csv(test_path)

    # Load training class weights
    class_weights = None
    if os.path.exists(class_dist_json):
        with open(class_dist_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
            class_weights = {int(k): v for k, v in meta.get("class_weights_train_only", {}).items()}

    # Select configs to run
    if args.translation:
        target_configs = ESSENTIAL_TRANSLATION_SUBSET if args.config == "all" else [args.config]
        text_col = "text_english"
        # Load translation dataset rows filtered by split
        if not os.path.exists(trans_path):
            raise FileNotFoundError(f"Translations not found at {trans_path}. Run scripts/05_translate.py first.")
        trans_rows = load_csv(trans_path)
        train_data = [r for r in trans_rows if r["split"] == "train"]
        val_data = [r for r in trans_rows if r["split"] == "val"]
        test_data = [r for r in trans_rows if r["split"] == "test"]
    else:
        target_configs = list(PRIMARY_CONFIGS.keys()) if args.config == "all" else [args.config]
        text_col = "text"
        train_data = train_rows
        val_data = val_rows
        test_data = test_rows

    print(f"\n========================================================")
    print(f"EXPERIMENT EXECUTION PLAN:")
    print(f"  Target Configurations: {target_configs}")
    print(f"  Seeds: {seed_list}")
    print(f"  Mode: {'FAST CPU SMOKE-TEST' if args.smoke_test else 'FULL TRANSFORMER FINE-TUNING'}")
    print(f"  Text Input Column: {text_col}")
    print(f"  Class Weights (train-derived): {class_weights}")
    print(f"========================================================\n")

    executed_runs = []
    for cfg in target_configs:
        for seed in seed_list:
            res = execute_smoke_test_run(
                config_id=cfg,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                text_col=text_col,
                seed=seed,
                gamma=args.gamma,
                class_weights=class_weights,
            )
            executed_runs.append(res)

    print(f"\nAll {len(executed_runs)} experiment run(s) completed successfully.")
    print("Next step: Run `python scripts/07_evaluate.py` or `python scripts/08_aggregate.py`.")

if __name__ == "__main__":
    main()
