"""
Data Splitting and Leakage Prevention Module
Implements grouped stratified 70/15/15 train/val/test splits with zero text/ID leakage.
Calculates class weights strictly from the training split.
"""

import csv
import hashlib
import json
import os
import random
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

from src.utils.constants import (
    DEFAULT_SPLIT_SEED,
    LABEL2ID,
    ID2LABEL,
    CLASS_NAMES,
)


def compute_split_hash(train_ids: List[int], val_ids: List[int], test_ids: List[int]) -> str:
    """Compute deterministic SHA-256 hash of split assignments."""
    content = f"train:{sorted(train_ids)}|val:{sorted(val_ids)}|test:{sorted(test_ids)}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def grouped_stratified_split(
    rows: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split rows into train/val/test respecting:
    1. group_id clustering (all near-duplicates with the same group_id stay in exactly one split).
    2. Stratification across the 5 class labels.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # Group rows by group_id
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    # Determine dominant label for each group
    group_meta = []
    for g_id, g_rows in groups.items():
        labels = [r["label_id"] for r in g_rows]
        dom_label = Counter(labels).most_common(1)[0][0]
        group_meta.append({
            "group_id": g_id,
            "rows": g_rows,
            "size": len(g_rows),
            "dom_label": dom_label,
            "label_counts": Counter(labels)
        })

    # Stratify groups by dominant label
    by_class = defaultdict(list)
    for gm in group_meta:
        by_class[gm["dom_label"]].append(gm)

    train_groups = []
    val_groups = []
    test_groups = []

    train_counts = Counter()
    val_counts = Counter()
    test_counts = Counter()

    for c in range(5):
        c_groups = by_class[c]
        rng.shuffle(c_groups)

        total_c_items = sum(g["size"] for g in c_groups)
        target_train = round(total_c_items * train_ratio)
        target_val = round(total_c_items * val_ratio)

        curr_train = 0
        curr_val = 0

        for gm in c_groups:
            if curr_train < target_train:
                train_groups.append(gm)
                curr_train += gm["size"]
                train_counts.update(gm["label_counts"])
            elif curr_val < target_val:
                val_groups.append(gm)
                curr_val += gm["size"]
                val_counts.update(gm["label_counts"])
            else:
                test_groups.append(gm)
                test_counts.update(gm["label_counts"])

    train_rows = [r for gm in train_groups for r in gm["rows"]]
    val_rows = [r for gm in val_groups for r in gm["rows"]]
    test_rows = [r for gm in test_groups for r in gm["rows"]]

    # Leakage assertions
    train_ids = set(r["id"] for r in train_rows)
    val_ids = set(r["id"] for r in val_rows)
    test_ids = set(r["id"] for r in test_rows)

    assert len(train_ids.intersection(val_ids)) == 0, "LEAKAGE DETECTED: train and val share IDs!"
    assert len(train_ids.intersection(test_ids)) == 0, "LEAKAGE DETECTED: train and test share IDs!"
    assert len(val_ids.intersection(test_ids)) == 0, "LEAKAGE DETECTED: val and test share IDs!"

    # Normalized text leakage assertion
    train_norm = set(r.get("norm_text", "") for r in train_rows if r.get("norm_text"))
    val_norm = set(r.get("norm_text", "") for r in val_rows if r.get("norm_text"))
    test_norm = set(r.get("norm_text", "") for r in test_rows if r.get("norm_text"))

    assert len(train_norm.intersection(val_norm)) == 0, "LEAKAGE DETECTED: train and val share normalized text!"
    assert len(train_norm.intersection(test_norm)) == 0, "LEAKAGE DETECTED: train and test share normalized text!"
    assert len(val_norm.intersection(test_norm)) == 0, "LEAKAGE DETECTED: val and test share normalized text!"

    # Group ID leakage assertion
    train_g = set(r["group_id"] for r in train_rows)
    val_g = set(r["group_id"] for r in val_rows)
    test_g = set(r["group_id"] for r in test_rows)

    assert len(train_g.intersection(val_g)) == 0, "LEAKAGE DETECTED: train and val share group_id!"
    assert len(train_g.intersection(test_g)) == 0, "LEAKAGE DETECTED: train and test share group_id!"
    assert len(val_g.intersection(test_g)) == 0, "LEAKAGE DETECTED: val and test share group_id!"

    split_hash = compute_split_hash(list(train_ids), list(val_ids), list(test_ids))

    # Calculate class weights strictly from the training split only!
    # Formula from Section 10.2: w_c = N_train / (K * n_c)
    n_train = len(train_rows)
    class_weights = {}
    for c in range(5):
        count_c = train_counts.get(c, 0)
        assert count_c > 0, f"Critical guard failure: Training split has 0 examples for class {c} ({ID2LABEL[c]})!"
        w_c = n_train / (5.0 * count_c)
        class_weights[c] = round(w_c, 4)

    total_n = len(rows)
    split_meta = {
        "split_seed": seed,
        "split_hash": split_hash,
        "total_rows": total_n,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_share": round((len(train_rows) / total_n) * 100, 2),
        "val_share": round((len(val_rows) / total_n) * 100, 2),
        "test_share": round((len(test_rows) / total_n) * 100, 2),
        "class_distribution": {
            "train": {ID2LABEL[c]: train_counts.get(c, 0) for c in range(5)},
            "val": {ID2LABEL[c]: val_counts.get(c, 0) for c in range(5)},
            "test": {ID2LABEL[c]: test_counts.get(c, 0) for c in range(5)},
        },
        "class_weights_train_only": class_weights,
        "leakage_verification": {
            "id_overlap": 0,
            "normalized_text_overlap": 0,
            "group_id_overlap": 0,
            "status": "PASSED_ZERO_LEAKAGE",
        }
    }

    return train_rows, val_rows, test_rows, split_meta
