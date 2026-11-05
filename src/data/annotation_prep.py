"""
Annotation Preparation Module
Prepares exact-deduplicated dataset, groups near-duplicates, creates blind annotation templates,
and provides LLM-assisted / dual-annotator suggestion generation with full disagreement logging.
"""

import csv
import hashlib
import json
import os
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from src.utils.constants import (
    LABEL2ID,
    ID2LABEL,
    ORIG_LABEL_MAP,
    DEFAULT_SPLIT_SEED,
)
from src.data.audit import normalize_text


def prepare_deduplicated_dataset(raw_csv_path: str) -> List[Dict[str, Any]]:
    """
    Load raw Hinglish dataset, strip trailing newlines, remove exact duplicates,
    and assign group_id to normalized-text near-duplicate clusters.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at: {raw_csv_path}")

    with open(raw_csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = [h.strip().lstrip('\ufeff') for h in next(reader)]
        rows = list(reader)

    # Map column indices
    col_idx = {}
    for i, h in enumerate(header):
        low = h.lower()
        if "text" in low:
            col_idx["text"] = i
        elif "sentiment" in low:
            col_idx["sentiment"] = i
        elif "label" in low:
            col_idx["label"] = i
        else:
            col_idx["raw_id"] = i

    exact_seen = set()
    cleaned_rows = []
    
    # Track normalized texts for cluster grouping
    norm_to_group = {}
    group_counter = 0

    for idx, r in enumerate(rows):
        raw_text = r[col_idx["text"]]
        # Strip only trailing newlines for exact text matching as per plan specification
        clean_text = raw_text.rstrip("\r\n")
        
        orig_sentiment = r[col_idx["sentiment"]].strip().lower()
        orig_label = int(r[col_idx["label"]].strip())

        # Exact duplicate filter (keep first occurrence: exactly 45 duplicates -> 2,721 unique)
        if clean_text in exact_seen:
            continue
        exact_seen.add(clean_text)

        norm = normalize_text(clean_text)
        if norm not in norm_to_group:
            norm_to_group[norm] = group_counter
            group_counter += 1
        group_id = norm_to_group[norm]

        cleaned_rows.append({
            "id": len(cleaned_rows),
            "orig_raw_id": r[col_idx.get("raw_id", 0)],
            "text": clean_text,
            "text_raw": raw_text,
            "orig_label": orig_label,
            "orig_sentiment": orig_sentiment,
            "group_id": group_id,
            "norm_text": norm,
        })

    return cleaned_rows


def create_annotation_template(cleaned_rows: List[Dict[str, Any]], output_path: str) -> None:
    """Save blind annotation template with original label hidden from primary annotator columns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "id",
        "group_id",
        "text",
        "annotator_A_label",
        "annotator_B_label",
        "adjudicated_label",
        "label_id",
        "flags",
        "disagreement",
        "notes",
    ]
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in cleaned_rows:
            writer.writerow({
                "id": r["id"],
                "group_id": r["group_id"],
                "text": r["text"],
                "annotator_A_label": "",
                "annotator_B_label": "",
                "adjudicated_label": "",
                "label_id": "",
                "flags": "",
                "disagreement": 0,
                "notes": "",
            })


def sample_stratified_pilot(cleaned_rows: List[Dict[str, Any]], n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample an initial stratified pilot set (~100 rows) by original polarity."""
    import random
    rng = random.Random(seed)

    by_sentiment = defaultdict(list)
    for r in cleaned_rows:
        by_sentiment[r["orig_sentiment"]].append(r)

    total = len(cleaned_rows)
    pilot = []
    for sentiment, items in by_sentiment.items():
        sample_size = round(n * (len(items) / total))
        sampled = rng.sample(items, min(sample_size, len(items)))
        pilot.extend(sampled)

    # Adjust to exact n if rounding caused slight difference
    if len(pilot) < n:
        remaining = [r for r in cleaned_rows if r not in pilot]
        pilot.extend(rng.sample(remaining, n - len(pilot)))
    elif len(pilot) > n:
        pilot = pilot[:n]

    return pilot


def sample_neutral_audit(cleaned_rows: List[Dict[str, Any]], fraction: float = 0.15, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample a 15% random subset of original Neutral rows for sentiment audit."""
    import random
    rng = random.Random(seed)
    neutrals = [r for r in cleaned_rows if r["orig_label"] == 1]
    k = max(1, round(len(neutrals) * fraction))
    return rng.sample(neutrals, k)
