"""
Evaluation and Minority-Class Collapse Metrics Module
Implements:
- Macro-F1 (primary metric), Weighted-F1, Accuracy
- Per-class Precision, Recall, F1, Support
- Predicted counts and Prediction-to-Support ratios
- 5x5 Confusion Matrix (raw and row-normalized)
- Strict and Partial Minority-Class Collapse detection
- Error Taxonomy quantification (intensity, neutral-weak, neutral-strong, polarity flip)
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

from src.utils.constants import (
    CLASS_NAMES,
    ID2LABEL,
    LABEL2ID,
    COLLAPSE_THRESHOLDS,
    ERROR_CATEGORIES,
)


def compute_confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 5) -> List[List[int]]:
    """Compute 5x5 confusion matrix where rows are true classes and columns are predicted classes."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for yt, yp in zip(y_true, y_pred):
        if 0 <= yt < num_classes and 0 <= yp < num_classes:
            cm[yt][yp] += 1
    return cm


def compute_row_normalized_cm(cm: List[List[int]]) -> List[List[float]]:
    """Compute row-normalized confusion matrix (percentages of true class)."""
    norm_cm = []
    for row in cm:
        total = sum(row)
        if total > 0:
            norm_cm.append([round(v / total, 4) for v in row])
        else:
            norm_cm.append([0.0] * len(row))
    return norm_cm


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics suite for 5-class classification.
    Safe handling of zero predictions via zero_division=0.
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have identical length"
    n_samples = len(y_true)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    norm_cm = compute_row_normalized_cm(cm)

    # Per-class metrics
    per_class = {}
    f1_list = []
    precision_list = []
    recall_list = []
    support_list = []

    correct_total = 0

    for c in range(num_classes):
        tp = cm[c][c]
        fn = sum(cm[c][j] for j in range(num_classes)) - tp
        fp = sum(cm[i][c] for i in range(num_classes)) - tp
        support = tp + fn
        pred_count = tp + fp

        correct_total += tp
        support_list.append(support)

        # Precision
        prec = tp / pred_count if pred_count > 0 else 0.0
        # Recall
        rec = tp / support if support > 0 else 0.0
        # F1
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        pred_ratio = pred_count / support if support > 0 else 0.0

        # Collapse detection
        is_strict_collapse = (pred_count == 0 and support > 0)
        is_partial_collapse = (
            (rec < COLLAPSE_THRESHOLDS["partial_collapse_recall_max"] or
             pred_ratio < COLLAPSE_THRESHOLDS["partial_collapse_ratio_max"])
            and support > 0
        )

        per_class[ID2LABEL[c]] = {
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
            "predicted_count": pred_count,
            "prediction_to_support_ratio": round(pred_ratio, 4),
            "strict_collapse": is_strict_collapse,
            "partial_collapse": is_partial_collapse,
        }

        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

    accuracy = correct_total / n_samples if n_samples > 0 else 0.0
    macro_f1 = sum(f1_list) / num_classes
    macro_precision = sum(precision_list) / num_classes
    macro_recall = sum(recall_list) / num_classes

    # Weighted F1
    total_support = sum(support_list)
    weighted_f1 = (
        sum(f1 * sup for f1, sup in zip(f1_list, support_list)) / total_support
        if total_support > 0 else 0.0
    )

    # Minority-class collapse summary
    strict_collapse_classes = [
        info["class_name"] for info in per_class.values() if info["strict_collapse"]
    ]
    partial_collapse_classes = [
        info["class_name"] for info in per_class.values() if info["partial_collapse"]
    ]

    # Error Taxonomy Breakdown
    total_errors = n_samples - correct_total
    error_breakdown = {}
    if total_errors > 0:
        for cat_name, pairs in ERROR_CATEGORIES.items():
            cat_count = 0
            for i, j in pairs:
                cat_count += cm[i][j] + cm[j][i]
            error_breakdown[cat_name] = {
                "count": cat_count,
                "share_of_errors": round((cat_count / total_errors) * 100, 2),
            }
    else:
        for cat_name in ERROR_CATEGORIES.keys():
            error_breakdown[cat_name] = {"count": 0, "share_of_errors": 0.0}

    results = {
        "n_samples": n_samples,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "per_class": per_class,
        "confusion_matrix_raw": cm,
        "confusion_matrix_normalized": norm_cm,
        "minority_collapse_summary": {
            "strict_collapse_count": len(strict_collapse_classes),
            "strict_collapse_classes": strict_collapse_classes,
            "partial_collapse_count": len(partial_collapse_classes),
            "partial_collapse_classes": partial_collapse_classes,
        },
        "error_taxonomy": error_breakdown,
    }

    return results
