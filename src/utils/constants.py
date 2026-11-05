"""
Core Constants and Configuration Specifications
Source of truth for all labels, model identifiers, losses, and collapse thresholds.
"""

from typing import Dict, List, Tuple

# Fixed Five-Class Label Mapping (Section 4 & Section 6.1)
LABEL2ID: Dict[str, int] = {
    "strongly_negative": 0,
    "weakly_negative": 1,
    "neutral": 2,
    "weakly_positive": 3,
    "strongly_positive": 4,
}

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

CLASS_NAMES: List[str] = [
    "Strongly Negative",
    "Weakly Negative",
    "Neutral",
    "Weakly Positive",
    "Strongly Positive",
]

NUM_CLASSES: int = 5

# Original 3-class mapping in Hinglish.csv (for provenance / audit only)
# 0 = negative, 1 = neutral, 2 = positive
# IMPORTANT: Never use original label as 5-class target!
ORIG_LABEL_MAP: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

# Transformer Backbones (Only two, Section 6)
MODELS: Dict[str, str] = {
    "muril": "google/muril-base-cased",
    "hingroberta": "l3cube-pune/hing-roberta",
}

# Primary Configurations (C1 - C6)
PRIMARY_CONFIGS: Dict[str, Dict[str, str]] = {
    "C1": {"model": "muril", "loss": "ce", "text": "hinglish"},
    "C2": {"model": "muril", "loss": "wce", "text": "hinglish"},
    "C3": {"model": "muril", "loss": "focal", "text": "hinglish"},
    "C4": {"model": "hingroberta", "loss": "ce", "text": "hinglish"},
    "C5": {"model": "hingroberta", "loss": "wce", "text": "hinglish"},
    "C6": {"model": "hingroberta", "loss": "focal", "text": "hinglish"},
}

# Translation Configurations (T1 - T6)
TRANSLATION_CONFIGS: Dict[str, Dict[str, str]] = {
    "T1": {"model": "muril", "loss": "ce", "text": "english", "counterpart": "C1"},
    "T2": {"model": "muril", "loss": "wce", "text": "english", "counterpart": "C2"},
    "T3": {"model": "muril", "loss": "focal", "text": "english", "counterpart": "C3"},
    "T4": {"model": "hingroberta", "loss": "ce", "text": "english", "counterpart": "C4"},
    "T5": {"model": "hingroberta", "loss": "wce", "text": "english", "counterpart": "C5"},
    "T6": {"model": "hingroberta", "loss": "focal", "text": "english", "counterpart": "C6"},
}

# Essential Translation Subset
ESSENTIAL_TRANSLATION_SUBSET: List[str] = ["T1", "T3", "T4", "T6"]

# Minority-Class Collapse Operational Definitions (Section 14.1)
COLLAPSE_THRESHOLDS = {
    "strict_collapse_max_pred": 0,          # 0 test predictions while support > 0
    "partial_collapse_recall_max": 0.10,    # Recall strictly below 0.10
    "partial_collapse_ratio_max": 0.50,     # Predicted count < 0.5 * true support
    "minority_class_share_threshold": 0.10, # Class training share < 10%
    "low_support_threshold": 30,            # Test support < 30
}

# Error Taxonomy Categories (Section 16.1)
# Unordered class index pairs
ERROR_CATEGORIES: Dict[str, List[Tuple[int, int]]] = {
    "intensity_confusion": [(0, 1), (3, 4)],              # Strong vs Weak within same polarity
    "neutral_weak_boundary": [(1, 2), (2, 3)],            # Neutral vs Weak sentiment
    "neutral_strong_skip": [(0, 2), (2, 4)],              # Neutral vs Strong sentiment
    "polarity_flip": [(0, 3), (0, 4), (1, 3), (1, 4)],     # Negative vs Positive classes
}

# Qualitative Error Phenonema (Section 16.2)
QUALITATIVE_ERROR_TAGS: List[str] = [
    "sarcasm_irony",
    "negation_scope",
    "informal_slang",
    "spelling_variation",
    "code_mixing_transliteration",
    "ambiguous_intensity",
    "weak_vs_strong_confusion",
    "mixed_context_dependent",
    "minority_class_misclassification",
    "translation_introduced_change",
    "possible_annotation_error",
]

# Random Seeds
DEFAULT_SPLIT_SEED: int = 42
DEFAULT_SEEDS: List[int] = [1, 2, 3]
FOCAL_GAMMA_CANDIDATES: List[float] = [0.5, 1.0, 2.0]
