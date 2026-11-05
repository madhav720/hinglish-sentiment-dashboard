"""
Transformer Sequence Classification Module
Supports:
- Model A: MuRIL (google/muril-base-cased)
- Model B: HingRoBERTa (l3cube-pune/hing-roberta)
Configures 5-class sequence classification heads, dynamic padding, tokenizer management,
and robust CPU/GPU hardware detection.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, List

from src.utils.constants import (
    MODELS,
    LABEL2ID,
    ID2LABEL,
    NUM_CLASSES,
)

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def get_device() -> str:
    """Detect available hardware (CUDA GPU or CPU)."""
    if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_transformer_backbone(
    model_key: str,
    num_labels: int = NUM_CLASSES,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load tokenizer and sequence classification model for either MuRIL or HingRoBERTa.
    Configures id2label and label2id mapping.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Allowed models: {list(MODELS.keys())}")

    hub_id = MODELS[model_key]
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "PyTorch and Hugging Face Transformers are required to load full transformer models. "
            "Please ensure torch and transformers are installed."
        )

    device_str = device if device is not None else get_device()
    print(f"[ModelLoader] Loading '{hub_id}' on device: {device_str}...")

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    config = AutoConfig.from_pretrained(
        hub_id,
        num_labels=num_labels,
        id2label={int(k): v for k, v in ID2LABEL.items()},
        label2id={k: int(v) for k, v in LABEL2ID.items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(hub_id, config=config)
    model.to(device_str)

    return tokenizer, model


def tokenize_batch(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 128,
) -> Dict[str, Any]:
    """Tokenize texts with truncation and dynamic padding."""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt" if TRANSFORMERS_AVAILABLE else "np",
    )
