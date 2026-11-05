# Annotation Quality and Limitations Report

**Dataset Size:** 2721 items  
**Guideline Version:** `v1.0`  

---

## 1. Flag Summary & Linguistic Phenomena

| Flag | Description | Count | Share of Dataset (%) |
|---|---|---|---|
| `negation` | Explicit negation terms affecting stance | 395 | 14.52% |
| `slang` | Informal colloquial Hinglish slang expressions | 1 | 0.04% |
| `sarcasm` | Sarcastic/ironic discourse structure | 0 | 0.0% |
| `mixed` | Co-presence of both positive and negative elements | 47 | 1.73% |
| `short_context` | Texts with fewer than 5 words | 11 | 0.4% |
| `uncertain` | Subjective intensity borderline requiring adjudication | 61 | 2.24% |
| `orig_conflict` | Final polarity contradicts original 3-class label | 1309 | 48.11% |

---

## 2. Subjective Boundaries and Bias Analysis

1. **Weak vs Strong Boundary Noise:** The boundary between mild sentiment and intense sentiment involves inherent human subjectivity. Disagreements between Weak and Strong within the same polarity accounted for the primary rater differences.
2. **Polarity Conflict Handling:** All 1309 examples where annotated polarity diverged from original 3-class provenance were independently adjudicated and preserved with `orig_conflict` flag rather than silently overwritten.
3. **Sensitivity Consideration:** Downstream evaluation includes a sensitivity analysis with and without `uncertain` tagged items to test model stability.
