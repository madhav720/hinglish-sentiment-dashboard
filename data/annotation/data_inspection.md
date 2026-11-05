# Dataset Audit Report: Hinglish Sentiment Dataset

**Source File:** `data\raw\Hinglish.csv`  
**Audit Status:** Verified against *Research Execution Plan (Revision 3.0)*

---

## 1. Summary Overview

| Metric | Measured Value | Plan Benchmark | Status |
|---|---|---|---|
| Total Raw Rows | **2766** | 2,766 | MATCH |
| Total Columns | **4** (`, text, sentiment, label`) | 4 | MATCH |
| Total Missing Values | **0** | 0 | MATCH |
| Exact Duplicate Texts | **45** | 45 | MATCH |
| Rows After Exact Dedup | **2721** | 2,721 | MATCH |
| Normalized Near-Duplicates | **66** | 66 (21 near + 45 exact) | MATCH |
| Empty Texts After Normalization | **0** | 0 | MATCH |
| Label-Sentiment Consistency | **100% Consistent** | 100% | MATCH |

---

## 2. Original Three-Class Distribution

> [!IMPORTANT]
> The original dataset uses a 3-class polarity scheme:
> - `0` = Negative
> - `1` = Neutral
> - `2` = Positive
> 
> **CRITICAL RULE**: The original integer label (`0/1/2`) conflicts with the 5-class target scheme (where `2` = Neutral). The original column is strictly preserved as `orig_label` and `orig_sentiment` and is NEVER directly reused as the 5-class target ID.

### Class Counts and Percentages

| Sentiment | Label ID | Raw Count ($N=2766$) | Raw Share | After Dedup ($N=2721$) | Share After Dedup | Benchmark Share |
|---|---|---|---|---|---|---|
| **Negative** | 0 | 545 | 19.7% | 538 | **19.77%** | 19.8% |
| **Neutral** | 1 | 1234 | 44.61% | 1205 | **44.29%** | 44.3% |
| **Positive** | 2 | 987 | 35.68% | 978 | **35.94%** | 35.9% |

---

## 3. Text Length Distribution

- **Mean Word Count:** 18.83 words
- **Median Word Count:** 14 words
- **Minimum Word Count:** 3 words
- **Maximum Word Count:** 157 words
- **Unusually Short Texts (< 3 words):** 0

---

## 4. Dataset Limitations and Domain Caveat

1. **Domain Mix:** A qualitative inspection reveals a mixture of formal, legal, scripture/moral commentary, and technical instructions transliterated into Roman-script Hindi alongside colloquial code-mixed sentences, rather than pure informal social media slang.
2. **Missing Intensity Dimension:** The original 3-class annotations lack intensity discrimination. Strongly Negative and Weakly Negative cannot be synthesized by rule; full 5-class re-annotation is required.
3. **Data Leakage Safeguard:** Near-duplicates (66 normalized matches) must be assigned cluster `group_id`s so they are strictly assigned to the same split.
