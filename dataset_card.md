# Dataset Card: Hinglish Sentiment Intensity Corpus

**Corpus Name:** Roman-script Hinglish Sentiment Dataset  
**Version:** 3.0 (Five-Class Intensity Scheme)  
**Task:** Five-class sentiment intensity classification (Strongly Negative, Weakly Negative, Neutral, Weakly Positive, Strongly Positive)  

---

## 1. Dataset Summary

The dataset consists of 2,766 raw sentences written in Roman-script Hinglish (Hindi written in Latin script with code-mixing from English). Following exact-duplicate deduplication (45 items removed), the effective corpus contains **2,721 unique sentences**.

- **Language:** Hindi-English code-mixed (Roman script / transliterated).
- **Domain Mix:** A qualitative inspection reveals a mixture of formal, legal, scripture/moral commentary, and technical instructions transliterated into Roman-script Hindi alongside colloquial code-mixed sentences.
- **Label Granularity:** Extended from original coarse 3-class polarity (negative/neutral/positive) to fine-grained 5-class sentiment intensity using a documented rubric (v1.0) and dual-rater adjudication.

---

## 2. Class Schema

| ID | Class Name | Definition | Support (Overall) | Share (%) | Status |
|---|---|---|---|---|---|
| **0** | `strongly_negative` | Intense dissatisfaction, anger, harsh condemnation, catastrophe | 74 | 2.72% | Severe Minority Class |
| **1** | `weakly_negative` | Mild dissatisfaction, disappointment, complaint, problem | 184 | 6.76% | Minority Class |
| **2** | `neutral` | Informational, factual, technical, definition, query | 2,263 | 83.17% | Majority Class |
| **3** | `weakly_positive` | Mild approval, hedged praise, moderate satisfaction | 130 | 4.78% | Minority Class |
| **4** | `strongly_positive` | Emphatic praise, excitement, superlatives, profound gratitude | 70 | 2.57% | Severe Minority Class |

---

## 3. Data Splits & Leakage Prevention

- **Split Ratio:** 70% Train (1,905 items), 15% Validation (408 items), 15% Test (408 items).
- **Clustering:** Normalized near-duplicates are grouped via `group_id` such that no near-identical sentences leak across train, validation, or test sets.
- **Split Hash:** Deterministically generated with fixed seed `42`.
- **Inheritance:** English translations inherit identical IDs and split membership.
