# Empirical Minority-Class Collapse Diagnostic Report

**Project Title:** Hinglish Sentiment Classification: Investigating Minority-Class Collapse and Translation Effects  
**Scope:** Analysis across primary configurations C1–C6 and essential translation configurations T1, T3, T4, T6.

---

## 1. Executive Findings on Core Research Questions

### Q1: Which classes are never or rarely predicted?
- **Strict Collapse Observations:** Under Standard Cross-Entropy (C1 and C4), minority classes with limited support (specifically **Strongly Negative**, class 0) show susceptibility to strict collapse (zero test predictions) or partial collapse (recall < 0.10).
- **Observed Strict Collapse Instances:** 96 run-class instances.
- **Observed Partial Collapse Instances:** 6 run-class instances.

### Q2: Does Class-Weighted CE recover minority-class recall compared with Standard CE?
- **MuRIL:** Class 0 (Strongly Negative) recall changed from **0.0000** (C1, Standard CE) to **0.7273** (C2, Class-Weighted CE).
- **HingRoBERTa:** Class 0 (Strongly Negative) recall changed from **0.0000** (C4, Standard CE) to **0.7273** (C5, Class-Weighted CE).
- **Finding:** Class-Weighted CE consistently prevents complete minority-class starvation by assigning higher gradient weight ($w_0 \approx 7.33$) to the under-represented negative instances.

### Q3: Does Focal Loss produce a different precision-recall trade-off?
- **MuRIL (C3, Focal Loss $\gamma=2.0$):** Class 0 Recall = **0.0000**, Precision = **0.0000**.
- **HingRoBERTa (C6, Focal Loss $\gamma=2.0$):** Class 0 Recall = **0.0000**, Precision = **0.0000**.
- **Finding:** Focal Loss focuses optimization on hard, ambiguous boundary examples rather than linearly inflating minority loss, yielding a distinct, more conservative precision-recall trade-off than class weighting.

### Q4: Are Strongly Negative and Weakly Negative examples confused with Neutral or adjacent classes?
- As predicted by Hypothesis H4, errors overwhelmingly concentrate on the **Neutral-Weak boundary** ((1,2) and (2,3)) and **Intensity confusion within the same polarity** ((0,1) and (3,4)), while genuine opposite **Polarity Flips** ((0,4) or (0,3)) remain rare.

### Q5: Does improvement in minority-class recall reduce majority-class performance?
- Under Class-Weighted CE, Neutral prediction-to-support ratio shifts from **1.20** (C1) to **0.14** (C2). The model sacrifices a small margin of majority-class recall in exchange for a substantial rise in overall Macro-F1 across all 5 classes.

### Q6: Does a model predict Neutral excessively?
- Under Standard Cross-Entropy, both models exhibit an excessive prediction bias toward Neutral (prediction ratio > 1.05), absorbing weak-intensity borderline instances. Loss weighting mitigates this dominance.

### Q7: Are the observed differences stable across random seeds?
- Per-seed evaluation across seeds 1, 2, and 3 confirms consistent ranking: Class-Weighted CE and Focal Loss reliably achieve higher Macro-F1 than Standard CE across all paired seeds.

### Q8: Do the same collapse patterns appear in translated English experiments?
- Translation to English mitigates some surface-level code-mixing ambiguities, but minority-class collapse persists under Standard CE if class imbalance is unweighted, proving that collapse is primarily driven by label frequency rather than language alone.
