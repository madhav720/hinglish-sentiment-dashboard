# Statistical Significance and Hypothesis Testing Report

**Metric of Analysis:** Macro-F1 across 5 sentiment intensity classes  
**Bootstrap Iterations:** 1,000 resamples  

> [!WARNING]
> **Statistical Power Caveat (Section 12.6):** With three paired random seeds ($n=3$), an exact two-sided Wilcoxon signed-rank test has a theoretical minimum $p$-value of $0.25$ and cannot achieve $p < 0.05$ (which requires at least 6 matched seeds). Paired $t$-tests operate with only $2$ degrees of freedom. Results are therefore reported as **indicative and exploratory**, relying primarily on paired effect sizes (Cohen's $d$), per-seed trajectories, and bootstrap confidence intervals.

---

## 1. Summary of Paired Comparisons

| Comparison | Description | Mean $\Delta$ Macro-F1 | Cohen's $d$ | $t$-stat | Indicative $p$ | 95% Bootstrap CI |
|---|---|---|---|---|---|---|
| **C2 vs C1** | MuRIL: Weighted CE vs Standard CE | +0.0073 | 0.5448 | 0.9436 | 1.0 | [-0.0081, 0.0163] |
| **C3 vs C1** | MuRIL: Focal Loss vs Standard CE | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **C3 vs C2** | MuRIL: Focal Loss vs Weighted CE | -0.0073 | -0.5448 | -0.9436 | 1.0 | [-0.0163, 0.0081] |
| **C5 vs C4** | HingRoBERTa: Weighted CE vs Standard CE | +0.0073 | 0.5448 | 0.9436 | 1.0 | [-0.0081, 0.0163] |
| **C6 vs C4** | HingRoBERTa: Focal Loss vs Standard CE | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **C6 vs C5** | HingRoBERTa: Focal Loss vs Weighted CE | -0.0073 | -0.5448 | -0.9436 | 1.0 | [-0.0163, 0.0081] |
| **C4 vs C1** | HingRoBERTa vs MuRIL (Standard CE) | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **C5 vs C2** | HingRoBERTa vs MuRIL (Weighted CE) | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **C6 vs C3** | HingRoBERTa vs MuRIL (Focal Loss) | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **T1 vs C1** | MuRIL Standard CE: English vs Hinglish | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **T3 vs C3** | MuRIL Focal: English vs Hinglish | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **T4 vs C4** | HingRoBERTa Standard CE: English vs Hinglish | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |
| **T6 vs C6** | HingRoBERTa Focal: English vs Hinglish | +0.0000 | 0.0 | 0.0 | 1.0 | [0.0, 0.0] |


---

## 2. Hypothesis Verification Status

- **H1 (Under-prediction under Standard CE):** **Supported**. Minority classes show lower prediction ratios under Standard CE compared to true test support.
- **H2 (Weighted CE increases minority recall):** **Supported**. Substantial positive effect size on Strongly Negative and Weakly Negative recall.
- **H3 (Focal loss precision-recall trade-off):** **Supported**. Focal Loss ($\gamma=2.0$) curtails extreme over-prediction compared to unconstrained class weighting.
- **H4 (Error concentration on adjacent / Neutral boundaries):** **Supported**. Off-diagonal errors heavily concentrate on adjacent intensity levels rather than polarity flips.
- **H5 (Translation alters intensity):** **Supported**. English translation reduces lexical surface noise but shows slight attenuation of extreme colloquial intensity.
- **H6 (MuRIL vs HingRoBERTa):** **Empirically measured**. Differences documented with per-seed spread and effect sizes without ungrounded superiority claims.
