# Gate G1 & Inter-Annotator Agreement Report

**Guideline Version:** `v1.0`  
**Dataset Size:** 2721 rows  
**Gate G1 Threshold:** Quadratic-Weighted $\kappa \ge 0.60$  
**Evaluation Status:** **PASS** (Quadratic-Weighted $\kappa = 0.9677$)

---

## 1. Agreement Metrics Summary

| Metric | Value | Threshold / Target | Status |
|---|---|---|---|
| **Raw Percent Agreement** | **97.76%** | $\ge 70.0\%$ | PASS |
| **Cohen's Unweighted $\kappa$** | **0.9242** | $> 0.50$ | PASS |
| **Quadratic-Weighted $\kappa$** | **0.9677** | $\ge 0.60$ | PASS (Gate G1 Passed) |
| Total Disagreements Adjudicated | 61 (2.24%) | All adjudicated with logged rationale | Complete |

---

## 2. Annotator Confusion Matrix (A vs B)

| Annotator A \ B | Strongly Neg (0) | Weakly Neg (1) | Neutral (2) | Weakly Pos (3) | Strongly Pos (4) |
|---|---|---|---|---|---|
| **Strongly Negative** | 74 | 0 | 0 | 0 | 0 |
| **Weakly Negative** | 30 | 154 | 0 | 0 | 0 |
| **Neutral** | 0 | 0 | 2263 | 17 | 0 |
| **Weakly Positive** | 0 | 0 | 0 | 99 | 14 |
| **Strongly Positive** | 0 | 0 | 0 | 0 | 70 |

*Note: High concentration along the diagonal and immediate adjacent off-diagonals demonstrates strong ordinal alignment.*

---

## 3. Resulting Five-Class Distribution

| ID | Class Label | Count | Share (%) | Minority Status (share < 10%) |
|---|---|---|---|---|
| 0 | `strongly_negative` | 74 | 2.72% | **MINORITY CLASS** |
| 1 | `weakly_negative` | 184 | 6.76% | **MINORITY CLASS** |
| 2 | `neutral` | 2263 | 83.17% | Majority Class |
| 3 | `weakly_positive` | 130 | 4.78% | Standard |
| 4 | `strongly_positive` | 70 | 2.57% | Standard |
| **Total** | | **2721** | **100.0%** | |
