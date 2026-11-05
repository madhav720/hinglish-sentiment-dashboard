# Split Verification and Leakage Audit Report

**Split Algorithm:** Grouped Stratified Split (Seed 42)  
**Split Hash (SHA-256):** `9400db5dde5db995213c95d36925b9d8792b7fb0cbf911b889f923c0439951d6`  
**Leakage Status:** **PASSED_ZERO_LEAKAGE**

---

## 1. Split Sizes and Proportions

| Split | Rows | Share (%) | Target Share (%) |
|---|---|---|---|
| **Train** | 1905 | 70.01% | 70.0% |
| **Validation** | 408 | 14.99% | 15.0% |
| **Test** | 408 | 14.99% | 15.0% |
| **Total** | **2721** | **100.0%** | **100.0%** |

---

## 2. Class Distribution Across Splits

| Class ID | Label Name | Train ($n=1905$) | Val ($n=408$) | Test ($n=408$) | Total ($n=2721$) | Train Class Weight |
|---|---|---|---|---|---|---|
| 0 | `strongly_negative` | 52 | 11 | 11 | 74 | **7.3269** |
| 1 | `weakly_negative` | 129 | 28 | 27 | 184 | **2.9535** |
| 2 | `neutral` | 1584 | 339 | 340 | 2263 | **0.2405** |
| 3 | `weakly_positive` | 91 | 20 | 19 | 130 | **4.1868** |
| 4 | `strongly_positive` | 49 | 10 | 11 | 70 | **7.7755** |

---

## 3. Leakage Guard Checks

1. **Row ID Disjointness:** Asserted $0$ ID overlap across Train, Val, and Test sets.
2. **Normalized Text Disjointness:** Asserted $0$ normalized text overlap across splits.
3. **Group ID Clustering:** All near-duplicates assigned the same `group_id` reside strictly within a single split.
4. **Class Weight Formula:** Computed strictly on training split as $w_c = \frac{N_{train}}{5 \cdot n_c}$.
