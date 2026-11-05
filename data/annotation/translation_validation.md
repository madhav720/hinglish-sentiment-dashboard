# Gate G3: Translation Validation and Label-Shift Audit Report

**Dataset Size:** 2721 translated sentences  
**Validation Sample Size:** 150 stratified rows  
**Gate G3 Status:** **PASS** (Polarity Preserved: 87.33%)

---

## 1. Automated Quality Metrics

| Check | Result | Plan Benchmark | Status |
|---|---|---|---|
| Total Translated Rows | **2721** | 2,721 | MATCH |
| Empty / Truncated English Translations | **0** | 0 | PASS |
| Empty / Truncated Hindi Translations | **0** | 0 | PASS |
| Mean Word Length Ratio (EN / Hinglish) | **0.99** | $0.80 - 1.25$ | PASS |
| Sample Polarity Preservation | **87.33%** | $\ge 80.0\%$ | PASS (Gate G3 Passed) |
| Sample Intensity Shift Rate | **14.67%** | Documented | PASS |

---

## 2. Qualitative Observations on Translation Effects

1. **Dual Translation Architecture:** Generates both Devanagari Hindi (हिंदी) and natural English paired translations for every Hinglish row.
2. **Intensity Neutralization:** Informal Hindi emphasis and emotional superlatives (*bahut*, *bilkul*, *hi*) occasionally lose slight intensity upon English translation, causing small shifts from Strong to Weak intensity.
3. **Negation Scope Preservation:** Sentences with explicit negation markers (*nahi*, *kabhi nahi*) maintained correct negation polarity in English and Hindi renderings.
4. **Loanword & Entity Retention:** Colloquial Hinglish idioms and English loanwords were harmonized into standard lexical equivalents while preserving the target sentiment stance.
