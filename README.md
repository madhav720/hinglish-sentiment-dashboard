# Hinglish Sentiment Classification: Investigating Minority-Class Collapse and Translation Effects

[![Tests](https://img.shields.io/badge/pytest-12%20passed-brightgreen.svg)]()
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![Status](https://img.shields.io/badge/Research%20Plan-Revision%203.0-orange.svg)]()

A complete, reproducible, end-to-end NLP research project investigating:
1. Five-class sentiment intensity classification of Roman-script Hinglish text.
2. Comparative performance between **MuRIL** (`google/muril-base-cased`) and **HingRoBERTa** (`l3cube-pune/hing-roberta`).
3. Mitigation of minority-class collapse using **Standard Cross-Entropy**, **Class-Weighted Cross-Entropy**, and **Focal Loss** ($\gamma \in \{0.5, 1.0, 2.0\}$).
4. Sentiment preservation and translation effects using paired English translations.
5. Error taxonomy breakdown and multi-seed statistical significance testing.

---

## Project Structure

```
├── data/
│   ├── raw/                      # Original Hinglish.csv (unmodified)
│   ├── annotation/               # Guidelines v1.0, logs, pilot samples, agreement reports
│   └── processed/                # Validated 5-class dataset, train/val/test splits, translations.csv
├── configs/                      # Experiment configuration JSONs
├── scripts/
│   ├── 01_data_audit.py          # Data inspection & plan verification
│   ├── 02_prepare_annotations.py # Deduplication & blind annotation template generation
│   ├── 03_validate_annotations.py# Dual-annotator agreement & 5-class dataset builder (Gate G1)
│   ├── 04_prepare_splits.py      # Leakage-safe grouped stratified splitting (Gate G2)
│   ├── 05_translate.py           # Paired English translation & audit (Gate G3)
│   ├── 06_train.py               # Experiment runner (C1-C6, T1-T6, seeds 1-3)
│   ├── 07_evaluate.py            # Detailed evaluation on saved test predictions
│   ├── 08_aggregate.py           # Results aggregation & master tables
│   ├── 09_collapse_analysis.py   # Empirical minority-class collapse diagnosis
│   ├── 10_error_analysis.py      # Error taxonomy & 60-item qualitative coding
│   ├── 11_statistical_tests.py   # Paired t-tests, Wilcoxon tests & bootstrap CIs
│   └── generate_figures.py       # Publication figures generator (Figures F1-F7)
├── src/
│   ├── data/                     # Audit, annotation, and splitting modules
│   ├── models/                   # Transformer loaders and trainer implementations
│   ├── losses/                   # Cross-entropy, class-weighted, and focal loss implementations
│   ├── evaluation/               # Metrics, collapse detection, and confusion matrix modules
│   ├── translation/              # Bilingual translation engine and quality auditing
│   └── utils/                    # Constants, mappings, and configuration definitions
├── models/                       # Best checkpoints saved per run ID
├── results/                      # Master CSVs, raw predictions, and markdown research reports
├── tables/                       # Publication tables (Tables R1 through R14)
├── figures/                      # High-resolution PNG (300 DPI) and PDF publication figures
├── logs/                         # Per-epoch training logs and run configurations
├── tests/                        # Automated unit and integration test suite
├── app.py                        # Interactive Streamlit research dashboard
├── reproducibility.md            # Detailed reproduction instructions
├── dataset_card.md               # Dataset provenance and schema card
└── requirements.txt              # Dependency specifications
```

---

## Quick Start & Verification

### 1. Run Automated Test Suite
```powershell
python -m pytest tests/test_pipeline.py -v
```

### 2. Run Complete Reproduction Workflow
```powershell
# 1. Audit raw dataset
python scripts/01_data_audit.py

# 2. Prepare blind annotation templates
python scripts/02_prepare_annotations.py

# 3. Validate annotations and build 5-class dataset
python scripts/03_validate_annotations.py

# 4. Generate leakage-free 70/15/15 grouped stratified splits
python scripts/04_prepare_splits.py

# 5. Generate paired English translations
python scripts/05_translate.py

# 6. Execute primary experiments (C1-C6 across seeds 1, 2, 3)
python scripts/06_train.py --config all --seeds 1,2,3 --smoke-test

# 7. Execute translation experiments (T1, T3, T4, T6 across seeds 1, 2, 3)
python scripts/06_train.py --translation --seeds 1,2,3 --smoke-test

# 8. Aggregate results & generate master tables
python scripts/08_aggregate.py

# 9. Perform minority-class collapse diagnosis
python scripts/09_collapse_analysis.py

# 10. Run error taxonomy analysis
python scripts/10_error_analysis.py

# 11. Run statistical significance tests
python scripts/11_statistical_tests.py

# 12. Generate all publication figures (PNG & PDF)
python scripts/generate_figures.py
```

### 3. Launch Interactive Research Dashboard
```powershell
streamlit run app.py
```
Or open the standalone web portal directly in any browser:
```powershell
start dashboard.html
```

#### New Capabilities in v3.1:
- **Batch CSV Translation & Detection:** Upload any `.csv` file with Roman-script Hinglish text to run simultaneous dual translation (Devanagari Hindi + fluent English) and 5-class sentiment intensity classification (`Strongly Negative`, `Weakly Negative`, `Neutral`, `Weakly Positive`, `Strongly Positive`).
- **Interactive Batch Analytics:** Live progress tracking, sentiment proportion breakdown bar, search & filter controls, and 1-click export of the enriched CSV with probabilities and confidence scores.
- **Enhanced Lexical & Phonetic Quality:** Expanded bilingual dictionary covering colloquial Hinglish e-commerce reviews, social comments, and negations.

---

## Research Integrity Standards

- **Zero Fabricated Data:** All metrics, confusion matrices, collapse flags, and figures are programmatically computed from actual saved experiment runs.
- **No Derivative Shortcut:** Strong and weak classes are annotated using documented semantic rubrics; the original 3-class label is never used as an intensity shortcut.
- **Strict Leakage Prevention:** Verified 0 ID, text, and group leakage across splits.
- **Training-Split Isolation:** Class weights and hyperparameter selections are calculated strictly on training and validation splits respectively; the test set is evaluated once.
