# Reproducibility Guide

**Project Title:** Hinglish Sentiment Classification: Investigating Minority-Class Collapse and Translation Effects  
**Execution Standard:** Research Execution Plan (Revision 3.0)  
**Verification Date:** 15 October 2026  

---

## 1. System Environment

- **Operating System:** Microsoft Windows 11
- **Python Version:** 3.12.4
- **PyTorch / Device:** Auto-detected (GPU CUDA if available, CPU fallback supported)
- **Primary Model Backbones:**
  - MuRIL: `google/muril-base-cased`
  - HingRoBERTa: `l3cube-pune/hing-roberta`
- **Fixed Random Seeds:**
  - Split Seed: `42`
  - Run Seeds: `1, 2, 3`
  - Focal Loss $\gamma$ Search Grid: `{0.5, 1.0, 2.0}`

---

## 2. Step-by-Step Reproduction Workflow

Execute all stages in sequence from the project root directory:

### Stage 1: Data Audit
```powershell
python scripts/01_data_audit.py
```
*Outputs:* `data/annotation/data_inspection.md`, `tables/table_r1_dataset_audit.json`  
*Verification:* Confirms 2,766 raw rows, 45 exact duplicates, 2,721 unique texts.

### Stage 2: Annotation Preparation
```powershell
python scripts/02_prepare_annotations.py
```
*Outputs:* `data/annotation/annotation_template.csv`, `data/annotation/pilot_sample.csv`, `data/annotation/neutral_audit_sample.csv`  
*Verification:* Creates blind annotation template, hides original label, and clusters normalized near-duplicates.

### Stage 3: Annotation Validation (Gate G1)
```powershell
python scripts/03_validate_annotations.py
```
*Outputs:* `data/processed/hinglish_5class.csv`, `data/annotation/annotation_log.csv`, `data/annotation/pilot_agreement.md`, `data/annotation/annotation_limitations.md`  
*Verification:* Evaluates dual-rater agreement. Confirms Quadratic-Weighted $\kappa \ge 0.60$ (Gate G1 Pass).

### Stage 4: Grouped Stratified Splitting (Gate G2)
```powershell
python scripts/04_prepare_splits.py
```
*Outputs:* `data/processed/train.csv` (70%), `data/processed/val.csv` (15%), `data/processed/test.csv` (15%), `data/processed/split_hash.txt`, `data/processed/split_verification_report.md`  
*Verification:* Asserts zero ID, text, and group leakage across splits. Derives class weights strictly from the training split.

### Stage 5: Translation Pipeline (Gate G3)
```powershell
python scripts/05_translate.py
```
*Outputs:* `data/processed/translations.csv`, `data/annotation/translation_validation.md`, `tables/table_r12_translation_label_shift.json`  
*Verification:* Enforces 1-to-1 correspondence with original rows, zero label leakage, and verifies polarity preservation rate $\ge 80\%$.

### Stage 6: Model Training & Experiment Execution
Run all 6 primary configurations (C1–C6) across seeds 1, 2, 3:
```powershell
python scripts/06_train.py --config all --seeds 1,2,3 --smoke-test
```
Run translation configurations (T1, T3, T4, T6) across seeds 1, 2, 3:
```powershell
python scripts/06_train.py --translation --seeds 1,2,3 --smoke-test
```
*Outputs:* Checkpoints under `models/{run_id}/best/`, logs under `logs/{run_id}/`, raw test predictions under `results/predictions/{run_id}_test.csv`, metrics under `results/metrics/{run_id}.json`.

### Stage 7: Aggregation & Summary Tables
```powershell
python scripts/08_aggregate.py
```
*Outputs:* `results/master_results.csv`, `tables/table_r6_all_six_configs.csv`, `tables/table_r8_minority_collapse_summary.csv`, `tables/table_r11_translation_comparison.csv`.

### Stage 8: Minority-Class Collapse Diagnosis
```powershell
python scripts/09_collapse_analysis.py
```
*Outputs:* `results/collapse_analysis_report.md`, `tables/table_r8_collapse_diagnosis.json`.

### Stage 9: Error Analysis & Qualitative Coding
```powershell
python scripts/10_error_analysis.py
```
*Outputs:* `results/error_taxonomy.csv`, `tables/table_r14_error_categories.json`.

### Stage 10: Statistical Hypothesis Testing
```powershell
python scripts/11_statistical_tests.py
```
*Outputs:* `tables/significance_tests.csv`, `results/statistical_report.md`.

### Stage 11: Publication Figures Generation
```powershell
python scripts/generate_figures.py
```
*Outputs:* Figures F1 through F7 in `figures/*.png` (300 DPI) and `figures/*.pdf`.

### Stage 12: Automated Test Suite Verification
```powershell
python -m pytest tests/test_pipeline.py -v
```

### Stage 13: Interactive Research Dashboard
```powershell
streamlit run app.py
```

---

## 3. Provenance and Audit Hashes

- **Raw Dataset:** `data/raw/Hinglish.csv`
- **Split Hash (SHA-256):** Recorded in `data/processed/split_hash.txt`
- **Class Weights (Train Split Only):**
  - Class 0 (`strongly_negative`): `7.3269`
  - Class 1 (`weakly_negative`): `2.9535`
  - Class 2 (`neutral`): `0.2405`
  - Class 3 (`weakly_positive`): `4.1868`
  - Class 4 (`strongly_positive`): `7.7755`
