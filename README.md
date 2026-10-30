# ML Assignment 2 — Bagging, Boosting and Bias-Variance on Face Recognition

| Field | Value |
|---|---|
| **Name** | Prem Kumar |
| **Roll Number** | M26CSA017 |
| **Seed** | 26017 |
| **Repository** | https://github.com/chandu-suggu/ml2ass |
| **Report** | `report/report.pdf` (CVPR format, 8 pages) |

---

## Overview

This repository contains the complete, reproducible implementation for Assignment 2 of the Machine Learning course.  
The assignment covers three questions on the Olivetti Faces dataset (40 identities, 10 images each, 64×64 grayscale):

| Question | Topic | Marks |
|---|---|---|
| Q1 | Decision-tree baseline and **Bagging from scratch** | 35 |
| Q2 | Multiclass **SAMME AdaBoost from scratch** | 35 |
| Q3 | **Bias² and Variance** as measured quantities | 30 |

---

## Repository Structure

```
ml2ass/
├── configs/
│   └── config.yaml            # Centralised config (seed, split ratios, hyperparameter grids)
├── data/
│   └── saved_split.npz        # Canonical 6-train / 2-val / 2-test split (seed 26017)
├── figures/                   # All generated figures (PNG)
├── logs/
│   ├── q1_results.json        # Q1 numerical outputs
│   ├── q2_results.json        # Q2 numerical outputs
│   └── q3_results.json        # Q3 numerical outputs
├── notebooks/                 # Original reference notebooks (ASSIGNMENT_2_1/2/3.ipynb)
├── report/
│   ├── report.typ             # Typst source for the CVPR-format report
│   ├── report.pdf             # Compiled 8-page PDF
│   ├── references.bib         # BibTeX references
│   └── figures/               # Figure copies used by the report
├── scripts/
│   ├── run_all.py             # Master reproduction script (runs Q1 → Q2 → Q3)
│   ├── run_q1.py              # Q1 experiments
│   ├── run_q2.py              # Q2 experiments
│   └── run_q3.py              # Q3 experiments
├── src/
│   ├── dataset.py             # Data loading, stratified split, masking
│   ├── bagging.py             # BaggedDecisionTreeEnsemble (from scratch)
│   ├── boosting.py            # SAMMEClassifier (from scratch)
│   └── bias_variance.py       # Bias² / Variance computation (30 refits)
├── environment.yml            # Conda environment file
├── requirements.txt           # pip requirements
├── reproduce.bat              # Windows reproduction commands
└── reproduce.sh               # Linux/macOS reproduction commands
```

---

## Setup

### Option A — Conda

```bash
conda env create -f environment.yml
conda activate ml2ass
```

### Option B — pip

```bash
pip install -r requirements.txt
```

**Python version:** 3.13.2  
**Key packages:** scikit-learn 1.9.1, numpy, matplotlib, scipy, typst (for report compilation)

---

## Reproduction

### Windows

```bat
reproduce.bat
```

### Linux / macOS

```bash
bash reproduce.sh
```

These scripts run `scripts/run_all.py` which executes Q1 → Q2 → Q3 in order, saves logs under `logs/`, and writes all figures to `figures/`.

### Run experiments individually

```bash
python scripts/run_q1.py
python scripts/run_q2.py
python scripts/run_q3.py
```

### Recompile the report PDF

```bash
typst compile report/report.typ report/report.pdf
```

---

## Key Results Summary

### Q1 — Bagging

| Model | Clean Test Acc | Masked Test Acc |
|---|---|---|
| Single Tree (depth=None, msl=4) | 66.25 % | 33.75 % |
| Bagged Ensemble (B=100) | 86.25 % | 75.00 % |

- Theoretical distinct bootstrap fraction: 1 − 1/e ≈ 63.2 % → **151.9 / 240** samples  
- OOB accuracy at B=100: **77.1 %**

### Q2 — SAMME AdaBoost

| Setting | Rounds | Val Acc | Clean Test Acc |
|---|---|---|---|
| Best (depth=4) | 50 | 50.00 % | 40.00 % |
| Bagging (B=100) | — | — | 86.25 % |

- Boosting converges quickly; heavily weighted sample is face index 189 (class 31).

### Q3 — Bias² & Variance (30 refits)

| Model | Bias² | Variance | Mean Test Acc |
|---|---|---|---|
| Single Tree | 0.3946 | 0.6296 | 48.79 % ± 3.69 % |
| Bagging (B=100) | 0.2094 | 0.2798 | 75.54 % ± 4.36 % |
| SAMME Boosting | 0.8923 | 0.0224 | 23.67 % ± 7.98 % |

---

## Dataset Split

The split is saved to `data/saved_split.npz` and reused by all scripts.  
Generated with `numpy.random.RandomState(26017)` (all numeric characters of roll number M26CSA017).  
Split ratio: **6 train / 2 validation / 2 test** per identity.

---

## Implementation Notes

- **Bagging** — `BaggedDecisionTreeEnsemble` draws bootstrap samples of full training-set size, computes OOB predictions using out-of-bag indices, pairwise disagreement via vectorised class-count approach, and vote margin as winner-fraction minus runner-up-fraction.  
- **Boosting** — `SAMMEClassifier` implements the SAMME update rule exactly:  
  α_t = log((1−ε_t)/ε_t) + log(K−1). Weak learners that are not better than random chance (ε_t ≥ 1 − 1/K) are skipped. Confidence uses normalised weighted vote q_k(x) = S_k(x) / Σ_t α_t.  
- **No PCA or dimensionality reduction** is used anywhere; raw flattened 4096-dim pixels only.  
- Models are trained exclusively on clean images; the masked test set is evaluation-only.

---

## License

For academic use only (ML course, 2026).
