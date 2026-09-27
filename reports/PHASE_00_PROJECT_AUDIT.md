# Phase 00 — Project Audit & Leaderboard Intelligence Report

**Project**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Date**: September 27, 2026  
**Status**: Completed — Awaiting Phase 1 Approval  
**Audit Author**: Engineering & Research Team  

---

## 1. Executive Summary

This audit establishes the baseline inventory, environment state, available tools, and competitive landscape prior to writing solution code. In strict adherence to our project governance principles:
* No solution architecture has been prematurely locked.
* Original data files remain strictly read-only.
* All decisions and findings are categorized as either **OFFICIAL**, **EXPERIMENTALLY VERIFIED**, or **UNKNOWN — REQUIRES VERIFICATION**.

---

## 2. Leaderboard Intelligence (from Official Portal Screenshot)

A live screenshot of the official competition leaderboard ("Top Gainers") was inspected.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OFFICIAL LEADERBOARD TOP GAINERS                         │
├──────┬──────────────────────┬────────────────────────────────┬──────────────┤
│ Rank │ Team Name            │ Institution / Campus           │ Score (F0.5) │
├──────┼──────────────────────┼────────────────────────────────┼──────────────┤
│ 1st  │ GG                   │ Lovely Professional Univ (LPU) │ **0.991811** │
│ 2nd  │ CDS_Team_iisc        │ IISc Bangalore                 │ **0.991483** │
│ 3rd  │ Team X               │ IIT Madras                     │ **0.991170** │
└──────┴──────────────────────┴────────────────────────────────┴──────────────┘
```

### Key Observations & Deductions:
1. **Extremely High Score Band ($> 0.991$)**:
   * The top teams on the public leaderboard have achieved scores above **$0.991$**.
   * *Deduction*: Under the Macro $F_{0.5}$ metric (which penalizes false merges $4\times$ heavier than misses), a score of $0.9918$ indicates that the matching problem on the public test subset contains a very high degree of deterministic or near-deterministic signal when text canonicalization, country partitioning, and high-precision candidate blocking are correctly aligned.
   * *Caution*: High public leaderboard scores also carry risk of public test set overfitting (e.g., threshold tuning to the public split). Our validation must guard against this via out-of-fold cross-validation on `source1_entity_id`.
2. **Competitive Landscape**:
   * Top university research teams (IISc Bangalore, IIT Madras, LPU) are actively submitting.
   * We do **not** assume or infer the internal architectures of these teams; our pipeline will be derived strictly from our empirical validation.

---

## 3. Existing Project Inventory

### 3.1 Existing Documentation & Knowledge Base (Reusable)
* `README.md` (24.1 KB): Official problem statement, schemas, and requirements. (Status: Complete, Version Controlled).
* `AMAZON_ML_2026_RESEARCH_REPORT.md` (36.5 KB): Comprehensive competition intelligence, metric derivation, and failure mode matrix. (Status: Complete).
* `LEARNING_RESOURCE_MAP.md` (6.1 KB): Component-to-literature mapping table. (Status: Complete).
* `READING_WHILE_BUILDING.md` (7.6 KB): Stage-by-stage engineering manual. (Status: Complete).
* `TECHNICAL_LIBRARY.md` (20.0 KB): Peer-reviewed bibliography and Python library inventory. (Status: Complete).
* `student_resource/README.md` (13.8 KB): Official problem statement and instructions from Unstop. (Status: Official Source of Truth).
* `student_resource/Documentation_template.md` (2.1 KB): Official methodology submission template. (Status: Ready to be filled).

### 3.2 Existing Tools & Scripts (Reusable)
* `student_resource/utils/validate_submission.py` (13.4 KB):
  * **Role**: Official submission formatting validator (Python stdlib only).
  * **Verification**: Checks exact headers, tab delimiters, valid test IDs, singleton representation, and candidate subset constraints.
  * **Reusability**: **100% Reusable**. Must be integrated as an automated pre-submission gate.

### 3.3 Existing Datasets (Strictly Read-Only)

| File Path | Size | Record Count | Status | Access Policy |
| :--- | :--- | :--- | :--- | :--- |
| `student_resource/dataset/train/train_source1.tsv` | 205.1 MB | 2,206,821 | Official Train | **READ-ONLY** |
| `student_resource/dataset/train/train_source2.tsv` | 477.8 MB | 5,034,616 | Official Train | **READ-ONLY** |
| `student_resource/dataset/train/train_source3.tsv` | 491.9 MB | 5,285,603 | Official Train | **READ-ONLY** |
| `student_resource/dataset/train/train_ground_truth.tsv` | 124.0 MB | 2,206,821 | Official Train | **READ-ONLY** |
| `student_resource/dataset/test/test_source1.tsv` | 170.9 MB | 1,732,544 | Official Test | **READ-ONLY** |
| `student_resource/dataset/test/test_source2.tsv` | 497.5 MB | 4,887,273 | Official Test | **READ-ONLY** |
| `student_resource/dataset/test/test_source3.tsv` | 494.1 MB | 5,082,316 | Official Test | **READ-ONLY** |

* **Total Disk Usage**: $\approx 2.47\text{ GB}$ of uncompressed TSVs.
* **Total Entities**: **24,229,173 records**.

### 3.4 Media & Non-Essential Files
* `6ab509c5b7036_ml_challenge_2026_video.mp4` (13.4 MB): Official video (inspected and documented). Excluded from git.
* `__MACOSX/` & `.DS_Store`: macOS artifact files. Excluded from git via `.gitignore`.

### 3.5 Existing Notebooks, Outputs, & Experiments
* **Notebooks**: None yet.
* **Outputs**: None yet (`output/` folder not yet created).
* **Experiments**: None yet (`experiments/EXPERIMENT_LOG.md` to be initialized).

---

## 4. Compute & Dependency Environment Audit

### 4.1 Python Interpreters on System
* System has `uv` package manager installed (`uv 0.11.22`).
* Available Python installations:
  * Python 3.13 (64-bit)
  * Python 3.12 (64-bit)
  * CPython 3.11.15 (64-bit)
* Active agent virtualenv: `Python 3.11.15`.
* Current package status in active venv:
  * `tqdm`: Available (`4.67.3`)
  * `polars`: Not installed
  * `pandas`: Not installed
  * `rapidfuzz`: Not installed
  * `lightgbm`: Not installed
  * `scikit-learn`: Not installed
  * `duckdb`: Not installed

### 4.2 Compute Constraints & Strategy
* **Local Machine**: Lightweight development, schema inspection, unit testing, validation scripts, and small-batch prototyping.
* **Large-Scale Execution (Training / Full Test Inference)**:
  * High-memory operations (e.g., full feature computation across millions of pairs) will be executed either in out-of-core streaming chunks via Polars on CPU or offloaded to Google Colab / AWS SageMaker using the team's $200 AWS credit allotment.
  * GPU is **NOT** required for the GBDT baseline; CPU histogram LightGBM runs with minimal RAM.

---

## 5. Proposed Project Architecture

To maintain strict modularity without unnecessary complexity, we propose creating the following minimal directory structure as phases advance:

```
Amazon_ML/
├── .gitignore                          # Excludes datasets (>100MB) & cache
├── README.md                           # Problem statement reference
├── AMAZON_ML_2026_RESEARCH_REPORT.md   # Research report
├── LEARNING_RESOURCE_MAP.md            # Literature mapping
├── READING_WHILE_BUILDING.md           # Engineering manual
├── TECHNICAL_LIBRARY.md                # Comprehensive bibliography
│
├── student_resource/                   # Official resources (READ-ONLY)
│   ├── dataset/
│   │   ├── train/                      # READ-ONLY raw train TSVs
│   │   └── test/                       # READ-ONLY raw test TSVs
│   ├── utils/
│   │   └── validate_submission.py      # Official validator
│   └── Documentation_template.md       # Final submission report template
│
├── src/                                # Core solution package
│   ├── __init__.py
│   ├── data/                           # Data loading & chunked streaming
│   ├── preprocessing/                  # Text & address canonicalization
│   ├── blocking/                       # Candidate generation & indexing
│   ├── features/                       # Pairwise feature extraction
│   ├── models/                         # GBDT classifier training & inference
│   ├── evaluation/                     # Local Macro F0.5 & candidate recall
│   └── submission/                     # Formatted output & zip packaging
│
├── reports/                            # Phase milestone reports
│   └── PHASE_00_PROJECT_AUDIT.md       # This document
│
├── experiments/                        # Experiment tracking
│   └── EXPERIMENT_LOG.md               # Master ledger of all runs
│
├── output/                             # Generated submission files
│   ├── matching_results.tsv            # Leaderboard matches
│   └── candidate_pairs.tsv             # Audited candidate set
│
└── requirements.txt                    # Pinned dependencies
```

---

## 6. What Can Be Reused vs. What Needs Building

1. **Directly Reused (Zero Code Needed)**:
   * Official validator (`student_resource/utils/validate_submission.py`).
   * Documentation template (`student_resource/Documentation_template.md`).
   * Raw dataset files (`student_resource/dataset/` read-only).
2. **To Be Built in Next Phases**:
   * **Phase 1**: Lightweight dataset profiler (`reports/PHASE_01_DATASET_PROFILE.md`).
   * **Phase 2**: Local evaluation harness implementing exact Macro $F_{0.5}$, singleton scoring, and candidate recall metrics (`src/evaluation/`).
   * **Phase 3**: Simple baseline (`BASELINE_001` — exact normalized name matching).

---

## 7. Decision Gate & Sign-Off

* [x] Project audit completed.
* [x] Leaderboard screenshot inspected and recorded.
* [x] Dataset files inventoried and confirmed read-only.
* [x] Environment and dependency status identified.
* [ ] **Gate**: Awaiting user approval to proceed to **Phase 1 (Dataset Profiling)** and **Phase 2 (Local Evaluator)**.
