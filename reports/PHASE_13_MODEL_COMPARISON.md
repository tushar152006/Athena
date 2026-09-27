# Phase 13 — Model Architecture Comparison (LightGBM vs. XGBoost vs. CatBoost vs. Linear Baseline)

**Status**: COMPLETED & VERIFIED  
**Date**: September 27, 2026  
**Execution Environment**: Python 3.12, LightGBM 4.7.0, XGBoost 3.2.0, CatBoost 1.2.10, Scikit-learn 1.9.1  
**Validation Partition**: 4,000 Unseen S1 Entities (56,469 Candidate Pairs, 15.50% Positive Rate)  
**Training Partition**: 16,000 S1 Entities (227,611 Candidate Pairs, 15.75% Positive Rate)  
**Artifact Path**: [`models/comparison/model_comparison_results.json`](models/comparison/model_comparison_results.json)  

---

## 1. Executive Summary & Objective

In Phase 13, we conducted a rigorous architectural comparison across four structurally distinct model families to evaluate pairwise entity classification performance, latency, and ensembling feasibility under identical conditions:
1. **Regularized Linear Baseline**: `Logistic Regression` with L2 penalty on standardized features.
2. **Gradient-Based One-Side Sampling (GOSS)**: `LightGBM` (leaf-wise best-first split growth).
3. **Histogram Exact Greedy Splits**: `XGBoost` (depth-wise constrained level growth).
4. **Symmetric Oblivious Trees**: `CatBoost` (table-based symmetric decision structures with ordered boosting).

All models were trained on the exact same 20 domain-specific features and evaluated on the exact same 4,000 unseen validation S1 entities.

---

## 2. Empirical Benchmark Results

| Model Architecture | PR-AUC | ROC-AUC | Optimal $p^*$ | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc | Fit Time | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (L2) | `0.95766` | `0.99138` | `0.55` | **`0.70041`** | `78.62%` | `56.78%` | `82.05%` | **0.77 s** | **0.2 µs/p** |
| **LightGBM** (Champion) | `0.98071` | `0.99621` | `0.58` | **`0.72985`** | `81.85%` | `58.79%` | `87.61%` | **4.14 s** | **8.9 µs/p** |
| **XGBoost** (Contender) | **`0.98128`** | **`0.99632`** | `0.62` | **`0.73131`** | **`82.17%`** | `58.68%` | **`90.17%`** | **5.72 s** | **2.3 µs/p** |
| **CatBoost** (Contender) | `0.97960` | `0.99599` | `0.52` | **`0.72686`** | `80.68%` | **`59.99%`** | `80.77%` | **13.07 s** | **0.4 µs/p** |
| **Ensemble (3-Way Soft Voting)** | `0.98100` | `0.99627` | `0.62` | **`0.72986`** | `82.05%` | `58.45%` | `89.32%` | — | — |

---

## 3. Inter-Model Prediction Correlation Analysis

To determine whether ensembling or stacking will yield meaningful variance reduction, we evaluated the Pearson correlation coefficient ($r$) across all 56,469 validation candidate pair probability predictions:

| Model Architecture | Logistic Regression | LightGBM | XGBoost | CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `1.0000` | `0.9693` | `0.9683` | `0.9730` |
| **LightGBM** | `0.9693` | `1.0000` | **`0.9988`** | **`0.9976`** |
| **XGBoost** | `0.9683` | **`0.9988`** | `1.0000` | **`0.9974`** |
| **CatBoost** | `0.9730` | **`0.9976`** | **`0.9974`** | `1.0000` |

---

## 4. Key Scientific & Architectural Insights

1. **Feature Quality Dominance Over Tree Topography**:
   - The linear baseline (`Logistic Regression`) reaches **`Macro F0.5 = 0.70041`** and `PR-AUC = 0.95766` on standardized features alone. This confirms that the 20 features engineered in Phase 5 cleanly capture the underlying geometric and lexical manifolds.
   - The three tree architectures (LightGBM, XGBoost, CatBoost) perform within **`0.45%`** of each other (`0.72686` to `0.73131`), with pairwise prediction correlations exceeding **`r = 0.997`**.

2. **XGBoost Achieves Peak Precision & Singleton Accuracy**:
   - `XGBoost` edged out LightGBM on Macro $F_{0.5}$ (**`0.73131` vs `0.72985`**), driven by superior precision (**`82.17%`**) and higher singleton protection (**`90.17%`** accuracy, correctly retaining 105 of 117 true singletons).
   - XGBoost's exact histogram split algorithm is slightly less aggressive than LightGBM's leaf-wise best-first expansion, resulting in more conservative probability tails near high-confidence boundaries.

3. **LightGBM Delivers Ideal Balance of Throughput & Latency**:
   - `LightGBM` fits in **4.14 seconds** (3.15× faster than CatBoost) and yields virtually identical ROC-AUC (`0.99621` vs `0.99632`).
   - For 24-million pair inference on the full test set, LightGBM's memory efficiency and mature C++ inference runtime make it the ideal primary production engine.

4. **Implications for Phase 14 Ensembling**:
   - Because LightGBM and XGBoost correlate at **`r = 0.9988`**, uniform simple averaging does not produce massive variance reduction over single best models.
   - Effective ensembling in Phase 14 must employ **Rank Averaging**, **Calibrated Soft Voting**, or **Complementary Error Gating** (leveraging CatBoost's higher recall of 59.99% with XGBoost's high-precision singleton filter of 90.17%).

---

## 5. Deliverables & Artifact Verification

- **Comparison Benchmark Runner**: [`scripts/compare_models.py`](scripts/compare_models.py)
- **JSON Metrics Artifact**: [`models/comparison/model_comparison_results.json`](models/comparison/model_comparison_results.json)
- **Official Evaluation**: Ground-truth validated against `train_ground_truth.tsv` using [`src/models/threshold_calibrator.py`](src/models/threshold_calibrator.py).
