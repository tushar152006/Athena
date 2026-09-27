# Phase 14 — Ensembling & Blending (Rank Averaging, Calibrated Soft Voting & Precision-Guarded Stacking)

**Status**: COMPLETED & VERIFIED  
**Date**: September 27, 2026  
**Execution Environment**: Python 3.12, LightGBM, XGBoost, CatBoost, Scikit-learn, Scipy  
**Validation Partition**: 4,000 Unseen S1 Entities (56,469 Candidate Pairs)  
**Training Partition**: 16,000 S1 Entities (227,611 Candidate Pairs)  
**Artifact Path**: [`models/ensemble/ensemble_config.json`](models/ensemble/ensemble_config.json)  

---

## 1. Executive Summary & Objective

In Phase 13, architectural benchmarking revealed distinct behavioral strengths across model families:
- **XGBoost**: Peak precision (`82.17%`) and highest singleton protection (`90.17%` accuracy).
- **LightGBM**: Highly balanced precision/recall tradeoff (`81.85%` / `58.79%`) with ultra-fast inference throughput.
- **CatBoost**: Highest raw recall (`59.99%`), capturing edge candidates missed by shallower trees.

Phase 14 designed, implemented, and empirically benchmarked three advanced ensembling paradigms to synthesize these complementary strengths:
1. **Calibrated Soft Voting**: Simplex-optimized linear probability blending ($w_{\text{lgb}}, w_{\text{xgb}}, w_{\text{cat}}$).
2. **Percentile Rank Averaging**: Borda count score mapping to eliminate inter-library probability calibration drift.
3. **Precision-Guarded Consensus Stacking**: A hierarchical policy employing an XGBoost singleton veto, 2-of-3 multi-model consensus promotion, and adaptive margin gap filtering ($\delta = 0.30$).

---

## 2. Empirical Benchmark Results

| Strategy | PR-AUC | ROC-AUC | Optimal Threshold | Macro $F_{0.5}$ | Macro Prec | Macro Rec | Singleton Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single: LightGBM** | `0.98071` | `0.99621` | `0.58` | `0.72985` | `81.85%` | `58.79%` | `87.61%` |
| **Single: XGBoost** | **`0.98128`** | **`0.99632`** | `0.62` | `0.73131` | `82.17%` | `58.68%` | **`90.17%`** |
| **Single: CatBoost** | `0.97914` | `0.99590` | `0.52` | `0.72591` | `80.57%` | **`59.95%`** | `79.92%` |
| **Ensemble: Equal Soft Voting (33/33/34)** | `0.98090` | `0.99625` | `0.58` | `0.72976` | `81.84%` | `58.80%` | `87.61%` |
| **Ensemble: Optimal Soft Voting (30/60/10)** | `0.98118` | `0.99631` | `0.58` | `0.73070` | `81.93%` | `58.91%` | `87.61%` |
| **Ensemble: Percentile Rank Averaging** | `0.98117` | `0.99630` | `0.85` | `0.72828` | `81.13%` | `59.60%` | `84.19%` |
| **Stacking: Precision-Guarded Consensus** | `0.98118` | `0.99631` | `0.58` | **`0.73260`** | **`82.80%`** | `58.04%` | `79.49%` |

---

## 3. Simplex Blending Weight Optimization

Grid search across the 3-model probability simplex identified the optimal linear mixture:

| LightGBM ($w_{\text{lgb}}$) | XGBoost ($w_{\text{xgb}}$) | CatBoost ($w_{\text{cat}}$) | Optimal $p^*$ | Macro $F_{0.5}$ | Macro Prec | Macro Rec |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.33` | `0.33` | `0.33` | `0.58` | `0.72976` | `81.84%` | `58.80%` |
| `0.50` | `0.35` | `0.15` | `0.58` | `0.73024` | `81.89%` | `58.83%` |
| `0.40` | `0.45` | `0.15` | `0.58` | `0.73006` | `81.87%` | `58.83%` |
| `0.35` | `0.50` | `0.15` | `0.58` | `0.73030` | `81.89%` | `58.87%` |
| `0.45` | `0.45` | `0.10` | `0.58` | `0.73007` | `81.88%` | `58.82%` |
| **`0.30`** | **`0.60`** | **`0.10`** | **`0.58`** | **`0.73070`** | **`81.93%`** | **`58.91%`** |
| `0.60` | `0.30` | `0.10` | `0.58` | `0.73049` | `81.93%` | `58.83%` |
| `0.40` | `0.40` | `0.20` | `0.60` | `0.72993` | `81.96%` | `58.67%` |
| `0.50` | `0.50` | `0.00` | `0.62` | `0.73069` | `82.10%` | `58.62%` |

The simplex surface strongly favors **XGBoost ($60\%$)** as the primary precision anchor, supported by **LightGBM ($30\%$)** and **CatBoost ($10\%$)**.

---

## 4. Architectural Analysis: Precision-Guarded Stacking

The champion policy—**Precision-Guarded Stacking**—achieved the top score of **`Macro F0.5 = 0.73260`** with **`82.80%` Precision**:
1. **XGBoost Hard Veto ($P_{\text{xgb}} < 0.25$)**: If XGBoost decisively rules out a candidate pair, it is suppressed even if CatBoost assigns marginal probability mass. This decisively truncates the false-positive tail.
2. **2-of-3 Multi-Model Consensus Promotion**: Candidates where at least two architectures output $P \ge 0.50$ are retained even if their blended probability dips slightly below $0.58$.
3. **Adaptive Margin Gap ($\delta = 0.30$)**: Enforces entity-level confidence grouping, retaining only matches within $0.30$ probability gap of the top candidate.

---

## 5. Deliverables & Artifact Verification

- **Ensemble Blender Module**: [`src/models/ensemble_blender.py`](src/models/ensemble_blender.py)
- **Runner Script**: [`scripts/run_ensemble_blend.py`](scripts/run_ensemble_blend.py)
- **JSON Configuration Artifact**: [`models/ensemble/ensemble_config.json`](models/ensemble/ensemble_config.json)
- **Research Report**: [`reports/PHASE_14_ENSEMBLING.md`](reports/PHASE_14_ENSEMBLING.md)
- **Ledger Entries**: EXP-014 logged in [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md).
