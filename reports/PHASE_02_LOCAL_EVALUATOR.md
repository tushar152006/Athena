# Phase 2 — Local Evaluator & Measurement Harness Report

**Challenge**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Document**: `reports/PHASE_02_LOCAL_EVALUATOR.md`  
**Date**: September 27, 2026  
**Status**: COMPLETE — [EXPERIMENTALLY VERIFIED & UNIT TESTED]  
**Primary Source of Truth**: Official Challenge Video, Rules, Dataset, and Validator  

---

## 1. Phase Objective & Governance

The objective of Phase 2 is to construct a production-grade, mathematically verified, standalone local evaluation engine (`src/evaluation/evaluator.py`) that implements:
1. The official competition metric: **Macro $F_{0.5}$ per Source 1 entity**.
2. Strict singleton boundary conditions ($0.0$ for hallucinations, $1.0$ for empty predictions).
3. Candidate generation / blocking metrics (Pairs Completeness, Reduction Ratio, Parsimony Distribution).
4. Sub-minute evaluation throughput across all $2,206,821$ training entities with low memory footprint ($\le 2.1\text{ GB}$).
5. Full compliance with the official submission format checker (`student_resource/utils/validate_submission.py`).

---

## 2. Mathematical Metric Formulation

### 2.1 Entity-Level Evaluation
For each Source 1 entity $i \in \{1, \dots, N_{S1}\}$:
* Let $T_i$ be the set of ground-truth matched IDs in $S_2 \cup S_3$.
* Let $P_i$ be the set of predicted matched IDs in $S_2 \cup S_3$.

$$\text{True Positives } (TP_i) = |P_i \cap T_i|$$
$$\text{False Positives } (FP_i) = |P_i \setminus T_i|$$
$$\text{False Negatives } (FN_i) = |T_i \setminus P_i|$$

#### Singleton & Edge Case Handling Rules:
* **Rule 1 (True Singleton Correctly Predicted)**:
  $$\text{If } |T_i| = 0 \text{ and } |P_i| = 0 \implies Precision_i = 1.0, \; Recall_i = 1.0, \; F_{0.5}^{(i)} = 1.0$$
* **Rule 2 (Singleton with Hallucinated False Positive)**:
  $$\text{If } |T_i| = 0 \text{ and } |P_i| > 0 \implies Precision_i = 0.0, \; Recall_i = 0.0, \; F_{0.5}^{(i)} = 0.0$$
* **Rule 3 (Matched Entity with Empty Prediction)**:
  $$\text{If } |T_i| > 0 \text{ and } |P_i| = 0 \implies Precision_i = 0.0, \; Recall_i = 0.0, \; F_{0.5}^{(i)} = 0.0$$
* **Rule 4 (Standard Matched Entity)**:
  When $|P_i| > 0$ and $|T_i| > 0$:
  $$Precision_i = \frac{TP_i}{|P_i|}, \quad Recall_i = \frac{TP_i}{|T_i|}$$
  $$F_{0.5}^{(i)} = \frac{(1 + 0.5^2) \cdot Precision_i \cdot Recall_i}{0.5^2 \cdot Precision_i + Recall_i} = \frac{1.25 \cdot Precision_i \cdot Recall_i}{0.25 \cdot Precision_i + Recall_i}$$
  If $Precision_i + Recall_i = 0$, $F_{0.5}^{(i)} = 0.0$.

### 2.2 Dataset-Level Macro Metric
$$\text{Macro } F_{0.5} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} F_{0.5}^{(i)}$$

$$\text{Macro Precision} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} Precision_i, \quad \text{Macro Recall} = \frac{1}{N_{S1}} \sum_{i=1}^{N_{S1}} Recall_i$$

$$\text{Singleton Accuracy} = \frac{1}{N_{\text{singleton}}} \sum_{i: |T_i|=0} \mathbb{I}(|P_i| = 0)$$

$$\text{Matched Macro } F_{0.5} = \frac{1}{N_{\text{matched}}} \sum_{i: |T_i|>0} F_{0.5}^{(i)}$$

---

## 3. Metric Dynamics: The 4x Precision Weighting

Under $\beta = 0.5$, precision is prioritized $4\times$ over recall:
$$\frac{\partial F_\beta / \partial P}{\partial F_\beta / \partial R} = \frac{1}{\beta^2} = \frac{1}{0.5^2} = 4.0$$

### Verified Demonstration:
* **Scenario A (High Precision, Low Recall)**: $P = 1.0, R = 0.5$
  $$F_{0.5} = \frac{1.25 \times 1.0 \times 0.5}{0.25 \times 1.0 + 0.5} = \frac{0.625}{0.750} = \frac{5}{6} \approx \mathbf{0.833333}$$
* **Scenario B (Low Precision, High Recall)**: $P = 0.5, R = 1.0$
  $$F_{0.5} = \frac{1.25 \times 0.5 \times 1.0}{0.25 \times 0.5 + 1.0} = \frac{0.625}{1.125} = \frac{5}{9} \approx \mathbf{0.555556}$$

**Conclusion**: A single false positive degrades $F_{0.5}$ far more than a missed match. Candidate post-processing and classification thresholds must be precision-biased ($\tau^* \ge 0.80 - 0.85$).

---

## 4. Candidate Generation / Blocking Metrics (Christen 2012)

To rigorously evaluate the blocking phase before running matching:
1. **Pairs Completeness (Candidate Recall)**:
   $$PC = \frac{\sum_i |C_i \cap T_i|}{\sum_i |T_i|} = \frac{|M \cap C|}{|M|}$$
2. **Reduction Ratio (RR)**:
   $$RR = 1 - \frac{\sum_i |C_i|}{N_{S1} \times (N_{S2} + N_{S3})}$$
3. **Candidate Parsimony Distribution**:
   Evaluates the organizer's secondary ranking criterion (smaller candidate set sizes per S1 entity rank higher beyond the leaderboard):
   * $\text{Mean}(|C_i|), \text{Median}(|C_i|), \text{Min}(|C_i|), \text{Max}(|C_i|)$
   * $P_{90}, P_{95}, P_{99}$ percentiles
   * Fraction of entities exceeding max target threshold ($|C_i| > 25$, $|C_i| > 50$)

---

## 5. Unit Test Verification Matrix

All unit tests in `tests/test_evaluator.py` were executed and verified:

| Test ID | Test Name | Expected Result | Actual Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| `test_01` | Perfect Prediction | Macro $F_{0.5} = 1.000000$ | $1.000000$ | **PASS** |
| `test_02` | Disjoint Predictions | Macro $F_{0.5} = 0.000000$ | $0.000000$ | **PASS** |
| `test_03` | Singleton with Empty Prediction | $F_{0.5} = 1.000000$ | $1.000000$ | **PASS** |
| `test_04` | Singleton with False Positive | $F_{0.5} = 0.000000$ | $0.000000$ | **PASS** |
| `test_05` | Matched with Empty Prediction | $F_{0.5} = 0.000000$ | $0.000000$ | **PASS** |
| `test_06` | Precision vs Recall Weighting | High P ($0.833333$) > Low P ($0.555556$) | $\Delta = +0.277778$ | **PASS** |
| `test_07` | Official Submission Validator Check | 0 Errors from `validate_submission.py` | 0 Errors | **PASS** |
| `test_08` | Candidate Blocking Evaluation | $PC = 0.666667$, $RR = 0.991667$ | Matches formula | **PASS** |

---

## 6. Large-Scale Performance & Scalability Benchmark

The evaluator was benchmarked on the full official training ground truth dataset (`train_ground_truth.tsv`, **2,206,821 records**, **7,638,365 true pairs**):

```
================ BENCHMARK FINAL METRICS ================
Total Benchmark Execution Time      : 40.60s
File Load Time (2,206,821 entities) : 7.62s
Prediction Eval (2.2M entities)     : 13.56s
Candidate Eval (2.2M entities)      : 8.77s
Noisy Eval (2.2M entities)          : 4.15s
Peak Process RSS                    : 2073.7 MB (2.03 GB)
=========================================================
```

### Self-Evaluation Verification (Ground Truth vs. Ground Truth):
* **Macro $F_{0.5}$ Score**: **$1.000000$**
* **Macro Precision**: **$1.000000$**
* **Macro Recall**: **$1.000000$**
* **Singleton Accuracy**: **$1.000000$** ($123,247 / 123,247$)
* **Matched Macro $F_{0.5}$**: **$1.000000$** ($2,083,574 / 2,083,574$)
* **Total True Positives**: **$7,638,365$**
* **Total False Positives**: **$0$**
* **Total False Negatives**: **$0$**

### Candidate Blocking Benchmark on True Matches:
* **Pairs Completeness**: **$1.000000$** ($7,638,365 / 7,638,365$)
* **Reduction Ratio**: **$0.99999966$**
* **Mean Candidates / Entity**: **$3.46$** (Median: $3.0$)
* **Min / Max Candidates**: $0$ / $11$
* **90th / 95th / 99th Percentile**: $6.0$ / $6.0$ / $8.0$
* **Entities with $>25$ Candidates**: **$0.00\%$**

---

## 7. Evidence-Based Implications for Phase 3+

1. **High Speed & Low Memory Overhead**: Evaluating 2.2M predictions takes **$13.56$ seconds** and **$2.03\text{ GB}$ RAM**. Local cross-validation cycles can run rapidly without GPU requirements.
2. **Threshold Guard**: Due to the severe penalty on false positives ($F_{0.5}$ drops to $0.555$ at $50\%$ precision even with $100\%$ recall), any decision layer in Phase 7/11 must set conservative confidence thresholds.
3. **Singleton Guard**: Any model predicting candidate matches for the $5.58\%$ singleton entities destroys their score from $1.0$ to $0.0$. A dedicated singleton detection rule or high minimum threshold is essential.

---

## 8. Reproducibility & Environment Details

* **Python Version**: 3.11.15
* **Dependencies**: `numpy>=1.26.0`, `scipy>=1.13.0`, `polars>=1.0.0`, `rapidfuzz>=3.9.0`, `psutil>=5.9.0`
* **Package Spec**: Managed via [`pyproject.toml`](file:///c:/Users/DELL/Downloads/Amazon_ML/pyproject.toml)
* **Execution Command**:
  ```bash
  uv run python -m unittest tests/test_evaluator.py
  uv run python scripts/benchmark_evaluator.py
  ```

---

## 9. Quality Control Checklist

- [x] Evaluator implemented in `src/evaluation/evaluator.py`
- [x] Official Macro $F_{0.5}$ metric accurately formulated
- [x] Singleton edge case scoring verified ($1.0$ for empty, $0.0$ for hallucination)
- [x] Pairs Completeness (Candidate Recall) implemented
- [x] Reduction Ratio and Parsimony distribution statistics implemented
- [x] All 8 unit tests in `tests/test_evaluator.py` passing
- [x] Benchmark on full 2.2M ground truth completed ($1.000000$ verified)
- [x] Formats compatible with official `student_resource/utils/validate_submission.py`
- [x] No modifications to original dataset files (strictly read-only)
- [x] Phase 2 report completed and committed to repository
