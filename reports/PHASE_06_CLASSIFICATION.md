# PHASE 6 — PAIRWISE CLASSIFIER & GRADIENT BOOSTING MODEL REPORT
**Amazon ML Challenge 2026 — Business Entity Resolution**  
**Stage III — Candidate Scoring & Classification**  
**Experiment ID**: `EXP-006`  
**Date**: September 27, 2026  
**Status**: COMPLETED  

---

## 1. Executive Summary & Objective

The objective of **Phase 6 (Pairwise Classifier & Gradient Boosting Model)** is to train, tune, and calibrate a high-discriminative gradient boosting pairwise scoring model on the candidate pairs generated in Phase 4 using the 20 domain-specific features developed in Phase 5.

Because the official competition evaluation metric is **Macro $F_{0.5}$ per Source 1 entity**, precision is weighted **twice as much as recall** ($\beta = 0.5$). The model must avoid speculative false-positive matching while accurately identifying true positive matches from the 6.47 : 1 negative-to-positive candidate pool.

### Headline Results:
* **Validation Macro $F_{0.5}$**: **`0.72953`** (vs Phase 3 Baseline Floor of `0.30361`, an absolute improvement of **`+0.42592`** or **`+140.28%`**).
* **Validation Macro Precision**: **`0.81925`** ($81.93\%$).
* **Validation Macro Recall**: **`0.58615`** ($58.62\%$).
* **Singleton Accuracy**: **`0.88034`** ($88.03\%$).
* **ROC-AUC**: **`0.9962`** | **PR-AUC**: **`0.9807`**.
* **Optimal Decision Threshold $p^*$**: **`0.60`**.

---

## 2. Experimental Setup & Split Methodology

To prevent any data leakage across candidate pairs:
1. **Entity-Stratified Group Splitting**:
   * We sampled **20,000 Source 1 entities** and their nominated candidate pairs from `output/blocking/candidate_pairs.tsv`.
   * Entities were partitioned into **16,000 Training S1 entities** ($80\%$) and **4,000 Validation S1 entities** ($20\%$).
   * Crucially, the split was performed strictly at the **Source 1 entity level**: no Source 1 entity present in the training set appears in the validation set.
2. **Dataset Dimensions**:
   * **Train Set**: 227,611 pairwise instances (35,845 True Positives, 191,766 Hard Negatives; $5.35 : 1$ negative ratio).
   * **Validation Set**: 56,469 pairwise instances (unseen Source 1 entities).
3. **Hardware & Execution Efficiency**:
   * Feature extraction: 227,611 train pairs extracted in **76.77s** (**2,965 pairs/sec**).
   * LightGBM fitting (350 trees): **34.2s**.
   * Peak RAM usage: $< 1.2\text{ GB}$.

---

## 3. Model Architecture & Hyperparameters

We implemented `LightGBMPairwiseClassifier` in `src/models/pairwise_classifier.py` using `LightGBM 4.7.0`:

```python
LightGBMPairwiseClassifier(
    objective="binary",
    n_estimators=350,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
)
```

Training was monitored with early stopping on validation binary logloss and ROC-AUC. Over 350 boosting rounds, validation binary logloss decreased monotonically from `0.0797` to `0.0567`, achieving a validation ROC-AUC of `0.996207`.

---

## 4. Decision Threshold Calibration Sweep

The official metric evaluates Macro $F_{0.5}$ per Source 1 entity:
$$F_{0.5} = \frac{(1 + 0.5^2) \times P \times R}{(0.5^2 \times P) + R} = \frac{1.25 \times P \times R}{0.25 P + R}$$

We swept classification probability thresholds $p \in [0.40, 0.95]$ on the 4,000 validation Source 1 entities:

| Threshold $p^*$ | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | Comments |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0.40 | 0.72292 | 0.79234 | 0.61542 | 0.73504 | Slightly loose; too many FP look-alikes |
| 0.45 | 0.72691 | 0.80224 | 0.60860 | 0.77350 | Precision climbs above 80% |
| 0.50 | 0.72925 | 0.80791 | 0.60470 | 0.81624 | Standard threshold |
| 0.55 | 0.72849 | 0.81564 | 0.58988 | 0.85470 | High precision |
| **0.60** | **0.72953** | **0.81925** | **0.58615** | **0.88034** | **GLOBAL OPTIMUM ($p^* = 0.60$)** |
| 0.65 | 0.72799 | 0.81947 | 0.58166 | 0.90598 | Peak precision plateau |
| 0.70 | 0.72411 | 0.81719 | 0.57532 | 0.91453 | Recall drop begins to outweigh precision |
| 0.75 | 0.71878 | 0.81480 | 0.56629 | 0.92308 | Overly conservative |
| 0.80 | 0.71331 | 0.81226 | 0.55681 | 0.93162 | Under-matching |
| 0.85 | 0.70422 | 0.80588 | 0.54466 | 0.97009 | Missing subtle variants |
| 0.90 | 0.69250 | 0.79688 | 0.52877 | 0.98291 | High singleton bias |
| 0.95 | 0.66003 | 0.77202 | 0.48819 | 0.99573 | Extreme under-prediction |

### Threshold Sweep Insights:
1. **The $p^* = 0.60$ Sweet Spot**: Because $\beta = 0.5$ penalizes precision mistakes twice as heavily, setting $p^* = 0.60$ lifts precision to **$81.93\%$** while preserving strong recall ($58.62\%$), yielding the peak Macro $F_{0.5} = \mathbf{0.72953}$.
2. **Singleton Accuracy**: As threshold increases, singleton accuracy rises from $73.5\%$ to $99.5\%$, confirming that higher thresholds correctly protect true singletons from false-positive matches.

---

## 5. Feature Importance Analysis (Gain & Split)

The trained LightGBM booster reveals the relative importance of all 20 features:

| Rank | Feature Name | Total Gain Importance | Split Count | Role in Decision Tree |
|:---:|:---|:---:|:---:|:---|
| 1 | **`addr_qgram_jaccard`** | **731,861.52** | 1,068 | **Dominant Discriminator**: Primary split feature separating look-alikes |
| 2 | **`addr_token_set_ratio`** | **426,184.39** | 1,060 | **Address Set Invariance**: Handles extra suite/building details |
| 3 | **`addr_token_sort_ratio`** | **126,811.81** | 982 | **Address Word Permutation**: Invariant to street token ordering |
| 4 | **`name_qgram_jaccard`** | **124,368.35** | 683 | **Fuzzy Substring Match**: Captures name typos and partial brands |
| 5 | **`addr_num_match`** | **98,144.96** | 500 | **Street Number Gate**: Directly cuts out same-name chain collisions |
| 6 | **`name_token_set_ratio`** | **74,719.45** | 707 | **Brand Core Alignment**: Resolves legal suffix differences |
| 7 | **`name_token_sort_ratio`** | **66,835.63** | 639 | **Name Token Order**: Matches transpositions |
| 8 | **`addr_available`** | **41,482.13** | 66 | **Address Presence Gate**: Direct route for missing address records |
| 9 | **`name_len_diff`** | **30,276.81** | 807 | **Length Discrepancy**: Penalizes mismatched name lengths |
| 10 | **`candidate_rank_in_entity`** | **24,109.09** | 682 | **Blocking Prior**: Rewards high-priority candidate nominations |
| 11 | **`name_jaro_winkler`** | **20,440.49** | 821 | **Prefix Weighted Fuzzy**: Standard phonetic/lexical check |
| 12 | **`addr_shared_words_count`** | **11,938.31** | 679 | **Significant Word Overlap**: Filters generic stopword overlap |
| 13 | **`franchise_collision_hazard`** | **9,545.94** | 39 | **Franchise Rejection**: Explicit hard-negative chain guard |
| 14 | **`name_levenshtein_ratio`** | **8,218.90** | 691 | **Normalized Levenshtein**: Edit distance refinement |
| 15 | **`name_len_ratio`** | **6,415.92** | 751 | **Proportion Regularization**: Ratio check on name lengths |
| 16 | **`name_first_token_match`** | **4,074.55** | 183 | **Anchor Token Verification**: First token exact match |
| 17 | **`addr_postal_match`** | **2,279.43** | 132 | **Postal Code Cross-Check**: High-confidence postal confirmation |
| 18 | **`multi_tenant_hazard`** | **55.08** | 7 | **Building Collision Guard**: Sparse check on commercial plazas |
| 19 | **`name_exact_clean`** | **23.12** | 3 | **Exact Match Short-Circuit**: Handled by higher-level continuous features |
| 20 | **`blocking_consensus_score`** | **0.00** | 0 | Neutral in current candidate pairs export |

---

## 6. Performance Comparison Across Project Phases

| Metric | Phase 3 Baseline Floor | Phase 4 Candidate Generation | Phase 6 LightGBM Model | Improvement over Baseline |
|:---|:---:|:---:|:---:|:---:|
| **Macro $F_{0.5}$** | `0.30361` | `0.27766` (raw candidates) | **`0.72953`** | **`+140.28%` (+0.42592)** |
| **Macro Precision** | `0.39363` | `0.26026` | **`0.81925`** | **`+108.13%` (+0.42562)** |
| **Macro Recall** | `0.20676` | `0.60176` | **`0.58615`** | **`+183.49%` (+0.37939)** |
| **Singleton Accuracy** | `0.62324` | `N/A` | **`0.88034`** | **`+41.25%` (+0.25710)** |
| **Pairs Completeness** | `0.18197` | `0.63382` | `0.63382` | **`+248.31%`** |

---

## 7. Artifacts & Deliverables Summary

1. **`src/models/__init__.py`**: Exporting `LightGBMPairwiseClassifier`.
2. **`src/models/pairwise_classifier.py`**: Production classifier with threshold calibration, early stopping, and model persistence.
3. **`scripts/train_classifier.py`**: End-to-end training pipeline with entity-stratified split and comprehensive reporting.
4. **`models/lightgbm_pairwise.txt`** & **`models/lightgbm_pairwise.meta.json`**: Persisted model booster and metadata.
5. **`reports/phase_06_train_results.json`**: Machine-readable training results and sweep metrics.
6. **`reports/PHASE_06_CLASSIFICATION.md`**: Complete experiment report.
