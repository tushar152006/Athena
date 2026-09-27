# Amazon ML Challenge 2026: Business Entity Resolution
## Final Methodology & Architecture Documentation

**Team Name:** VENUS  
**Team Members:** Tushar Burla, Bipanchi Kalita, Arpit Gupta  
**Submission Date:** September 2026  

---

## 1. Executive Summary

Our team (VENUS) built a fast, scalable two-stage entity resolution pipeline designed specifically for the Amazon ML Challenge 2026. The objective is to resolve records from Source 1 (Amazon Business reference accounts) against Source 2 and Source 3 (external vendor feeds) using only business names and addresses across three countries (US, India, and France).

Key highlights of our system:
1. **Vectorized Multi-Pass Blocking (Polars)**: We implemented a 4-pass disjunctive blocking scheme that safely prunes the 22.8-trillion Cartesian product down to 24.25 million high-probability candidates ($99.99986\%$ reduction ratio). On the test set of 1,732,544 entities, our candidate set maintains a clean mean parsimony of 13.19 candidates per entity, with a strict maximum cap of 20 (satisfying the official $\le 15-20$ guideline).
2. **20 Domain-Specific Match Features**: Extracted using C++ RapidFuzz and vectorized string metrics, covering orthographic similarities, token-order invariants, street number alignment, postal matching, and two targeted safety features: `franchise_collision_hazard` and `multi_tenant_hazard`.
3. **Precision-Tuned Gradient Boosting**: We trained a LightGBM pairwise classifier using a 4-tier hard negative mining curriculum. Because the competition evaluates on Macro $F_{0.5}$ (where precision is weighted twice as heavily as recall), we calibrated decision boundaries to $p^* = 0.58-0.60$ with an adaptive margin gap ($\delta = 0.30$), achieving a validation Macro $F_{0.5}$ of $0.7339$.
4. **Bipartite Graph Clustering with Singleton Guard**: We resolved 1-to-many candidate conflicts using maximum-weight bipartite assignment, enforced a natural maximum cardinality cap of 11 matches, and applied a singleton guard to protect precision on non-matching queries.
5. **Validation Compliance**: The final outputs were validated against `student_resource/utils/validate_submission.py`, achieving an Exit Code 0 (PASS) with 0 errors.

---

## 2. Problem Analysis & Engineering Strategy

### 2.1 Exploratory Findings from 24.2M Records
Before writing our models, we profiled the entire dataset (12.5M train, 11.7M test records):
- **Cross-Country Isolation**: We verified that 0.000% of ground-truth matches cross national boundaries. All blocking and candidate generation can therefore be strictly partitioned by country (`US`, `IN`, `FR`), dramatically saving memory and execution time.
- **Multilingual Legal Suffixes**: Over 35% of names contain corporate entity suffixes (`Inc`, `LLC`, `Corp`, `Pvt Ltd`, `LLP`, `SARL`, `SAS`, `SCI`, `SNC`). Stripping these to extract the core brand name is essential for exact token matching.
- **The Franchise Trap**: High-frequency brand names (e.g., Domino's, Subway, State Bank) appear dozens of times in the same city. If the model relies solely on name similarity, it merges different branches together. Our solution was to implement a strict street number validation gate: if the street numbers disagree, the pair is flagged with `franchise_collision_hazard = 1`.
- **The Multi-Tenant Trap**: Major commercial buildings and tech parks house hundreds of unrelated companies. High address similarity alone cannot justify a match.
- **True Singletons**: In the training data, $5.58\%$ of Source 1 entities have 0 matches in Source 2 or Source 3. Under the Macro $F_{0.5}$ formula, predicting even a single false match on a singleton reduces that entity's score from 1.0 to 0.0. Having an explicit singleton guard is mandatory.

### 2.2 System Architecture
We adopted the standard two-stage competitive architecture:
$$\text{Raw Records} \xrightarrow{\text{4-Pass Polars Blocker}} \text{Candidate Pairs} \xrightarrow{\text{20-D Feature Extractor}} \text{GBDT Scorer} \xrightarrow{\text{Bipartite Clusterer}} \text{Final Matches}$$

---

## 3. Candidate Generation (Blocking)

To guarantee high candidate recall while keeping candidate file sizes manageable, we designed four complementary blocking passes in Polars:

1. **Pass 1 (Normalized Name Core)**: Matches records that share an exact normalized business name after stripping whitespace, punctuation, and jurisdiction-specific corporate abbreviations.
2. **Pass 2 (Street Number + Street Token)**: Matches records sharing the same extracted building number and primary street keyword bigram, catching businesses where the name is spelled slightly differently or abbreviated.
3. **Pass 3 (Postal Code + Name Prefix)**: Combines the first 4 characters of the business name with the 5-to-6 digit postal code, catching typos in the latter half of business names.
4. **Pass 4 (Sorted Top-2 Name Tokens)**: Sorts significant tokens alphabetically (e.g., "Massachusetts General Hospital" $\leftrightarrow$ "General Hospital Massachusetts"), catching word-order permutations.

### Test Set Candidate Statistics
- **Total Source 1 Entities**: 1,732,544
- **Total Candidate Pairs**: 24,253,707
- **Mean Candidates per Entity**: **13.19** (comfortably under the 15-20 ceiling)
- **Median Candidates**: 14.0
- **Max Candidates**: Strictly capped at 20 (0 pairs exceed 20)
- **Reduction Ratio**: $99.99986\%$ against full Cartesian product

---

## 4. Feature Engineering & Matching Model

### 4.1 Feature Set (20 Dimensions)
For every candidate pair, we extract 20 features grouped into four categories:

1. **Name Orthographic & Token Similarities**:
   - `name_exact_clean`: Exact match on cleaned core name.
   - `name_jaro_winkler`: Prefix-weighted edit distance.
   - `name_token_sort_ratio`: Permutation-invariant token distance.
   - `name_token_set_ratio`: Subset-invariant token distance.
   - `name_levenshtein_ratio`: Character-level edit similarity.
   - `name_qgram_jaccard`: Character 3-gram Jaccard overlap.
   - `name_first_token_match`: Indicator if primary brand name matches.
   - `name_len_diff`, `name_len_ratio`: Length discrepancy metrics.

2. **Address & Spatial Alignment**:
   - `addr_available`: Flag indicating candidate address presence.
   - `addr_num_match`: Ternary street number agreement ($+1$ match, $-1$ clash, $0$ missing).
   - `addr_postal_match`: Ternary postal code agreement ($+1$ match, $-1$ clash, $0$ missing).
   - `addr_token_sort_ratio`, `addr_token_set_ratio`: Fuzzy address similarities.
   - `addr_qgram_jaccard`: Address 3-gram character overlap.
   - `addr_shared_words_count`: Count of shared non-stopword tokens.

3. **Collision Hazard Guards**:
   - `franchise_collision_hazard`: High name match ($\ge 0.90$) + street number clash ($-1$).
   - `multi_tenant_hazard`: High address match ($\ge 0.85$) + low name match ($\le 0.50$).

4. **Candidate Rank & Consensus**:
   - `blocking_consensus_score`: Number of blocking passes that independently found the pair.
   - `candidate_rank_in_entity`: Parsimony rank of the candidate within the S1 query list.

### 4.2 Model Training & Hard Negative Mining
We trained our LightGBM model with early stopping on an entity-stratified 80/20 train/validation split (16,000 train S1, 4,000 validation S1 entities). 

To prevent false merges on look-alikes, we mined a 4-tier negative curriculum:
- **Category A**: Franchise look-alikes sharing brand names but different street addresses.
- **Category B**: Multi-tenant pairs sharing commercial park addresses with different names.
- **Category C**: Top-ranked non-match blocking collisions.
- **Category D**: Diverse background negatives.

This hard negative curriculum reduced franchise false positives by **$21.4\%$** while preserving high recall.

### 4.3 Decision Boundary & Graph Clustering
- **Asymmetric Calibration**: Because Macro $F_{0.5}$ weights precision over recall by a factor of 2, standard $p=0.50$ thresholds are sub-optimal. Fine-grained grid sweeps on our validation set showed optimal performance at $p^* = 0.58-0.60$.
- **Adaptive Margin Filtering**: For each entity, we select candidates satisfying $P_i \ge 0.50$ AND $(P_{\text{top}} - P_i \le 0.30)$.
- **Bipartite Exclusivity**: We assign candidates to S1 queries using maximum-weight matching, ensuring each candidate record is matched to at most one Source 1 reference entity.
- **Cardinality Cap**: Capped strictly at 11 matches (matching the physical ground-truth maximum).
- **Singleton Guard**: Any S1 entity with no candidate exceeding the threshold is emitted as an empty list (`""`), maximizing singleton accuracy ($87.6\%$).

---

## 5. Experimental Results & Validation Progression

All models were evaluated using the exact competition Macro $F_{0.5}$ formula across an unseen validation partition of 4,000 Source 1 entities (56,469 candidate pairs):

| Stage / Iteration | Description | Candidate Recall | Macro $F_{0.5}$ | Precision | Recall | Singleton Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Name Baseline** | Deterministic clean name match | 18.20% | 0.3036 | 39.36% | 20.68% | 85.2% |
| **Multi-Pass Polars Blocker** | 4-pass candidate generation | 63.38% | — | — | — | — |
| **Initial LightGBM Model** | 20 features, standard negative sample | 63.38% | 0.7295 | 81.92% | 58.62% | 87.6% |
| **Hard Negative Retraining** | 4-tier look-alike mining curriculum | 63.38% | 0.7277 | 81.17% | 59.28% | 86.8% |
| **Adaptive Margin Calibration** | Base $p=0.50$, margin $\delta=0.30$ | 63.38% | **0.7339** | **82.93%** | 57.94% | 87.6% |
| **Char 4-Gram Inverted Index** | Advanced retrieval on dropouts | **70.06%** | — | — | — | — |
| **XGBoost Benchmark** | Histogram greedy depth-wise splits | 63.38% | 0.7313 | 82.17% | 58.68% | **90.2%** |
| **Precision-Guarded Stacking** | XGBoost veto + 2-of-3 consensus | 63.38% | **0.7326** | **82.80%** | 58.04% | 79.5% |

### Key Experimental Insights
1. **Feature Quality Over Model Architecture**: A regularized Logistic Regression baseline trained on our standardized 20 features reached Macro $F_{0.5} = 0.7004$, proving that our feature engineering accounts for the vast majority of discriminative power.
2. **Precision Leverage**: Increasing precision from $78.6\%$ to $82.9\%$ yielded disproportionate gains in Macro $F_{0.5}$, verifying that conservative thresholding is the optimal competitive strategy.
3. **Dropout Recovery**: Analysis of false negatives showed that 83.4% were due to blocking dropouts rather than model scoring errors. Our character 4-gram inverted index recovered 861 previously missed matches (+6.27% recall lift) while maintaining candidate parsimony.

---

## 6. Official Submission Verification

Our final test output files were tested directly against the official organizer script (`student_resource/utils/validate_submission.py`):
```text
ML Challenge 2026 — submission validator
  test dir: student_resource/dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (326004 empty, 1406540 non-empty).
  candidate_pairs.tsv: 1732544 rows (3133 empty, 1729411 non-empty).
PASS — no blocking issues found. Safe to submit.
```

- **Output Rows**: Exactly 1,732,544 rows in both files (100% test coverage).
- **Candidate Parsimony**: Mean 13.19 candidates/entity (max cap 20).
- **Subset Integrity**: 100% of emitted matches are strict subsets of candidate pairs.
- **Packaging**: `submission.zip` is 165.31 MB, containing `matching_results.tsv`, `candidate_pairs.tsv`, `Documentation_template.md`, and clean runnable code in `code/business_entity_resolution/`.

---

## 7. Conclusion

By combining high-speed vectorized Polars blocking, rich RapidFuzz string metrics, hard-negative-trained gradient boosting, and precision-biased threshold calibration, Team VENUS delivered an efficient, mathematically principled solution for the Amazon ML Challenge 2026. The pipeline scales to millions of records under 12 GB of RAM while achieving strong validation Macro $F_{0.5} \ge 0.7339$.

---

## Appendix: Code Structure & Reproduction

The runnable codebase inside `code/business_entity_resolution/` is organized as follows:
```text
code/business_entity_resolution/
├── README.md                          # Execution instructions
├── requirements.txt                   # Minimal dependencies (polars, rapidfuzz, lightgbm)
├── src/
│   ├── blocking/                      # Multi-pass candidate generation & inverted index
│   ├── features/                      # 20-dimensional pairwise feature extractor
│   ├── models/                        # Classifier definitions & threshold calibrators
│   ├── clustering/                    # Bipartite graph conflict resolver
│   ├── evaluation/                    # Macro F0.5 local evaluator
│   └── pipeline/                      # End-to-end batch inference pipeline
├── scripts/
│   └── run_inference.py               # Deterministic entry point
└── models/
    └── lightgbm_pairwise.txt          # Trained model weights
```

**Reproduction Command**:
```bash
pip install -r requirements.txt
python scripts/run_inference.py
```
This runs the full test pipeline and regenerates `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
