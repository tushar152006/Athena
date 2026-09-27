# Master Experiment Ledger — Amazon ML Challenge 2026

Every single experiment, ablation, and model trial must be logged here chronologically. No experiment may be overwritten or deleted.

---

## Experiment Summary Table

| Exp ID | Date | Pipeline Description | Candidate Recall | Mean Candidates / S1 | Macro $F_{0.5}$ | Precision | Recall | Runtime | RAM | Decision |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-000* | 2026-09-27 | Project Audit Verification | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | 45s | 1.1GB | **AUDIT COMPLETED** |
| *EXP-001* | 2026-09-27 | Full Data Forensics & Noise Profiling | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | 3m 18s | 2.8GB | **FORENSICS COMPLETED** |
| *EXP-002* | 2026-09-27 | Local Evaluator Engine & 2.2M Benchmark | 1.000000 | 3.46 | 1.000000 | 1.000000 | 1.000000 | 13.56s | 2.03GB | **EVALUATOR VERIFIED** |
| *EXP-003* | 2026-09-27 | Deterministic Exact Baseline Matcher | 0.181969 | 2.83 | 0.303609 | 0.393628 | 0.206765 | 113.33s | 3.65GB | **BASELINE ESTABLISHED** |
| *EXP-004* | 2026-09-27 | Multi-Pass Candidate Generation & Blocking | 0.633817 | 14.20 | 0.277664 | 0.260261 | 0.601758 | 122.85s | 4.60GB | **BLOCKING ACCEPTED** |
| *EXP-006* | 2026-09-27 | LightGBM Pairwise Classifier (p*=0.60) | 0.633817 | 14.20 | 0.729530 | 0.819250 | 0.586150 | 119.5s | 1.2GB | **MODEL ACCEPTED (+140.3%)** |
| *EXP-007* | *Planned* | Global Graph Clustering & Singleton Guard | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pending Phase 7 |


---

## Detailed Experiment Logs

### EXP-000: Project Audit Verification
* **Date**: September 27, 2026
* **Objective**: Establish exact dataset dimensions, integrity, and directory governance.
* **Code Version**: `main` commit `f43965f`
* **Dataset**: Official `student_resource/dataset/`
* **Decision**: **AUDIT COMPLETED**.

### EXP-001: Phase 1 — Comprehensive Data Forensics & Noise Profiling
* **Date**: September 27, 2026
* **Objective**: Measure column missingness, exact string equality rates, noise typologies, multilingual distribution, and ground-truth match cardinality across all 24.2M records.
* **Script**: [`scripts/profile_data.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/profile_data.py)
* **Runtime**: 3 minutes 18 seconds (Polars LazyFrame streaming, CPU-only).
* **RAM Peak**: 2.8 GB.
* **Key Measured Facts [EXPERIMENTALLY VERIFIED]**:
  * Total Universe: 24,229,173 records across 7 files.
  * Cross-country match rate: **0.0000%** (0 out of 7,638,365 pairs).
  * Missing addresses: 0.0% in S1; 3.36% in S2; 3.33% in S3.
  * Case disparity: S1 is Title Case; 17.6% of S2 names and 55.8% of S2 addresses are ALL-CAPS.
  * Exact matching coverage on ground truth: Exact Name match = 10.69%; Exact Address match = 6.79%.
  * Multilingual presence: 4.53% of Indian ground-truth pairs have candidate names in Indic vernacular scripts (Devanagari, Tamil, etc.).
  * Singletons: 123,247 S1 entities (5.58%) have 0 matches; remaining 94.42% average 3.666 matches (max 11).
  * Test distribution shift: France introduced in Test set (15.0% of S1, ~1.43M records in S2/S3).
* **Decision**: **FORENSICS ACCEPTED**. Machine-readable reports generated in `reports/phase_01/` and comprehensive report in `reports/PHASE_01_DATA_FORENSICS.md`.

### EXP-002: Phase 2 — Local Evaluator Verification & Full-Scale Benchmark
* **Date**: September 27, 2026
* **Objective**: Build and benchmark high-throughput Macro F0.5 evaluator and candidate generation metrics on full 2,206,821 training ground-truth entities.
* **Modules**: [`src/evaluation/evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/evaluation/evaluator.py), [`tests/test_evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/tests/test_evaluator.py), [`scripts/benchmark_evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/benchmark_evaluator.py)
* **Test Suite**: 8 unit tests covering edge cases, singleton rules, beta=0.5 weighting asymmetry, and official validator checks. Result: **100% PASS (8/8)**.
* **Benchmark Performance on 2,206,821 Records**:
  * File load time: **7.62s**
  * Evaluation throughput: **13.56s** for 2.21M entities (~162,000 entities/sec).
  * Peak memory: **2.03 GB RSS**.
  * Self-evaluation verification: Macro $F_{0.5} = \mathbf{1.000000}$, Precision = $\mathbf{1.000000}$, Recall = $\mathbf{1.000000}$, Singleton Accuracy = $\mathbf{1.000000}$.
  * Candidate blocking metrics: Pairs Completeness = $\mathbf{1.000000}$, Reduction Ratio = $\mathbf{0.99999966}$, Parsimony (Mean: 3.46, P90: 6.0, P99: 8.0, >25: 0.00%).
* **Decision**: **EVALUATOR ADOPTED AS CANONICAL GROUND TRUTH**.

### EXP-003: Phase 3 — Deterministic Exact Baseline Matcher
* **Date**: September 27, 2026
* **Objective**: Measure empirical performance floor using exact normalized string matching on `(country, norm_name)`.
* **Modules**: [`src/baseline/exact_matcher.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/baseline/exact_matcher.py), [`scripts/run_baseline.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/run_baseline.py)
* **Official Validator Status**: **PASS (0 Errors, 0 Warnings)** on 2,206,821 emitted rows.
* **Measured Baseline Performance on Full Training Set**:
  * **Macro $F_{0.5}$ Score**: **`0.303609`** (The baseline floor to beat).
  * **Macro Precision**: **`0.393628`** ($39.36\%$).
  * **Macro Recall**: **`0.206765`** ($20.68\%$).
  * **Singleton Accuracy**: **`0.623236`** ($62.32\%$).
  * **Matched Macro $F_{0.5}$**: **`0.284703`**.
  * **Candidate Pairs Completeness**: **`0.181969`** ($18.20\%$ recall; captures 1,389,945 of 7.64M true pairs).
  * **Candidate Reduction Ratio**: **`0.99999973`**.
  * **Candidate Parsimony**: Mean **`2.83`**, Median **`1.0`**, P90 **`10.0`**, $>25$ cands **`0.00%`**.
* **Resource Profile**: Total pipeline wall time **113.33s** (Matcher: 31.4s, Validator: 18.0s, Evaluator: 7.6s), Peak RAM **3.65 GB**.
* **Key Findings**: Exact matching misses $81.8\%$ of true matches due to legal suffixes, Indic transliteration, and typos. In addition, exact matching produces $4.85\text{M}$ false positives due to identical franchise names across different locations.
* **Decision**: **BASELINE ESTABLISHED**. Serves as canonical benchmark for Phase 4 Blocking.

### EXP-004: Phase 4 — Multi-Pass Candidate Generation & Blocking
* **Date**: September 27, 2026
* **Objective**: Build a high-recall, parsimonious multi-pass candidate blocking engine combining canonical name cores, street numbers + street tokens, postal codes, and sorted name tokens.
* **Modules**: [`src/blocking/multi_pass_blocker.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/blocking/multi_pass_blocker.py), [`scripts/run_blocking.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/run_blocking.py)
* **Official Validator Status**: **PASS (0 Errors, 0 Warnings)** on both `candidate_pairs.tsv` and `matching_results.tsv`.
* **Measured Blocking Efficiency on Full 2.21M Training Set**:
  * **Candidate Pairs Completeness (Recall)**: **`0.633817`** ($63.38\%$; **`4,841,325`** of $7,638,365$ true matches captured — a **$+248.3\%$ boost** over Baseline).
  * **Candidate Reduction Ratio ($RR$)**: **`0.99999862`** (eliminates $99.99986\%$ of all pairwise comparisons).
  * **Total Candidates Generated**: **`31,339,885`** pairs across 2.21M S1 entities.
  * **Candidate Parsimony**: Mean **`14.20`**, Median **`17.0`**, Max **`20`**, $>25$ cands **`0.00%`** (strictly zero organizer audit violations).
  * **Pass Contributions**: Pass 1 (Name Core): 13.12M, Pass 2 (Street+Num): 12.95M, Pass 3 (Postal+Prefix): 0.34M, Pass 4 (Sorted Tokens): 21.42M.
* **Resource Profile**: Total pipeline wall time **122.85s**, Streaming evaluation time **16.75s**, Peak process RAM **4.60 GB**.
* **Key Findings**: Multi-pass blocking dramatically expands coverage across legal suffix variants and street-level matches. The resulting candidate pool establishes a clean $6.47 : 1$ negative-to-positive ratio ready for pairwise feature engineering and classification.
* **Decision**: **BLOCKING ACCEPTED**. Candidate pairs will serve as the candidate set for Phase 5 Feature Engineering.

### EXP-005: Phase 5 — Domain-Specific Feature Engineering & Correlation Benchmark
* **Date**: September 27, 2026
* **Objective**: Design, implement, and benchmark a 20-dimensional pairwise feature vector capturing fuzzy name similarities, address component agreements, and cross-field collision hazards (franchise look-alikes and multi-tenant buildings).
* **Modules**: [`src/features/feature_extractor.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/features/feature_extractor.py), [`scripts/benchmark_features.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/benchmark_features.py), [`reports/PHASE_05_FEATURE_ENGINEERING.md`](file:///c:/Users/DELL/Downloads/Amazon_ML/reports/PHASE_05_FEATURE_ENGINEERING.md)
* **Measured Benchmark on 70,056 Candidate Pairs (5,000 S1 Entities)**:
  * **Extraction Throughput**: **`2,129` pairs/sec** on single CPU core.
  * **Numerical Stability**: 0 NaNs, 0 Infs across all 20 feature columns.
  * **Top Discriminative Signals**:
    * `addr_shared_words_count`: Spearman $\rho = \mathbf{+0.5689}$, Pearson $r = +0.5919$ (TP mean 3.47 vs Neg mean 0.65).
    * `addr_qgram_jaccard`: Spearman $\rho = \mathbf{+0.5593}$, Pearson $r = \mathbf{+0.7492}$ (TP mean 0.667 vs Neg mean 0.099).
    * `addr_token_set_ratio`: Spearman $\rho = \mathbf{+0.5468}$, Pearson $r = +0.6615$ (TP mean 0.884 vs Neg mean 0.442).
    * `addr_num_match`: Spearman $\rho = \mathbf{+0.4202}$, Pearson $r = +0.4275$ (TP mean +0.636 vs Neg mean -0.403).
    * `candidate_rank_in_entity`: Spearman $\rho = \mathbf{-0.3270}$ (earlier blocking candidates have 3x higher true positive rate).
    * `franchise_collision_hazard`: Spearman $\rho = \mathbf{-0.1920}$ (3.5x higher frequency in hard negatives: 33.5% vs 9.6%).
* **Key Findings**: Address features provide massive orthogonal separation against hard negatives that pass through name-based blocking. Street number matching is the decisive precision gatekeeper against look-alike franchise chains.
* **Decision**: **FEATURES ACCEPTED**. Adopted as the feature set for Phase 6 LightGBM classification.
* **Next Step**: Phase 6 — Pairwise Classifier & Gradient Boosting Model (Awaiting User Prompt).



---

## Detailed Experiment Logs

### EXP-000: Project Audit Verification
* **Date**: September 27, 2026
* **Objective**: Establish exact dataset dimensions, integrity, and directory governance.
* **Code Version**: `main` commit `f43965f`
* **Dataset**: Official `student_resource/dataset/`
* **Decision**: **AUDIT COMPLETED**.

### EXP-001: Phase 1 — Comprehensive Data Forensics & Noise Profiling
* **Date**: September 27, 2026
* **Objective**: Measure column missingness, exact string equality rates, noise typologies, multilingual distribution, and ground-truth match cardinality across all 24.2M records.
* **Script**: [`scripts/profile_data.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/profile_data.py)
* **Runtime**: 3 minutes 18 seconds (Polars LazyFrame streaming, CPU-only).
* **RAM Peak**: 2.8 GB.
* **Key Measured Facts [EXPERIMENTALLY VERIFIED]**:
  * Total Universe: 24,229,173 records across 7 files.
  * Cross-country match rate: **0.0000%** (0 out of 7,638,365 pairs).
  * Missing addresses: 0.0% in S1; 3.36% in S2; 3.33% in S3.
  * Case disparity: S1 is Title Case; 17.6% of S2 names and 55.8% of S2 addresses are ALL-CAPS.
  * Exact matching coverage on ground truth: Exact Name match = 10.69%; Exact Address match = 6.79%.
  * Multilingual presence: 4.53% of Indian ground-truth pairs have candidate names in Indic vernacular scripts (Devanagari, Tamil, etc.).
  * Singletons: 123,247 S1 entities (5.58%) have 0 matches; remaining 94.42% average 3.666 matches (max 11).
  * Test distribution shift: France introduced in Test set (15.0% of S1, ~1.43M records in S2/S3).
* **Decision**: **FORENSICS ACCEPTED**. Machine-readable reports generated in `reports/phase_01/` and comprehensive report in `reports/PHASE_01_DATA_FORENSICS.md`.

### EXP-002: Phase 2 — Local Evaluator Verification & Full-Scale Benchmark
* **Date**: September 27, 2026
* **Objective**: Build and benchmark high-throughput Macro F0.5 evaluator and candidate generation metrics on full 2,206,821 training ground-truth entities.
* **Modules**: [`src/evaluation/evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/evaluation/evaluator.py), [`tests/test_evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/tests/test_evaluator.py), [`scripts/benchmark_evaluator.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/benchmark_evaluator.py)
* **Test Suite**: 8 unit tests covering edge cases, singleton rules, beta=0.5 weighting asymmetry, and official validator checks. Result: **100% PASS (8/8)**.
* **Benchmark Performance on 2,206,821 Records**:
  * File load time: **7.62s**
  * Evaluation throughput: **13.56s** for 2.21M entities (~162,000 entities/sec).
  * Peak memory: **2.03 GB RSS**.
  * Self-evaluation verification: Macro $F_{0.5} = \mathbf{1.000000}$, Precision = $\mathbf{1.000000}$, Recall = $\mathbf{1.000000}$, Singleton Accuracy = $\mathbf{1.000000}$.
  * Candidate blocking metrics: Pairs Completeness = $\mathbf{1.000000}$, Reduction Ratio = $\mathbf{0.99999966}$, Parsimony (Mean: 3.46, P90: 6.0, P99: 8.0, >25: 0.00%).
* **Decision**: **EVALUATOR ADOPTED AS CANONICAL GROUND TRUTH**.

### EXP-003: Phase 3 — Deterministic Exact Baseline Matcher
* **Date**: September 27, 2026
* **Objective**: Measure empirical performance floor using exact normalized string matching on `(country, norm_name)`.
* **Modules**: [`src/baseline/exact_matcher.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/baseline/exact_matcher.py), [`scripts/run_baseline.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/run_baseline.py)
* **Official Validator Status**: **PASS (0 Errors, 0 Warnings)** on 2,206,821 emitted rows.
* **Measured Baseline Performance on Full Training Set**:
  * **Macro $F_{0.5}$ Score**: **`0.303609`** (The baseline floor to beat).
  * **Macro Precision**: **`0.393628`** ($39.36\%$).
  * **Macro Recall**: **`0.206765`** ($20.68\%$).
  * **Singleton Accuracy**: **`0.623236`** ($62.32\%$).
  * **Matched Macro $F_{0.5}$**: **`0.284703`**.
  * **Candidate Pairs Completeness**: **`0.181969`** ($18.20\%$ recall; captures 1,389,945 of 7.64M true pairs).
  * **Candidate Reduction Ratio**: **`0.99999973`**.
  * **Candidate Parsimony**: Mean **`2.83`**, Median **`1.0`**, P90 **`10.0`**, $>25$ cands **`0.00%`**.
* **Resource Profile**: Total pipeline wall time **113.33s** (Matcher: 31.4s, Validator: 18.0s, Evaluator: 7.6s), Peak RAM **3.65 GB**.
* **Key Findings**: Exact matching misses $81.8\%$ of true matches due to legal suffixes, Indic transliteration, and typos. In addition, exact matching produces $4.85\text{M}$ false positives due to identical franchise names across different locations.
* **Decision**: **BASELINE ESTABLISHED**. Serves as canonical benchmark for Phase 4 Blocking.

### EXP-004: Phase 4 — Multi-Pass Candidate Generation & Blocking
* **Date**: September 27, 2026
* **Objective**: Build a high-recall, parsimonious multi-pass candidate blocking engine combining canonical name cores, street numbers + street tokens, postal codes, and sorted name tokens.
* **Modules**: [`src/blocking/multi_pass_blocker.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/blocking/multi_pass_blocker.py), [`scripts/run_blocking.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/run_blocking.py)
* **Official Validator Status**: **PASS (0 Errors, 0 Warnings)** on both `candidate_pairs.tsv` and `matching_results.tsv`.
* **Measured Blocking Efficiency on Full 2.21M Training Set**:
  * **Candidate Pairs Completeness (Recall)**: **`0.633817`** ($63.38\%$; **`4,841,325`** of $7,638,365$ true matches captured — a **$+248.3\%$ boost** over Baseline).
  * **Candidate Reduction Ratio ($RR$)**: **`0.99999862`** (eliminates $99.99986\%$ of all pairwise comparisons).
  * **Total Candidates Generated**: **`31,339,885`** pairs across 2.21M S1 entities.
  * **Candidate Parsimony**: Mean **`14.20`**, Median **`17.0`**, Max **`20`**, $>25$ cands **`0.00%`** (strictly zero organizer audit violations).
  * **Pass Contributions**: Pass 1 (Name Core): 13.12M, Pass 2 (Street+Num): 12.95M, Pass 3 (Postal+Prefix): 0.34M, Pass 4 (Sorted Tokens): 21.42M.
* **Resource Profile**: Total pipeline wall time **122.85s**, Streaming evaluation time **16.75s**, Peak process RAM **4.60 GB**.
* **Key Findings**: Multi-pass blocking dramatically expands coverage across legal suffix variants and street-level matches. The resulting candidate pool establishes a clean $6.47 : 1$ negative-to-positive ratio ready for pairwise feature engineering and classification.
* **Decision**: **FEATURES ACCEPTED**. Adopted as the feature set for Phase 6 LightGBM classification.
* **Next Step**: Phase 6 — Pairwise Classifier & Gradient Boosting Model (`EXP-006`).

### EXP-006: Phase 6 — LightGBM Pairwise Classifier & Macro F0.5 Calibration
* **Date**: September 27, 2026
* **Objective**: Train a 350-tree LightGBM pairwise scoring model on candidate pairs with an entity-stratified split (16k train S1, 4k val S1) and optimize the classification threshold $p^*$ specifically for Macro $F_{0.5}$.
* **Modules**: [`src/models/pairwise_classifier.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/src/models/pairwise_classifier.py), [`scripts/train_classifier.py`](file:///c:/Users/DELL/Downloads/Amazon_ML/scripts/train_classifier.py), [`reports/PHASE_06_CLASSIFICATION.md`](file:///c:/Users/DELL/Downloads/Amazon_ML/reports/PHASE_06_CLASSIFICATION.md)
* **Model Checkpoint**: [`models/lightgbm_pairwise.txt`](file:///c:/Users/DELL/Downloads/Amazon_ML/models/lightgbm_pairwise.txt)
* **Measured Validation Results on 4,000 Unseen S1 Entities (56,469 Candidate Pairs)**:
  * **Validation ROC-AUC**: **`0.9962`**
  * **Validation PR-AUC**: **`0.9807`**
  * **Optimal Threshold $p^*$**: **`0.60`**
  * **Macro $F_{0.5}$ Score**: **`0.729530`** (A **`+140.28%`** relative improvement and **`+0.42592`** absolute gain over Phase 3 Baseline Floor of `0.303609`).
  * **Macro Precision**: **`0.819250`** ($81.93\%$, **`+108.13%`** over Baseline).
  * **Macro Recall**: **`0.586150`** ($58.62\%$, **`+183.49%`** over Baseline).
  * **Singleton Accuracy**: **`0.880340`** ($88.03\%$, **`+41.25%`** over Baseline).
* **Resource Profile**: Total pipeline wall time **119.5s** (Feature extraction: 76.8s train + 7.6s val; Training: 34.2s), Peak RAM **1.2 GB**.
* **Key Findings**: Address features dominate the gradient boosting splits (`addr_qgram_jaccard` Gain = 731k, `addr_token_set_ratio` Gain = 426k, `addr_num_match` Gain = 98k). The calibrated threshold of $p^* = 0.60$ strikes the optimal trade-off for the $\beta=0.5$ precision weighting.
* **Decision**: **MODEL ACCEPTED**. Baseline floor surpassed by +140.3%. Adopted for Phase 7 Graph Clustering and Singleton Resolution.
* **Next Step**: Phase 7 — Global Graph Clustering & Singleton Resolution (Awaiting User Prompt).




