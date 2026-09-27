# Project Checklist — Amazon ML Challenge 2026

## Official Requirements
- [x] Problem statement verified (`README.md`, video, portal)
- [x] Dataset structure verified (3 sources train/test, ground truth TSV)
- [x] Submission format verified (`matching_results.tsv` schema, tab-delimited)
- [x] Validator understood (`student_resource/utils/validate_submission.py`)
- [x] F0.5 implementation verified (Macro-average, singleton rule, $\beta=0.5$)
- [x] Candidate-pair requirements verified (`candidate_pairs.tsv` audited, parsimony rule)
- [x] Model-license restriction verified (MIT / Apache 2.0, $\le 8$B parameters)
- [x] External-data restriction verified (Strictly zero external lookups/APIs)

## Data
- [x] Training data profiled (2.21M S1, 5.03M S2, 5.29M S3; 12.53M total)
- [x] Test data profiled (1.73M S1, 4.89M S2, 5.08M S3; 11.70M total)
- [x] Missing values analyzed (0% null in S1; 3.3% missing addresses in S2/S3)
- [x] Duplicates analyzed (0 exact composite duplicates in S1; 30.2% name overlap, 3.5% address overlap)
- [x] Country distribution analyzed (Train: US 60%, India 40%; Test: India 47%, US 38%, France 15%)
- [x] Name noise analyzed (Documented in text_statistics.csv & noise_examples.csv)
- [x] Address noise analyzed (Documented in text_statistics.csv & noise_examples.csv)
- [x] Ground truth analyzed (7,638,365 true match pairs, mean 3.461 matches/entity)
- [x] Singleton distribution analyzed (123,247 singletons / 5.58% in train ground truth)

## Evaluation
- [x] Local evaluator implemented (`src/evaluation/evaluator.py`)
- [x] Evaluator tested (8 unit tests in `tests/test_evaluator.py`, all PASS)
- [x] Macro F0.5 verified against manual examples (Exact entity-level $\beta=0.5$ macro formulation)
- [x] Singleton scoring verified (1.0 for empty, 0.0 for hallucination)
- [x] Candidate recall implemented (Pairs Completeness $|M \cap C| / |M|$)
- [x] Candidate-size statistics implemented (Mean, Median, P90, P95, P99, Max)
- [x] Reduction ratio implemented ($1 - |C| / (|S_1| \times (|S_2| + |S_3|))$)


## Baseline
- [x] Baseline implemented (`src/baseline/exact_matcher.py` & `scripts/run_baseline.py`)
- [x] Baseline measured (Macro $F_{0.5} = 0.303609$, Candidate Recall = $0.181969$, Precision = $0.393628$, Recall = $0.206765$)
- [x] Baseline documented (`reports/PHASE_03_BASELINE.md`)


## Blocking
- [ ] Blocking strategy A (Exact clean name)
- [ ] Blocking strategy B (Name prefix + postal code)
- [ ] Blocking strategy C (Street number + street shingle)
- [ ] Multi-pass blocking (Disjunctive union within country partition)
- [ ] Candidate recall measured
- [ ] Candidate size measured (Target $\le 20$ candidates/S1)
- [ ] Blocking errors analyzed

## Features
- [ ] Name features (Jaro-Winkler, Token sort/set, Levenshtein, Q-grams)
- [ ] Address features (Token sort/set, Street name, City, Landmark)
- [ ] Cross-field features (Street number match/mismatch, Postal match, Digit Jaccard)
- [ ] Feature ablation

## Model
- [ ] Baseline classifier (Logistic Regression)
- [ ] Gradient boosting experiment (LightGBM)
- [ ] Hard negatives (Look-alikes, same address / similar name)
- [ ] Calibration/threshold analysis

## Optimization
- [ ] Threshold optimization (Sweep $p \in [0.60, 0.95]$)
- [ ] Error analysis (FP, FN, Singleton FP, Blocking misses)
- [ ] Singleton analysis (Singleton guard threshold)
- [ ] False-positive analysis
- [ ] False-negative analysis

## Advanced
- [ ] Retrieval experiment (TF-IDF / BM25 inverted index)
- [ ] Embedding experiment if justified (Only if CPU baseline requires semantic help)
- [ ] Hybrid experiment if justified
- [ ] Ablation study

## Submission
- [ ] `matching_results.tsv` generated
- [ ] `candidate_pairs.tsv` generated
- [ ] Official validator passes (`python3 utils/validate_submission.py`)
- [ ] Final matches subset of candidates
- [ ] Every S1 represented (1,732,544 rows in test)
- [ ] No duplicates
- [ ] Reproducibility tested
- [ ] Final methodology completed (`Documentation_template.md`)
- [ ] Submission package tested (`.zip` structure verified)

---

### Status Summary (Phase 0 Audit Completion)
* **Completed**:
  * Phase 0 Project & Resource Audit.
  * Official Problem Statement & Rules Verification.
  * Raw Dataset Verification (Line & Record counts, Country breakdown, Singleton rates).
  * 0.0000% Cross-Country Match Proof.
* **Not completed**:
  * Phase 1 Full Data Forensics (Deep field-level noise analysis).
  * Phase 2 Local Evaluator (`src/evaluation/`).
  * Phase 3 Simple Baseline (`EXP-BASELINE`).
* **Blocked**: None.
* **New risks**:
  * France out-of-distribution shift in test set (15% of test S1) requires country-agnostic normalization.
  * Official candidate parsimony scoring penalty (bloated candidate sets hurt final ranking).
* **Next recommended task**:
  * Execute Phase 1 Data Forensics (`reports/PHASE_01_DATA_FORENSICS.md`).
