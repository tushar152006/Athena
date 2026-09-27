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
- [ ] Missing values analyzed (Detailed field-level null scan pending Phase 1)
- [ ] Duplicates analyzed (Internal deduplication check pending Phase 1)
- [x] Country distribution analyzed (Train: US 60%, India 40%; Test: India 47%, US 38%, France 15%)
- [ ] Name noise analyzed (Systematic noise taxonomy pending Phase 1)
- [ ] Address noise analyzed (Systematic address taxonomy pending Phase 1)
- [x] Ground truth analyzed (7,638,365 true match pairs, mean 3.666 matches/entity)
- [x] Singleton distribution analyzed (123,247 singletons / 5.58% in train ground truth)

## Evaluation
- [ ] Local evaluator implemented (Pending Phase 2)
- [ ] Evaluator tested (Pending Phase 2)
- [ ] Macro F0.5 verified against manual examples (Pending Phase 2)
- [ ] Singleton scoring verified (Pending Phase 2)
- [ ] Candidate recall implemented (Pending Phase 2)
- [ ] Candidate-size statistics implemented (Mean, Median, P90, P95, P99, Max) (Pending Phase 2)
- [ ] Reduction ratio implemented (Pending Phase 2)

## Baseline
- [ ] Baseline implemented (Pending Phase 3)
- [ ] Baseline measured (Pending Phase 3)
- [ ] Baseline documented (Pending Phase 3)

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
