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
| *EXP-004* | *Planned* | Multi-Channel Candidate Blocking | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pending Phase 4 |

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
* **Next Step**: Phase 4 — Candidate Generation & Multi-Pass Blocking (Awaiting User Prompt).


