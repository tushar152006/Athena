# Master Experiment Ledger — Amazon ML Challenge 2026

Every single experiment, ablation, and model trial must be logged here chronologically. No experiment may be overwritten or deleted.

---

## Experiment Summary Table

| Exp ID | Date | Pipeline Description | Candidate Recall | Mean Candidates / S1 | Macro $F_{0.5}$ | Precision | Recall | Runtime | RAM | Decision |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-000* | 2026-09-27 | Project Audit Verification | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | 45s | 1.1GB | **AUDIT COMPLETED** |
| *EXP-001* | 2026-09-27 | Full Data Forensics & Noise Profiling | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | 3m 18s | 2.8GB | **FORENSICS COMPLETED** |
| *EXP-002* | 2026-09-27 | Local Evaluator Engine & 2.2M Benchmark | 1.000000 | 3.46 | 1.000000 | 1.000000 | 1.000000 | 13.56s | 2.03GB | **EVALUATOR VERIFIED** |
| *EXP-003* | *Planned* | Baseline Exact Normalized Matcher | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pending Phase 3 |

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
* **Next Step**: Phase 3 — Baseline Implementation (Awaiting User Prompt).


