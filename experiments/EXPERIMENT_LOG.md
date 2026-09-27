# Master Experiment Ledger — Amazon ML Challenge 2026

Every single experiment, ablation, and model trial must be logged here chronologically. No experiment may be overwritten or deleted.

---

## Experiment Summary Table

| Exp ID | Date | Pipeline Description | Candidate Recall | Mean Candidates / S1 | Macro $F_{0.5}$ | Precision | Recall | Runtime | RAM | Decision |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-000* | 2026-09-27 | Project Audit & Data Forensics Verification | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | — | — | **AUDIT COMPLETED** |
| *EXP-001* | *Planned* | Baseline Exact Normalized Matcher | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pending Phase 3 |

---

## Detailed Experiment Logs

### EXP-000: Project Audit & Raw Data Profile
* **Date**: September 27, 2026
* **Objective**: Establish exact dataset dimensions, country distributions, singleton proportions, and cross-country matching integrity.
* **Hypothesis**: Sources adhere to clean tab-delimited schemas; cross-country matches are negligible or zero.
* **Code Version**: `main` commit `f43965f`
* **Dataset**: Official `student_resource/dataset/`
* **Measured Facts [EXPERIMENTALLY VERIFIED]**:
  * Training records: $12,527,040$ ($S_1$: 2.21M, $S_2$: 5.03M, $S_3$: 5.29M).
  * Test records: $11,702,133$ ($S_1$: 1.73M, $S_2$: 4.89M, $S_3$: 5.08M).
  * Singletons in ground truth: $123,247$ ($5.58\%$).
  * Non-singletons in ground truth: $2,083,574$ ($94.42\%$, mean $3.666$ matches).
  * Cross-country true match pairs: **EXACTLY 0 out of 7,638,365 pairs ($0.0000\%$)**.
  * Test set country distribution: India ($46.8\%$), US ($38.3\%$), France ($15.0\%$).
* **Decision**: **KEEP & ADOPT**. Country partitioning is established as a 100% loss-free candidate generation constraint.
* **Next Step**: Build Phase 1 Data Forensics Report and Phase 2 Local Evaluator.
