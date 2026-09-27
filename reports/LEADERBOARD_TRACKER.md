# Leaderboard Tracker — Amazon ML Challenge 2026

**Metric**: Macro-Averaged $F_{0.5}$ (Precision-Weighted, $\beta=0.5$)  
**Official Portal**: Unstop (`unstop.com`)  

---

## 1. Official Leaderboard Snapshot

*Timestamp / Date Captured*: September 27, 2026 (~15:10 IST)  
*Source*: Official Portal Screenshot ("Top Gainers")  

| Rank | Team Name | Academic Institution / Campus | Public Leaderboard Score ($F_{0.5}$) | Observations / Notes |
| :---: | :--- | :--- | :---: | :--- |
| **1st** | **GG** | Lovely Professional University (LPU), Punjab | **0.991811** | Current visible highest score. |
| **2nd** | **CDS_Team_iisc** | Indian Institute of Science (IISc), Bangalore | **0.991483** | Top research institution team. |
| **3rd** | **Team X** | Indian Institute of Technology (IIT), Madras | **0.991170** | Top engineering campus team. |
| — | **Athena (Our Team)** | *To be submitted* | — | In development phase. |

---

## 2. Technical Deductions from Visible Leaderboard Data

1. **Tight High-Score Cluster ($> 0.991$)**:
   * The top 3 teams are separated by less than $0.00065$ in Macro $F_{0.5}$ ($0.99117$ to $0.99181$).
   * *Inference*: The public test set possesses very strong deterministic or near-deterministic signal when:
     * Country partitioning is strictly enforced.
     * Text canonicalization handles abbreviations and legal entity suffixes cleanly.
     * High-precision blocking avoids candidate drop while keeping candidate sets tight.
2. **Submissions Drive Public vs. Private Split**:
   * Public leaderboard rankings reflect performance on a subset of `test_source1.tsv`.
   * Final rankings are decided on the **Private Leaderboard** (the remaining test set subset) **PLUS** the audited candidate generation parsimony score from `candidate_pairs.tsv`.
3. **Strategic Rule**:
   * We will **not** overfit our decision threshold to match a specific public leaderboard score. All models and thresholds will be anchored strictly in out-of-fold `GroupKFold` cross-validation on `source1_entity_id`.

---

## 3. Submission History Log

*(To be updated after each leaderboard submission)*

| Submission # | Date / Time | Pipeline Version | Local CV Macro $F_{0.5}$ | Public LB Score | Difference ($\Delta$) | Action / Decision Taken |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| *Pending* | — | EXP-BASELINE | — | — | — | Awaiting Phase 3 baseline implementation. |
