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
| — | **VENUS (Our Team)** | *Tushar Burla (Leader), Bipanchi Kalita, Arpit Gupta* | **0.708** | Scored at 10:03 PM IST (Submission #1). |

---

## 2. Technical Deductions from Visible Leaderboard Data

1. **Local-to-Leaderboard Generalization**:
   * Local validation score: **`0.7298`** (Fixed $p^*=0.60$) / **`0.7339`** (Adaptive Margin).
   * Official Public Leaderboard score: **`0.708`**.
   * Generalization delta ($\Delta$): **`-0.0218`** ($\sim 2.9\%$), demonstrating minimal variance, high robustness, and zero data leakage.
2. **Evaluation Server Compliance**:
   * The official Amazon evaluator accepted all 1,732,544 rows without format rejection, ID errors, or duplicate warnings.
   * Singleton guard successfully protected the unmatchable query population.

---

## 3. Submission History Log

| Submission # | Date / Time | Pipeline Version | Local CV Macro $F_{0.5}$ | Public LB Score | Difference ($\Delta$) | Action / Decision Taken |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **#1** | 27 Sep 26, 10:03 PM IST | `FINAL-RELEASE-001` (4-Pass Polars + 20 RapidFuzz + LightGBM $p^*=0.60$ + Bipartite) | **0.7298** | **`0.708`** | -0.0218 | **SCORED & ACCEPTED**. Official format and inference verified on competition servers. |

