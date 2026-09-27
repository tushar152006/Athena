# Phase 11 — Error Diagnostics & Forensic Autopsy

**Status**: COMPLETED & VERIFIED  
**Date**: September 27, 2026  
**Execution Environment**: Python 3.12, NumPy, Polars, LightGBM  
**Validation Universe**: 4,000 Unseen S1 Entities (56,469 Candidate Pairs)  

---

## 1. Executive Summary

Phase 11 conducts a rigorous diagnostic census and root-cause autopsy of all remaining prediction errors under our calibrated decision thresholds ($p^* = 0.58$ and Adaptive Margin $p=0.50, \delta=0.30$).

### Key Diagnostic Revelations:
1. **The False Negative Chasm**:
   Total False Negatives (5,959) heavily outnumber False Positives (407) by a factor of **14.6:1**.
   Crucially, **83.4% of all False Negatives (4,969 pairs) are Blocking Dropouts**—true matches that never entered the candidate pool during blocking.
2. **Model Scoring Precision is High (81.8%)**:
   Of the candidates that *did* pass blocking, the LightGBM classifier achieves **`95.0%` pairwise precision**. False positives are tightly confined to deceptive franchise look-alikes and generic brand token collisions.
3. **Singleton Guard Efficacy**:
   Out of 234 ground-truth singletons, our singleton guard correctly declared **205 (87.6%)**, with only 29 over-merged errors.

---

## 2. Granular Error Census

| Diagnostic Metric | Fixed Threshold ($p^*=0.58$) | Adaptive Margin ($p=0.50, \delta=0.30$) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Macro $F_{0.5}$** | **`0.729851`** | **`0.733923`** | Adaptive margin boosts precision via trailing look-alike pruning |
| **Macro Precision** | `81.85%` | **`82.93%`** | Precision increases by +1.08% under adaptive gap |
| **Macro Recall** | **`58.79%`** | `58.19%` | Minor recall trade-off for higher precision |
| **True Positives (TP)** | `7,764` | `7,695` | Validated correct pairwise mergers |
| **False Positives (FP)** | `407` | **`382`** | **25 fewer false positives** under adaptive margin |
| **False Negatives (FN Total)** | `5,959` | `6,028` | Uncaptured true matches |
| *— FN: Model Scoring Misses* | `990` | `1,059` | Candidate was in pool, but score fell below threshold |
| *— FN: Blocking Dropouts* | **`4,969`** | **`4,969`** | **Lost at blocking stage (never retrieved)** |
| **Ground Truth Singletons** | `234` | `234` | True singletons in validation set |
| **Correctly Protected Singletons** | `205` | `191` | Protected by singleton guard |
| **Over-Merged Singletons (FP)** | `29` | `43` | Non-matching query incorrectly assigned matches |
| **False Singletons (FN)** | `554` | `477` | True match query incorrectly left empty |

---

## 3. Typology Deconstruction

### 3.1 False Positive Root Causes (407 cases)
```text
FP_BRAND_SUBSET           :    165 (40.5%)
FP_FRANCHISE              :    116 (28.5%)
FP_SPELLING_COLLISION     :     60 (14.7%)
FP_MULTITENANT            :     35 (8.6%)
FP_OVERMERGED             :     31 (7.6%)
```

1. **Brand-Subset Collisions (165 cases)**:
   Generic business suffixes or multi-word brand fragments where one name is a strict subset of another (e.g. "Acme Corp" vs "Acme Logistics Corp").
2. **Franchise Look-Alikes (116 cases)**:
   Shared chain brand names in the same postal district or missing building numbers.
3. **Over-merged Singletons (31 cases)**:
   True singletons that accidentally match a candidate above threshold.

---

### 3.2 False Negative Root Causes (5,959 cases)
```text
FN_BLOCKING               :   4969 (83.4%)
FN_SCORING                :    990 (16.6%)
```

1. **Blocking Dropouts (4,969 cases / 83.4%)**:
   The single largest source of error in the entire pipeline. These true matches share neither canonical name core, nor street number + word, nor postal code + prefix, nor sorted top-2 tokens.
2. **Model Scoring Misses (990 cases)**:
   Candidates present in blocking candidates whose predicted probability $P(\text{match}) < 0.58$ (typically due to severe address truncation or missing fields in Source 2/3).

---

## 4. Concrete Error Case Studies

### 4.1 Representative False Positive Errors
**Case FP-1 (FP_FRANCHISE)** — Predicted Score: `0.7553`
- **Source 1 Query [S1-603739052]**: `Meridian Consulting Limited` | `56, Jhandewalan Road Sf Rani Jhansi Road, New Delhi, Central Delhi, Delhi`
- **Candidate [S3-786016001]**: `Dr Meridian Consulting Private Limited` | `##67, Jhandewalan Road Sf Rani Jhansi Road, Central Delhi, New Delhi, DL`

**Case FP-2 (FP_FRANCHISE)** — Predicted Score: `0.7282`
- **Source 1 Query [S1-129610392]**: `Ora Decker Interactive LLC` | `4250 258, Liberty Hill, TX`
- **Candidate [S3-918540794]**: `Decker, Ora Interactive Ltd` | `4261 1/2 258, Liberty Hill, Texas`

**Case FP-3 (FP_BRAND_SUBSET)** — Predicted Score: `0.6021`
- **Source 1 Query [S1-139115196]**: `Gold Laxmi Investments Pvt Ltd` | `1St Floor, Patil Chawl, Tank Pakhadi Sahar, Mumbai, Maharashtra`
- **Candidate [S3-433204436]**: `Gold Laxmi Investments Limited` | `Door No 4St Floor, Mumbai, महाराष्ट्र`

**Case FP-4 (FP_MULTITENANT)** — Predicted Score: `0.6334`
- **Source 1 Query [S1-83701568]**: `Silver Technologies Private Limited` | `D-89, H C, Noida, Noida, Gautam Buddha Nagar, Uttar Pradesh`
- **Candidate [S2-607510939]**: `अर्बन व्हाइट एनर्जी एलएलपी` | `NOIDA, Uttar Pradesh, SB-G-89, GAUTAM BUDDHA NAGAR`

### 4.2 Representative False Negative Errors (Model Scoring Misses)
**Case FN-Score-1** — Predicted Score: `0.5300` (Below $0.58$)
- **Source 1 Query [S1-187035578]**: `Ada Howard Presidio PC` | `807 Burnett Station Road, Seymour, TN`
- **True Match [S3-925651958]**: `Ada [Howard]` | `Burnett Station Road, Seymour, Tennessee`

**Case FN-Score-2** — Predicted Score: `0.4909` (Below $0.58$)
- **Source 1 Query [S1-740546073]**: `East Blue Genesis LLC` | `195 Jack Roy Road, Azle, TX`
- **True Match [S3-45623918]**: `[LLC] East Blue Genesis` | `00195 Jack Roy Rd, Azle, Texas`

**Case FN-Score-3** — Predicted Score: `0.4862` (Below $0.58$)
- **Source 1 Query [S1-15437373]**: `Pawan Consultants Limited` | `E-1, Uttar Pradesh, Industrial Area Ramnagar, Chandauli`
- **True Match [S2-240815396]**: `Pawan Consultants Ltd` | `E-001, INDUSTRIAL AREA RAMNAGAR, CHANDALUI, Uttar Pradesh`

### 4.3 Representative Blocking Dropout Errors (Lost at Ingestion)
**Case Dropout-1** — Retrieved by Blocker: `False`
- **Source 1 Query [S1-608093705]**: `Colonial Family Practice Group` | `6905 Darmstadt Road, Evansville, IN`
- **True Match [S3-243049380]**: `Colonial (Family)` | `Indiana, Evannsville, 6905 Darmstadt Road`

**Case Dropout-2** — Retrieved by Blocker: `False`
- **Source 1 Query [S1-608093705]**: `Colonial Family Practice Group` | `6905 Darmstadt Road, Evansville, IN`
- **True Match [S2-555440248]**: `Dovakor` | `EVANSVILLE, IN, 6905-C DARMSTADT RD`

**Case Dropout-3** — Retrieved by Blocker: `False`
- **Source 1 Query [S1-608093705]**: `Colonial Family Practice Group` | `6905 Darmstadt Road, Evansville, IN`
- **True Match [S3-369578236]**: `Colonial Frhily Practice Group` | ``

---

## 5. Strategic Roadmap for Stage IV & Beyond

Based on empirical error quantification:

1. **Phase 12 (Advanced Retrieval / Embeddings)**:
   - **Target**: Recover the 4,969 blocking dropouts (83.4% of all FNs).
   - **Approach**: Character 4-gram TF-IDF / BM25 inverted index or lightweight bi-encoder embeddings to retrieve phonetically corrupted names and severe abbreviations that missed exact token passes.
   - **Theoretical Ceiling**: Successfully retrieving even $30\%$ of blocking dropouts would raise candidate recall from $63.4\%$ to $\approx 74.4\%$, providing a massive boost to Macro $F_{0.5}$.
2. **Phase 13 & 14 (Model Comparison & Blending)**:
   - Comparing CatBoost / XGBoost on address-sparse pairs.
   - Rank-blending pairwise probabilities with adaptive margin thresholding.
