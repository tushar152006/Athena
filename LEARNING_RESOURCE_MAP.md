# Amazon ML Challenge 2026 — Learning Resource Map

This document maps every core technical component of the **Business Entity Resolution** pipeline to its highest-yield book chapters, academic papers, and technical guides.

---

## 1. Project Component Mapping Table

| Topic | Best Resource | Specific Chapters / Sections | Priority | Project Pipeline Component | Primary Objective / What to Learn |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Entity Resolution Fundamentals** | Peter Christen — *Data Matching* (Springer, 2012) | Chapters 1 & 2 ("Introduction", "The Data Matching Process") | **MUST READ** | Entire System Design | End-to-end framing: indexing $\to$ comparison $\to$ classification $\to$ evaluation. |
| **Data Cleaning & Normalization** | Peter Christen — *Data Matching* (Springer, 2012) | Chapter 3 ("Data Pre-Processing and Quality") | **MUST READ** | Preprocessing (`src/normalize.py`) | Canonicalization, legal suffix stripping, accent folding, street abbreviation expansion. |
| **Blocking & Candidate Generation** | Peter Christen — *Data Matching* (Springer, 2012) | Chapter 4 ("Indexing / Blocking") | **MUST READ** | Candidate Generation (`candidate_pairs.tsv`) | Inverted indexes, multi-channel disjunctive blocking, candidate recall ceiling vs. reduction ratio. |
| **Candidate Retrieval & Indexing** | Manning, Raghavan, Schütze — *Introduction to Information Retrieval* (Cambridge, 2008) | Chapters 1, 2, & 6 ("Boolean Retrieval", "Inverted Index", "Scoring & Vector Space") | **HIGH VALUE** | Candidate Retrieval | Fast token inverted indexes, BM25 / TF-IDF scoring, pruning high-frequency stopwords. |
| **String & Phonetic Comparison** | Peter Christen — *Data Matching* (Springer, 2012) | Chapter 5 ("Comparison Functions") | **MUST READ** | Feature Extraction (`src/features.py`) | Jaro-Winkler, Levenshtein, Damerau, Q-gram Jaccard, SoftTFIDF, Double Metaphone. |
| **Approximate String Matching** | Papadakis et al. — *Comparative Analysis of Approx. String Matching* (IEEE TKDE, 2020) | Full Survey Paper (Sections 3 & 4) | **HIGH VALUE** | Feature Extraction | Benchmarks on string similarities: speed, accuracy, token vs. character level trade-offs. |
| **Probabilistic Linkage Theory** | Fellegi & Sunter — *A Theory for Record Linkage* (JASA, 1969) | Mathematical Derivation (pp. 1183–1195) | **HIGH VALUE** | Decision Logic | Understanding likelihood ratios, match ($M$) vs. non-match ($U$) log-odds. |
| **Gradient Boosted Matching** | Ke et al. — *LightGBM: A Highly Efficient GBDT* (NeurIPS, 2017) | Section 2 & 3 ("GOSS" and "EFB") | **MUST READ** | Matching Model (`src/model.py`) | Histogram binning, high-throughput pairwise binary classification, sub-second inference. |
| **Evaluation & F0.5 Optimization** | C. J. Van Rijsbergen — *Information Retrieval* (Butterworth, 1979) | Chapter 7 ("Evaluation") | **MUST READ** | Threshold Calibration (`src/threshold.py`) | Derivation of $F_\beta$, precision weighting, Bayes optimal decision threshold under $\beta=0.5$. |
| **International Address Parsing** | Libpostal Documentation & Barrientos et al. (2018) | International address syntax patterns | **MUST READ** | Address Normalization | Handling French (`Rue`, `Avenue`) vs. Indian (landmarks) vs. US street formats. |
| **High-Performance Data Processing** | Polars Official User Guide (Ritchie Vink et al., 2024) | "Lazy API", "Streaming", "Expressions" | **MUST READ** | Data Ingestion & Joins | Processing 24M records without RAM overflow via out-of-core streaming execution. |
| **Fast Vectorized String Matching** | RapidFuzz Documentation (Max Bachmann, 2024) | `rapidfuzz.distance`, `rapidfuzz.fuzz` | **MUST READ** | Feature Computation | C++ SIMD-accelerated string metrics computing 100k comparisons/sec in pure Python. |
| **Validation Without Leakage** | Scikit-Learn Documentation (2024) | `GroupKFold` & Cross-Validation Guide | **MUST READ** | Validation Framework (`src/validate.py`) | Entity-level grouping on `source1_entity_id` preventing optimistic test data leakage. |
| **Deep Learning for ER (Context)** | Mudgal et al. — *Deep Learning for Entity Matching* (SIGMOD, 2018) | Empirical Evaluation (Sections 4 & 5) | **REFERENCE** | Architecture Trade-offs | Why GBDTs outperform or match Transformers on tabular text while running $100\times$ faster. |

---

## 2. Component-by-Component Responsibility Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE STAGES & CORE REFERENCES                      │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ Pipeline Stage           │ Primary Reference Document / Chapter             │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ 1. Data Ingestion        │ Polars Streaming Guide (Out-of-core lazy scans)  │
│ 2. Text Normalization    │ Christen (2012) Chapter 3 (Preprocessing)        │
│ 3. Country Partitioning  │ Empirical Proof (0.0000% cross-country matches)  │
│ 4. Multi-Channel Block   │ Christen (2012) Chapter 4 (Indexing/Blocking)    │
│ 5. Feature Engineering   │ Christen (2012) Chapter 5 (Comparison Functions) │
│ 6. Model Training        │ LightGBM Docs + Ke et al. (NeurIPS 2017)         │
│ 7. Thresholding (F0.5)   │ Van Rijsbergen (1979) Chapter 7 (F-measure)      │
│ 8. Submission Packaging  │ Official `validate_submission.py` CLI script     │
└──────────────────────────┴──────────────────────────────────────────────────┘
```
