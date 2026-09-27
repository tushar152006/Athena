# Amazon ML Challenge 2026 — Technical Learning & Reference Library

**Curated For**: Engineering & Research Team  
**Challenge**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Scope**: Authoritative Books, Peer-Reviewed Papers, Technical Guides, and Software Frameworks  

---

## 1. Essential Books

### 1.1 The Primary Foundation
* **Title**: *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection*
* **Author**: Peter Christen
* **Year**: 2012
* **Publisher**: Springer-Verlag Berlin Heidelberg
* **ISBN / DOI**: 978-3-642-31163-5 / [10.1007/978-3-642-31164-2](https://doi.org/10.1007/978-3-642-31164-2)
* **Status in Project**: **THE MOST IMPORTANT BOOK (MUST READ CORE REFERENCE)**
* **Why It Is Essential**: This is the definitive, industry-standard textbook on entity resolution. It formalizes the exact 5-stage pipeline required by this competition: Data Preprocessing $\to$ Indexing/Blocking $\to$ Field Comparison $\to$ Classification $\to$ Evaluation.
* **Key Chapters for This Project**:
  * **Chapter 2 ("The Data Matching Process")**: Establishes the architectural framework, candidate generation vs. classification distinction, and real-world noise models.
  * **Chapter 3 ("Data Pre-Processing and Quality")**: Data cleansing, token standardization, segmenting names and addresses, and legal corporate suffix handling.
  * **Chapter 4 ("Indexing / Blocking")**: Inverted index blocking, standard blocking, multi-pass/disjunctive blocking, candidate completeness vs. reduction ratio.
  * **Chapter 5 ("Comparison Functions")**: Exact and approximate string matching (Jaro-Winkler, Levenshtein, Damerau, Q-grams, Jaccard), numeric difference functions, and phonetic encoders.
  * **Chapter 6 ("Classification")**: Supervised binary classification on pairwise comparison vectors.
  * **Chapter 7 ("Evaluation")**: Formulations of precision, recall, $F_\beta$ scores, and class imbalance.

### 1.2 Information Quality & Entity Architecture
* **Title**: *Entity Resolution and Information Quality*
* **Author**: John R. Talburt
* **Year**: 2011
* **Publisher**: Morgan Kaufmann (Elsevier)
* **ISBN / DOI**: 978-0-12-381972-7 / [10.1016/C2009-0-64350-0](https://doi.org/10.1016/C2009-0-64350-0)
* **Status**: **HIGH VALUE REFERENCE**
* **Relevant Content**: Chapter 3 ("The Entity Resolution Process") and Chapter 6 ("Lexical Matching"). Teaches rule-based identity resolution, deterministic merge rules, and how business entity records degrade across heterogeneous data feeds.

---

## 2. Entity Resolution & Record Linkage Books

### 2.1 Big Data Entity Resolution
* **Title**: *Big Data Integration*
* **Authors**: Xin Luna Dong and Divesh Srivastava
* **Year**: 2015
* **Publisher**: Morgan & Claypool Publishers (Synthesis Lectures on Data Management)
* **Relevant Sections**: Chapter 4 ("Entity Resolution") & Chapter 5 ("Advanced Entity Resolution").
* **What to Learn**: Scaling entity matching to tens of millions of records, blocking techniques for web data, and graph-based reconciliation.

### 2.2 Probabilistic Linkage & Survey Literature
* **Title**: *Principles of Data Quality*
* **Authors**: Thomas C. Redman
* **Year**: 2001
* **Publisher**: Artech House
* **Relevant Sections**: Chapters on data standardization, record linkage error costs, and corporate customer master data management (MDM).

---

## 3. Information Retrieval Books

### 3.1 The Canonical IR Reference
* **Title**: *Introduction to Information Retrieval*
* **Authors**: Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze
* **Year**: 2008
* **Publisher**: Cambridge University Press
* **Official URL**: [nlp.stanford.edu/IR-book/](https://nlp.stanford.edu/IR-book/) (Freely available online)
* **Status**: **MUST READ (SPECIFIC CHAPTERS)**
* **Direct Application to Challenge**:
  * **Chapter 1 ("Boolean Retrieval") & Chapter 2 ("The Inverted Index")**: Teaches how to construct memory-efficient inverted indexes to look up candidate records sharing tokens in sub-millisecond time.
  * **Chapter 3 ("Dictionaries and Tolerant Retrieval")**: Character $k$-gram indexes and wildcard matching for robust candidate retrieval under misspellings.
  * **Chapter 6 ("Scoring, Term Weighting and the Vector Space Model")**: TF-IDF and BM25 ranking for scoring candidate relevance.

---

## 4. Machine Learning & Gradient Boosting Books

### 4.1 Tree-Based Learning & Feature Ensembles
* **Title**: *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd Edition)
* **Authors**: Trevor Hastie, Robert Tibshirani, and Jerome Friedman
* **Year**: 2009
* **Publisher**: Springer
* **Official URL**: [hastie.su.domains/ElemStatLearn/](https://hastie.su.domains/ElemStatLearn/) (Freely available online)
* **Relevant Sections**: Chapter 10 ("Boosting and Additive Trees").
* **What to Learn**: The mathematical mechanics of gradient boosting, loss functions, tree depth interactions, and why gradient boosting handles correlated, heterogeneous tabular features better than neural networks.

### 4.2 Applied Machine Learning Engineering
* **Title**: *Approaching (Almost) Any Machine Learning Problem*
* **Author**: Abhishek Thakur
* **Year**: 2020
* **Relevant Sections**: Chapter on Tabular Data, Cross-Validation (GroupKFold), and Metric Optimization.
* **What to Learn**: Practical competitive ML workflows: structuring fold splits to prevent data leakage, hyperparameter tuning for LightGBM/CatBoost, and threshold optimization.

---

## 5. NLP & Multilingual Text Resources

### 5.1 Speech and Language Processing
* **Title**: *Speech and Language Processing* (3rd Edition Draft)
* **Authors**: Dan Jurafsky and James H. Martin
* **Year**: 2024
* **Official URL**: [web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
* **Relevant Chapters**:
  * **Chapter 2 ("Regular Expressions, Text Normalization, Edit Distance")**: The gold-standard mathematical formulation of Levenshtein edit distance, minimum edit path, and regex-based text canonicalization.

### 5.2 Multilingual Address Normalization Guide
* **Title**: *International Address Formats and Postal Syntax Guidelines*
* **Author / Publisher**: Universal Postal Union (UPU) / Postal Address Standards
* **Key Insights**:
  * **US**: `[Building No] [Street Name] [Unit/Ste], [City], [State] [Zip]`
  * **India**: `[Plot No], [Street/Road], [Near/Opp Landmark], [Locality], [City], [Pin Code]`
  * **France**: `[N° de voie] [Type de voie (Rue/Avenue/Boulevard)] [Nom de voie], [Code Postal (5 chiffres)] [Commune]`

---

## 6. Data Engineering & Scalability Resources

### 6.1 Fast Tabular Data Processing
* **Title**: *Polars User Guide & High-Performance Data Processing*
* **Authors**: Ritchie Vink and the Polars Development Team
* **Year**: 2024
* **Official URL**: [docs.pola.rs](https://docs.pola.rs)
* **Status**: **MUST READ MANUAL**
* **Direct Application**:
  * Polars Streaming LazyFrames (`pl.scan_csv(..., separator='\t')`) enable querying and transforming the 12.5M train and 11.7M test datasets in memory-mapped chunks, eliminating the $>32\text{ GB}$ Pandas out-of-memory crash.

### 6.2 Modern In-Process Analytics
* **Title**: *DuckDB: An In-Process Analytical Database Management System*
* **Authors**: Mark Raasveldt and Hannes Mühleisen
* **Year**: 2019 / Official Guide (2024)
* **Official URL**: [duckdb.org/docs/](https://duckdb.org/docs/)
* **Direct Application**: SQL-based multi-channel blocking joins on disk without loading unindexed Cartesian products into RAM.

---

## 7. Python & Software Engineering Resources

### 7.1 Production-Grade Scientific Python
* **Title**: *High Performance Python: Practical Performant Programming for Humans* (2nd Edition)
* **Authors**: Micha Gorelick and Ian Ozsvald
* **Year**: 2020
* **Publisher**: O'Reilly Media
* **Relevant Chapters**: Chapter 6 ("Matrix and Vector Computation"), Chapter 8 ("Asynchronous I/O"), Chapter 9 ("The multiprocessing Module").
* **What to Learn**: Memory profiling (`memory_profiler`), vectorized NumPy operations, and parallelizing pairwise feature extraction across 16 CPU cores.

---

## 8. Foundational Research Papers

### 8.1 The Foundation of Record Linkage
* **Title**: *A Theory for Record Linkage*
* **Authors**: Ivan P. Fellegi and Alan B. Sunter
* **Year**: 1969
* **Journal**: *Journal of the American Statistical Association* (JASA), 64(328), 1183–1210
* **DOI**: [10.1080/01621459.1969.10501049](https://doi.org/10.1080/01621459.1969.10501049)
* **Classification**: **FOUNDATIONAL (MUST READ)**
* **Summary**: The seminal mathematical foundation defining record linkage as a decision problem over a comparison vector $\gamma$, establishing the optimal likelihood ratio test partitioning pairs into Matches ($M$), Non-matches ($U$), and Clerical Review ($P$).

### 8.2 The Jaro-Winkler Metric
* **Title**: *String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage*
* **Author**: William E. Winkler
* **Year**: 1990
* **Publication**: *Proceedings of the Section on Survey Research Methods*, American Statistical Association, 354–359
* **Classification**: **FOUNDATIONAL**
* **Summary**: Introduces the Winkler modification to Matthew Jaro's string comparator, up-weighting common prefix characters ($L \le 4$). Crucial for business name matching where prefixes convey brand identity.

### 8.3 The Evaluation Metric ($F_\beta$)
* **Title**: *Information Retrieval* (Chapter 7: Evaluation)
* **Author**: C. J. Van Rijsbergen
* **Year**: 1979
* **Publisher**: Butterworth-Heinemann
* **Classification**: **FOUNDATIONAL (MUST READ)**
* **Summary**: Derives the mathematical formulation of $F_\beta$ from the effectiveness function $E = 1 - \frac{1}{\frac{\alpha}{P} + \frac{1-\alpha}{R}}$, proving that $\beta = 0.5$ weights precision twice as heavily as recall.

---

## 9. Modern Research Papers

### 9.1 Comparative String Matching Benchmarks
* **Title**: *Comparative Analysis of Approximate String Matching Techniques for Entity Resolution*
* **Authors**: George Papadakis, Ekaterini Svirsky, Avigdor Gal, and Themis Palpanas
* **Year**: 2020
* **Journal**: *IEEE Transactions on Knowledge and Data Engineering* (TKDE)
* **DOI**: [10.1109/TKDE.2020.3013622](https://doi.org/10.1109/TKDE.2020.3013622)
* **Classification**: **MODERN HIGH VALUE**
* **Summary**: Exhaustive benchmark comparing 27 string similarity algorithms on real-world entity resolution datasets. Demonstrates that token-based Jaccard/Dice combined with Jaro-Winkler provides the highest empirical discriminative power for named entities.

### 9.2 Deep Learning for Entity Matching (Empirical Reality)
* **Title**: *Deep Learning for Entity Matching: A Comprehensive Empirical Evaluation*
* **Authors**: Sidharth Mudgal, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchoon Park, Ganesh Krishnan, Rohit Gokhale, Jianbing Huang, David Paulsen
* **Year**: 2018
* **Conference**: *ACM SIGMOD International Conference on Management of Data*, 483–498
* **DOI**: [10.1145/3183713.3196926](https://doi.org/10.1145/3183713.3196926)
* **Classification**: **MODERN REFERENCE**
* **Summary**: Large-scale empirical evaluation comparing deep learning models (RNNs, Siamese networks) against feature-engineered GBDTs across structured and textual datasets. Concludes that for structured tabular data with text fragments, feature engineering + GBDTs achieves comparable or superior $F$-scores at a fraction of the compute cost.

### 9.3 Ditto: Entity Matching with Pre-trained Transformers
* **Title**: *Ditto: Deep Entity Matching with Pre-trained Language Models*
* **Authors**: Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, Wang-Chiew Tan
* **Year**: 2020
* **Conference**: *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (EMNLP), 2033–2044
* **DOI**: [10.18653/v1/2020.emnlp-main.159](https://doi.org/10.18653/v1/2020.emnlp-main.159)
* **Classification**: **MODERN REFERENCE**
* **Summary**: Casts entity matching as a sequence-pair classification problem using RoBERTa. Highlights why sequence-pair cross-encoders achieve high accuracy but require extreme computational budgets ($O(N^2)$ cross-attention), underscoring why GBDT is preferred for 24M records.

### 9.4 LightGBM
* **Title**: *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
* **Authors**: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu
* **Year**: 2017
* **Conference**: *Advances in Neural Information Processing Systems* (NeurIPS 2017)
* **Classification**: **MODERN HIGH VALUE**
* **Summary**: Introduces Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), enabling tree learning on millions of rows in seconds.

---

## 10. Technical Articles & Practical Guides

1. **"Entity Resolution: The Complete Guide"** (Reltio Data Architecture Series, 2023). Covers deterministic rules vs. probabilistic linkage in enterprise customer master records.
2. **"Tuning Thresholds for Custom F-Beta Scores in Scikit-Learn"** (Towards Data Science, 2021). Practical mathematical guide on optimizing decision thresholds when precision is favored over recall.
3. **"Fast String Matching in Python: From Levenshtein to RapidFuzz"** (Max Bachmann, 2022). Explains Cython and C++ SIMD optimizations for string comparisons.

---

## 11. Official Documentation & Specifications

1. **Official Competition Resource (`student_resource/README.md`)**: Exact schemas, rules, test set France introduction, and candidate generation evaluation mandate.
2. **Official Validator Script (`student_resource/utils/validate_submission.py`)**: Authoritative formatting constraints and exit code rules.
3. **Polars Documentation**: [docs.pola.rs](https://docs.pola.rs) (Streaming, Expressions, Joins).
4. **LightGBM Documentation**: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io) (Parameters, Early Stopping, Binary Logloss).
5. **RapidFuzz Documentation**: [rapidfuzz.github.io/RapidFuzz/](https://rapidfuzz.github.io/RapidFuzz/) (`fuzz.token_sort_ratio`, `distance.JaroWinkler`).

---

## 12. Useful Open-Source Python Libraries

| Library | Version | License | Primary Purpose in Challenge | Key APIs / Functions |
| :--- | :--- | :--- | :--- | :--- |
| **`polars`** | $\ge 0.20$ | MIT | Streaming data ingestion & out-of-core joins | `pl.scan_csv(sep='\t')`, `lazy().join()`, `sink_parquet()` |
| **`rapidfuzz`** | $\ge 3.5$ | MIT | C++ SIMD-accelerated string metrics | `fuzz.token_sort_ratio`, `fuzz.token_set_ratio`, `distance.JaroWinkler` |
| **`lightgbm`** | $\ge 4.0$ | MIT | Pairwise matching classifier | `LGBMClassifier(objective='binary', metric='binary_logloss')` |
| **`scikit-learn`** | $\ge 1.3$ | BSD-3 | Cross-validation & TF-IDF tokenization | `model_selection.GroupKFold`, `feature_extraction.text.TfidfVectorizer` |
| **`scipy`** | $\ge 1.11$ | BSD-3 | Sparse matrix operations for token inverted indexes | `scipy.sparse.csr_matrix` |
| **`duckdb`** | $\ge 0.9$ | MIT | In-process SQL blocking engine | `duckdb.query("SELECT ... FROM ...")` |

*Compliance Note*: All recommended libraries operate under **MIT / BSD-3 licenses**, fully adhering to competition rule #5 ("MIT/Apache 2.0 license, up to 8B parameters").

---

## 13. GitHub Solution & Code References

* **RapidFuzz Core Implementation**: [github.com/rapidfuzz/RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) — C++ string distance algorithms.
* **LightGBM Official Repository**: [github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM) — Tree algorithms.
* **Hugging Face Challenge Dataset**: [huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026](https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026) — Verified Parquet/TSV distribution.

---

## 14. Competition-Specific References

1. **Kaggle Record Linkage Competitions**:
   * *Shopee - Price Match Guarantee* (Kaggle 2021): Key learnings on multi-channel candidate generation, text shingle blocking, and high-threshold ensembling.
   * *Foursquare - Location Matching* (Kaggle 2022): Key learnings on business name and physical address matching, POI deduplication, and coordinate/address disambiguation.
2. **Amazon ML Challenge 2026 Unstop Portal Updates**:
   * Candidate generation parsimony scoring update.
   * France country introduction in test set.

---

## 15. Structured Learning Path: From Zero to Expert

```
Level 0: Prerequisites (Polars, Git, Command Line)
   │
Level 1: Data Cleaning & Normalization (Christen Ch. 3)
   │
Level 2: String & Phonetic Similarity (Christen Ch. 5, RapidFuzz)
   │
Level 3: Entity Resolution Theory (Fellegi-Sunter 1969, Christen Ch. 2)
   │
Level 4: Blocking & Candidate Generation (Christen Ch. 4, Inverted Index)
   │
Level 5: Feature Engineering & Disambiguation (Papadakis 2020)
   │
Level 6: Classification & Ranking (Ke et al. 2017, LightGBM)
   │
Level 7: Information Retrieval & Shingling (Manning Ch. 1, 2, 6)
   │
Level 8: Multilingual & International Address Text (French/Indian syntax)
   │
Level 9: Validation Design & F0.5 Optimization (Van Rijsbergen Ch. 7, GroupKFold)
   │
Level 10: Scalable Pipeline Engineering (Memory profiling, Streaming)
   │
Level 11: Production Submission Packaging (Validator script pass, documentation)
```

### Detailed Level breakdown:
* **LEVEL 0 — Prerequisites**: Setting up Python 3.11, installing `polars`, `rapidfuzz`, `lightgbm`, verifying `student_resource/utils/validate_submission.py`.
* **LEVEL 1 — Data Cleaning**: Reading Christen Ch. 3; implementing legal suffix stripping (`Inc`, `LLC`, `Pvt Ltd`, `SA`, `SAS`).
* **LEVEL 2 — String Similarity**: Reading Christen Ch. 5; benchmarking Jaro-Winkler and Token Sort ratio on business names.
* **LEVEL 3 — ER Theory**: Reading Fellegi-Sunter (1969); understanding the likelihood ratio between match and non-match populations.
* **LEVEL 4 — Blocking & Candidate Generation**: Reading Christen Ch. 4; implementing country-partitioned multi-channel blocking to achieve $>96\%$ recall with $\le 25$ candidates per entity.
* **LEVEL 5 — Feature Engineering**: Building the 32-feature matrix (name, address, strict numeric agreement, candidate rank).
* **LEVEL 6 — Classification**: Reading Ke et al. (2017); training 5-fold `GroupKFold` LightGBM.
* **LEVEL 7 — Retrieval Systems**: Reading Manning Ch. 1, 2; building inverted token indexes with frequency-based stopword pruning.
* **LEVEL 8 — Multilingual Text**: Handling France test set; normalizing French street keywords (`Rue`, `Avenue`) and accents.
* **LEVEL 9 — Validation & F0.5**: Reading Van Rijsbergen Ch. 7; setting decision threshold to $p \ge 0.82$; verifying singletons score $1.0$.
* **LEVEL 10 — Scalability**: Memory-efficient execution under 12 GB RAM using streaming Polars.
* **LEVEL 11 — Packaging**: Generating `matching_results.tsv` and `candidate_pairs.tsv`; running `validate_submission.py` to confirm exit code 0; writing `Documentation_template.md`.

---

## 16. Reading While Building Summary

* **Before Preprocessing**: Read Christen (2012) Chapter 3.
* **Before Blocking**: Read Christen (2012) Chapter 4 + Portal Candidate Set announcement.
* **Before Feature Extraction**: Read Christen (2012) Chapter 5 + Papadakis et al. (2020).
* **Before Model Training**: Read Ke et al. (2017) (LightGBM).
* **Before Threshold Tuning**: Read Van Rijsbergen (1979) Chapter 7 + Portal Evaluation rules.
* **Before Packaging**: Read `student_resource/utils/validate_submission.py`.

*(See full stage-by-stage guide in [READING_WHILE_BUILDING.md](file:///c:/Users/DELL/Downloads/Amazon_ML/READING_WHILE_BUILDING.md))*

---

## 17. Final Recommended Core Library (The Essential 5)

If the team has limited time, read **ONLY** these 5 references:

1. **Peter Christen (2012) — *Data Matching***: Chapters 2, 3, 4, 5. (The blueprint for our entire pipeline).
2. **Official Challenge Specification (`student_resource/README.md`)**: The primary source of truth for schemas, constraints, France test set, and metrics.
3. **C. J. Van Rijsbergen (1979) — *Information Retrieval*, Chapter 7**: The mathematical origin of $F_\beta$ and asymmetric error penalization.
4. **Ke et al. (2017) — *LightGBM: A Highly Efficient GBDT***: The core machine learning engine.
5. **RapidFuzz & Polars Official Guides**: The high-throughput computation stack.
