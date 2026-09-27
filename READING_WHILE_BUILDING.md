# Reading While Building Guide — Just-In-Time Engineering Manual

**Target Audience**: Engineering & Data Science Team  
**Philosophy**: *Read only what you need immediately prior to implementing each component.*

---

## Stage 0: Before Dataset Analysis & Profiling

### What to Read:
1. **Peter Christen — *Data Matching* (2012): Chapter 2 ("The Data Matching Process")**
   * *Focus*: Sections 2.1–2.4 (The standard 5-step pipeline: Preprocessing $\to$ Indexing $\to$ Field Comparison $\to$ Classification $\to$ Evaluation).
   * *Why*: Establishes clean mental boundaries between candidate generation (blocking) and matching models so the team does not conflate candidate retrieval with final classification.
2. **Official Dataset File Specifications (`student_resource/README.md`)**
   * *Focus*: Column schemas (`entity_id`, `business_name`, `business_address`, `country`), singletons ($5.58\%$), and the France test set out-of-distribution shift.

### Immediate Action Item:
* Run profiling scripts using **Polars** to verify record counts ($12.5\text{M}$ train, $11.7\text{M}$ test), missing values, and zero cross-country matches without loading full TSVs into Pandas RAM.

---

## Stage 1: Before Text Normalization & Preprocessing

### What to Read:
1. **Peter Christen — *Data Matching* (2012): Chapter 3 ("Data Pre-Processing and Quality")**
   * *Focus*: Section 3.3 ("Data Cleansing and Standardization") & Section 3.4 ("Segmentation and Parsing").
   * *Why*: Learn deterministic lookup dictionaries for legal corporate entity suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SA`, `SAS`, `SARL`) and street types (`Street`, `Road`, `Rue`, `Avenue`).
2. **Unicode Standard Annex #15: Unicode Normalization Forms (NFC / NFKD)**
   * *Focus*: Stripping diacritics and combining characters cleanly across English and French text while preserving alphanumeric tokens.

### Immediate Action Item:
* Implement `src/normalize.py`:
  * Lowercase folding.
  * Unicode NFKD normalization (folding accented French vowels while retaining base characters).
  * Legal suffix canonicalization (mapping abbreviations to a uniform token).
  * Street abbreviation expansion.
  * Explicit digit extraction for address numbers and postal codes.

---

## Stage 2: Before Blocking & Candidate Generation

### What to Read:
1. **Peter Christen — *Data Matching* (2012): Chapter 4 ("Indexing / Blocking")**
   * *Focus*: Section 4.2 ("Traditional Blocking"), Section 4.3 ("Standard Inverted Index Blocking"), Section 4.7 ("Multi-Pass / Disjunctive Blocking").
   * *Why*: Understand the mathematical relationship between the **Recall Ceiling** ($PC = |M \cap C| / |M|$) and the **Reduction Ratio** ($RR$). Discover why a single blocking key always fails and why disjunctive multi-channel blocking guarantees $>95\%$ recall.
2. **Official Evaluation Announcement on Candidate Generation**:
   * *Focus*: *"The approach that generates a smaller candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private leaderboard."*
   * *Why*: Enforce a hard cap of $\le 20 - 25$ candidates per entity. Do not bloat candidate sets.

### Immediate Action Item:
* Implement `src/blocking.py`:
  * Step 1: Partition records strictly by `country` (100% loss-free, cuts 61.1% of pairs).
  * Step 2: Channel 1 — Clean exact name.
  * Step 3: Channel 2 — First 6 characters of name + postal code.
  * Step 4: Channel 3 — Normalized street number + street name shingle.
  * Step 5: Channel 4 — Rare brand token (IDF $> 4.5$).
  * Step 6: Union channels, deduplicate, and enforce top-$K$ cap ($K=25$). Output `candidate_pairs.tsv`.

---

## Stage 3: Before Feature Engineering

### What to Read:
1. **Peter Christen — *Data Matching* (2012): Chapter 5 ("Comparison Functions")**
   * *Focus*: Section 5.3 ("String Comparison") & Section 5.4 ("Numeric and Date Comparison").
   * *Why*: Master Jaro-Winkler (prefix bias for names), Levenshtein ratio, token sort vs. token set ratios, and character $q$-grams.
2. **Papadakis et al. (IEEE TKDE, 2020): *Comparative Analysis of Approximate String Matching***
   * *Focus*: Table 2 & Section 4 (Empirical performance benchmarks across string algorithms).
   * *Why*: Identify the fastest, most discriminative metrics for Python implementation (`rapidfuzz.fuzz.token_sort_ratio`, `rapidfuzz.distance.JaroWinkler`).

### Immediate Action Item:
* Implement `src/features.py`:
  * Compute the 32 orthogonal features across candidate pairs.
  * Group 1: Name features (10 features).
  * Group 2: Address features (12 features).
  * Group 3: Strict numeric verification (6 features — street number match, postal code match, digit Jaccard).
  * Group 4: Meta & rank interactions (4 features — candidate rank in block, margin to next candidate).

---

## Stage 4: Before Model Training

### What to Read:
1. **Ke et al. (NeurIPS, 2017): *LightGBM: A Highly Efficient Gradient Boosting Decision Tree***
   * *Focus*: Section 2 ("Gradient-based One-Side Sampling - GOSS") & Section 3 ("Exclusive Feature Bundling - EFB").
   * *Why*: Understand why histogram binning enables LightGBM to score millions of candidate feature vectors in seconds with minimal RAM.
2. **Christen (2012): Chapter 6 ("Classification"), Section 6.4 ("Supervised Classification")**
   * *Focus*: Formulating pairwise entity matching as binary classification (1 = True Match, 0 = Non-Match/Look-alike).

### Immediate Action Item:
* Implement `src/train.py`:
  * Setup 5-fold `GroupKFold` grouped strictly on `source1_entity_id` to eliminate entity leakage.
  * Maintain exact $5.58\%$ singleton ratio per fold.
  * Train LightGBM with binary logloss (`objective='binary'`, `metric='binary_logloss'`, `learning_rate=0.05`, `num_leaves=63`, `n_estimators=600`).
  * Log out-of-fold predicted probabilities for threshold calibration.

---

## Stage 5: Before Threshold Optimization ($F_{0.5}$)

### What to Read:
1. **C. J. Van Rijsbergen (1979): *Information Retrieval*, Chapter 7 ("Evaluation")**
   * *Focus*: Derivation of $F_\beta$ and the cost-utility matrix under asymmetric error penalties.
   * *Why*: Understand why the optimal decision threshold shifts from $p^* = 0.50$ (in $F_1$) up to $p^* = \frac{1}{1 + 0.25} = 0.80$ (in $F_{0.5}$).
2. **Official Video Slide 5 & `student_resource/README.md` Evaluation Section**
   * *Focus*: The singleton rule ($1.0$ if empty, $0.0$ if any false match emitted) and the $4\times$ penalty ratio on false merges.

### Immediate Action Item:
* Implement `src/threshold.py`:
  * Evaluate out-of-fold Macro $F_{0.5}$ across threshold values $p \in [0.60, 0.95]$ in increments of $0.01$.
  * Implement the **Singleton Guard**: If the top candidate for a Source 1 entity has $\hat{p} < p^*$, predict an empty match list.
  * Verify that singletons achieve $>95\%$ prediction accuracy on validation data.

---

## Stage 6: Before Final Validation & Submission Packaging

### What to Read:
1. **Official Submission Validator (`student_resource/utils/validate_submission.py`)**
   * *Focus*: Lines 116–205 (Validation rules: exact TSV headers, no commas in delimiters, no self-matches, no trailing commas, valid test IDs).
2. **Official Packaging Guide (`student_resource/README.md` Lines 147–179)**
   * *Focus*: Exact `.zip` directory structure:
     `output/matching_results.tsv`, `output/candidate_pairs.tsv`, `code/business_entity_resolution/`, `Documentation_template.md`.

### Immediate Action Item:
* Execute validation command:
  ```bash
  python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test
  ```
* Ensure it returns `PASS` (exit code `0`).
* Assemble the final `.zip` package and verify archive integrity.
