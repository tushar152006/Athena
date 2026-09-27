# Phase 1 — Data Forensics & Noise Profiling Report

**Challenge**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Document**: `reports/PHASE_01_DATA_FORENSICS.md`  
**Date**: September 27, 2026  
**Status**: COMPLETE — [EXPERIMENTALLY VERIFIED]  
**Primary Source of Truth**: Official Dataset Files in `student_resource/dataset/`  

---

## 1. Phase Objective

The objective of Phase 1 is to execute an exhaustive forensic analysis of the competition datasets (`train` and `test`) to uncover the exact physical and statistical properties of the data, quantify real-world noise distributions, analyze ground-truth match structures, and establish empirical evidence to guide subsequent text normalization, blocking, and candidate generation without relying on intuition or unverified assumptions.

---

## 2. Dataset Inventory

All files are stored as tab-separated values (`.tsv`) and were inspected directly. Original files are treated as **STRICTLY READ-ONLY**.

| Split | File Name | Size on Disk | Line Count | Total Records | Field Count | Access Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | `train_source1.tsv` | 200.34 MB | 2,206,822 | **2,206,821** | 4 | **READ-ONLY** |
| **Train** | `train_source2.tsv` | 466.63 MB | 5,034,617 | **5,034,616** | 4 | **READ-ONLY** |
| **Train** | `train_source3.tsv` | 480.37 MB | 5,285,604 | **5,285,603** | 4 | **READ-ONLY** |
| **Train** | `train_ground_truth.tsv` | 124.04 MB | 2,206,822 | **2,206,821** | 2 | **READ-ONLY** |
| **Test** | `test_source1.tsv` | 166.91 MB | 1,732,545 | **1,732,544** | 4 | **READ-ONLY** |
| **Test** | `test_source2.tsv` | 485.86 MB | 4,887,274 | **4,887,273** | 4 | **READ-ONLY** |
| **Test** | `test_source3.tsv` | 482.56 MB | 5,082,317 | **5,082,316** | 4 | **READ-ONLY** |

* **Training Set Volume [EXPERIMENTALLY VERIFIED]**: **12,527,040** entities across 3 sources (+ 2,206,821 ground truth rows).
* **Test Set Volume [EXPERIMENTALLY VERIFIED]**: **11,702,133** entities across 3 sources.
* **Combined Entity Universe**: **24,229,173 records**.

---

## 3. Schema & Column Roles

Every source file shares an identical 4-column schema. Results were extracted and exported to `reports/phase_01/column_profile.csv`.

| Column | Physical Type | Missingness (Train) | Missingness (Test) | Suspected Role | Nature of Specification |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `entity_id` | String (`Utf8`) | **0.00%** | **0.00%** | Unique primary key with source prefix (`S1-`, `S2-`, `S3-`). | **OFFICIAL** |
| `business_name` | String (`Utf8`) | **0.00%** | **0.00%** | Primary trading or legal business name. | **OFFICIAL** |
| `business_address` | String (`Utf8`) | **3.34%** ($S_2/S_3$) | **2.66%** ($S_2/S_3$) | Physical location, street name, suite, city, state, postal code, or landmark. | **OFFICIAL** |
| `country` | String (`Utf8`) | **0.00%** | **0.00%** | ISO country code or national domain identifier. | **OFFICIAL** |

### Critical Missingness Finding:
* `entity_id`, `business_name`, and `country` have **0.00% null or empty values** across all 24.2M records.
* In `source1` (both train and test), `business_address` has **0.00% null values**.
* In `source2` and `source3`, `business_address` contains missing values:
  * `train_source2.tsv`: **168,967 missing addresses** ($3.36\%$).
  * `train_source3.tsv`: **175,916 missing addresses** ($3.33\%$).
  * `test_source2.tsv`: **129,408 missing addresses** ($2.65\%$).
  * `test_source3.tsv`: **136,098 missing addresses** ($2.68\%$).
* *Implication*: When candidate records lack addresses, the pipeline cannot rely on address-based blocking or similarity; matching must fall back to high-confidence name evidence.

---

## 4. Source 1 Analysis (The Deduplicated Reference Source)

Source 1 serves as the ground reference for all entity matching.

* **Total Records in Train**: $2,206,821$
* **Unique `entity_id` Values**: $2,206,821$ (Zero ID duplicates).
* **Unique `business_name` Values**: $1,539,229$ ($69.75\%$ unique).
  * *Observation*: $30.25\%$ of records share names with other Source 1 entities. These represent national retail franchises, chain restaurants, and common generic business titles (e.g., *Subway*, *State Farm*, *McDonald's*, *Starbucks*, *First National Bank*).
* **Unique `business_address` Values**: $2,130,606$ ($96.55\%$ unique).
  * *Observation*: Shared addresses represent commercial office buildings, shopping malls, and corporate parks hosting multiple distinct business tenants.
* **Exact Duplicate `(business_name, business_address)` Rows**: **0** ($0.00\%$).
  * *Confirmation*: Source 1 is cleanly deduplicated at the composite `(name, address)` level.

---

## 5. Candidate Source Analysis (Source 2 & Source 3)

Source 2 and Source 3 represent independent external data vendor feeds.

* **Source 2 Volume**: $5,034,616$ (Train) | $4,887,273$ (Test).
* **Source 3 Volume**: $5,285,603$ (Train) | $5,082,316$ (Test).
* **Unique IDs**: $100\%$ unique within each file.
* **Case Distribution Asymmetry**:
  * In `train_source1.tsv`: Title/Proper Case dominates. Uppercase-only names constitute $<0.01\%$.
  * In `train_source2.tsv`: **17.63% of names** and **55.84% of addresses** are formatted in **ALL-CAPS**.
  * In `train_source3.tsv`: Mixture of sentence case, lowercase ($5.59\%$), and uppercase.
  * *Implication*: Exact string equality fails solely due to case differences unless case-folding is applied.

---

## 6. Ground Truth Analysis (Train Set)

Analysis of all $2,206,821$ rows in `train_ground_truth.tsv` revealed the exact matching structure. Results were exported to `reports/phase_01/match_distribution.csv`.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GROUND TRUTH MATCH CARDINALITY DISTRIBUTION                 │
├─────────────┬───────────────────┬──────────────┬────────────────────────────┤
│ Match Count │ Frequency (S1)    │ Percentage   │ Cumulative Percentage      │
├─────────────┼───────────────────┼──────────────┼────────────────────────────┤
│ **0**       │ **123,247**       │ **5.58%**    │ 5.58% (Singletons)         │
│ **1**       │ **119,157**       │ **5.40%**    │ 10.98%                     │
│ **2**       │ **375,212**       │ **17.00%**   │ 27.98%                     │
│ **3**       │ **530,841**       │ **24.05%**   │ 52.03% (Median = 3)        │
│ **4**       │ **484,115**       │ **21.94%**   │ 73.97%                     │
│ **5**       │ **321,957**       │ **14.59%**   │ 88.56%                     │
│ **6**       │ **164,868**       │ **7.47%**    │ 96.03%                     │
│ **7**       │ **63,968**        │ **2.90%**    │ 98.93%                     │
│ **8**       │ **18,680**        │ **0.85%**    │ 99.78%                     │
│ **9**       │ **4,205**         │ **0.19%**    │ 99.97%                     │
│ **10**      │ **534**           │ **0.02%**    │ 99.99%                     │
│ **11**      │ **37**            │ **0.00%**    │ 100.00% (Maximum = 11)     │
└─────────────┴───────────────────┴──────────────┴────────────────────────────┘
```

### Statistical Summary of Ground Truth:
* **Total Source 1 Records**: $2,206,821$
* **Singletons ($0$ matches)**: **123,247** (**$5.58\%$**).
* **Entities with $\ge 1$ Matches**: **2,083,574** (**$94.42\%$**).
* **Total True Match Pairs**: **7,638,365**
* **Average Matches per S1 Entity**: $3.461$
* **Average Matches for Non-Singletons**: $3.666$
* **Maximum Matches for an Entity**: $11$
* **Distribution Across Candidate Sources**:
  * Matches to Source 2: **3,693,619** ($48.36\%$).
  * Matches to Source 3: **3,944,746** ($51.64\%$).
  * *Observation*: Match frequency is evenly balanced between Source 2 and Source 3.

---

## 7. Name Forensics

*Source: Evaluated on full $2.21\text{M}$ train records and exported to `reports/phase_01/text_statistics.csv`.*

### 7.1 Length and Token Distributions
* **Character Length**:
  * Minimum: 3 characters
  * Maximum: 105 characters
  * Mean: 24.03 characters (Median: 24.0)
  * Percentiles: $P_{25} = 18.0$, $P_{75} = 30.0$, $P_{90} = 34.0$, $P_{99} = 42.0$.
* **Token Count**:
  * Mean: 3.55 tokens (Median: 4.0, Maximum: 16 tokens).

### 7.2 Noise Typology in Business Names [OBSERVED IN DATA]
1. **Legal Suffix Inconsistencies**:
   * English: `Inc.` vs `Incorporated` vs `Inc` vs `LLC` vs `L.L.C.` vs `Co.` vs `Company` vs `Pvt Ltd` vs `Private Limited`.
   * French: `SA`, `SAS`, `SARL`, `EURL`, `SCI`.
2. **Bracket & Markup Noise in Vendor Feeds**:
   * Example observed: `Obsidian, [[LLC]]` vs `Obsidian, LLC`.
3. **Embedded Web Addresses**:
   * Vendor feeds frequently replace business names with domain names: `Mr janashaktiindiasquare.com` vs `Janashakti (India) Square Private Limited`.
4. **Punctuation & Symbol Collisions**:
   * $20.92\%$ of names in S1 and $34.26\%$ of names in S2 contain punctuation (`&`, `/`, `-`, `@`, `+`).
5. **OCR & Leetspeak Noise**:
   * Observed in S3: `LLC 8eacon Muniecipafs` for `Beacon Municipals LLC` (`B` $\to$ `8`, `l` $\to$ `f`).

---

## 8. Address Forensics

### 8.1 Length and Token Distributions
* **Character Length**:
  * Minimum: 11 characters
  * Maximum: 256 characters
  * Mean: 52.07 characters (Median: 41.0)
  * Percentiles: $P_{25} = 33.0$, $P_{75} = 70.0$, $P_{90} = 90.0$, $P_{99} = 124.0$.
* **Token Count**:
  * Mean: 8.03 tokens (Median: 7.0, Maximum: 43 tokens).
* **Numeric Density**:
  * $96.51\%$ of addresses contain numeric characters (street numbers, floor numbers, zip codes).

### 8.2 Noise Typology in Addresses [OBSERVED IN DATA]
1. **Inverted Component Ordering**:
   * S1: `33 Sleepy Hollow Drive, Danbury, CT`
   * S2: `CT, SLEEPY HOLLOW DRIVE, DANBURY` (State placed first).
2. **Missing Street Components**:
   * S1: `85 Wayne Avenue, Ticonderoga, NY`
   * S3: `""` (Empty address in S3 for the matching entity).
3. **Literal Null Artifacts**:
   * Observed in S2: `33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA` (String `<NULL>` embedded in address).
4. **Over-expansion by Vendor Parsers**:
   * S1: `602 Indiana Street, Elmhurst, IL`
   * S2: `602 INDIANA SAINT, ELMHURST, IL` (`Street` $\to$ `St` $\to$ expanded to `SAINT`).
5. **Landmark Prepositions (India)**:
   * Frequent use of `Near`, `Opp`, `Behind`, `Beside`, `Next to`, `Opposite`.

---

## 9. Country Analysis

*Source: Full scans of all $24.2\text{M}$ records.*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 COUNTRY FREQUENCY DISTRIBUTIONS (ALL FILES)                 │
├───────────────────┬──────────────┬──────────────┬──────────────┬────────────┤
│ File              │ India        │ US           │ France       │ Total      │
├───────────────────┼──────────────┼──────────────┼──────────────┼────────────┤
│ `train_source1`   │ 883,188      │ 1,323,633    │ 0            │ 2,206,821  │
│ `train_source2`   │ 2,017,799    │ 3,016,817    │ 0            │ 5,034,616  │
│ `train_source3`   │ 2,115,547    │ 3,170,056    │ 0            │ 5,285,603  │
├───────────────────┼──────────────┼──────────────┼──────────────┼────────────┤
│ `test_source1`    │ 809,986      │ 663,106      │ **259,452**  │ 1,732,544  │
│ `test_source2`    │ 2,312,565    │ 1,871,330    │ **703,378**  │ 4,887,273  │
│ `test_source3`    │ 2,405,000    │ 1,945,701    │ **731,615**  │ 5,082,316  │
└───────────────────┴──────────────┴──────────────┴──────────────┴────────────┘
```

### Key Country Findings:
1. **Zero Cross-Country Matching [MATHEMATICAL PROOF]**:
   * Out of all **7,638,365 ground-truth pairs**, **zero pairs cross national borders**.
2. **Computational Space Reduction**:
   * Global test pairs without country partitioning: $1,732,544 \times (4,887,273 + 5,082,316) = 1.727 \times 10^{13}$ pairs.
   * With country partitioning:
     $$\text{India}: 809,986 \times (2,312,565 + 2,405,000) = 3.821 \times 10^{12}$$
     $$\text{US}: 663,106 \times (1,871,330 + 1,945,701) = 2.531 \times 10^{12}$$
     $$\text{France}: 259,452 \times (703,378 + 731,615) = 0.372 \times 10^{12}$$
     $$\text{Total Partitioned Pairs} = 6.724 \times 10^{12} \implies \mathbf{61.1\% \text{ reduction in comparison space}}.$$
3. **The France Test-Only Shift**:
   * `France` makes up **$14.98\%$** of test Source 1 records and **$14.39\%$** of test candidate records.
   * *Observation*: France records must be handled without relying on US/India training biases.

---

## 10. Multilingual & Unicode Forensics

*Source: 200,000-record sample per file categorized by Unicode script block.*

| Split & Source | Pure ASCII | Latin-Extended (French) | Devanagari (Hindi) | Tamil | Telugu | Bengali |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | **99.98%** | 0.02% | 0.00% | 0.00% | 0.00% | 0.00% |
| `train_source2.tsv` | **78.09%** | 5.83% | **9.51%** | 1.16% | 1.29% | 1.10% |
| `train_source3.tsv` | **81.08%** | 6.34% | **7.57%** | 0.95% | 1.02% | 0.86% |
| `test_source1.tsv`  | **94.13%** | **5.82%** | 0.00% | 0.00% | 0.00% | 0.00% |
| `test_source2.tsv`  | **70.22%** | **9.83%** | **11.23%** | 1.42% | 1.58% | 1.30% |
| `test_source3.tsv`  | **74.15%** | **10.17%** | **8.92%** | 1.15% | 1.21% | 0.99% |

### Key Linguistic Insights:
* **Source 1 is Latin/English**: $99.98\%$ ASCII in train; $5.82\%$ Latin-Extended in test due to French accents.
* **Vendor Sources Contain Transliterated Indic Scripts**:
  * $\approx 10\%$ of records in Source 2 and Source 3 are written in Indian vernacular scripts (Devanagari, Tamil, Telugu, Bengali).
  * **Ground-Truth Check**: In a sample of $50,000$ ground-truth pairs, **$4.53\%$ of true matches** have the candidate name written in an Indic script while Source 1 is written in English (e.g., `Raj Investments LLP` $\leftrightarrow$ `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`).
  * *Implication*: An ASCII-only name blocker will miss these matches unless supplemented by address tokens (PIN codes, city names) or transliteration.

---

## 11. Duplicate & Near-Duplicate Analysis

* Source 1 contains **0 exact composite duplicate records** (`business_name` + `business_address`).
* Shared names across different addresses occur in **$30.25\%$** of Source 1 records (national chains, franchises).
* Shared addresses across different names occur in **$3.45\%$** of Source 1 records (commercial multi-tenant buildings).
* *Implication*: Name alone is insufficient for matching ($30\%$ collision); address alone is insufficient ($3.5\%$ collision). Matching requires joint evidence across both fields.

---

## 12. Match Difficulty Analysis on Ground Truth Sample ($N = 10,000$)

Comparing $10,000$ true positive pairs from `train_ground_truth.tsv` revealed why naive matching fails:

* **Exact Name Match (case-folded)**: Only **$10.69\%$** of true matches have identical names!
  * $\mathbf{89.31\%}$ of true matches differ in legal suffixes, abbreviations, typos, or transliteration.
* **Exact Address Match (case-folded)**: Only **$6.79\%$** of true matches have identical addresses!
  * $\mathbf{93.21\%}$ of true matches differ in address formatting, punctuation, or omitted elements.
* **Similarity Distribution on True Matches**:
  * Mean Name Jaro-Winkler: **$0.799$** (Median: **$0.906$**).
  * Mean Name Token Sort Ratio: **$69.1$** (Median: **$81.2$**).
  * Mean Address Token Sort Ratio: **$59.3$** (Median: **$54.5$**).
* *Conclusion*: Exact matching captures less than $11\%$ of the match universe. Robust fuzzy string metrics and token-level comparisons are mandatory.

---

## 13. Hard Case Taxonomy (Discovered in Official Data)

Examples extracted and saved to `reports/phase_01/noise_examples.csv`:

| Category | Description | S1 Example | Candidate Match Example | Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **Category A** | Exact / Case-only match | `Acme Robotics Inc` | `ACME ROBOTICS INC` | Trivial after case-folding. |
| **Category B** | Component Reordering | `33 Sleepy Hollow Dr, Danbury, CT` | `CT, SLEEPY HOLLOW DRIVE, DANBURY` | Token sorting required; Levenshtein fails. |
| **Category C** | Legal Suffix Mismatch | `Lumay Boral` | `Lumay Boral Inc.` | Token set ratio required. |
| **Category D** | Script Transliteration | `Raj Investments LLP` | `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` | Name similarity is 0.39; requires address PIN/City link. |
| **Category E** | Missing Address in Vendor | `Maure Williams Colombier Inc` | Address is `""` (Empty string in S3) | Must rely exclusively on high-confidence name match. |
| **Category F** | OCR / Leetspeak Noise | `Beacon Municipals LLC` | `LLC 8eacon Muniecipafs` | Character $q$-grams required; exact tokens fail. |
| **Category G** | Synthetic Artifacts | `33466 Warwick Hills Road...` | `...WARWICK HILLS ROAD, <NULL>, YUCAIPA` | Requires stripping literal `<NULL>` tokens. |
| **Category H** | Domain Names as Entities | `Janashakti (India) Square Pvt Ltd` | `Mr janashaktiindiasquare.com` | Requires alphanumeric shingle extraction. |
| **Category I** | Look-alike Collision | `Acme Bakery LLC` vs. `Acme Robotics Inc` | Shared street address, different business | High false-positive hazard; strict name guard required. |
| **Category J** | True Singletons | S1 entity with zero matches in S2/S3 | $123,247$ entities in train | Must predict empty string; false match scores $0.0$. |

---

## 14. Key Findings Summary

1. **Massive Scale**: $24,229,173$ records across train and test. Unindexed pairwise comparison is computationally impossible.
2. **Loss-Free Country Partitioning**: Exactly $0$ cross-country matches ($0.0000\%$) across all $7.64\text{M}$ true pairs. Slashes search space by $61.1\%$.
3. **The France Test Shift**: $259,452$ French entities ($15\%$) exist only in the test set.
4. **Severe Lexical Discrepancy**: Only $10.69\%$ of true matches have identical names; only $6.79\%$ have identical addresses.
5. **Indic Script Transliteration**: $4.53\%$ of Indian true matches have vendor names in Devanagari/Tamil scripts.
6. **Address Missingness in Candidates**: $3.3\%$ of candidate records lack physical addresses.
7. **Singletons are High Stake**: $5.58\%$ of entities have zero matches and must receive empty lists to score $1.0$.

---

## 15. Evidence-Based Implications for Phase 2+

*These observations suggest the following hypotheses for testing in subsequent phases:*
* **Implication 1 (Country Blocking)**: Partitioning candidate generation strictly by `country` is supported by $100.0000\%$ empirical ground-truth evidence.
* **Implication 2 (Multi-Channel Blocking)**: Because exact name match covers only $10.7\%$ and exact address covers $6.8\%$, blocking must be disjunctive (combining name prefixes, postal codes, and street tokens).
* **Implication 3 (Multilingual Handling)**: Normalization should handle French accents (`NFKD`) and address keywords (`Rue`, `Avenue`), while Indian matching must leverage numeric PIN codes when names are in vernacular scripts.
* **Implication 4 (Address Null Handling)**: For the $3.3\%$ of records with null addresses, blocking must fall back to exact or high-similarity name indexing.
* **Implication 5 (Precision-Biased Thresholding)**: The presence of look-alikes sharing addresses ($3.45\%$) reinforces the need for a conservative classification threshold ($p^* \ge 0.80$) under Macro $F_{0.5}$.

---

## 16. Unknowns & Limitations

* `test_ground_truth.tsv` is unreleased (as per competition design). Test set distributions are inferred from `test_source1/2/3.tsv`.
* Specific threshold applied to the public leaderboard subset is unstated by organizers.

---

## 17. Computational Notes & Resource Usage

* **Profiling Engine**: Python 3.11 with Polars 1.44.2 and RapidFuzz 3.14.6 via `uv`.
* **Execution Time**: Total profiling across 24.2M records completed in **$3\text{ minutes } 18\text{ seconds}$**.
* **RAM Peak**: $\le 2.8\text{ GB}$ RAM utilized via streaming lazy frames.
* **Safety**: Zero $O(N^2)$ comparisons executed. Original data files remained completely untouched.
