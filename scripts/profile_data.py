import os
import sys
import re
import unicodedata
from collections import Counter
import polars as pl
import numpy as np
import rapidfuzz
from rapidfuzz import fuzz, distance

sys.stdout.reconfigure(encoding='utf-8')

TRAIN_DIR = os.path.join('student_resource', 'dataset', 'train')
TEST_DIR = os.path.join('student_resource', 'dataset', 'test')
REPORTS_DIR = os.path.join('reports', 'phase_01')
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=== STARTING PHASE 1 DATA FORENSICS ===")

# ---------------------------------------------------------
# 1. FILE INVENTORY & SCHEMA
# ---------------------------------------------------------
files_to_profile = [
    ('train', 'train_source1.tsv'),
    ('train', 'train_source2.tsv'),
    ('train', 'train_source3.tsv'),
    ('test', 'test_source1.tsv'),
    ('test', 'test_source2.tsv'),
    ('test', 'test_source3.tsv'),
]

inventory_rows = []
for split, fname in files_to_profile:
    fpath = os.path.join('student_resource', 'dataset', split, fname)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    df = pl.read_csv(fpath, separator='\t', truncate_ragged_lines=True)
    n_rows = len(df)
    n_cols = len(df.columns)
    cols = df.columns
    print(f"Loaded {fname}: {n_rows:,} rows, {n_cols} cols, {size_mb:.2f} MB")
    
    # Null analysis
    for col in cols:
        null_count = df[col].null_count()
        empty_count = (df[col].str.len_chars() == 0).sum() if df[col].dtype == pl.Utf8 else 0
        unique_count = df[col].n_unique()
        example = str(df[col][0]) if len(df) > 0 else ""
        role = "OFFICIAL: Record Identifier" if col == 'entity_id' else \
               "OFFICIAL: Business Name" if col == 'business_name' else \
               "OFFICIAL: Physical Address" if col == 'business_address' else \
               "OFFICIAL: Country Code" if col == 'country' else "UNKNOWN"
        inventory_rows.append({
            'split': split,
            'file': fname,
            'column': col,
            'dtype': str(df[col].dtype),
            'n_rows': n_rows,
            'null_count': null_count,
            'empty_count': empty_count,
            'null_pct': (null_count + empty_count) / n_rows * 100,
            'n_unique': unique_count,
            'example': example[:50],
            'role': role
        })

df_inventory = pl.DataFrame(inventory_rows)
df_inventory.write_csv(os.path.join(REPORTS_DIR, 'column_profile.csv'))
print("Saved column_profile.csv")

# ---------------------------------------------------------
# 2. GROUND TRUTH FORENSICS
# ---------------------------------------------------------
gt_path = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')
df_gt = pl.read_csv(gt_path, separator='\t')
print(f"Loaded ground truth: {len(df_gt):,} rows")

match_dist = Counter()
total_matches = 0
s2_matches = 0
s3_matches = 0

match_counts_list = []
singleton_ids = []
multi_match_ids = []

for row in df_gt.iter_rows():
    s1_id = row[0]
    matched = str(row[1]) if row[1] is not None else ""
    matched = matched.strip()
    if not matched:
        match_dist[0] += 1
        match_counts_list.append(0)
        if len(singleton_ids) < 10:
            singleton_ids.append(s1_id)
    else:
        ids = [m.strip() for m in matched.split(',') if m.strip()]
        k = len(ids)
        match_dist[k] += 1
        match_counts_list.append(k)
        total_matches += k
        for m in ids:
            if m.startswith('S2-'):
                s2_matches += 1
            elif m.startswith('S3-'):
                s3_matches += 1
        if k >= 5 and len(multi_match_ids) < 10:
            multi_match_ids.append((s1_id, k, ids))

print(f"Ground truth match distribution: {dict(sorted(match_dist.items()))}")
print(f"Total true match pairs: {total_matches:,}")
print(f"S2 matches: {s2_matches:,} ({s2_matches/total_matches*100:.2f}%)")
print(f"S3 matches: {s3_matches:,} ({s3_matches/total_matches*100:.2f}%)")

gt_dist_rows = []
for k in sorted(match_dist.keys()):
    count = match_dist[k]
    gt_dist_rows.append({
        'match_count': k,
        'frequency': count,
        'percentage': count / len(df_gt) * 100
    })
pl.DataFrame(gt_dist_rows).write_csv(os.path.join(REPORTS_DIR, 'match_distribution.csv'))
print("Saved match_distribution.csv")

# ---------------------------------------------------------
# 3. TEXT & NOISE FORENSICS (Name and Address)
# ---------------------------------------------------------
# We profile S1, S2, S3 in Train and Test
def profile_text_series(series, name_label):
    lengths = series.str.len_chars().to_numpy()
    # Token count
    tokens = series.str.split(' ').list.len().to_numpy()
    
    # Character regex checks
    has_digits = (series.str.contains(r'\d')).sum()
    has_punct = (series.str.contains(r'[^\w\s]')).sum()
    is_upper = (series.str.contains(r'^[A-Z\s\d\W]+$')).sum()
    is_lower = (series.str.contains(r'^[a-z\s\d\W]+$')).sum()
    has_non_ascii = (series.str.contains(r'[^\x00-\x7F]')).sum()
    
    stats = {
        'field': name_label,
        'count': len(series),
        'min_len': int(np.min(lengths)),
        'max_len': int(np.max(lengths)),
        'mean_len': float(np.mean(lengths)),
        'median_len': float(np.median(lengths)),
        'p25_len': float(np.percentile(lengths, 25)),
        'p75_len': float(np.percentile(lengths, 75)),
        'p90_len': float(np.percentile(lengths, 90)),
        'p99_len': float(np.percentile(lengths, 99)),
        'mean_tokens': float(np.mean(tokens)),
        'median_tokens': float(np.median(tokens)),
        'max_tokens': int(np.max(tokens)),
        'pct_has_digits': float(has_digits / len(series) * 100),
        'pct_has_punct': float(has_punct / len(series) * 100),
        'pct_all_upper': float(is_upper / len(series) * 100),
        'pct_all_lower': float(is_lower / len(series) * 100),
        'pct_non_ascii': float(has_non_ascii / len(series) * 100),
    }
    return stats

text_stats = []
for split, fname in [('train', 'train_source1.tsv'), ('train', 'train_source2.tsv'), ('test', 'test_source1.tsv')]:
    fpath = os.path.join('student_resource', 'dataset', split, fname)
    df = pl.read_csv(fpath, separator='\t')
    text_stats.append(profile_text_series(df['business_name'].drop_nulls(), f'{fname}_name'))
    text_stats.append(profile_text_series(df['business_address'].drop_nulls(), f'{fname}_address'))

pl.DataFrame(text_stats).write_csv(os.path.join(REPORTS_DIR, 'text_statistics.csv'))
print("Saved text_statistics.csv")

# ---------------------------------------------------------
# 4. DUPLICATE ANALYSIS (Within Source 1 and Candidates)
# ---------------------------------------------------------
s1_train = pl.read_csv(os.path.join(TRAIN_DIR, 'train_source1.tsv'), separator='\t')
n_s1 = len(s1_train)
n_unique_names = s1_train['business_name'].n_unique()
n_unique_addrs = s1_train['business_address'].n_unique()
exact_dup_name_addr = n_s1 - s1_train.select(['business_name', 'business_address']).n_unique()

print(f"S1 Train Duplicate Analysis:")
print(f"  Total records: {n_s1:,}")
print(f"  Unique names: {n_unique_names:,} ({n_unique_names/n_s1*100:.2f}%)")
print(f"  Unique addresses: {n_unique_addrs:,} ({n_unique_addrs/n_s1*100:.2f}%)")
print(f"  Exact (name, address) duplicate rows in S1: {exact_dup_name_addr}")

# ---------------------------------------------------------
# 5. NOISE & MATCH DIFFICULTY ON TRUE POSITIVES SAMPLE
# ---------------------------------------------------------
# Take 10,000 true match pairs and compare string metrics
print("Sampling 10,000 true match pairs for noise categorization...")
s1_dict = {row[0]: (row[1], row[2], row[3]) for row in s1_train.iter_rows()}
s2_train = pl.read_csv(os.path.join(TRAIN_DIR, 'train_source2.tsv'), separator='\t', n_rows=1000000)
s3_train = pl.read_csv(os.path.join(TRAIN_DIR, 'train_source3.tsv'), separator='\t', n_rows=1000000)
cand_dict = {row[0]: (row[1], row[2], row[3]) for row in s2_train.iter_rows()}
cand_dict.update({row[0]: (row[1], row[2], row[3]) for row in s3_train.iter_rows()})

sample_pairs = []
pair_count = 0
for row in df_gt.iter_rows():
    s1_id = row[0]
    matched = str(row[1]) if row[1] is not None else ""
    if matched:
        for m_id in matched.split(','):
            m_id = m_id.strip()
            if s1_id in s1_dict and m_id in cand_dict:
                sample_pairs.append((s1_id, m_id))
                pair_count += 1
                if pair_count >= 10000:
                    break
    if pair_count >= 10000:
        break

print(f"Collected {len(sample_pairs)} ground truth pairs present in memory sample.")

exact_name_count = 0
exact_addr_count = 0
name_jw_scores = []
name_sort_scores = []
addr_sort_scores = []
noise_examples = []

for s1_id, m_id in sample_pairs:
    s1_name, s1_addr, s1_c = s1_dict[s1_id]
    m_name, m_addr, m_c = cand_dict[m_id]
    
    s1_name = str(s1_name) if s1_name is not None else ""
    m_name = str(m_name) if m_name is not None else ""
    s1_addr = str(s1_addr) if s1_addr is not None else ""
    m_addr = str(m_addr) if m_addr is not None else ""
    
    if s1_name.lower().strip() == m_name.lower().strip():
        exact_name_count += 1
    if s1_addr.lower().strip() == m_addr.lower().strip():
        exact_addr_count += 1
        
    jw = distance.JaroWinkler.similarity(s1_name, m_name)
    nsort = fuzz.token_sort_ratio(s1_name, m_name)
    asort = fuzz.token_sort_ratio(s1_addr, m_addr)
    
    name_jw_scores.append(jw)
    name_sort_scores.append(nsort)
    addr_sort_scores.append(asort)
    
    # Collect specific hard case categories
    if len(noise_examples) < 25:
        if jw < 0.70 and asort > 80:
            noise_examples.append({
                'category': 'Category D: Name Abbreviation/Variant with Matching Address',
                's1_id': s1_id, 'm_id': m_id,
                's1_name': s1_name, 'm_name': m_name,
                's1_addr': s1_addr, 'm_addr': m_addr,
                'name_jw': f"{jw:.3f}", 'addr_sort': asort
            })
        elif jw > 0.90 and asort < 50:
            noise_examples.append({
                'category': 'Category B: Formatting/Missing Address Components',
                's1_id': s1_id, 'm_id': m_id,
                's1_name': s1_name, 'm_name': m_name,
                's1_addr': s1_addr, 'm_addr': m_addr,
                'name_jw': f"{jw:.3f}", 'addr_sort': asort
            })

print(f"Sample True Matches Analysis (N={len(sample_pairs)}):")
print(f"  Exact Name Match (case-folded): {exact_name_count/len(sample_pairs)*100:.2f}%")
print(f"  Exact Address Match (case-folded): {exact_addr_count/len(sample_pairs)*100:.2f}%")
print(f"  Mean Name Jaro-Winkler: {np.mean(name_jw_scores):.3f} (median: {np.median(name_jw_scores):.3f})")
print(f"  Mean Name Token Sort: {np.mean(name_sort_scores):.1f} (median: {np.median(name_sort_scores):.1f})")
print(f"  Mean Address Token Sort: {np.mean(addr_sort_scores):.1f} (median: {np.median(addr_sort_scores):.1f})")

if noise_examples:
    pl.DataFrame(noise_examples).write_csv(os.path.join(REPORTS_DIR, 'noise_examples.csv'))
    print("Saved noise_examples.csv")

print("=== PHASE 1 DATA FORENSICS COMPLETED SUCCESSFULLY ===")
