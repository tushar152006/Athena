"""
Inspect true pairs and test candidate generation blocking keys
"""

import re
import polars as pl

# Load sample ground truth
gt = pl.read_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=20000)
# filter to non-empty
gt = gt.filter(pl.col("matched_entity_ids") != "")
s1_sample_ids = set(gt["source1_entity_id"])

# explode to (s1_id, matched_id)
gt_pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
print(f"Total test pairs in sample: {len(gt_pairs):,}")

# Load S1 matching sample
s1 = pl.read_csv("student_resource/dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s1_sample_ids)))
s2_ids = set(gt_pairs.filter(pl.col("matched_entity_ids").str.starts_with("S2-"))["matched_entity_ids"])
s3_ids = set(gt_pairs.filter(pl.col("matched_entity_ids").str.starts_with("S3-"))["matched_entity_ids"])

s2 = pl.read_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s2_ids)))
s3 = pl.read_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(s3_ids)))
cand = pl.concat([s2, s3])

# Join S1 and Cand on true pairs
joined = gt_pairs.join(s1, left_on="source1_entity_id", right_on="entity_id", suffix="_s1").join(
    cand, left_on="matched_entity_ids", right_on="entity_id", suffix="_cand"
)
print(f"Loaded {len(joined):,} ground-truth joined pairs")

# Now let's test different blocking keys
legal_suffixes = r"\b(inc|incorporated|llc|corp|corporation|co|company|ltd|limited|pvt|private|llp|sa|sas|sarl|eurl|sci|snc)\b"

def clean_name(name):
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(legal_suffixes, "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def extract_postal(addr):
    if not addr:
        return ""
    m = re.findall(r"\b\d{5,6}\b", addr)
    return m[-1] if m else ""

def get_first_token(name):
    c = clean_name(name)
    parts = c.split()
    return parts[0] if parts else ""

def get_name_prefix(name, k=4):
    c = clean_name(name)
    return c[:k] if len(c) >= k else c

def get_sorted_tokens(name, k=2):
    c = clean_name(name)
    parts = sorted(c.split()[:k])
    return "_".join(parts)

# Let's test on the joined pairs:
n = len(joined)
hits_pass1 = 0 # clean name exact
hits_pass2 = 0 # postal + first token
hits_pass3 = 0 # postal + name prefix
hits_pass4 = 0 # sorted 2 tokens
hits_union = 0

for row in joined.iter_rows(named=True):
    n1 = clean_name(row["business_name"])
    n2 = clean_name(row["business_name_cand"])
    
    p1 = extract_postal(row["business_address"])
    p2 = extract_postal(row["business_address_cand"])
    
    ft1 = get_first_token(row["business_name"])
    ft2 = get_first_token(row["business_name_cand"])
    
    pref1 = get_name_prefix(row["business_name"], 4)
    pref2 = get_name_prefix(row["business_name_cand"], 4)
    
    st1 = get_sorted_tokens(row["business_name"], 2)
    st2 = get_sorted_tokens(row["business_name_cand"], 2)
    
    m1 = (n1 == n2 and len(n1) > 1)
    m2 = (p1 == p2 and p1 != "" and ft1 == ft2 and ft1 != "")
    m3 = (p1 == p2 and p1 != "" and pref1 == pref2 and pref1 != "")
    m4 = (st1 == st2 and st1 != "")
    
    if m1: hits_pass1 += 1
    if m2: hits_pass2 += 1
    if m3: hits_pass3 += 1
    if m4: hits_pass4 += 1
    if m1 or m2 or m3 or m4: hits_union += 1

print(f"Sample Size: {n}")
print(f"Pass 1 (Canonical Name Root): {hits_pass1/n*100:.2f}%")
print(f"Pass 2 (Postal + First Token): {hits_pass2/n*100:.2f}%")
print(f"Pass 3 (Postal + Name Prefix 4): {hits_pass3/n*100:.2f}%")
print(f"Pass 4 (Sorted Top-2 Tokens): {hits_pass4/n*100:.2f}%")
print(f"Disjunctive Union (Recall): {hits_union/n*100:.2f}%")
