"""
Amazon ML Challenge 2026 - Feature Extraction Benchmark & Validation
Stage III - Candidate Scoring & Classification (Phase 5)

This script:
1. Samples 150,000 candidate pairs from output/blocking/candidate_pairs.tsv.
2. Extracts 20-dimensional pairwise feature vectors via PairwiseFeatureExtractor.
3. Measures extraction throughput (pairs/second).
4. Verifies numerical stability (zero NaNs, zero Infs, clean missing address handling).
5. Computes Pearson and Spearman rank correlation with ground-truth labels.
6. Computes distribution shifts between True Positives and Hard Negatives.
7. Outputs summary metrics for reports/PHASE_05_FEATURE_ENGINEERING.md.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
from scipy import stats

from src.features.feature_extractor import FEATURE_NAMES, PairwiseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BenchmarkFeatures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
CANDIDATE_FILE = PROJECT_ROOT / "output" / "blocking" / "candidate_pairs.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth mapping S1 -> Set of matched candidate IDs."""
    logger.info("Loading ground truth from %s...", gt_path)
    gt: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        s1_col = 0
        cand_col = 1
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                s1_id = parts[s1_col]
                cand_ids = [c.strip() for c in parts[cand_col].split(",") if c.strip()]
                gt[s1_id] = set(cand_ids)
    logger.info("Loaded %d ground truth mappings.", len(gt))
    return gt


def sample_candidate_pairs(
    candidate_path: Path,
    num_s1: int = 10000,
) -> Tuple[List[Dict], Set[str], Set[str]]:
    """Sample candidate pairs from candidate TSV file."""
    logger.info("Sampling %d entities from %s...", num_s1, candidate_path)
    sampled_pairs: List[Dict] = []
    needed_s1: Set[str] = set()
    needed_cands: Set[str] = set()

    count = 0
    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                cand_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
                needed_s1.add(s1_id)
                for rank, cand_id in enumerate(cand_ids, start=1):
                    needed_cands.add(cand_id)
                    sampled_pairs.append({
                        "s1_id": s1_id,
                        "cand_id": cand_id,
                        "rank": float(rank),
                        "consensus": 1.0,
                    })
                count += 1
                if count >= num_s1:
                    break

    logger.info(
        "Sampled %d S1 entities -> %d total candidate pairs (%d unique candidates).",
        len(needed_s1),
        len(sampled_pairs),
        len(needed_cands),
    )
    return sampled_pairs, needed_s1, needed_cands


def load_entity_attributes(
    needed_s1: Set[str],
    needed_cands: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    """Load name and address for needed S1 and candidate entities."""
    logger.info("Loading entity metadata for %d S1 and %d candidates...", len(needed_s1), len(needed_cands))
    
    # 1. Load S1 metadata
    s1_meta: Dict[str, Tuple[str, str]] = {}
    s1_df = (
        pl.scan_csv(DATA_DIR / "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_s1)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s1_df.iter_rows():
        s1_meta[row[0]] = (row[1] or "", row[2] or "")

    # 2. Load S2 metadata
    cand_meta: Dict[str, Tuple[str, str]] = {}
    s2_df = (
        pl.scan_csv(DATA_DIR / "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_cands)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s2_df.iter_rows():
        cand_meta[row[0]] = (row[1] or "", row[2] or "")

    # 3. Load S3 metadata
    remaining_cands = needed_cands - set(cand_meta.keys())
    if remaining_cands:
        s3_df = (
            pl.scan_csv(DATA_DIR / "train_source3.tsv", separator="\t")
            .filter(pl.col("entity_id").is_in(list(remaining_cands)))
            .select(["entity_id", "business_name", "business_address"])
            .collect()
        )
        for row in s3_df.iter_rows():
            cand_meta[row[0]] = (row[1] or "", row[2] or "")

    logger.info("Loaded metadata: %d S1, %d Candidates.", len(s1_meta), len(cand_meta))
    return s1_meta, cand_meta


def run_benchmark():
    t_start = time.perf_counter()
    logger.info("Starting Phase 5 Feature Extraction Benchmark...")

    # Step 1: Load Ground Truth
    gt = load_ground_truth(GT_FILE)

    # Step 2: Sample candidate pairs
    sampled_pairs, needed_s1, needed_cands = sample_candidate_pairs(CANDIDATE_FILE, num_s1=5000)

    # Step 3: Load metadata
    s1_meta, cand_meta = load_entity_attributes(needed_s1, needed_cands)

    # Enrich pair dicts with names & addresses
    enriched_pairs: List[Dict] = []
    for pair in sampled_pairs:
        s1_id = pair["s1_id"]
        cand_id = pair["cand_id"]
        s1_name, s1_addr = s1_meta.get(s1_id, ("", ""))
        cand_name, cand_addr = cand_meta.get(cand_id, ("", ""))
        enriched_pairs.append({
            "s1_id": s1_id,
            "cand_id": cand_id,
            "s1_name": s1_name,
            "s1_addr": s1_addr,
            "cand_name": cand_name,
            "cand_addr": cand_addr,
            "rank": pair["rank"],
            "consensus": pair["consensus"],
        })

    # Step 4: Extract features & measure throughput
    num_pairs = len(enriched_pairs)
    logger.info("Extracting 20 features for %d candidate pairs...", num_pairs)
    
    t_extract_start = time.perf_counter()
    feature_vector = PairwiseFeatureExtractor.extract_batch(enriched_pairs, ground_truth=gt)
    extract_time = time.perf_counter() - t_extract_start
    throughput = num_pairs / extract_time

    logger.info("Extraction complete: %.2f seconds (%.0f pairs/sec).", extract_time, throughput)

    # Step 5: Numerical stability check
    nan_count = int(np.isnan(feature_vector.features).sum())
    inf_count = int(np.isinf(feature_vector.features).sum())
    logger.info("Numerical Stability Check: NaN count = %d, Inf count = %d", nan_count, inf_count)

    # Step 6: Labels & Distribution Analysis
    y = feature_vector.labels
    X = feature_vector.features
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_ratio = (n_pos / num_pairs) * 100.0
    logger.info("Pair labels: %d True Positives (%.2f%%), %d Negatives (%.2f%%)", n_pos, pos_ratio, n_neg, 100 - pos_ratio)

    # Step 7: Correlation Analysis
    correlations = []
    for idx, name in enumerate(FEATURE_NAMES):
        feat_col = X[:, idx]
        # Pearson
        try:
            r_p, _ = stats.pearsonr(feat_col, y)
            r_p = float(r_p) if not np.isnan(r_p) else 0.0
        except Exception:
            r_p = 0.0

        # Spearman
        try:
            r_s, _ = stats.spearmanr(feat_col, y)
            r_s = float(r_s) if not np.isnan(r_s) else 0.0
        except Exception:
            r_s = 0.0

        # Means and stds
        pos_mean = float(np.mean(feat_col[y == 1]))
        pos_std = float(np.std(feat_col[y == 1]))
        neg_mean = float(np.mean(feat_col[y == 0]))
        neg_std = float(np.std(feat_col[y == 0]))

        correlations.append({
            "feature": name,
            "pearson_r": r_p,
            "spearman_rho": r_s,
            "pos_mean": pos_mean,
            "pos_std": pos_std,
            "neg_mean": neg_mean,
            "neg_std": neg_std,
        })

    # Sort by absolute Spearman rho descending
    correlations.sort(key=lambda item: abs(item["spearman_rho"]), reverse=True)

    # Print Table (ASCII safe)
    print("\n" + "=" * 92)
    print(f"{'Feature Name':<28} | {'Spearman rho':<12} | {'Pearson r':<11} | {'TP Mean (Std)':<18} | {'Neg Mean (Std)':<18}")
    print("-" * 92)
    for c in correlations:
        tp_str = f"{c['pos_mean']:.3f} ({c['pos_std']:.3f})"
        neg_str = f"{c['neg_mean']:.3f} ({c['neg_std']:.3f})"
        print(f"{c['feature']:<28} | {c['spearman_rho']:>12.4f} | {c['pearson_r']:>11.4f} | {tp_str:<18} | {neg_str:<18}")
    print("=" * 92 + "\n")

    # Save results to JSON
    benchmark_summary = {
        "num_pairs_evaluated": num_pairs,
        "extraction_time_sec": round(extract_time, 3),
        "throughput_pairs_per_sec": round(throughput, 1),
        "num_true_positives": n_pos,
        "num_negatives": n_neg,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "feature_correlations": correlations,
    }

    out_json = PROJECT_ROOT / "reports" / "phase_05_benchmark_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logger.info("Saved benchmark summary to %s", out_json)


if __name__ == "__main__":
    run_benchmark()
