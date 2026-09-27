"""
Amazon ML Challenge 2026 - Global Graph Clustering & Singleton Evaluation
Stage IV - Global Consistency & Clustering (Phase 7)

This script:
1. Loads the trained LightGBM pairwise model from models/lightgbm_pairwise.txt.
2. Samples 5,000 unseen validation S1 entities and scores their candidate pairs.
3. Evaluates unclustered baseline predictions vs BipartiteGraphClusterer.
4. Sweeps singleton guard thresholds to measure precision boost and singleton accuracy.
5. Verifies megacluster suppression and candidate conflict resolution.
6. Measures Macro F0.5 before vs after global graph clustering.
7. Emits reports/phase_07_clustering_results.json for documentation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl

from src.clustering.graph_clusterer import BipartiteGraphClusterer, ClustererConfig
from src.evaluation.evaluator import EntityResolutionEvaluator
from src.features.feature_extractor import PairwiseFeatureExtractor
from src.models.pairwise_classifier import LightGBMPairwiseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateClustering")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
CANDIDATE_FILE = PROJECT_ROOT / "output" / "blocking" / "candidate_pairs.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_pairwise.txt"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth mapping S1 -> Set of matched candidate IDs."""
    logger.info("Loading ground truth from %s...", gt_path)
    gt: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                s1_id = parts[0]
                cand_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
                gt[s1_id] = set(cand_ids)
    logger.info("Loaded %d ground truth mappings.", len(gt))
    return gt


def sample_validation_entities(
    candidate_path: Path,
    num_entities: int = 5000,
    seed: int = 999,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Sample fresh validation S1 entities and their candidate nominations."""
    logger.info("Sampling %d validation entities from %s (seed=%d)...", num_entities, candidate_path, seed)
    candidates_by_s1: Dict[str, List[str]] = {}

    # Skip first 20,000 lines (used in Phase 6 training) to guarantee unseen test entities!
    skip_count = 20000
    with open(candidate_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        # Skip
        for _ in range(skip_count):
            _ = f.readline()
        # Collect fresh validation sample
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                cand_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
                candidates_by_s1[s1_id] = cand_ids
                if len(candidates_by_s1) >= num_entities:
                    break

    val_s1_list = list(candidates_by_s1.keys())
    logger.info("Sampled %d fresh unseen validation S1 entities.", len(val_s1_list))
    return val_s1_list, candidates_by_s1


def load_metadata_for_entities(
    needed_s1: Set[str],
    needed_cands: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    """Load metadata for needed S1 and candidate IDs."""
    logger.info("Loading metadata for %d S1 and %d candidates...", len(needed_s1), len(needed_cands))
    
    # S1 metadata
    s1_meta: Dict[str, Tuple[str, str]] = {}
    s1_df = (
        pl.scan_csv(DATA_DIR / "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_s1)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s1_df.iter_rows():
        s1_meta[row[0]] = (row[1] or "", row[2] or "")

    # S2 metadata
    cand_meta: Dict[str, Tuple[str, str]] = {}
    s2_df = (
        pl.scan_csv(DATA_DIR / "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_cands)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s2_df.iter_rows():
        cand_meta[row[0]] = (row[1] or "", row[2] or "")

    # S3 metadata
    remaining = needed_cands - set(cand_meta.keys())
    if remaining:
        s3_df = (
            pl.scan_csv(DATA_DIR / "train_source3.tsv", separator="\t")
            .filter(pl.col("entity_id").is_in(list(remaining)))
            .select(["entity_id", "business_name", "business_address"])
            .collect()
        )
        for row in s3_df.iter_rows():
            cand_meta[row[0]] = (row[1] or "", row[2] or "")

    logger.info("Loaded metadata: %d S1, %d Candidates.", len(s1_meta), len(cand_meta))
    return s1_meta, cand_meta


def evaluate_match_dict(
    matches: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    val_s1_universe: Set[str],
) -> Dict[str, float]:
    """Compute Macro F0.5, Precision, Recall, and Singleton Accuracy."""
    beta = 0.5
    beta_sq = beta ** 2
    f05_multiplier = 1.0 + beta_sq

    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    num_entities = len(val_s1_universe)

    singleton_correct = 0
    total_singletons = 0

    for s1 in val_s1_universe:
        true_set = ground_truth.get(s1, set())
        pred_set = set(matches.get(s1, []))

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                total_f05 += 1.0
                total_p += 1.0
                total_r += 1.0
            continue

        if len(pred_set) == 0:
            continue

        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        total_p += p
        total_r += r

        denom = (beta_sq * p) + r
        if denom > 0:
            f05 = (f05_multiplier * p * r) / denom
            total_f05 += f05

    macro_f05 = total_f05 / num_entities if num_entities > 0 else 0.0
    macro_p = total_p / num_entities if num_entities > 0 else 0.0
    macro_r = total_r / num_entities if num_entities > 0 else 0.0
    singleton_acc = singleton_correct / total_singletons if total_singletons > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
    }


def run_clustering_evaluation():
    t_start = time.perf_counter()
    logger.info("Starting Phase 7 Global Graph Clustering & Singleton Evaluation...")

    # Step 1: Load Ground Truth and Model
    gt = load_ground_truth(GT_FILE)
    classifier = LightGBMPairwiseClassifier.load(MODEL_PATH)
    logger.info("Loaded LightGBM model from %s (optimal threshold = %.2f)", MODEL_PATH, classifier.optimal_threshold)

    # Step 2: Sample fresh validation S1 entities
    val_s1_list, candidates_by_s1 = sample_validation_entities(CANDIDATE_FILE, num_entities=5000, seed=999)
    val_s1_universe = set(val_s1_list)

    needed_cands = set()
    for s1 in val_s1_list:
        for c in candidates_by_s1.get(s1, []):
            needed_cands.add(c)

    # Step 3: Load metadata
    s1_meta, cand_meta = load_metadata_for_entities(val_s1_universe, needed_cands)

    # Step 4: Build pairs and extract features
    logger.info("Building pair records for scoring...")
    pairs: List[Dict] = []
    for s1 in val_s1_list:
        cand_list = candidates_by_s1.get(s1, [])
        s1_name, s1_addr = s1_meta.get(s1, ("", ""))
        for rank, cand in enumerate(cand_list, start=1):
            c_name, c_addr = cand_meta.get(cand, ("", ""))
            pairs.append({
                "s1_id": s1,
                "cand_id": cand,
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": c_name,
                "cand_addr": c_addr,
                "rank": float(rank),
                "consensus": 1.0,
            })

    logger.info("Extracting features for %d validation pairs...", len(pairs))
    fv = PairwiseFeatureExtractor.extract_batch(pairs, ground_truth=gt)
    logger.info("Extraction done in %.2f s.", fv.extraction_time)

    # Step 5: Score pairs with LightGBM
    logger.info("Scoring pairs with LightGBM model...")
    scores = classifier.predict_proba(fv.features)
    s1_ids = [p[0] for p in fv.pair_ids]
    cand_ids = [p[1] for p in fv.pair_ids]

    # Baseline: Unclustered threshold matching (p >= 0.60)
    unclustered_matches: Dict[str, List[str]] = {s1: [] for s1 in val_s1_universe}
    for s1, cand, score in zip(s1_ids, cand_ids, scores):
        if score >= classifier.optimal_threshold:
            unclustered_matches[s1].append(cand)

    unclustered_metrics = evaluate_match_dict(unclustered_matches, gt, val_s1_universe)
    logger.info(
        "UNCLUSTERED (p=%.2f) -> Macro F0.5: %.5f | Precision: %.5f | Recall: %.5f | Singleton Acc: %.5f",
        classifier.optimal_threshold,
        unclustered_metrics["macro_f05"],
        unclustered_metrics["macro_precision"],
        unclustered_metrics["macro_recall"],
        unclustered_metrics["singleton_accuracy"],
    )

    # Step 6: Graph Clustering & Singleton Guard Sweeps
    singleton_guards = [0.55, 0.60, 0.62, 0.65, 0.70]
    clustering_experiments = []

    best_cluster_f05 = -1.0
    best_config = None
    best_clustered_matches = None
    best_cluster_eval = None
    best_cluster_metrics = None

    for guard_thresh in singleton_guards:
        cfg = ClustererConfig(
            match_threshold=0.60,
            singleton_guard_threshold=guard_thresh,
            max_matches_per_entity=11,
            enforce_candidate_exclusivity=True,
        )
        clusterer = BipartiteGraphClusterer(config=cfg)
        clustered_matches, cluster_metrics = clusterer.cluster_pairs(
            s1_ids=s1_ids,
            cand_ids=cand_ids,
            scores=scores,
            all_s1_universe=val_s1_universe,
        )
        eval_res = evaluate_match_dict(clustered_matches, gt, val_s1_universe)

        exp_record = {
            "match_threshold": cfg.match_threshold,
            "singleton_guard": guard_thresh,
            "macro_f05": eval_res["macro_f05"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "singleton_accuracy": eval_res["singleton_accuracy"],
            "conflicts_resolved": cluster_metrics.candidate_conflicts_resolved,
            "singleton_rate": cluster_metrics.singleton_rate,
            "mean_matches": cluster_metrics.mean_matches_per_entity,
            "max_matches": cluster_metrics.max_matches_per_entity,
            "cluster_size_distribution": cluster_metrics.cluster_size_distribution,
        }
        clustering_experiments.append(exp_record)

        logger.info(
            "CLUSTERED (guard=%.2f) -> Macro F0.5: %.5f | Precision: %.5f | Recall: %.5f | Singleton Acc: %.5f | Conflicts: %d",
            guard_thresh,
            eval_res["macro_f05"],
            eval_res["macro_precision"],
            eval_res["macro_recall"],
            eval_res["singleton_accuracy"],
            cluster_metrics.candidate_conflicts_resolved,
        )

        if eval_res["macro_f05"] > best_cluster_f05:
            best_cluster_f05 = eval_res["macro_f05"]
            best_config = cfg
            best_clustered_matches = clustered_matches
            best_cluster_eval = eval_res
            best_cluster_metrics = cluster_metrics

    # Step 7: Print Comparison Table
    print("\n" + "=" * 96)
    print(f"{'Configuration':<26} | {'Macro F0.5':<11} | {'Precision':<11} | {'Recall':<11} | {'Singleton Acc':<14} | {'Conflicts':<10}")
    print("-" * 96)
    print(
        f"{'Unclustered (p=0.60)':<26} | {unclustered_metrics['macro_f05']:>10.5f} | {unclustered_metrics['macro_precision']:>10.5f} | {unclustered_metrics['macro_recall']:>10.5f} | {unclustered_metrics['singleton_accuracy']:>13.5f} | {'N/A':>10}"
    )
    for exp in clustering_experiments:
        is_best = " *" if exp["macro_f05"] == best_cluster_f05 else ""
        label = f"Clustered (guard={exp['singleton_guard']:.2f}){is_best}"
        print(
            f"{label:<26} | {exp['macro_f05']:>10.5f} | {exp['macro_precision']:>10.5f} | {exp['macro_recall']:>10.5f} | {exp['singleton_accuracy']:>13.5f} | {exp['conflicts_resolved']:>10}"
        )
    print("=" * 96 + "\n")

    # Step 8: Print Cluster Size Distribution
    print("=" * 60)
    print("CLUSTERED MATCH SET SIZE DISTRIBUTION (BEST CONFIG):")
    print(f"{'Size (Matches)':<18} | {'Entity Count':<16} | {'Percentage':<12}")
    print("-" * 60)
    total_val_entities = len(val_s1_universe)
    for size, cnt in best_cluster_metrics.cluster_size_distribution.items():
        pct = (cnt / total_val_entities) * 100.0
        print(f"{size:<18} | {cnt:<16} | {pct:>10.2f}%")
    print("=" * 60 + "\n")

    # Step 9: Save results JSON
    summary_results = {
        "num_val_entities": len(val_s1_universe),
        "num_val_pairs": len(pairs),
        "unclustered_metrics": unclustered_metrics,
        "best_clustered_metrics": best_cluster_eval,
        "best_cluster_summary": {
            "match_threshold": best_config.match_threshold,
            "singleton_guard": best_config.singleton_guard_threshold,
            "max_matches": best_config.max_matches_per_entity,
            "conflicts_resolved": best_cluster_metrics.candidate_conflicts_resolved,
            "singleton_rate": best_cluster_metrics.singleton_rate,
            "mean_matches": best_cluster_metrics.mean_matches_per_entity,
            "max_matches": best_cluster_metrics.max_matches_per_entity,
            "cluster_size_distribution": best_cluster_metrics.cluster_size_distribution,
        },
        "all_experiments": clustering_experiments,
        "total_runtime_sec": round(time.perf_counter() - t_start, 2),
    }

    out_json = REPORTS_DIR / "phase_07_clustering_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    logger.info("Saved Phase 7 clustering summary to %s", out_json)

    # Step 10: Comparison vs Baseline Floor
    baseline_f05 = 0.303609
    gain_over_baseline = best_cluster_eval["macro_f05"] - baseline_f05
    gain_over_unclustered = best_cluster_eval["macro_f05"] - unclustered_metrics["macro_f05"]
    logger.info("=" * 65)
    logger.info("PHASE 7 CLUSTERED MACRO F0.5:   %.5f", best_cluster_eval["macro_f05"])
    logger.info("PHASE 6 UNCLUSTERED MACRO F0.5: %.5f (%+.5f)", unclustered_metrics["macro_f05"], gain_over_unclustered)
    logger.info("PHASE 3 BASELINE FLOOR:         %.5f (%+.5f / +%.2f%%)", baseline_f05, gain_over_baseline, (gain_over_baseline / baseline_f05) * 100.0)
    logger.info("SINGLETON ACCURACY:             %.5f", best_cluster_eval["singleton_accuracy"])
    logger.info("=" * 65)


if __name__ == "__main__":
    run_clustering_evaluation()
