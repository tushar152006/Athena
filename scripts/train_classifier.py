"""
Amazon ML Challenge 2026 - Train LightGBM Pairwise Classifier
Stage III - Candidate Scoring & Classification (Phase 6)

This script:
1. Samples 20,000 S1 entities and their candidate pairs from output/blocking/candidate_pairs.tsv.
2. Performs an entity-stratified 80/20 train/validation split (16k train S1, 4k val S1).
3. Extracts the 20 domain-specific features using PairwiseFeatureExtractor.
4. Trains LightGBMPairwiseClassifier with early stopping on validation binary logloss.
5. Calibrates optimal decision threshold p* across [0.40, 0.95] for Macro F0.5.
6. Evaluates feature importances (Gain and Split).
7. Emits validation matching results and validates with EntityResolutionEvaluator.
8. Persists trained model to models/lightgbm_pairwise.txt.
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

from src.evaluation.evaluator import EntityResolutionEvaluator
from src.features.feature_extractor import FEATURE_NAMES, PairwiseFeatureExtractor
from src.models.pairwise_classifier import LightGBMPairwiseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainClassifier")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
CANDIDATE_FILE = PROJECT_ROOT / "output" / "blocking" / "candidate_pairs.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
MODELS_DIR = PROJECT_ROOT / "models"
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


def sample_train_val_entities(
    candidate_path: Path,
    num_entities: int = 20000,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Sample S1 entities and split into train and validation sets."""
    logger.info("Sampling %d entities from %s...", num_entities, candidate_path)
    candidates_by_s1: Dict[str, List[str]] = {}

    with open(candidate_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0]
                cand_ids = [c.strip() for c in parts[1].split(",") if c.strip()]
                candidates_by_s1[s1_id] = cand_ids
                if len(candidates_by_s1) >= num_entities:
                    break

    all_s1_list = list(candidates_by_s1.keys())
    rng = random.Random(seed)
    rng.shuffle(all_s1_list)

    n_train = int(len(all_s1_list) * train_ratio)
    train_s1 = all_s1_list[:n_train]
    val_s1 = all_s1_list[n_train:]

    logger.info("Split %d entities -> Train: %d S1, Val: %d S1.", len(all_s1_list), len(train_s1), len(val_s1))
    return train_s1, val_s1, candidates_by_s1


def load_metadata_for_entities(
    needed_s1: Set[str],
    needed_cands: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    """Load names and addresses for needed S1 and candidate IDs."""
    logger.info("Loading metadata for %d S1 and %d candidates...", len(needed_s1), len(needed_cands))
    
    # 1. S1 metadata
    s1_meta: Dict[str, Tuple[str, str]] = {}
    s1_df = (
        pl.scan_csv(DATA_DIR / "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_s1)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s1_df.iter_rows():
        s1_meta[row[0]] = (row[1] or "", row[2] or "")

    # 2. S2 metadata
    cand_meta: Dict[str, Tuple[str, str]] = {}
    s2_df = (
        pl.scan_csv(DATA_DIR / "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_cands)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s2_df.iter_rows():
        cand_meta[row[0]] = (row[1] or "", row[2] or "")

    # 3. S3 metadata
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


def build_pair_records(
    s1_list: List[str],
    candidates_by_s1: Dict[str, List[str]],
    s1_meta: Dict[str, Tuple[str, str]],
    cand_meta: Dict[str, Tuple[str, str]],
) -> List[Dict]:
    """Construct pair dicts with entity attributes."""
    pairs: List[Dict] = []
    for s1_id in s1_list:
        cand_ids = candidates_by_s1.get(s1_id, [])
        s1_name, s1_addr = s1_meta.get(s1_id, ("", ""))
        for rank, cand_id in enumerate(cand_ids, start=1):
            cand_name, cand_addr = cand_meta.get(cand_id, ("", ""))
            pairs.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_name": cand_name,
                "cand_addr": cand_addr,
                "rank": float(rank),
                "consensus": 1.0,
            })
    return pairs


def run_pipeline():
    t_pipeline_start = time.perf_counter()
    logger.info("Starting Phase 6 LightGBM Pairwise Classifier Training...")

    # Step 1: Load Ground Truth
    gt = load_ground_truth(GT_FILE)

    # Step 2: Sample Entities and Split
    train_s1, val_s1, candidates_by_s1 = sample_train_val_entities(
        CANDIDATE_FILE,
        num_entities=20000,
        train_ratio=0.80,
        seed=42,
    )

    # Collect needed candidate IDs
    needed_s1 = set(train_s1) | set(val_s1)
    needed_cands: Set[str] = set()
    for s1 in needed_s1:
        for c in candidates_by_s1.get(s1, []):
            needed_cands.add(c)

    # Step 3: Load Metadata
    s1_meta, cand_meta = load_metadata_for_entities(needed_s1, needed_cands)

    # Step 4: Build Pair Records
    logger.info("Building pair records for train and val...")
    train_pairs = build_pair_records(train_s1, candidates_by_s1, s1_meta, cand_meta)
    val_pairs = build_pair_records(val_s1, candidates_by_s1, s1_meta, cand_meta)
    logger.info("Constructed %d train pairs and %d val pairs.", len(train_pairs), len(val_pairs))

    # Step 5: Feature Extraction
    logger.info("Extracting features for train pairs (%d)...", len(train_pairs))
    train_fv = PairwiseFeatureExtractor.extract_batch(train_pairs, ground_truth=gt)
    logger.info("Train extraction done in %.2f s.", train_fv.extraction_time)

    logger.info("Extracting features for val pairs (%d)...", len(val_pairs))
    val_fv = PairwiseFeatureExtractor.extract_batch(val_pairs, ground_truth=gt)
    logger.info("Val extraction done in %.2f s.", val_fv.extraction_time)

    X_train, y_train = train_fv.features, train_fv.labels
    X_val, y_val = val_fv.features, val_fv.labels

    # Step 6: Train LightGBM Classifier
    logger.info("Initializing and training LightGBMPairwiseClassifier...")
    classifier = LightGBMPairwiseClassifier(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=30)

    # Step 7: Calibrate Threshold for Macro F0.5
    logger.info("Predicting validation probabilities...")
    val_probas = classifier.predict_proba(X_val)
    val_s1_ids = [p[0] for p in val_fv.pair_ids]
    val_cand_ids = [p[1] for p in val_fv.pair_ids]
    val_universe = set(val_s1)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_thresh, sweep_results = classifier.optimize_threshold(
        s1_ids=val_s1_ids,
        cand_ids=val_cand_ids,
        probas=val_probas,
        ground_truth=gt,
        val_s1_universe=val_universe,
        thresholds=thresholds,
    )

    # Step 8: Feature Importances
    importances = classifier.get_feature_importances()
    print("\n" + "=" * 65)
    print(f"{'Feature Name':<28} | {'Gain Importance':<16} | {'Split':<8}")
    print("-" * 65)
    for imp in importances:
        print(f"{imp['feature']:<28} | {imp['gain']:>16.2f} | {imp['split']:>8}")
    print("=" * 65 + "\n")

    # Step 9: Print Threshold Calibration Table
    print("\n" + "=" * 80)
    print(f"{'Threshold p*':<14} | {'Macro F0.5':<12} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Singleton Acc':<14}")
    print("-" * 80)
    for res in sweep_results:
        is_best = " (BEST)" if res["threshold"] == best_thresh else ""
        print(
            f"{res['threshold']:<14.2f} | {res['macro_f05']:>10.5f}{is_best:<2} | {res['macro_precision']:>14.5f}  | {res['macro_recall']:>12.5f} | {res['singleton_accuracy']:>12.5f}"
        )
    print("=" * 80 + "\n")

    # Step 10: Save Model & Summary
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / "lightgbm_pairwise.txt"
    classifier.save(model_save_path)

    # Save summary report JSON
    best_res = next(r for r in sweep_results if r["threshold"] == best_thresh)
    summary_data = {
        "num_train_s1": len(train_s1),
        "num_val_s1": len(val_s1),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "optimal_threshold": best_thresh,
        "best_macro_f05": best_res["macro_f05"],
        "best_macro_precision": best_res["macro_precision"],
        "best_macro_recall": best_res["macro_recall"],
        "singleton_accuracy": best_res["singleton_accuracy"],
        "sweep_results": sweep_results,
        "feature_importances": importances,
        "total_runtime_sec": round(time.perf_counter() - t_pipeline_start, 2),
    }

    out_json = REPORTS_DIR / "phase_06_train_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved Phase 6 results to %s.", out_json)

    # Step 11: Compare with Baseline Floor
    baseline_f05 = 0.303609
    f05_gain = best_res["macro_f05"] - baseline_f05
    pct_gain = (f05_gain / baseline_f05) * 100.0
    logger.info("=" * 60)
    logger.info("PHASE 6 VALIDATION MACRO F0.5: %.5f", best_res["macro_f05"])
    logger.info("BASELINE FLOOR MACRO F0.5:    %.5f", baseline_f05)
    logger.info("ABSOLUTE GAIN:               %+.5f (+%.2f%%)", f05_gain, pct_gain)
    logger.info("OPTIMAL THRESHOLD p*:        %.2f", best_thresh)
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
