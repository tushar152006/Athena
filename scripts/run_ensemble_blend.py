"""
Amazon ML Challenge 2026 - Phase 14 Ensemble & Blending Evaluation Runner
Stage IV - Diagnostics & Iterative Improvement

Trains the 3 champion gradient boosted architectures (LightGBM, XGBoost, CatBoost)
and systematically evaluates multiple ensembling and stacking policies on unseen validation entities:
1. Single Best Model Baselines (LightGBM, XGBoost, CatBoost)
2. Equal-Weight Soft Voting (1/3, 1/3, 1/3)
3. Simplex-Optimized Calibrated Soft Voting (w_lgb, w_xgb, w_cat)
4. Percentile Rank Averaging (Borda Count Score)
5. Precision-Guarded Stacking (XGBoost Singleton Veto + 2-of-3 Consensus + Adaptive Margin)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from src.features.feature_extractor import FEATURE_NAMES, PairwiseFeatureExtractor
from src.models.ensemble_blender import EnsembleBlender, EnsembleWeights
from src.models.threshold_calibrator import ThresholdCalibrator, ThresholdMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EnsembleRunner")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset" / "train"
CANDIDATE_FILE = BASE_DIR / "output" / "blocking" / "candidate_pairs.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
OUTPUT_DIR = BASE_DIR / "models" / "ensemble"
REPORTS_DIR = BASE_DIR / "reports"


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
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
            elif len(parts) == 1 and parts[0]:
                gt[parts[0]] = set()
    logger.info("Loaded ground truth for %d S1 entities.", len(gt))
    return gt


def sample_train_val_entities(
    candidate_path: Path,
    num_entities: int = 20000,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    logger.info("Sampling %d entities from %s (seed=%d)...", num_entities, candidate_path, seed)
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
    logger.info("Loading metadata for %d S1 and %d candidates...", len(needed_s1), len(needed_cands))
    s1_meta: Dict[str, Tuple[str, str]] = {}
    s1_df = (
        pl.scan_csv(DATA_DIR / "train_source1.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_s1)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s1_df.iter_rows():
        s1_meta[row[0]] = (row[1] or "", row[2] or "")

    cand_meta: Dict[str, Tuple[str, str]] = {}
    s2_df = (
        pl.scan_csv(DATA_DIR / "train_source2.tsv", separator="\t")
        .filter(pl.col("entity_id").is_in(list(needed_cands)))
        .select(["entity_id", "business_name", "business_address"])
        .collect()
    )
    for row in s2_df.iter_rows():
        cand_meta[row[0]] = (row[1] or "", row[2] or "")

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


def main():
    t_start = time.perf_counter()
    logger.info("=" * 80)
    logger.info("PHASE 14 - ENSEMBLING, BLENDING & PRECISION-GUARDED STACKING")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    gt = load_ground_truth(GT_FILE)
    train_s1, val_s1, candidates_by_s1 = sample_train_val_entities(
        CANDIDATE_FILE,
        num_entities=20000,
        train_ratio=0.80,
        seed=42,
    )

    needed_s1 = set(train_s1) | set(val_s1)
    needed_cands: Set[str] = set()
    for s1 in needed_s1:
        for c in candidates_by_s1.get(s1, []):
            needed_cands.add(c)

    s1_meta, cand_meta = load_metadata_for_entities(needed_s1, needed_cands)

    train_pairs = build_pair_records(train_s1, candidates_by_s1, s1_meta, cand_meta)
    val_pairs = build_pair_records(val_s1, candidates_by_s1, s1_meta, cand_meta)
    logger.info("Train pairs: %d | Val pairs: %d", len(train_pairs), len(val_pairs))

    # 2. Extract Features
    logger.info("Extracting features for train pairs...")
    train_fv = PairwiseFeatureExtractor.extract_batch(train_pairs, ground_truth=gt)
    logger.info("Extracting features for val pairs...")
    val_fv = PairwiseFeatureExtractor.extract_batch(val_pairs, ground_truth=gt)

    X_train, y_train = train_fv.features, train_fv.labels
    X_val, y_val = val_fv.features, val_fv.labels

    val_s1_ids = [p[0] for p in val_fv.pair_ids]
    val_cand_ids = [p[1] for p in val_fv.pair_ids]
    val_universe = set(val_s1)
    calibrator = ThresholdCalibrator(ground_truth=gt, all_s1_universe=val_universe)

    # 3. Train Models
    logger.info("Training LightGBM...")
    t0 = time.perf_counter()
    clf_lgb = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf_lgb.fit(X_train, y_train)
    p_lgb = clf_lgb.predict_proba(X_val)[:, 1]
    logger.info("LightGBM fit in %.2fs. PR-AUC: %.5f", time.perf_counter() - t0, average_precision_score(y_val, p_lgb))

    logger.info("Training XGBoost...")
    t0 = time.perf_counter()
    clf_xgb = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf_xgb.fit(X_train, y_train)
    p_xgb = clf_xgb.predict_proba(X_val)[:, 1]
    logger.info("XGBoost fit in %.2fs. PR-AUC: %.5f", time.perf_counter() - t0, average_precision_score(y_val, p_xgb))

    logger.info("Training CatBoost...")
    t0 = time.perf_counter()
    clf_cat = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        eval_metric="Logloss",
        random_seed=42,
        thread_count=-1,
        verbose=0,
    )
    clf_cat.fit(X_train, y_train)
    p_cat = clf_cat.predict_proba(X_val)[:, 1]
    logger.info("CatBoost fit in %.2fs. PR-AUC: %.5f", time.perf_counter() - t0, average_precision_score(y_val, p_cat))

    probas_dict = {
        "LightGBM": p_lgb,
        "XGBoost": p_xgb,
        "CatBoost": p_cat,
    }

    threshold_grid = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]

    # Benchmark Individual Models
    benchmarks: Dict[str, dict] = {}
    for name, p in probas_dict.items():
        best_metric, _ = calibrator.sweep_global_grid(val_s1_ids, val_cand_ids, p, threshold_grid)
        benchmarks[name] = {
            "strategy": f"Single: {name}",
            "pr_auc": round(float(average_precision_score(y_val, p)), 5),
            "roc_auc": round(float(roc_auc_score(y_val, p)), 5),
            "opt_threshold": float(best_metric.threshold),
            "macro_f05": round(float(best_metric.macro_f05), 5),
            "macro_precision": round(float(best_metric.macro_precision), 5),
            "macro_recall": round(float(best_metric.macro_recall), 5),
            "singleton_accuracy": round(float(best_metric.singleton_accuracy), 5),
        }

    # Strategy 1: Equal-Weight Soft Voting (1/3, 1/3, 1/3)
    p_equal = EnsembleBlender.soft_vote(probas_dict)
    best_eq, _ = calibrator.sweep_global_grid(val_s1_ids, val_cand_ids, p_equal, threshold_grid)
    benchmarks["Equal_Soft_Vote"] = {
        "strategy": "Ensemble: Equal Soft Voting (33/33/34)",
        "pr_auc": round(float(average_precision_score(y_val, p_equal)), 5),
        "roc_auc": round(float(roc_auc_score(y_val, p_equal)), 5),
        "opt_threshold": float(best_eq.threshold),
        "macro_f05": round(float(best_eq.macro_f05), 5),
        "macro_precision": round(float(best_eq.macro_precision), 5),
        "macro_recall": round(float(best_eq.macro_recall), 5),
        "singleton_accuracy": round(float(best_eq.singleton_accuracy), 5),
    }

    # Strategy 2: Simplex-Optimized Calibrated Soft Voting
    best_w, best_opt_t, best_opt_f05, sweep_records = EnsembleBlender.sweep_blend_weights(
        val_s1_ids, val_cand_ids, probas_dict, calibrator, thresholds=threshold_grid
    )
    p_opt = EnsembleBlender.soft_vote(probas_dict, weights=best_w)
    best_opt_metric, _ = calibrator.sweep_global_grid(val_s1_ids, val_cand_ids, p_opt, threshold_grid)
    benchmarks["Optimal_Soft_Vote"] = {
        "strategy": f"Ensemble: Optimal Soft Voting ({int(best_w['LightGBM']*100)}/{int(best_w['XGBoost']*100)}/{int(best_w['CatBoost']*100)})",
        "weights": best_w,
        "pr_auc": round(float(average_precision_score(y_val, p_opt)), 5),
        "roc_auc": round(float(roc_auc_score(y_val, p_opt)), 5),
        "opt_threshold": float(best_opt_metric.threshold),
        "macro_f05": round(float(best_opt_metric.macro_f05), 5),
        "macro_precision": round(float(best_opt_metric.macro_precision), 5),
        "macro_recall": round(float(best_opt_metric.macro_recall), 5),
        "singleton_accuracy": round(float(best_opt_metric.singleton_accuracy), 5),
    }

    # Strategy 3: Percentile Rank Averaging
    p_rank = EnsembleBlender.rank_average(probas_dict, weights=best_w)
    rank_threshold_grid = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98]
    best_rank, _ = calibrator.sweep_global_grid(val_s1_ids, val_cand_ids, p_rank, rank_threshold_grid)
    benchmarks["Rank_Averaging"] = {
        "strategy": "Ensemble: Percentile Rank Averaging (Borda Count)",
        "pr_auc": round(float(average_precision_score(y_val, p_rank)), 5),
        "roc_auc": round(float(roc_auc_score(y_val, p_rank)), 5),
        "opt_threshold": float(best_rank.threshold),
        "macro_f05": round(float(best_rank.macro_f05), 5),
        "macro_precision": round(float(best_rank.macro_precision), 5),
        "macro_recall": round(float(best_rank.macro_recall), 5),
        "singleton_accuracy": round(float(best_rank.singleton_accuracy), 5),
    }

    # Strategy 4: Precision-Guarded Stacking (XGBoost Singleton Veto + 2-of-3 Consensus + Adaptive Margin)
    stack_preds = EnsembleBlender.precision_guarded_predict(
        s1_ids=val_s1_ids,
        cand_ids=val_cand_ids,
        probas_lgb=p_lgb,
        probas_xgb=p_xgb,
        probas_cat=p_cat,
        all_s1_universe=val_universe,
        base_threshold=0.58,
        singleton_gate_threshold=0.45,
        veto_xgb_threshold=0.25,
        consensus_boost_threshold=0.50,
        margin_delta=0.30,
        max_matches_per_entity=11,
    )
    stack_eval = calibrator.evaluate_predictions(stack_preds)
    benchmarks["Precision_Guarded_Stacking"] = {
        "strategy": "Stacking: XGBoost Gate + 2-of-3 Consensus + Adaptive Margin",
        "pr_auc": round(float(average_precision_score(y_val, p_opt)), 5),
        "roc_auc": round(float(roc_auc_score(y_val, p_opt)), 5),
        "opt_threshold": 0.58,
        "macro_f05": round(float(stack_eval["macro_f05"]), 5),
        "macro_precision": round(float(stack_eval["macro_precision"]), 5),
        "macro_recall": round(float(stack_eval["macro_recall"]), 5),
        "singleton_accuracy": round(float(stack_eval["singleton_accuracy"]), 5),
    }

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 115)
    print(f"{'Ensemble / Blending Strategy':<45} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Opt Thresh':<10} | {'Macro F0.5':<10} | {'Prec':<7} | {'Rec':<7} | {'Singl Acc':<9}")
    print("-" * 115)
    for k, b in benchmarks.items():
        print(f"{b['strategy']:<45} | {b['pr_auc']:>8.5f} | {b['roc_auc']:>8.5f} | {b['opt_threshold']:>10.2f} | {b['macro_f05']:>10.5f} | {b['macro_precision']*100:>6.2f}% | {b['macro_recall']*100:>6.2f}% | {b['singleton_accuracy']*100:>8.2f}%")
    print("=" * 115 + "\n")

    # Print Simplex Weight Search Table
    print("\n" + "=" * 85)
    print(f"{'LightGBM':<10} | {'XGBoost':<10} | {'CatBoost':<10} | {'Opt Thresh':<12} | {'Macro F0.5':<12} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 85)
    for rec in sweep_records:
        w = rec["weights"]
        is_best = " *" if rec["macro_f05"] == best_opt_f05 else ""
        print(f"{w['LightGBM']:<10.2f} | {w['XGBoost']:<10.2f} | {w['CatBoost']:<10.2f} | {rec['best_threshold']:>12.2f} | {rec['macro_f05']:>12.5f}{is_best:<2} | {rec['macro_precision']*100:>7.2f}% | {rec['macro_recall']*100:>7.2f}%")
    print("=" * 85 + "\n")

    # 5. Persist JSON artifact
    artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_entities": len(val_universe),
        "validation_pairs": len(val_pairs),
        "optimal_weights": best_w,
        "optimal_threshold": best_opt_t,
        "benchmarks": benchmarks,
        "weight_sweep": sweep_records,
    }
    with open(OUTPUT_DIR / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    logger.info("Saved ensemble configuration to %s", OUTPUT_DIR / "ensemble_config.json")

    total_time = time.perf_counter() - t_start
    logger.info("Phase 14 completed in %.2f s (%.2f min).", total_time, total_time / 60)


if __name__ == "__main__":
    main()
