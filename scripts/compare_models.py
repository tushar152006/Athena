"""
Amazon ML Challenge 2026 - Phase 13 Model Architecture Comparison
Stage IV - Diagnostics & Iterative Improvement

Compares four distinct model architectures under strictly identical feature sets
and entity-stratified train/val splits:
1. Baseline: Logistic Regression (L2 regularized, standard-scaled linear model)
2. Champion: LightGBM (Gradient-based one-side sampling, leaf-wise tree growth)
3. Contender: XGBoost (Histogram-based exact greedy depth-wise tree growth)
4. Contender: CatBoost (Symmetric oblivious decision trees)

Evaluates:
- Macro F0.5 at optimal threshold p* and default p=0.50
- Macro Precision, Macro Recall, Singleton Accuracy
- Pairwise PR-AUC, ROC-AUC, Binary Logloss
- Training Wall Clock Time and Inference Latency (pairs/sec)
- Inter-Model Probability Correlation Matrix (Ensemble Feasibility)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import catboost as cb
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluator import EntityResolutionEvaluator
from src.features.feature_extractor import FEATURE_NAMES, PairwiseFeatureExtractor
from src.models.threshold_calibrator import ThresholdCalibrator, ThresholdMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModelComparison")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "student_resource" / "dataset" / "train"
CANDIDATE_FILE = BASE_DIR / "output" / "blocking" / "candidate_pairs.tsv"
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
OUTPUT_DIR = BASE_DIR / "models" / "comparison"
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


def optimize_threshold_for_probas(
    s1_ids: List[str],
    cand_ids: List[str],
    probas: np.ndarray,
    calibrator: ThresholdCalibrator,
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float, Dict[str, float], List[Dict]]:
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]
    best_metric, sweep_list = calibrator.sweep_global_grid(s1_ids, cand_ids, probas, thresholds)
    sweep_dicts = [asdict(m) for m in sweep_list]
    return best_metric.threshold, best_metric.macro_f05, asdict(best_metric), sweep_dicts


def evaluate_at_threshold(
    s1_ids: List[str],
    cand_ids: List[str],
    probas: np.ndarray,
    calibrator: ThresholdCalibrator,
    threshold: float = 0.50,
) -> Dict[str, float]:
    _, sweep_list = calibrator.sweep_global_grid(s1_ids, cand_ids, probas, [threshold])
    return asdict(sweep_list[0])


def main():
    t_start = time.perf_counter()
    logger.info("=" * 80)
    logger.info("PHASE 13 - COMPREHENSIVE MODEL COMPARISON BENCHMARK")
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

    logger.info("Feature matrix: Train shape %s, Val shape %s. Positive rate: Train %.2f%%, Val %.2f%%",
                X_train.shape, X_val.shape, y_train.mean() * 100, y_val.mean() * 100)

    # Prepare standard scaler for linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Model Definitions
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(C=1.0, max_iter=500, random_state=42, solver="lbfgs"),
            "scaled": True,
            "description": "L2 Regularized Linear Model with Feature Standardization",
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(
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
            ),
            "scaled": False,
            "description": "GOSS Leaf-wise Gradient Boosted Decision Trees",
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(
                n_estimators=350,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "scaled": False,
            "description": "Histogram Exact Greedy Depth-wise Boosted Trees",
        },
        "CatBoost": {
            "model": cb.CatBoostClassifier(
                iterations=350,
                learning_rate=0.05,
                depth=6,
                eval_metric="Logloss",
                random_seed=42,
                thread_count=-1,
                verbose=0,
            ),
            "scaled": False,
            "description": "Symmetric Oblivious Decision Trees (Ordered Boosting)",
        },
    }

    results: Dict[str, dict] = {}
    probas_dict: Dict[str, np.ndarray] = {}

    for name, cfg in models.items():
        logger.info("-" * 60)
        logger.info("Training Model: %s (%s)", name, cfg["description"])
        clf = cfg["model"]
        x_tr = X_train_scaled if cfg["scaled"] else X_train
        x_vl = X_val_scaled if cfg["scaled"] else X_val

        # Measure Fit Time
        t_fit_start = time.perf_counter()
        clf.fit(x_tr, y_train)
        fit_time = time.perf_counter() - t_fit_start
        logger.info("[%s] Fit completed in %.2f s", name, fit_time)

        # Measure Inference Time
        t_inf_start = time.perf_counter()
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_vl)[:, 1]
        else:
            probs = clf.predict(x_vl)
        inf_time = time.perf_counter() - t_inf_start
        latency_us = (inf_time / len(x_vl)) * 1_000_000
        throughput = len(x_vl) / max(0.001, inf_time)
        probas_dict[name] = probs

        # Pairwise metrics
        roc_auc = float(roc_auc_score(y_val, probs))
        pr_auc = float(average_precision_score(y_val, probs))
        b_loss = float(log_loss(y_val, probs))

        # Default threshold p=0.50
        def_metrics = evaluate_at_threshold(
            val_s1_ids, val_cand_ids, probs, calibrator, threshold=0.50
        )

        # Threshold sweep for optimal p*
        best_p, best_f05, best_metrics, sweep = optimize_threshold_for_probas(
            val_s1_ids, val_cand_ids, probs, calibrator
        )

        results[name] = {
            "model_name": name,
            "description": cfg["description"],
            "fit_time_seconds": round(fit_time, 2),
            "inf_time_seconds": round(inf_time, 4),
            "inf_latency_us_per_pair": round(latency_us, 2),
            "inf_throughput_pairs_sec": round(throughput, 1),
            "roc_auc": round(roc_auc, 5),
            "pr_auc": round(pr_auc, 5),
            "logloss": round(b_loss, 5),
            "default_p050": {
                "macro_f05": round(def_metrics["macro_f05"], 5),
                "macro_precision": round(def_metrics["macro_precision"], 5),
                "macro_recall": round(def_metrics["macro_recall"], 5),
                "singleton_accuracy": round(def_metrics["singleton_accuracy"], 5),
            },
            "optimal_p": {
                "threshold": round(best_p, 2),
                "macro_f05": round(best_f05, 5),
                "macro_precision": round(best_metrics["macro_precision"], 5),
                "macro_recall": round(best_metrics["macro_recall"], 5),
                "singleton_accuracy": round(best_metrics["singleton_accuracy"], 5),
            },
            "sweep": sweep,
        }

        logger.info(
            "[%s] PR-AUC: %.5f | ROC-AUC: %.5f | Optimal p*=%.2f: Macro F0.5=%.5f (P=%.4f, R=%.4f) | Fit: %.2fs",
            name, pr_auc, roc_auc, best_p, best_f05,
            best_metrics["macro_precision"], best_metrics["macro_recall"], fit_time
        )

    # 4. Inter-Model Probability Correlation Analysis
    corr_matrix = {}
    model_names = list(models.keys())
    for m1 in model_names:
        corr_matrix[m1] = {}
        for m2 in model_names:
            corr = float(np.corrcoef(probas_dict[m1], probas_dict[m2])[0, 1])
            corr_matrix[m1][m2] = round(corr, 4)

    # Simple Equal-Weight Ensemble Evaluation (LightGBM + CatBoost + XGBoost)
    ensemble_probs = (
        probas_dict["LightGBM"] + probas_dict["CatBoost"] + probas_dict["XGBoost"]
    ) / 3.0
    ens_roc_auc = float(roc_auc_score(y_val, ensemble_probs))
    ens_pr_auc = float(average_precision_score(y_val, ensemble_probs))
    best_ens_p, best_ens_f05, best_ens_metrics, ens_sweep = optimize_threshold_for_probas(
        val_s1_ids, val_cand_ids, ensemble_probs, calibrator
    )
    ens_p50 = evaluate_at_threshold(
        val_s1_ids, val_cand_ids, ensemble_probs, calibrator, threshold=0.50
    )

    ensemble_result = {
        "model_name": "Ensemble (LGBM + CatBoost + XGBoost)",
        "description": "Equal-weight Soft Voting of 3 Gradient Boosted Tree Families",
        "roc_auc": round(ens_roc_auc, 5),
        "pr_auc": round(ens_pr_auc, 5),
        "default_p050": {
            "macro_f05": round(ens_p50["macro_f05"], 5),
            "macro_precision": round(ens_p50["macro_precision"], 5),
            "macro_recall": round(ens_p50["macro_recall"], 5),
            "singleton_accuracy": round(ens_p50["singleton_accuracy"], 5),
        },
        "optimal_p": {
            "threshold": round(best_ens_p, 2),
            "macro_f05": round(best_ens_f05, 5),
            "macro_precision": round(best_ens_metrics["macro_precision"], 5),
            "macro_recall": round(best_ens_metrics["macro_recall"], 5),
            "singleton_accuracy": round(best_ens_metrics["singleton_accuracy"], 5),
        },
    }

    # 5. Print Comparison Table
    print("\n" + "=" * 115)
    print(f"{'Model Architecture':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Opt p*':<7} | {'Opt F0.5':<9} | {'Opt Prec':<9} | {'Opt Rec':<8} | {'Fit (s)':<7} | {'Latency':<11}")
    print("-" * 115)
    for name, r in results.items():
        opt = r["optimal_p"]
        lat = f"{r['inf_latency_us_per_pair']:.1f} µs/p"
        print(f"{name:<22} | {r['pr_auc']:>8.5f} | {r['roc_auc']:>8.5f} | {opt['threshold']:>7.2f} | {opt['macro_f05']:>9.5f} | {opt['macro_precision']:>9.4f} | {opt['macro_recall']:>8.4f} | {r['fit_time_seconds']:>7.2f} | {lat:>11}")
    
    ens_opt = ensemble_result["optimal_p"]
    print("-" * 115)
    print(f"{'Ensemble (3-Way)':<22} | {ensemble_result['pr_auc']:>8.5f} | {ensemble_result['roc_auc']:>8.5f} | {ens_opt['threshold']:>7.2f} | {ens_opt['macro_f05']:>9.5f} | {ens_opt['macro_precision']:>9.4f} | {ens_opt['macro_recall']:>8.4f} | {'N/A':>7} | {'Ensemble':>11}")
    print("=" * 115 + "\n")

    # Print Probability Correlation Matrix
    print("\n" + "=" * 70)
    print("INTER-MODEL PREDICTION PROBABILITY CORRELATION MATRIX (Pearson r):")
    print("-" * 70)
    header = f"{'Model':<22} | " + " | ".join(f"{m[:10]:<10}" for m in model_names)
    print(header)
    print("-" * 70)
    for m1 in model_names:
        row = f"{m1:<22} | " + " | ".join(f"{corr_matrix[m1][m2]:>10.4f}" for m2 in model_names)
        print(row)
    print("=" * 70 + "\n")

    # 6. Save JSON artifact
    comparison_artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_universe": len(val_s1),
        "validation_candidate_pairs": len(val_pairs),
        "models": results,
        "ensemble_preview": ensemble_result,
        "correlation_matrix": corr_matrix,
    }
    with open(OUTPUT_DIR / "model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_artifact, f, indent=2)
    logger.info("Saved comparison results to %s", OUTPUT_DIR / "model_comparison_results.json")

    total_time = time.perf_counter() - t_start
    logger.info("Phase 13 model comparison finished in %.2f s (%.2f min).", total_time, total_time / 60)


if __name__ == "__main__":
    main()
