"""
Amazon ML Challenge 2026 - Ensemble & Blending Engine
Stage IV - Diagnostics & Iterative Improvement (Phase 14)

Provides advanced ensembling strategies combining diverse gradient-boosted families:
1. Calibrated Soft Voting (Simplex-weighted linear probability blending).
2. Percentile Rank Averaging (Borda count rank transformation to neutralize calibration drift).
3. Precision-Guarded Consensus Stacking (High-precision singleton veto + multi-model consensus promotion).
4. Adaptive-Margin Boundary Resolution (Entity-level confidence margin filtering on ensemble scores).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy.stats import rankdata

from src.models.threshold_calibrator import ThresholdCalibrator, ThresholdMetrics

logger = logging.getLogger(__name__)


@dataclass
class EnsembleWeights:
    w_lightgbm: float = 0.40
    w_xgboost: float = 0.45
    w_catboost: float = 0.15

    def to_array(self) -> np.ndarray:
        arr = np.array([self.w_lightgbm, self.w_xgboost, self.w_catboost], dtype=np.float64)
        return arr / np.sum(arr)


class EnsembleBlender:
    """
    Multi-model ensemble blender supporting Soft Voting, Rank Averaging,
    and Precision-Guarded Stacking.
    """

    def __init__(self, weights: Optional[EnsembleWeights] = None):
        self.weights = weights or EnsembleWeights()

    @staticmethod
    def soft_vote(
        probas_dict: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Compute weighted linear combination of model probability arrays.
        """
        if weights is None:
            weights = {k: 1.0 / len(probas_dict) for k in probas_dict}

        total_weight = sum(weights.values())
        norm_weights = {k: w / total_weight for k, w in weights.items()}

        keys = list(probas_dict.keys())
        blended = np.zeros_like(probas_dict[keys[0]], dtype=np.float64)

        for k in keys:
            blended += norm_weights.get(k, 0.0) * probas_dict[k]

        return np.clip(blended, 0.0, 1.0)

    @staticmethod
    def rank_average(
        probas_dict: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Transform predicted probabilities into empirical percentile ranks [0, 1]
        and compute weighted rank average. Neutralizes probability calibration differences.
        """
        if weights is None:
            weights = {k: 1.0 / len(probas_dict) for k in probas_dict}

        total_weight = sum(weights.values())
        norm_weights = {k: w / total_weight for k, w in weights.items()}

        keys = list(probas_dict.keys())
        n = len(probas_dict[keys[0]])
        blended_ranks = np.zeros(n, dtype=np.float64)

        for k in keys:
            raw_p = probas_dict[k]
            # Convert to percentile rank in [0, 1]
            ranks = (rankdata(raw_p, method="average") - 1.0) / max(1, n - 1)
            blended_ranks += norm_weights.get(k, 0.0) * ranks

        return np.clip(blended_ranks, 0.0, 1.0)

    @staticmethod
    def precision_guarded_predict(
        s1_ids: List[str],
        cand_ids: List[str],
        probas_lgb: np.ndarray,
        probas_xgb: np.ndarray,
        probas_cat: np.ndarray,
        all_s1_universe: Set[str],
        base_threshold: float = 0.58,
        singleton_gate_threshold: float = 0.45,
        veto_xgb_threshold: float = 0.25,
        consensus_boost_threshold: float = 0.50,
        margin_delta: float = 0.30,
        max_matches_per_entity: int = 11,
    ) -> Dict[str, List[str]]:
        """
        Hierarchical stacked decision policy:
        1. Pre-group candidate predictions by S1 entity.
        2. XGBoost Singleton Gate: If max(p_xgb) < singleton_gate_threshold, declare entity a singleton.
        3. Consensus & Veto:
           - Weighted soft vote score P_blend = 0.40 * p_lgb + 0.45 * p_xgb + 0.15 * p_cat
           - XGBoost Veto: If p_xgb < veto_xgb_threshold, candidate is discarded (prevents false merges).
           - 2-of-3 Consensus Promotion: If >= 2 models have p >= consensus_boost_threshold, candidate qualified.
        4. Adaptive Confidence Margin: Filter candidates within delta of the top-ranked candidate.
        5. Physical match cardinality cap (<= 11 matches).
        """
        # Group indices by S1
        grouped: Dict[str, List[int]] = {s1: [] for s1 in all_s1_universe}
        for idx, s1 in enumerate(s1_ids):
            if s1 in grouped:
                grouped[s1].append(idx)

        # Weighted blend
        p_blend = 0.40 * probas_lgb + 0.45 * probas_xgb + 0.15 * probas_cat

        final_predictions: Dict[str, List[str]] = {}

        for s1, idx_list in grouped.items():
            if not idx_list:
                final_predictions[s1] = []
                continue

            # Step 1: XGBoost Singleton Protection
            max_xgb_p = max(probas_xgb[i] for i in idx_list)
            if max_xgb_p < singleton_gate_threshold:
                # High-confidence singleton
                final_predictions[s1] = []
                continue

            # Step 2: Candidate qualification with Veto and Consensus
            qualified: List[Tuple[str, float]] = []
            for i in idx_list:
                c = cand_ids[i]
                p_b = float(p_blend[i])
                p_x = float(probas_xgb[i])
                p_l = float(probas_lgb[i])
                p_c = float(probas_cat[i])

                # XGBoost Hard Veto
                if p_x < veto_xgb_threshold:
                    continue

                # Multi-model consensus count (>= 0.50)
                votes = (p_l >= consensus_boost_threshold) + (p_x >= consensus_boost_threshold) + (p_c >= consensus_boost_threshold)

                # Qualification criterion
                if p_b >= base_threshold or votes >= 2:
                    qualified.append((c, p_b))

            if not qualified:
                final_predictions[s1] = []
                continue

            # Sort by blend score descending
            qualified.sort(key=lambda x: x[1], reverse=True)
            top_score = qualified[0][1]

            # Step 3: Adaptive Margin Filter
            selected = [
                c for c, sc in qualified
                if (top_score - sc) <= margin_delta
            ]

            # Step 4: Cardinality Cap
            if len(selected) > max_matches_per_entity:
                selected = selected[:max_matches_per_entity]

            final_predictions[s1] = selected

        return final_predictions

    @staticmethod
    def sweep_blend_weights(
        s1_ids: List[str],
        cand_ids: List[str],
        probas_dict: Dict[str, np.ndarray],
        calibrator: ThresholdCalibrator,
        weight_grid: Optional[List[Dict[str, float]]] = None,
        thresholds: Optional[Sequence[float]] = None,
    ) -> Tuple[Dict[str, float], float, float, List[Dict]]:
        """
        Grid search over simplex blending weights to maximize validation Macro F0.5.
        """
        if weight_grid is None:
            # Focused simplex grid for LightGBM, XGBoost, CatBoost
            weight_grid = [
                {"LightGBM": 0.333, "XGBoost": 0.333, "CatBoost": 0.334},
                {"LightGBM": 0.50, "XGBoost": 0.35, "CatBoost": 0.15},
                {"LightGBM": 0.40, "XGBoost": 0.45, "CatBoost": 0.15},
                {"LightGBM": 0.35, "XGBoost": 0.50, "CatBoost": 0.15},
                {"LightGBM": 0.45, "XGBoost": 0.45, "CatBoost": 0.10},
                {"LightGBM": 0.30, "XGBoost": 0.60, "CatBoost": 0.10},
                {"LightGBM": 0.60, "XGBoost": 0.30, "CatBoost": 0.10},
                {"LightGBM": 0.40, "XGBoost": 0.40, "CatBoost": 0.20},
                {"LightGBM": 0.50, "XGBoost": 0.50, "CatBoost": 0.00},
            ]

        if thresholds is None:
            thresholds = [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]

        best_weights = weight_grid[0]
        best_threshold = 0.58
        best_f05 = -1.0
        sweep_records: List[Dict] = []

        for w in weight_grid:
            blended = EnsembleBlender.soft_vote(probas_dict, weights=w)
            best_metric, _ = calibrator.sweep_global_grid(s1_ids, cand_ids, blended, thresholds)

            record = {
                "weights": w,
                "best_threshold": float(best_metric.threshold),
                "macro_f05": float(best_metric.macro_f05),
                "macro_precision": float(best_metric.macro_precision),
                "macro_recall": float(best_metric.macro_recall),
                "singleton_accuracy": float(best_metric.singleton_accuracy),
            }
            sweep_records.append(record)

            if best_metric.macro_f05 > best_f05:
                best_f05 = best_metric.macro_f05
                best_threshold = float(best_metric.threshold)
                best_weights = w

        return best_weights, best_threshold, best_f05, sweep_records
