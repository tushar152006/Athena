"""
Amazon ML Challenge 2026 - Precision-Biased Threshold Optimization & Calibration Engine
Stage III - Feature Engineering & Core Model (Phase 10)

This module provides high-performance calibration and threshold optimization for Macro F0.5:
1. Fine-grained global grid sweep (step 0.01 across [0.40, 0.95]).
2. Adaptive confidence-margin calibration (top-1 gap filtering: p1 - pi <= delta).
3. Candidate density / parsimony trade-off analysis.
4. Comparative model selection between Phase 6 baseline and Phase 9 hard-negative models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdMetrics:
    """Evaluation metrics for a specific decision threshold configuration."""
    threshold: float
    macro_f05: float
    macro_precision: float
    macro_recall: float
    singleton_accuracy: float
    total_matches: int
    mean_matches_per_entity: float
    singletons_declared: int


@dataclass(frozen=True)
class AdaptiveMarginMetrics:
    """Evaluation metrics for adaptive margin-gap thresholding."""
    base_threshold: float
    margin_delta: float
    macro_f05: float
    macro_precision: float
    macro_recall: float
    singleton_accuracy: float
    total_matches: int
    mean_matches_per_entity: float


class ThresholdCalibrator:
    """
    Optimizes decision boundaries for Macro F0.5 under asymmetric precision weighting.
    """

    BETA: float = 0.5
    BETA_SQ: float = 0.25

    def __init__(self, ground_truth: Dict[str, Set[str]], all_s1_universe: Set[str]):
        self.ground_truth = ground_truth
        self.all_s1_universe = all_s1_universe
        self.n_total_entities = len(all_s1_universe)

    def evaluate_predictions(
        self,
        predicted_matches: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Compute official Macro F0.5 and auxiliary metrics on predicted match sets.
        """
        total_p = 0.0
        total_r = 0.0
        total_f05 = 0.0
        correct_singletons = 0
        total_gt_singletons = 0
        total_matches = 0
        singletons_declared = 0

        gt = self.ground_truth
        beta_sq = self.BETA_SQ

        for s1_id in self.all_s1_universe:
            true_set = gt.get(s1_id, set())
            pred_list = predicted_matches.get(s1_id, [])
            pred_set = set(pred_list)
            k_pred = len(pred_set)
            k_true = len(true_set)
            total_matches += k_pred

            if k_pred == 0:
                singletons_declared += 1

            if k_true == 0:
                total_gt_singletons += 1
                if k_pred == 0:
                    correct_singletons += 1
                    total_p += 1.0
                    total_r += 1.0
                    total_f05 += 1.0
                else:
                    total_p += 0.0
                    total_r += 0.0
                    total_f05 += 0.0
                continue

            if k_pred == 0:
                total_p += 0.0
                total_r += 0.0
                total_f05 += 0.0
                continue

            tp = len(pred_set & true_set)
            prec = tp / k_pred
            rec = tp / k_true

            total_p += prec
            total_r += rec

            denom = (beta_sq * prec) + rec
            if denom > 0:
                f05 = (1.0 + beta_sq) * (prec * rec) / denom
            else:
                f05 = 0.0
            total_f05 += f05

        n = float(self.n_total_entities)
        return {
            "macro_f05": total_f05 / n,
            "macro_precision": total_p / n,
            "macro_recall": total_r / n,
            "singleton_accuracy": correct_singletons / max(1, total_gt_singletons),
            "total_matches": total_matches,
            "mean_matches_per_entity": total_matches / n,
            "singletons_declared": singletons_declared,
        }

    def sweep_global_grid(
        self,
        s1_ids: Sequence[str],
        cand_ids: Sequence[str],
        probas: np.ndarray,
        thresholds: Optional[Sequence[float]] = None,
    ) -> Tuple[ThresholdMetrics, List[ThresholdMetrics]]:
        """
        Execute fine-grained grid search across candidate thresholds.
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.linspace(0.40, 0.90, 51)]

        logger.info("Sweeping %d global thresholds from %.2f to %.2f...", len(thresholds), thresholds[0], thresholds[-1])

        # Pre-group candidates by S1 entity
        candidates_by_s1: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in self.all_s1_universe}
        for s1, cand, p in zip(s1_ids, cand_ids, probas):
            if s1 in candidates_by_s1:
                candidates_by_s1[s1].append((cand, float(p)))

        results: List[ThresholdMetrics] = []
        best_metric: Optional[ThresholdMetrics] = None

        for t in thresholds:
            pred_matches: Dict[str, List[str]] = {}
            for s1, cands in candidates_by_s1.items():
                matched = [c for c, p in cands if p >= t]
                # Enforce physical max cap 11
                if len(matched) > 11:
                    matched = matched[:11]
                pred_matches[s1] = matched

            eval_res = self.evaluate_predictions(pred_matches)
            metric = ThresholdMetrics(
                threshold=t,
                macro_f05=eval_res["macro_f05"],
                macro_precision=eval_res["macro_precision"],
                macro_recall=eval_res["macro_recall"],
                singleton_accuracy=eval_res["singleton_accuracy"],
                total_matches=int(eval_res["total_matches"]),
                mean_matches_per_entity=eval_res["mean_matches_per_entity"],
                singletons_declared=int(eval_res["singletons_declared"]),
            )
            results.append(metric)

            if best_metric is None or metric.macro_f05 > best_metric.macro_f05:
                best_metric = metric

        logger.info(
            "Optimal Global Threshold: p* = %.2f -> Macro F0.5 = %.6f (Prec: %.4f, Rec: %.4f, Singletons: %.4f)",
            best_metric.threshold,
            best_metric.macro_f05,
            best_metric.macro_precision,
            best_metric.macro_recall,
            best_metric.singleton_accuracy,
        )
        return best_metric, results

    def sweep_adaptive_margin(
        self,
        s1_ids: Sequence[str],
        cand_ids: Sequence[str],
        probas: np.ndarray,
        base_thresholds: Sequence[float] = (0.50, 0.55, 0.60, 0.65),
        margin_deltas: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 1.0),
    ) -> Tuple[AdaptiveMarginMetrics, List[AdaptiveMarginMetrics]]:
        """
        Evaluate confidence-gap adaptive thresholding:
        An S1 entity accepts top candidate C1 if P(C1) >= base_threshold.
        Subsequent candidates Ci (i >= 2) are accepted iff:
            P(Ci) >= base_threshold AND (P(C1) - P(Ci)) <= delta
        """
        logger.info("Sweeping adaptive margin thresholds (base=%s, deltas=%s)...", base_thresholds, margin_deltas)

        # Pre-group and sort candidates per S1 by probability descending
        candidates_by_s1: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in self.all_s1_universe}
        for s1, cand, p in zip(s1_ids, cand_ids, probas):
            if s1 in candidates_by_s1:
                candidates_by_s1[s1].append((cand, float(p)))

        for s1 in candidates_by_s1:
            candidates_by_s1[s1].sort(key=lambda x: x[1], reverse=True)

        results: List[AdaptiveMarginMetrics] = []
        best_metric: Optional[AdaptiveMarginMetrics] = None

        for base_t in base_thresholds:
            for delta in margin_deltas:
                pred_matches: Dict[str, List[str]] = {}
                for s1, cands in candidates_by_s1.items():
                    if not cands or cands[0][1] < base_t:
                        pred_matches[s1] = []
                        continue

                    top_p = cands[0][1]
                    accepted: List[str] = [cands[0][0]]

                    for c, p in cands[1:11]:  # max cap 11
                        if p >= base_t and (top_p - p) <= delta:
                            accepted.append(c)

                    pred_matches[s1] = accepted

                eval_res = self.evaluate_predictions(pred_matches)
                metric = AdaptiveMarginMetrics(
                    base_threshold=base_t,
                    margin_delta=delta,
                    macro_f05=eval_res["macro_f05"],
                    macro_precision=eval_res["macro_precision"],
                    macro_recall=eval_res["macro_recall"],
                    singleton_accuracy=eval_res["singleton_accuracy"],
                    total_matches=int(eval_res["total_matches"]),
                    mean_matches_per_entity=eval_res["mean_matches_per_entity"],
                )
                results.append(metric)

                if best_metric is None or metric.macro_f05 > best_metric.macro_f05:
                    best_metric = metric

        logger.info(
            "Optimal Adaptive Margin: Base=%.2f, Delta=%.2f -> Macro F0.5 = %.6f (Prec: %.4f, Rec: %.4f)",
            best_metric.base_threshold,
            best_metric.margin_delta,
            best_metric.macro_f05,
            best_metric.macro_precision,
            best_metric.macro_recall,
        )
        return best_metric, results
