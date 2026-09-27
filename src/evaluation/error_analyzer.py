"""
Amazon ML Challenge 2026 - Comprehensive Error Diagnostics & Autopsy Engine
Stage IV - Diagnostics & Iterative Improvement (Phase 11)

This module conducts a granular diagnostic census and root-cause attribution:
1. False Positive (FP) categorization: Franchise look-alikes, multi-tenant co-locations, spelling collisions.
2. False Negative (FN) categorization: Model scoring misses vs. blocking dropouts.
3. Singleton error dichotomy: False singletons vs. over-merged singletons.
4. Feature distribution attribution for false positives and false negatives.
5. Actionable roadmap and potential recovery bounds for Phase 12 (Retrieval / Embeddings).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.features.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class ErrorCensus:
    """Comprehensive quantification of errors across validation universe."""
    total_entities: int = 0
    total_gt_matches: int = 0
    total_pred_matches: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    fn_scoring_miss: int = 0
    fn_blocking_dropout: int = 0
    gt_singletons: int = 0
    correct_singletons: int = 0
    false_singletons: int = 0
    overmerged_singletons: int = 0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f05: float = 0.0


@dataclass
class ErrorCase:
    """Detailed metadata and feature vector for an error instance."""
    s1_id: str
    cand_id: str
    s1_name: str
    s1_addr: str
    cand_name: str
    cand_addr: str
    score: float
    error_type: str  # 'FP_FRANCHISE', 'FP_MULTITENANT', 'FP_OTHER', 'FN_SCORING', 'FN_BLOCKING'
    features: Optional[Dict[str, float]] = None


class ErrorAnalyzer:
    """
    Performs forensic autopsy of entity resolution predictions against ground truth.
    """

    def __init__(
        self,
        ground_truth: Dict[str, Set[str]],
        s1_universe: Set[str],
        s1_meta: Dict[str, Tuple[str, str]],
        cand_meta: Dict[str, Tuple[str, str]],
    ):
        self.gt = ground_truth
        self.universe = s1_universe
        self.s1_meta = s1_meta
        self.cand_meta = cand_meta

    def analyze(
        self,
        candidates_by_s1: Dict[str, List[str]],
        predicted_matches: Dict[str, List[str]],
        scores_by_pair: Dict[Tuple[str, str], float],
        features_by_pair: Optional[Dict[Tuple[str, str], Sequence[float]]] = None,
    ) -> Tuple[ErrorCensus, List[ErrorCase], List[ErrorCase]]:
        """
        Run complete error audit and extract representative diagnostic cases.
        """
        census = ErrorCensus(total_entities=len(self.universe))
        fp_cases: List[ErrorCase] = []
        fn_cases: List[ErrorCase] = []

        total_prec = 0.0
        total_rec = 0.0
        total_f05 = 0.0
        beta_sq = 0.25

        for s1 in self.universe:
            true_set = self.gt.get(s1, set())
            pred_list = predicted_matches.get(s1, [])
            pred_set = set(pred_list)
            cand_list = candidates_by_s1.get(s1, [])
            cand_set = set(cand_list)

            k_true = len(true_set)
            k_pred = len(pred_set)

            census.total_gt_matches += k_true
            census.total_pred_matches += k_pred

            # Singleton accounting
            if k_true == 0:
                census.gt_singletons += 1
                if k_pred == 0:
                    census.correct_singletons += 1
                    total_prec += 1.0
                    total_rec += 1.0
                    total_f05 += 1.0
                else:
                    census.overmerged_singletons += 1
                    # Over-merged singleton: every pred is FP
                    for cid in pred_set:
                        census.total_fp += 1
                        fp_cases.append(self._make_error_case(s1, cid, scores_by_pair, features_by_pair, "FP_OVERMERGED"))
                continue

            if k_pred == 0:
                census.false_singletons += 1
                # False singleton: all true matches were missed
                for cid in true_set:
                    census.total_fn += 1
                    if cid in cand_set:
                        census.fn_scoring_miss += 1
                        fn_cases.append(self._make_error_case(s1, cid, scores_by_pair, features_by_pair, "FN_SCORING"))
                    else:
                        census.fn_blocking_dropout += 1
                        fn_cases.append(self._make_error_case(s1, cid, scores_by_pair, features_by_pair, "FN_BLOCKING"))
                continue

            # Non-singleton entity evaluation
            tp_set = pred_set & true_set
            fp_set = pred_set - true_set
            fn_set = true_set - pred_set

            census.total_tp += len(tp_set)
            census.total_fp += len(fp_set)
            census.total_fn += len(fn_set)

            prec = len(tp_set) / k_pred
            rec = len(tp_set) / k_true
            total_prec += prec
            total_rec += rec

            denom = (beta_sq * prec) + rec
            f05 = ((1.0 + beta_sq) * prec * rec / denom) if denom > 0 else 0.0
            total_f05 += f05

            # Categorize FPs
            for cid in fp_set:
                fp_case = self._classify_fp(s1, cid, scores_by_pair, features_by_pair)
                fp_cases.append(fp_case)

            # Categorize FNs
            for cid in fn_set:
                if cid in cand_set:
                    census.fn_scoring_miss += 1
                    fn_case = self._make_error_case(s1, cid, scores_by_pair, features_by_pair, "FN_SCORING")
                else:
                    census.fn_blocking_dropout += 1
                    fn_case = self._make_error_case(s1, cid, scores_by_pair, features_by_pair, "FN_BLOCKING")
                fn_cases.append(fn_case)

        n = float(census.total_entities)
        census.macro_precision = total_prec / n
        census.macro_recall = total_rec / n
        census.macro_f05 = total_f05 / n

        logger.info(
            "Audit complete: TP=%d, FP=%d, FN=%d (Scoring=%d, Dropout=%d). Singletons: Correct=%d, False=%d, OverMerged=%d",
            census.total_tp,
            census.total_fp,
            census.total_fn,
            census.fn_scoring_miss,
            census.fn_blocking_dropout,
            census.correct_singletons,
            census.false_singletons,
            census.overmerged_singletons,
        )

        return census, fp_cases, fn_cases

    def _make_error_case(
        self,
        s1: str,
        cand: str,
        scores_by_pair: Dict[Tuple[str, str], float],
        features_by_pair: Optional[Dict[Tuple[str, str], Sequence[float]]],
        error_type: str,
    ) -> ErrorCase:
        s1_n, s1_a = self.s1_meta.get(s1, ("", ""))
        c_n, c_a = self.cand_meta.get(cand, ("", ""))
        score = scores_by_pair.get((s1, cand), 0.0)

        feats_dict = None
        if features_by_pair and (s1, cand) in features_by_pair:
            raw_f = features_by_pair[(s1, cand)]
            feats_dict = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_f)}

        return ErrorCase(
            s1_id=s1,
            cand_id=cand,
            s1_name=s1_n,
            s1_addr=s1_a,
            cand_name=c_n,
            cand_addr=c_a,
            score=score,
            error_type=error_type,
            features=feats_dict,
        )

    def _classify_fp(
        self,
        s1: str,
        cand: str,
        scores_by_pair: Dict[Tuple[str, str], float],
        features_by_pair: Optional[Dict[Tuple[str, str], Sequence[float]]],
    ) -> ErrorCase:
        case = self._make_error_case(s1, cand, scores_by_pair, features_by_pair, "FP_OTHER")
        if case.features:
            f = case.features
            if f.get("franchise_collision_hazard", 0.0) == 1.0 or (f.get("name_token_sort_ratio", 0.0) >= 0.85 and f.get("addr_num_match", 0.0) == -1.0):
                case.error_type = "FP_FRANCHISE"
            elif f.get("multi_tenant_hazard", 0.0) == 1.0 or (f.get("addr_token_sort_ratio", 0.0) >= 0.85 and f.get("name_token_sort_ratio", 0.0) <= 0.50):
                case.error_type = "FP_MULTITENANT"
            elif f.get("name_token_set_ratio", 0.0) >= 0.90:
                case.error_type = "FP_BRAND_SUBSET"
            else:
                case.error_type = "FP_SPELLING_COLLISION"
        return case
