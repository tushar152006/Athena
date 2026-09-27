"""
Amazon ML Challenge 2026 - LightGBM Pairwise Classifier
Stage III - Candidate Scoring & Classification (Phase 6)

This module provides a production-grade LightGBM classifier specifically tuned
for pairwise entity resolution and calibrated for Macro F0.5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)


class LightGBMPairwiseClassifier:
    """
    LightGBM pairwise scoring model for entity matching candidate pairs.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.feature_names = list(FEATURE_NAMES)
        self.optimal_threshold: float = 0.70
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_samples=min_child_samples,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        self.is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ) -> "LightGBMPairwiseClassifier":
        """
        Train the LightGBM classifier with early stopping on validation set.
        """
        eval_set = None
        callbacks = []

        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=50),
            ]

        logger.info(
            "Fitting LightGBM model on %d train pairs (y=1: %d, y=0: %d)...",
            len(y_train),
            int((y_train == 1).sum()),
            int((y_train == 0).sum()),
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=["train", "val"] if eval_set else None,
            eval_metric=["binary_logloss", "auc"],
            callbacks=callbacks,
        )
        self.is_fitted = True

        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            val_pr_auc = average_precision_score(y_val, val_preds)
            logger.info("Validation ROC-AUC: %.4f | PR-AUC: %.4f", val_auc, val_pr_auc)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class probabilities P(is_match=1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self, "booster") and self.booster is not None:
            return np.asarray(self.booster.predict(X), dtype=np.float32)
        elif hasattr(self.model, "booster_") and self.model.booster_ is not None:
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=np.float32)
        elif hasattr(self.model, "_Booster") and self.model._Booster is not None:
            return np.asarray(self.model._Booster.predict(X), dtype=np.float32)
        return np.asarray(self.model.predict_proba(X)[:, 1], dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary match decisions using specified or calibrated threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probas = self.predict_proba(X)
        return (probas >= thresh).astype(np.int8)

    def get_feature_importances(self) -> List[Dict[str, Union[str, float]]]:
        """
        Return feature importances by split count and total gain.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        booster = self.booster if hasattr(self, "booster") and self.booster is not None else self.model.booster_
        split_imp = booster.feature_importance(importance_type="split")
        gain_imp = booster.feature_importance(importance_type="gain")

        importances = []
        for i, name in enumerate(self.feature_names):
            importances.append({
                "feature": name,
                "split": int(split_imp[i]),
                "gain": float(gain_imp[i]),
            })

        # Sort by gain descending
        importances.sort(key=lambda x: x["gain"], reverse=True)
        return importances

    @staticmethod
    def evaluate_threshold_macro_f05(
        s1_ids: Sequence[str],
        cand_ids: Sequence[str],
        probas: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        val_s1_universe: Set[str],
        threshold: float,
    ) -> Dict[str, float]:
        """
        Evaluate full competition Macro F0.5 per S1 entity at a given threshold.
        """
        beta = 0.5
        beta_sq = beta ** 2
        f05_multiplier = 1.0 + beta_sq  # 1.25

        # Group predicted matches above threshold by S1 entity
        pred_matches: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_universe}
        for s1, cand, p in zip(s1_ids, cand_ids, probas):
            if p >= threshold:
                pred_matches[s1].add(cand)

        total_f05 = 0.0
        total_p = 0.0
        total_r = 0.0
        num_entities = len(val_s1_universe)

        singleton_correct = 0
        total_singletons = 0

        for s1 in val_s1_universe:
            true_set = ground_truth.get(s1, set())
            pred_set = pred_matches.get(s1, set())

            # Singleton evaluation
            if len(true_set) == 0:
                total_singletons += 1
                if len(pred_set) == 0:
                    singleton_correct += 1
                    total_f05 += 1.0
                    total_p += 1.0
                    total_r += 1.0
                continue

            # Non-singleton true match
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
            "threshold": threshold,
            "macro_f05": macro_f05,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "singleton_accuracy": singleton_acc,
        }

    def optimize_threshold(
        self,
        s1_ids: Sequence[str],
        cand_ids: Sequence[str],
        probas: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        val_s1_universe: Set[str],
        thresholds: Sequence[float] = (
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        ),
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Sweep threshold values to find optimal threshold maximizing Macro F0.5.
        """
        logger.info("Sweeping %d classification thresholds for Macro F0.5...", len(thresholds))
        sweep_results = []
        best_thresh = 0.70
        best_f05 = -1.0

        for thresh in thresholds:
            res = self.evaluate_threshold_macro_f05(
                s1_ids=s1_ids,
                cand_ids=cand_ids,
                probas=probas,
                ground_truth=ground_truth,
                val_s1_universe=val_s1_universe,
                threshold=thresh,
            )
            sweep_results.append(res)
            logger.info(
                "Threshold %.2f -> Macro F0.5: %.5f | Precision: %.5f | Recall: %.5f",
                thresh,
                res["macro_f05"],
                res["macro_precision"],
                res["macro_recall"],
            )
            if res["macro_f05"] > best_f05:
                best_f05 = res["macro_f05"]
                best_thresh = thresh

        self.optimal_threshold = best_thresh
        logger.info("Optimal threshold selected: %.2f (Macro F0.5 = %.5f)", best_thresh, best_f05)
        return best_thresh, sweep_results

    def save(self, filepath: Union[str, Path]) -> None:
        """Save model booster and threshold to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        # Save metadata sidecar
        meta_path = path.with_suffix(".meta.json")
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "optimal_threshold": self.optimal_threshold,
                "feature_names": self.feature_names,
            }, f, indent=2)
        logger.info("Saved model to %s (and metadata to %s)", path, meta_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LightGBMPairwiseClassifier":
        """Load model booster and metadata from disk."""
        path = Path(filepath)
        booster = lgb.Booster(model_file=str(path))
        classifier = cls()
        classifier.booster = booster
        classifier.is_fitted = True

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                classifier.optimal_threshold = meta.get("optimal_threshold", 0.70)
                classifier.feature_names = meta.get("feature_names", list(FEATURE_NAMES))
        logger.info("Loaded model from %s (optimal threshold = %.2f)", path, classifier.optimal_threshold)
        return classifier
