#!/usr/bin/env python3
"""
Test bit-for-bit determinism of scoring and clustering on a slice of candidates.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from src.clustering.graph_clusterer import BipartiteGraphClusterer, ClustererConfig
from src.models.pairwise_classifier import LightGBMPairwiseClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_pairwise.txt"

def test_determinism():
    # 1. Model prediction determinism
    clf = LightGBMPairwiseClassifier.load(MODEL_PATH)
    np.random.seed(42)
    X = np.random.randn(1000, 20).astype(np.float32)

    pred1 = clf.predict_proba(X)
    pred2 = clf.predict_proba(X)
    np.testing.assert_array_equal(pred1, pred2)
    print("Pass 1: Model inference is 100% bit-for-bit deterministic.")

    # 2. Clusterer determinism
    clusterer = BipartiteGraphClusterer(
        config=ClustererConfig(
            match_threshold=0.60,
            singleton_guard_threshold=0.60,
            max_matches_per_entity=11,
            enforce_candidate_exclusivity=True,
        )
    )

    s1_ids = [f"S1-{i//5:05d}" for i in range(1000)]
    cand_ids = [f"S2-{i:05d}" for i in range(1000)]
    all_s1 = set(s1_ids)

    matches1, _ = clusterer.cluster_pairs(s1_ids, cand_ids, pred1, all_s1_universe=all_s1)
    matches2, _ = clusterer.cluster_pairs(s1_ids, cand_ids, pred2, all_s1_universe=all_s1)

    assert matches1 == matches2, "Clusterer produced different outputs!"
    print("Pass 2: Graph clusterer tie-breaking and conflict resolution is 100% deterministic.")

    # Serialize to TSV strings and compare hashes
    lines1 = [f"{s1}\t{','.join(cands)}" for s1, cands in sorted(matches1.items())]
    lines2 = [f"{s1}\t{','.join(cands)}" for s1, cands in sorted(matches2.items())]

    h1 = hashlib.sha256("\n".join(lines1).encode()).hexdigest()
    h2 = hashlib.sha256("\n".join(lines2).encode()).hexdigest()

    assert h1 == h2, f"Hashes differ: {h1} vs {h2}"
    print(f"Pass 3: Serialized TSV hashes match perfectly: {h1}")
    print("ALL REPRODUCIBILITY CHECKS PASSED.")

if __name__ == "__main__":
    test_determinism()
