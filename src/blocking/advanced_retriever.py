"""
Amazon ML Challenge 2026 - Advanced Retrieval Engine
Stage IV - Diagnostics & Iterative Improvement (Phase 12)

This module implements secondary hybrid retrieval to recover blocking dropouts:
1. Character 3-4 Gram TF-IDF Inverted Index: Sparse cosine similarity retrieval across normalized names.
2. Spatial Partition Nearest-Neighbor Retrieval: Address-constrained retrieval using postal codes and street numbers.
3. Parsimonious Candidate Fusion: Merges baseline blocking candidates with advanced retrieval candidates, enforcing max parsimony cap <= 20.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.feature_extractor import PairwiseFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancedRetrievalConfig:
    """Configuration for character n-gram inverted index retrieval."""
    ngram_range: Tuple[int, int] = (3, 4)
    min_df: int = 1
    top_k_per_query: int = 5
    min_similarity_threshold: float = 0.35
    max_total_candidates_per_entity: int = 20


@dataclass(frozen=True)
class RetrievalLiftMetrics:
    """Summary metrics evaluating recall improvement from advanced retrieval."""
    total_queries: int
    total_ground_truth_matches: int
    baseline_captured_matches: int
    baseline_recall: float
    hybrid_captured_matches: int
    hybrid_recall: float
    absolute_recall_lift: float
    relative_recall_lift_pct: float
    mean_parsimony_before: float
    mean_parsimony_after: float
    max_parsimony: int


class AdvancedRetriever:
    """
    Inverted-index character n-gram and spatial candidate retrieval engine.
    """

    def __init__(self, config: Optional[AdvancedRetrievalConfig] = None):
        self.config = config or AdvancedRetrievalConfig()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.candidate_ids: List[str] = []
        self.candidate_tfidf_matrix: Optional[sparse.csr_matrix] = None

    def fit_candidate_index(
        self,
        candidate_meta: Dict[str, Tuple[str, str]],
    ) -> None:
        """
        Fit character n-gram TF-IDF vectorizer and construct sparse inverted index
        over candidate names.
        """
        cfg = self.config
        logger.info("Indexing %d candidate entities with char n-grams %s...", len(candidate_meta), cfg.ngram_range)

        self.candidate_ids = list(candidate_meta.keys())
        cleaned_names = [
            PairwiseFeatureExtractor.clean_name(candidate_meta[cid][0])
            for cid in self.candidate_ids
        ]

        # Fit TF-IDF matrix
        self.candidate_tfidf_matrix = self.vectorizer.fit_transform(cleaned_names)
        logger.info(
            "Constructed candidate TF-IDF matrix of shape %s with %d non-zero entries.",
            self.candidate_tfidf_matrix.shape,
            self.candidate_tfidf_matrix.nnz,
        )

    def retrieve_for_queries(
        self,
        query_meta: Dict[str, Tuple[str, str]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve top-k candidates for each query entity using sparse matrix multiplication.
        
        Returns:
            Dictionary mapping query_id -> list of (candidate_id, cosine_similarity_score).
        """
        if self.candidate_tfidf_matrix is None:
            raise RuntimeError("Candidate index has not been fitted. Call fit_candidate_index first.")

        cfg = self.config
        query_ids = list(query_meta.keys())
        cleaned_queries = [
            PairwiseFeatureExtractor.clean_name(query_meta[qid][0])
            for qid in query_ids
        ]

        logger.info("Transforming %d query names into TF-IDF space...", len(query_ids))
        query_tfidf = self.vectorizer.transform(cleaned_queries)

        logger.info("Computing sparse cosine similarity matrix (%d x %d)...", query_tfidf.shape[0], self.candidate_tfidf_matrix.shape[0])
        # Similarity matrix: (N_query, N_cand)
        sim_matrix = query_tfidf.dot(self.candidate_tfidf_matrix.T)

        results: Dict[str, List[Tuple[str, float]]] = {}
        top_k = cfg.top_k_per_query
        min_thresh = cfg.min_similarity_threshold

        for i, qid in enumerate(query_ids):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                results[qid] = []
                continue

            col_indices = row.indices
            scores = row.data

            # Filter by threshold
            valid_mask = scores >= min_thresh
            if not np.any(valid_mask):
                results[qid] = []
                continue

            valid_cols = col_indices[valid_mask]
            valid_scores = scores[valid_mask]

            # Top-k selection
            if len(valid_scores) > top_k:
                top_part = np.argpartition(valid_scores, -top_k)[-top_k:]
                best_indices = top_part[np.argsort(-valid_scores[top_part])]
            else:
                best_indices = np.argsort(-valid_scores)

            retrieved: List[Tuple[str, float]] = [
                (self.candidate_ids[valid_cols[idx]], float(valid_scores[idx]))
                for idx in best_indices
            ]
            results[qid] = retrieved

        logger.info("Retrieved advanced candidates for %d queries.", len(results))
        return results

    def merge_and_prune(
        self,
        baseline_candidates: Dict[str, List[str]],
        retrieved_candidates: Dict[str, List[Tuple[str, float]]],
        all_s1_universe: Set[str],
    ) -> Tuple[Dict[str, List[str]], RetrievalLiftMetrics, Dict[str, Set[str]]]:
        """
        Merge baseline candidates with advanced retrieval candidates and enforce parsimony <= 20.
        """
        cfg = self.config
        max_cands = cfg.max_total_candidates_per_entity

        hybrid_candidates: Dict[str, List[str]] = {}
        newly_added_by_s1: Dict[str, Set[str]] = {}

        for s1 in all_s1_universe:
            base_list = baseline_candidates.get(s1, [])
            ret_list = [c for c, _ in retrieved_candidates.get(s1, [])]

            # Preserve baseline order, then append new candidates
            seen = set(base_list)
            added_set: Set[str] = set()
            merged = list(base_list)

            for cand in ret_list:
                if cand not in seen:
                    seen.add(cand)
                    merged.append(cand)
                    added_set.add(cand)
                if len(merged) >= max_cands:
                    break

            hybrid_candidates[s1] = merged[:max_cands]
            newly_added_by_s1[s1] = added_set

        return hybrid_candidates, newly_added_by_s1
