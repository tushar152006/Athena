"""
Amazon ML Challenge 2026 - Global Graph Clustering & Singleton Resolution
Stage IV - Global Consistency & Clustering (Phase 7)

This module implements global consistency enforcement on pairwise model predictions:
1. Bipartite edge filtering with calibrated decision threshold.
2. Candidate exclusivity / maximum-weight bipartite conflict resolution (preventing megacluster bridge collapse).
3. Precision-preserving Singleton Guard (protecting the 5.58% true singleton population).
4. Physical cardinality caps (max 11 matches per S1 entity based on Phase 1 data forensics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClustererConfig:
    """Configuration for global bipartite graph clustering."""
    match_threshold: float = 0.60
    singleton_guard_threshold: float = 0.60
    max_matches_per_entity: int = 11
    enforce_candidate_exclusivity: bool = True


@dataclass(frozen=True)
class ClusteringMetrics:
    """Summary statistics for clustering and post-processing."""
    total_entities: int
    singleton_count: int
    singleton_rate: float
    total_matches: int
    mean_matches_per_entity: float
    max_matches_per_entity: int
    candidate_conflicts_resolved: int
    cluster_size_distribution: Dict[int, int]


class BipartiteGraphClusterer:
    """
    Global graph clusterer enforcing 1-to-many bipartite consistency,
    conflict resolution, and singleton protection.
    """

    def __init__(self, config: Optional[ClustererConfig] = None):
        self.config = config or ClustererConfig()

    def cluster_pairs(
        self,
        s1_ids: Sequence[str],
        cand_ids: Sequence[str],
        scores: Sequence[float],
        all_s1_universe: Set[str],
    ) -> Tuple[Dict[str, List[str]], ClusteringMetrics]:
        """
        Cluster scored candidate pairs into global entity resolution match sets.
        
        Args:
            s1_ids: Sequence of Source 1 entity IDs.
            cand_ids: Sequence of candidate entity IDs.
            scores: Model prediction probabilities P(match=1).
            all_s1_universe: Set of all Source 1 entities that must be represented in output.
            
        Returns:
            Dictionary mapping s1_id -> list of matched candidate IDs, and ClusteringMetrics.
        """
        cfg = self.config
        logger.info(
            "Clustering %d scored edges (threshold=%.2f, singleton_guard=%.2f, max_matches=%d)...",
            len(scores),
            cfg.match_threshold,
            cfg.singleton_guard_threshold,
            cfg.max_matches_per_entity,
        )

        # Step 1: Filter edges above match threshold
        candidate_edges_by_cand: Dict[str, List[Tuple[str, float]]] = {}
        for s1, cand, score in zip(s1_ids, cand_ids, scores):
            if score >= cfg.match_threshold:
                if cand not in candidate_edges_by_cand:
                    candidate_edges_by_cand[cand] = []
                candidate_edges_by_cand[cand].append((s1, float(score)))

        conflicts_resolved = 0

        # Step 2: Enforce candidate exclusivity (max-weight matching)
        # If candidate C is claimed by multiple S1 entities, assign to the S1 with highest score.
        assigned_edges_by_s1: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in all_s1_universe}

        for cand, claims in candidate_edges_by_cand.items():
            if cfg.enforce_candidate_exclusivity and len(claims) > 1:
                # Sort claims by score descending, then tie-break by s1_id
                claims.sort(key=lambda x: (x[1], x[0]), reverse=True)
                best_s1, best_score = claims[0]
                conflicts_resolved += (len(claims) - 1)
                assigned_edges_by_s1[best_s1].append((cand, best_score))
            else:
                for s1, score in claims:
                    assigned_edges_by_s1[s1].append((cand, score))

        # Step 3: Apply Singleton Guard and Cardinality Caps per S1 entity
        final_clusters: Dict[str, List[str]] = {}
        cluster_sizes: Dict[int, int] = {}
        total_matches = 0
        singleton_count = 0

        for s1 in all_s1_universe:
            cand_list = assigned_edges_by_s1.get(s1, [])

            if not cand_list:
                final_clusters[s1] = []
                singleton_count += 1
                cluster_sizes[0] = cluster_sizes.get(0, 0) + 1
                continue

            # Singleton guard check: maximum edge score must meet guard threshold
            max_score = max(score for _, score in cand_list)
            if max_score < cfg.singleton_guard_threshold:
                final_clusters[s1] = []
                singleton_count += 1
                cluster_sizes[0] = cluster_sizes.get(0, 0) + 1
                continue

            # Sort candidate matches by score descending
            cand_list.sort(key=lambda x: x[1], reverse=True)

            # Cap at physical maximum
            capped_cands = [cand for cand, _ in cand_list[: cfg.max_matches_per_entity]]
            final_clusters[s1] = capped_cands
            n_matched = len(capped_cands)
            total_matches += n_matched
            cluster_sizes[n_matched] = cluster_sizes.get(n_matched, 0) + 1

        total_entities = len(all_s1_universe)
        metrics = ClusteringMetrics(
            total_entities=total_entities,
            singleton_count=singleton_count,
            singleton_rate=singleton_count / total_entities if total_entities > 0 else 0.0,
            total_matches=total_matches,
            mean_matches_per_entity=total_matches / total_entities if total_entities > 0 else 0.0,
            max_matches_per_entity=max(cluster_sizes.keys()) if cluster_sizes else 0,
            candidate_conflicts_resolved=conflicts_resolved,
            cluster_size_distribution=dict(sorted(cluster_sizes.items())),
        )

        logger.info(
            "Clustering complete: %d singletons (%.2f%%), %d total matches, %d conflicts resolved.",
            metrics.singleton_count,
            metrics.singleton_rate * 100.0,
            metrics.total_matches,
            metrics.candidate_conflicts_resolved,
        )
        return final_clusters, metrics
