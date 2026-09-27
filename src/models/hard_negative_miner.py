"""
Amazon ML Challenge 2026 - Hard Negative Mining Engine
Stage III - Feature Engineering & Core Model (Phase 9)

This module implements targeted hard negative mining to resolve ambiguous look-alikes:
1. Category A: Franchise Look-Alikes (High Name Similarity + Conflicting Street Number/Postal Code).
2. Category B: Multi-Tenant Co-locations (High Address Similarity + Low/Mismatched Name).
3. Category C: High-Confidence Blocking False Positives (Candidates ranked top 1-3 by blocker that are non-matches).
4. Category D: Standard Diverse Negatives (to preserve general background distribution).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from src.features.feature_extractor import PairwiseFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MiningCurriculumConfig:
    """Configuration for hard negative sampling curriculum."""
    target_negative_ratio: float = 7.0  # 7 negatives per 1 positive
    cat_a_franchise_weight: float = 0.35
    cat_b_multitenant_weight: float = 0.25
    cat_c_top_blocking_weight: float = 0.25
    cat_d_diverse_weight: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class MiningStatistics:
    """Detailed summary of mined pair curriculum."""
    total_positives: int
    total_negatives: int
    negative_to_positive_ratio: float
    cat_a_franchise_count: int
    cat_b_multitenant_count: int
    cat_c_top_blocking_count: int
    cat_d_diverse_count: int
    total_pairs: int


class HardNegativeMiner:
    """
    Identifies, categorizes, and samples deceptive non-matching candidate pairs
    to construct an augmented training curriculum for gradient boosting disambiguation.
    """

    DIGITS_REGEX = re.compile(r"\b\d+\b")
    POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")

    def __init__(self, config: Optional[MiningCurriculumConfig] = None):
        self.config = config or MiningCurriculumConfig()

    @classmethod
    def extract_street_num(cls, address: str) -> Optional[str]:
        """Extract first integer sequence from address."""
        nums = cls.DIGITS_REGEX.findall(address)
        return nums[0] if nums else None

    @classmethod
    def extract_postal_code(cls, address: str) -> Optional[str]:
        """Extract 5-6 digit postal code from address."""
        postals = cls.POSTAL_REGEX.findall(address)
        return postals[-1] if postals else None

    def mine_curriculum(
        self,
        s1_ids: List[str],
        candidates_by_s1: Dict[str, List[str]],
        s1_meta: Dict[str, Tuple[str, str]],
        cand_meta: Dict[str, Tuple[str, str]],
        ground_truth: Dict[str, Set[str]],
    ) -> Tuple[List[Dict], MiningStatistics]:
        """
        Mine hard negative training curriculum across given entities.
        
        Returns:
            List of pair dictionaries (ready for PairwiseFeatureExtractor.extract_batch)
            and MiningStatistics summary.
        """
        cfg = self.config
        rng = random.Random(cfg.seed)

        logger.info("Starting Hard Negative Mining across %d S1 entities...", len(s1_ids))

        pos_pairs: List[Dict] = []
        cat_a_pairs: List[Dict] = []
        cat_b_pairs: List[Dict] = []
        cat_c_pairs: List[Dict] = []
        cat_d_pairs: List[Dict] = []

        # Pre-clean metadata lookups to accelerate categorization
        logger.info("Pre-cleaning entity strings for fast categorization...")
        s1_clean: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}
        for s1_id in s1_ids:
            name, addr = s1_meta.get(s1_id, ("", ""))
            cname = PairwiseFeatureExtractor.clean_name(name)
            caddr = PairwiseFeatureExtractor.clean_address(addr)
            s1_clean[s1_id] = (
                cname,
                caddr,
                self.extract_street_num(caddr),
                self.extract_postal_code(caddr),
            )

        cand_clean: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}
        all_cands_needed: Set[str] = set()
        for s1_id in s1_ids:
            for c in candidates_by_s1.get(s1_id, []):
                all_cands_needed.add(c)
            # Also ensure true matches are present
            for c in ground_truth.get(s1_id, set()):
                all_cands_needed.add(c)

        for c_id in all_cands_needed:
            name, addr = cand_meta.get(c_id, ("", ""))
            cname = PairwiseFeatureExtractor.clean_name(name)
            caddr = PairwiseFeatureExtractor.clean_address(addr)
            cand_clean[c_id] = (
                cname,
                caddr,
                self.extract_street_num(caddr),
                self.extract_postal_code(caddr),
            )

        # Categorize pairs
        for s1_id in s1_ids:
            s1_name_raw, s1_addr_raw = s1_meta.get(s1_id, ("", ""))
            s1_cname, s1_caddr, s1_snum, s1_post = s1_clean[s1_id]
            gt_matches = ground_truth.get(s1_id, set())

            # 1. Collect True Positives
            # Include blocking candidates that match GT, plus any GT matches missed by blocking
            c_list = candidates_by_s1.get(s1_id, [])
            c_set = set(c_list)

            for rank, cand_id in enumerate(c_list, start=1):
                cand_name_raw, cand_addr_raw = cand_meta.get(cand_id, ("", ""))
                if cand_id in gt_matches:
                    pos_pairs.append({
                        "s1_id": s1_id,
                        "cand_id": cand_id,
                        "s1_name": s1_name_raw,
                        "s1_addr": s1_addr_raw,
                        "cand_name": cand_name_raw,
                        "cand_addr": cand_addr_raw,
                        "rank": float(rank),
                        "consensus": 1.0,
                        "label": 1,
                    })
                else:
                    # It is a candidate negative! Analyze hardness:
                    cand_cname, cand_caddr, cand_snum, cand_post = cand_clean[cand_id]
                    
                    # Quick name similarity
                    name_sim = JaroWinkler.similarity(s1_cname, cand_cname) if (s1_cname and cand_cname) else 0.0
                    
                    # Number / postal match indicators
                    snum_match = (s1_snum == cand_snum) if (s1_snum and cand_snum) else None
                    post_match = (s1_post == cand_post) if (s1_post and cand_post) else None

                    pair_dict = {
                        "s1_id": s1_id,
                        "cand_id": cand_id,
                        "s1_name": s1_name_raw,
                        "s1_addr": s1_addr_raw,
                        "cand_name": cand_name_raw,
                        "cand_addr": cand_addr_raw,
                        "rank": float(rank),
                        "consensus": 1.0,
                        "label": 0,
                    }

                    # Category A: Franchise Look-Alike
                    # Very high name similarity (>= 0.85) but street number mismatch or postal mismatch
                    if name_sim >= 0.85 and (snum_match is False or post_match is False):
                        cat_a_pairs.append(pair_dict)
                    # Category B: Multi-Tenant Co-location
                    # High address similarity / same street number & postal code, but low name similarity (< 0.55)
                    elif (snum_match is True or post_match is True) and name_sim <= 0.55:
                        cat_b_pairs.append(pair_dict)
                    # Category C: Top-Ranked Blocking Collision (Rank 1 to 3)
                    elif rank <= 3:
                        cat_c_pairs.append(pair_dict)
                    else:
                        # Category D: Diverse general negatives
                        cat_d_pairs.append(pair_dict)

            # Ensure ground-truth matches missed by blocking are included as positive examples
            for gt_id in gt_matches:
                if gt_id not in c_set and gt_id in cand_meta:
                    cand_name_raw, cand_addr_raw = cand_meta[gt_id]
                    pos_pairs.append({
                        "s1_id": s1_id,
                        "cand_id": gt_id,
                        "s1_name": s1_name_raw,
                        "s1_addr": s1_addr_raw,
                        "cand_name": cand_name_raw,
                        "cand_addr": cand_addr_raw,
                        "rank": 20.0,
                        "consensus": 1.0,
                        "label": 1,
                    })

        n_pos = len(pos_pairs)
        n_target_neg = int(n_pos * cfg.target_negative_ratio)

        logger.info(
            "Identified candidate pools: Positives=%d | Cat A (Franchise)=%d | Cat B (MultiTenant)=%d | Cat C (TopBlocking)=%d | Cat D (Diverse)=%d",
            n_pos,
            len(cat_a_pairs),
            len(cat_b_pairs),
            len(cat_c_pairs),
            len(cat_d_pairs),
        )

        # Allocate negative quotas
        target_a = int(n_target_neg * cfg.cat_a_franchise_weight)
        target_b = int(n_target_neg * cfg.cat_b_multitenant_weight)
        target_c = int(n_target_neg * cfg.cat_c_top_blocking_weight)
        target_d = n_target_neg - (target_a + target_b + target_c)

        sampled_a = rng.sample(cat_a_pairs, min(len(cat_a_pairs), target_a))
        sampled_b = rng.sample(cat_b_pairs, min(len(cat_b_pairs), target_b))
        sampled_c = rng.sample(cat_c_pairs, min(len(cat_c_pairs), target_c))
        
        # If any specialized category had fewer pairs than quota, fill remainder from Cat D
        remaining_quota = n_target_neg - (len(sampled_a) + len(sampled_b) + len(sampled_c))
        sampled_d = rng.sample(cat_d_pairs, min(len(cat_d_pairs), max(0, remaining_quota)))

        final_negatives = sampled_a + sampled_b + sampled_c + sampled_d
        rng.shuffle(final_negatives)

        final_curriculum = pos_pairs + final_negatives
        rng.shuffle(final_curriculum)

        stats = MiningStatistics(
            total_positives=n_pos,
            total_negatives=len(final_negatives),
            negative_to_positive_ratio=round(len(final_negatives) / max(1, n_pos), 2),
            cat_a_franchise_count=len(sampled_a),
            cat_b_multitenant_count=len(sampled_b),
            cat_c_top_blocking_count=len(sampled_c),
            cat_d_diverse_count=len(sampled_d),
            total_pairs=len(final_curriculum),
        )

        logger.info(
            "Curriculum assembled: %d total pairs (%d Pos, %d Neg -> %.2f:1 ratio). Franchise: %d, MultiTenant: %d, TopBlocking: %d, Diverse: %d",
            stats.total_pairs,
            stats.total_positives,
            stats.total_negatives,
            stats.negative_to_positive_ratio,
            stats.cat_a_franchise_count,
            stats.cat_b_multitenant_count,
            stats.cat_c_top_blocking_count,
            stats.cat_d_diverse_count,
        )

        return final_curriculum, stats
