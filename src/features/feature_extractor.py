"""
Amazon ML Challenge 2026 - Domain-Specific Pairwise Feature Extractor
Stage III - Candidate Scoring & Classification (Phase 5)

This module extracts a compact, high-discriminative 20-dimensional feature vector
for pairwise candidate classification.
Features:
1. Lexical & phonetic name similarities (RapidFuzz C++ engine).
2. Address component agreements & spatial indicators.
3. Cross-field collision guards (franchise detector & multi-tenant detector).
4. Candidate ranking & blocking consensus metadata.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein


FEATURE_NAMES = [
    "name_exact_clean",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_levenshtein_ratio",
    "name_qgram_jaccard",
    "name_first_token_match",
    "name_len_diff",
    "name_len_ratio",
    "addr_available",
    "addr_num_match",
    "addr_postal_match",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_qgram_jaccard",
    "addr_shared_words_count",
    "franchise_collision_hazard",
    "multi_tenant_hazard",
    "blocking_consensus_score",
    "candidate_rank_in_entity",
]


@dataclass(frozen=True)
class FeatureVector:
    """Stores the extracted feature matrix and metadata."""
    feature_names: List[str]
    features: np.ndarray  # Shape: (N, 20), dtype float32
    pair_ids: List[Tuple[str, str]]  # (source1_id, candidate_id)
    labels: Optional[np.ndarray] = None  # (N,), dtype int8 (0 or 1)
    extraction_time: float = 0.0


class PairwiseFeatureExtractor:
    """
    High-throughput pairwise feature extractor for entity resolution.
    """

    LEGAL_SUFFIXES_REGEX = re.compile(
        r"\b(inc|incorporated|llc|corp|corporation|co|company|ltd|limited|"
        r"pvt|private|llp|sa|sas|sarl|eurl|sci|snc)\b",
        re.IGNORECASE,
    )
    PUNCT_REGEX = re.compile(r"[^\w\s]")
    SPACES_REGEX = re.compile(r"\s+")
    DIGITS_REGEX = re.compile(r"\b\d+\b")
    POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")
    STOPWORDS_ADDR = {
        "road", "street", "avenue", "drive", "lane", "suite", "floor", "blvd",
        "near", "behind", "opp", "opposite", "flr", "apt", "block", "cross", "main"
    }

    @classmethod
    def clean_name(cls, name: Optional[str]) -> str:
        """Strip legal suffixes, punctuation, and normalize whitespace."""
        if not name:
            return ""
        s = name.lower()
        s = cls.PUNCT_REGEX.sub(" ", s)
        s = cls.LEGAL_SUFFIXES_REGEX.sub(" ", s)
        s = cls.SPACES_REGEX.sub(" ", s).strip()
        return s

    @classmethod
    def clean_address(cls, addr: Optional[str]) -> str:
        """Lowercase and normalize address string."""
        if not addr:
            return ""
        s = addr.lower()
        s = cls.PUNCT_REGEX.sub(" ", s)
        s = cls.SPACES_REGEX.sub(" ", s).strip()
        return s

    @staticmethod
    def get_qgrams(s: str, q: int = 3) -> Set[str]:
        """Extract character q-grams."""
        if len(s) < q:
            return {s} if s else set()
        return {s[i : i + q] for i in range(len(s) - q + 1)}

    @classmethod
    def qgram_jaccard(cls, s1: str, s2: str, q: int = 3) -> float:
        """Compute Jaccard similarity of character q-grams."""
        if not s1 or not s2:
            return 0.0
        q1 = cls.get_qgrams(s1, q)
        q2 = cls.get_qgrams(s2, q)
        union_len = len(q1 | q2)
        if union_len == 0:
            return 0.0
        return len(q1 & q2) / union_len

    @classmethod
    def extract_pair_features(
        cls,
        name1: str,
        addr1: str,
        name2: str,
        addr2: str,
        rank: float = 1.0,
        consensus: float = 1.0,
    ) -> List[float]:
        """
        Compute the 20-dimensional feature vector for a single pair.
        """
        # Preprocessing
        cname1 = cls.clean_name(name1)
        cname2 = cls.clean_name(name2)
        caddr1 = cls.clean_address(addr1)
        caddr2 = cls.clean_address(addr2)

        len1 = len(cname1)
        len2 = len(cname2)

        # A. Business Name Features
        name_exact = 1.0 if (cname1 == cname2 and len1 > 0) else 0.0
        name_jw = float(JaroWinkler.similarity(cname1, cname2)) if (len1 > 0 and len2 > 0) else 0.0
        name_tsr = float(fuzz.token_sort_ratio(cname1, cname2)) / 100.0 if (len1 > 0 and len2 > 0) else 0.0
        name_tset = float(fuzz.token_set_ratio(cname1, cname2)) / 100.0 if (len1 > 0 and len2 > 0) else 0.0
        name_lev = float(Levenshtein.normalized_similarity(cname1, cname2)) if (len1 > 0 and len2 > 0) else 0.0
        name_qg = cls.qgram_jaccard(cname1, cname2, 3)

        toks1 = cname1.split()
        toks2 = cname2.split()
        ft_match = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0
        len_diff = float(abs(len1 - len2))
        max_len = max(len1, len2)
        len_ratio = float(min(len1, len2) / max_len) if max_len > 0 else 1.0

        # B. Address Features
        has_addr2 = 1.0 if bool(caddr2) else 0.0

        # Street number matching
        nums1 = cls.DIGITS_REGEX.findall(caddr1)
        nums2 = cls.DIGITS_REGEX.findall(caddr2)
        if nums1 and nums2:
            num_match = 1.0 if nums1[0] == nums2[0] else -1.0
        else:
            num_match = 0.0

        # Postal code matching
        postals1 = cls.POSTAL_REGEX.findall(caddr1)
        postals2 = cls.POSTAL_REGEX.findall(caddr2)
        if postals1 and postals2:
            postal_match = 1.0 if postals1[-1] == postals2[-1] else -1.0
        else:
            postal_match = 0.0

        if has_addr2 and caddr1:
            addr_tsr = float(fuzz.token_sort_ratio(caddr1, caddr2)) / 100.0
            addr_tset = float(fuzz.token_set_ratio(caddr1, caddr2)) / 100.0
            addr_qg = cls.qgram_jaccard(caddr1, caddr2, 3)

            addr_toks1 = caddr1.split()
            addr_toks2 = caddr2.split()
            words1 = {w for w in addr_toks1 if len(w) >= 4 and w not in cls.STOPWORDS_ADDR}
            words2 = {w for w in addr_toks2 if len(w) >= 4 and w not in cls.STOPWORDS_ADDR}
            shared_words = float(len(words1 & words2))
        else:
            addr_tsr = 0.0
            addr_tset = 0.0
            addr_qg = 0.0
            shared_words = 0.0

        # C. Cross-Field Collision & Governance Features
        # Franchise hazard: high name similarity (>= 0.90) but different street numbers (-1.0)
        franchise_hazard = 1.0 if (name_tsr >= 0.90 and num_match == -1.0) else 0.0

        # Multi-tenant hazard: high address similarity (>= 0.85) but low name similarity (<= 0.50)
        multi_tenant_hazard = 1.0 if (addr_tsr >= 0.85 and name_tsr <= 0.50) else 0.0

        return [
            name_exact,
            name_jw,
            name_tsr,
            name_tset,
            name_lev,
            name_qg,
            ft_match,
            len_diff,
            len_ratio,
            has_addr2,
            num_match,
            postal_match,
            addr_tsr,
            addr_tset,
            addr_qg,
            shared_words,
            franchise_hazard,
            multi_tenant_hazard,
            float(consensus),
            float(rank),
        ]

    @classmethod
    def extract_batch(
        cls,
        pairs: List[Dict],
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ) -> FeatureVector:
        """
        Extract features for a batch of candidate pairs.
        
        Each item in `pairs` must be a dict with:
        's1_id', 's1_name', 's1_addr', 'cand_id', 'cand_name', 'cand_addr',
        and optionally 'rank', 'consensus'.
        """
        t0 = time.perf_counter()
        n = len(pairs)
        features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
        pair_ids: List[Tuple[str, str]] = []
        labels = np.empty(n, dtype=np.int8) if ground_truth is not None else None

        for i, item in enumerate(pairs):
            s1_id = item["s1_id"]
            cand_id = item["cand_id"]
            pair_ids.append((s1_id, cand_id))

            rank = item.get("rank", 1.0)
            consensus = item.get("consensus", 1.0)

            feat = cls.extract_pair_features(
                name1=item.get("s1_name", ""),
                addr1=item.get("s1_addr", ""),
                name2=item.get("cand_name", ""),
                addr2=item.get("cand_addr", ""),
                rank=rank,
                consensus=consensus,
            )
            features[i] = feat

            if ground_truth is not None:
                true_set = ground_truth.get(s1_id, set())
                labels[i] = 1 if cand_id in true_set else 0

        runtime = time.perf_counter() - t0
        return FeatureVector(
            feature_names=FEATURE_NAMES,
            features=features,
            pair_ids=pair_ids,
            labels=labels,
            extraction_time=runtime,
        )
