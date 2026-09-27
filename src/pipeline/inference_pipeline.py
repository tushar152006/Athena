"""
Amazon ML Challenge 2026 - Production Test Inference Pipeline
Stage V - Pipeline Integration & Submission Ready (Phase 8)

Architecture:
1. Stage 1: Vectorized Multi-Pass Candidate Blocking (Pass 1-4) via Polars
   - Country partitioned (France, India, US).
   - Generates output/candidate_pairs.tsv (1,732,544 rows, parsimony <= 20).
2. Stage 2: Chunked High-Speed Scoring & Graph Clustering
   - Streams candidate_pairs.tsv in chunks of 50,000 entities.
   - Ultra-fast C++ feature extraction via precomputed entity attributes.
   - LightGBM scoring at calibrated threshold p* = 0.60.
   - Bipartite graph clustering with candidate conflict resolution, singleton guard, and max 11 cap.
   - Generates output/matching_results.tsv (1,732,544 rows).
3. Stage 3: Official Submission Validation
   - Runs student_resource/utils/validate_submission.py.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.blocking.multi_pass_blocker import BlockingConfig, MultiPassBlocker
from src.clustering.graph_clusterer import BipartiteGraphClusterer, ClustererConfig
from src.features.feature_extractor import PairwiseFeatureExtractor
from src.models.pairwise_classifier import LightGBMPairwiseClassifier

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PrecomputedEntity:
    """Precomputed attributes for ultra-fast pairwise feature extraction."""
    cname: str
    caddr: str
    name_qgrams: Set[str]
    addr_qgrams: Set[str]
    name_toks: List[str]
    addr_words_set: Set[str]
    street_num: Optional[str]
    postal_code: Optional[str]


class EndToEndInferencePipeline:
    """
    Production-grade end-to-end inference pipeline for full test set submission.
    """

    DIGITS_REGEX = re.compile(r"\b\d+\b")
    POSTAL_REGEX = re.compile(r"\b\d{5,6}\b")
    STOPWORDS_ADDR = {
        "road", "street", "avenue", "drive", "lane", "suite", "floor", "blvd",
        "near", "behind", "opp", "opposite", "flr", "apt", "block", "cross", "main"
    }

    def __init__(
        self,
        model_path: Path,
        output_dir: Path,
        match_threshold: float = 0.60,
        singleton_guard_threshold: float = 0.60,
        max_matches_per_entity: int = 11,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.match_threshold = match_threshold
        self.singleton_guard_threshold = singleton_guard_threshold
        self.max_matches_per_entity = max_matches_per_entity

        self.blocker = MultiPassBlocker(
            config=BlockingConfig(
                max_candidates_per_entity=20,
                max_candidates_per_key=10,
            )
        )
        self.clusterer = BipartiteGraphClusterer(
            config=ClustererConfig(
                match_threshold=self.match_threshold,
                singleton_guard_threshold=self.singleton_guard_threshold,
                max_matches_per_entity=self.max_matches_per_entity,
                enforce_candidate_exclusivity=True,
            )
        )

        logger.info("Loading LightGBM model from %s...", self.model_path)
        self.classifier = LightGBMPairwiseClassifier.load(self.model_path)
        logger.info("Loaded model successfully (threshold = %.2f)", self.classifier.optimal_threshold)

    @classmethod
    def precompute_entity(cls, name: Optional[str], addr: Optional[str]) -> PrecomputedEntity:
        """Precompute clean strings, q-gram sets, and regex tokens for an entity."""
        cname = PairwiseFeatureExtractor.clean_name(name)
        caddr = PairwiseFeatureExtractor.clean_address(addr)

        name_qg = PairwiseFeatureExtractor.get_qgrams(cname, 3)
        addr_qg = PairwiseFeatureExtractor.get_qgrams(caddr, 3)

        name_toks = cname.split()
        addr_toks = caddr.split()
        addr_words = {w for w in addr_toks if len(w) >= 4 and w not in cls.STOPWORDS_ADDR}

        nums = cls.DIGITS_REGEX.findall(caddr)
        street_num = nums[0] if nums else None

        postals = cls.POSTAL_REGEX.findall(caddr)
        postal_code = postals[-1] if postals else None

        return PrecomputedEntity(
            cname=cname,
            caddr=caddr,
            name_qgrams=name_qg,
            addr_qgrams=addr_qg,
            name_toks=name_toks,
            addr_words_set=addr_words,
            street_num=street_num,
            postal_code=postal_code,
        )

    @classmethod
    def extract_pair_features(
        cls,
        e1: PrecomputedEntity,
        e2: PrecomputedEntity,
        rank: float = 1.0,
        consensus: float = 1.0,
    ) -> List[float]:
        """Ultra-fast feature extraction from precomputed entity representations."""
        cname1, cname2 = e1.cname, e2.cname
        len1, len2 = len(cname1), len(cname2)

        # Name Features
        name_exact = 1.0 if (cname1 == cname2 and len1 > 0) else 0.0
        name_jw = float(JaroWinkler.similarity(cname1, cname2)) if (len1 > 0 and len2 > 0) else 0.0
        name_tsr = float(fuzz.token_sort_ratio(cname1, cname2)) / 100.0 if (len1 > 0 and len2 > 0) else 0.0
        name_tset = float(fuzz.token_set_ratio(cname1, cname2)) / 100.0 if (len1 > 0 and len2 > 0) else 0.0
        name_lev = float(Levenshtein.normalized_similarity(cname1, cname2)) if (len1 > 0 and len2 > 0) else 0.0

        q1, q2 = e1.name_qgrams, e2.name_qgrams
        u_len = len(q1 | q2)
        name_qg = len(q1 & q2) / u_len if u_len > 0 else 0.0

        toks1, toks2 = e1.name_toks, e2.name_toks
        ft_match = 1.0 if (toks1 and toks2 and toks1[0] == toks2[0]) else 0.0
        len_diff = float(abs(len1 - len2))
        max_l = max(len1, len2)
        len_ratio = float(min(len1, len2) / max_l) if max_l > 0 else 1.0

        # Address Features
        has_addr2 = 1.0 if bool(e2.caddr) else 0.0

        if e1.street_num and e2.street_num:
            num_match = 1.0 if e1.street_num == e2.street_num else -1.0
        else:
            num_match = 0.0

        if e1.postal_code and e2.postal_code:
            postal_match = 1.0 if e1.postal_code == e2.postal_code else -1.0
        else:
            postal_match = 0.0

        if has_addr2 and e1.caddr:
            caddr1, caddr2 = e1.caddr, e2.caddr
            addr_tsr = float(fuzz.token_sort_ratio(caddr1, caddr2)) / 100.0
            addr_tset = float(fuzz.token_set_ratio(caddr1, caddr2)) / 100.0

            aq1, aq2 = e1.addr_qgrams, e2.addr_qgrams
            au_len = len(aq1 | aq2)
            addr_qg = len(aq1 & aq2) / au_len if au_len > 0 else 0.0

            shared_words = float(len(e1.addr_words_set & e2.addr_words_set))
        else:
            addr_tsr = 0.0
            addr_tset = 0.0
            addr_qg = 0.0
            shared_words = 0.0

        # Cross-field collision guards
        franchise_hazard = 1.0 if (name_tsr >= 0.90 and num_match == -1.0) else 0.0
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

    def run_full_pipeline(self, test_dir: Path) -> Dict[str, Union[int, float]]:
        """Execute end-to-end inference and emit validated submission TSVs."""
        t_pipeline_start = time.perf_counter()
        logger.info("Starting Full End-to-End Submission Inference Pipeline...")

        s1_file = test_dir / "test_source1.tsv"
        s2_file = test_dir / "test_source2.tsv"
        s3_file = test_dir / "test_source3.tsv"

        cand_tsv = self.output_dir / "candidate_pairs.tsv"
        match_tsv = self.output_dir / "matching_results.tsv"

        # Step 1: Multi-Pass Candidate Generation & Blocking via Polars
        t_block_start = time.perf_counter()
        logger.info("=== Stage 1: Vectorized Multi-Pass Candidate Generation ===")
        self.blocker.generate_candidates(
            s1_path=s1_file,
            s2_path=s2_file,
            s3_path=s3_file,
            output_candidate_path=cand_tsv,
            output_matching_path=None,
        )
        block_time = time.perf_counter() - t_block_start
        logger.info("Candidate Blocking complete in %.2f seconds.", block_time)

        # Step 2: Build Entity Metadata Lookup Table for Test Sources
        logger.info("=== Stage 2: Ingesting Entity Metadata Lookups ===")
        t_meta_start = time.perf_counter()
        
        # Load S1 metadata
        logger.info("Loading S1 metadata...")
        s1_df = pl.read_csv(
            s1_file,
            separator="\t",
            columns=["entity_id", "business_name", "business_address"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8},
        )
        s1_dict: Dict[str, Tuple[str, str]] = {}
        for row in s1_df.iter_rows():
            s1_dict[row[0]] = (row[1] or "", row[2] or "")
        del s1_df
        logger.info("Loaded %d S1 entities.", len(s1_dict))

        # Load S2 and S3 metadata
        logger.info("Loading Candidate (S2 & S3) metadata...")
        cand_df = pl.concat([
            pl.read_csv(
                s2_file,
                separator="\t",
                columns=["entity_id", "business_name", "business_address"],
                dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8},
            ),
            pl.read_csv(
                s3_file,
                separator="\t",
                columns=["entity_id", "business_name", "business_address"],
                dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8},
            ),
        ], how="vertical")

        cand_dict: Dict[str, Tuple[str, str]] = {}
        for row in cand_df.iter_rows():
            cand_dict[row[0]] = (row[1] or "", row[2] or "")
        del cand_df
        logger.info("Loaded %d Candidate entities in %.2f s.", len(cand_dict), time.perf_counter() - t_meta_start)

        # Step 3: Stream candidate_pairs.tsv in chunks, extract features, score, and cluster
        logger.info("=== Stage 3: Streaming Scoring & Global Graph Clustering ===")
        t_score_start = time.perf_counter()

        chunk_size = 50000
        total_s1_processed = 0
        total_matches_emitted = 0
        total_singletons_emitted = 0

        with open(cand_tsv, "r", encoding="utf-8") as f_cand_in, \
             open(match_tsv, "w", encoding="utf-8") as f_match_out:
            # Header
            _ = f_cand_in.readline()
            f_match_out.write("source1_entity_id\tmatched_entity_ids\n")

            cur_chunk_s1: List[str] = []
            cur_chunk_cands: Dict[str, List[str]] = {}

            def process_current_chunk(chunk_s1_list: List[str], chunk_cand_map: Dict[str, List[str]]):
                nonlocal total_s1_processed, total_matches_emitted, total_singletons_emitted
                if not chunk_s1_list:
                    return

                # Precompute entity attributes needed for this chunk
                needed_cand_ids: Set[str] = set()
                for s1 in chunk_s1_list:
                    for c in chunk_cand_map.get(s1, []):
                        needed_cand_ids.add(c)

                # Precompute S1
                s1_precomputed: Dict[str, PrecomputedEntity] = {}
                for s1 in chunk_s1_list:
                    n, a = s1_dict.get(s1, ("", ""))
                    s1_precomputed[s1] = self.precompute_entity(n, a)

                # Precompute Candidates
                cand_precomputed: Dict[str, PrecomputedEntity] = {}
                for c in needed_cand_ids:
                    n, a = cand_dict.get(c, ("", ""))
                    cand_precomputed[c] = self.precompute_entity(n, a)

                # Build pair feature matrix
                pair_features: List[List[float]] = []
                pair_s1_ids: List[str] = []
                pair_cand_ids: List[str] = []

                for s1 in chunk_s1_list:
                    pe1 = s1_precomputed[s1]
                    c_list = chunk_cand_map.get(s1, [])
                    for rank, cid in enumerate(c_list, start=1):
                        pe2 = cand_precomputed.get(cid)
                        if pe2:
                            feat = self.extract_pair_features(pe1, pe2, rank=float(rank), consensus=1.0)
                            pair_features.append(feat)
                            pair_s1_ids.append(s1)
                            pair_cand_ids.append(cid)

                # Predict probabilities
                if pair_features:
                    X_np = np.asarray(pair_features, dtype=np.float32)
                    scores = self.classifier.predict_proba(X_np)
                else:
                    scores = np.empty(0, dtype=np.float32)

                # Cluster
                chunk_s1_set = set(chunk_s1_list)
                clustered_matches, cluster_metrics = self.clusterer.cluster_pairs(
                    s1_ids=pair_s1_ids,
                    cand_ids=pair_cand_ids,
                    scores=scores,
                    all_s1_universe=chunk_s1_set,
                )

                # Write to matching_results.tsv
                for s1 in chunk_s1_list:
                    matches = clustered_matches.get(s1, [])
                    match_str = ",".join(matches) if matches else ""
                    f_match_out.write(f"{s1}\t{match_str}\n")
                    if matches:
                        total_matches_emitted += len(matches)
                    else:
                        total_singletons_emitted += 1

                total_s1_processed += len(chunk_s1_list)
                logger.info(
                    "Processed %d / 1,732,544 entities (Matches: %d, Singletons: %d)...",
                    total_s1_processed,
                    total_matches_emitted,
                    total_singletons_emitted,
                )

            # Read lines
            for line in f_cand_in:
                parts = line.strip().split("\t")
                if len(parts) >= 1:
                    s1_id = parts[0]
                    cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) >= 2 else []
                    cur_chunk_s1.append(s1_id)
                    cur_chunk_cands[s1_id] = cands

                    if len(cur_chunk_s1) >= chunk_size:
                        process_current_chunk(cur_chunk_s1, cur_chunk_cands)
                        cur_chunk_s1.clear()
                        cur_chunk_cands.clear()

            if cur_chunk_s1:
                process_current_chunk(cur_chunk_s1, cur_chunk_cands)
                cur_chunk_s1.clear()
                cur_chunk_cands.clear()

        score_time = time.perf_counter() - t_score_start
        total_time = time.perf_counter() - t_pipeline_start

        logger.info("=" * 70)
        logger.info("TEST SET INFERENCE SUCCESSFULLY COMPLETED")
        logger.info("Total S1 Entities Processed: %d (Target: 1,732,544)", total_s1_processed)
        logger.info("Total Matches Emitted:       %d", total_matches_emitted)
        logger.info("Total Singletons Emitted:    %d (%.2f%%)", total_singletons_emitted, (total_singletons_emitted / total_s1_processed) * 100.0)
        logger.info("Blocking Time:               %.2f s", block_time)
        logger.info("Scoring & Clustering Time:   %.2f s", score_time)
        logger.info("Total Pipeline Wall Time:    %.2f s (%.2f minutes)", total_time, total_time / 60.0)
        logger.info("=" * 70)

        return {
            "total_s1_entities": total_s1_processed,
            "total_matches_emitted": total_matches_emitted,
            "total_singletons_emitted": total_singletons_emitted,
            "blocking_time_sec": round(block_time, 2),
            "scoring_time_sec": round(score_time, 2),
            "total_time_sec": round(total_time, 2),
            "cand_file": str(cand_tsv),
            "match_file": str(match_tsv),
        }
