"""
Amazon ML Challenge 2026 - Multi-Pass Candidate Generation & Blocking Engine
Stage II - Candidate Generation & Blocking (Phase 4)

This module implements a disjunctive multi-pass blocking architecture:
1. Country Partition Guard: Strict 0.0000% cross-country matching partition.
2. Pass 1 - Canonical Name Core: Suffix-stripped normalized name matching.
3. Pass 2 - Street Number + Street Token: Physical location key for heavily corrupted names & URLs.
4. Pass 3 - Postal Code + Name Prefix: Pinpoint regional key for multilingual Indic transliteration.
5. Pass 4 - Sorted Name Tokens: Word reordering and permutation invariant key.
6. Disjunctive Union & Parsimony Truncation: Merges candidate streams and caps candidates per entity (K <= 20).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import polars as pl


@dataclass
class BlockingConfig:
    max_candidates_per_entity: int = 20
    max_candidates_per_key: int = 15
    enable_pass1_name_core: bool = True
    enable_pass2_addr_street: bool = True
    enable_pass3_postal_prefix: bool = True
    enable_pass4_sorted_tokens: bool = True


class MultiPassBlocker:
    """
    High-recall, parsimonious multi-pass blocker for large-scale entity resolution.
    """

    LEGAL_SUFFIXES_REGEX = (
        r"\b(inc|incorporated|llc|corp|corporation|co|company|ltd|limited|"
        r"pvt|private|llp|sa|sas|sarl|eurl|sci|snc)\b"
    )
    STREET_STOPWORDS_REGEX = (
        r"\b(road|street|avenue|drive|lane|suite|floor|blvd|near|behind|"
        r"opp|opposite|flr|apt|block|cross|main)\b"
    )

    def __init__(self, config: Optional[BlockingConfig] = None):
        self.config = config or BlockingConfig()

    def extract_blocking_keys(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract multi-pass blocking keys using vectorized Polars operations.
        
        Input DataFrame must have: ['entity_id', 'business_name', 'business_address', 'country']
        """
        # Step 1: Base clean name and address
        df_clean = df.with_columns([
            pl.col("business_name").fill_null("").str.to_lowercase()
            .str.replace_all(r"[^\w\s]", "")
            .str.replace_all(self.LEGAL_SUFFIXES_REGEX, "")
            .str.replace_all(r"\s+", " ").str.strip_chars()
            .alias("name_core"),

            pl.col("business_address").fill_null("").str.to_lowercase()
            .alias("addr_clean"),
        ])

        # Step 2: Extract individual key components
        df_keys = df_clean.with_columns([
            # Pass 1 Key: Canonical name core
            pl.col("name_core").alias("key_pass1"),

            # Pass 2 Key: Street number + primary street word
            pl.concat_str([
                pl.col("addr_clean").str.extract(r"\b(\d+)\b", 1).fill_null(""),
                pl.lit("_"),
                pl.col("addr_clean")
                .str.replace_all(self.STREET_STOPWORDS_REGEX, "")
                .str.extract(r"([a-z]{4,})", 1).fill_null(""),
            ]).alias("key_pass2"),

            # Pass 3 Key: Postal code (5-6 digits) + first 3 letters of name
            pl.concat_str([
                pl.col("addr_clean").str.extract(r"\b(\d{5,6})\b", 1).fill_null(""),
                pl.lit("_"),
                pl.col("name_core").str.slice(0, 3).fill_null(""),
            ]).alias("key_pass3"),

            # Pass 4 Key: Sorted first 2 significant words of name
            pl.col("name_core")
            .str.split(" ")
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
            .list.slice(0, 2)
            .list.sort()
            .list.join("_")
            .alias("key_pass4"),
        ])

        return df_keys.select([
            "entity_id",
            "country",
            "key_pass1",
            "key_pass2",
            "key_pass3",
            "key_pass4",
        ])

    def generate_candidates(
        self,
        s1_path: Union[str, Path],
        s2_path: Union[str, Path],
        s3_path: Union[str, Path],
        output_candidate_path: Union[str, Path],
        output_matching_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Optional[Path], Dict[str, int]]:
        """
        Execute disjunctive multi-pass blocking across S1, S2, and S3.
        
        Args:
            s1_path: Path to Source 1 TSV.
            s2_path: Path to Source 2 TSV.
            s3_path: Path to Source 3 TSV.
            output_candidate_path: Destination for candidate_pairs.tsv.
            output_matching_path: Destination for matching_results.tsv (optional).
            
        Returns:
            Tuple of (Path to candidate TSV, Path to matching TSV, dict with pass statistics).
        """
        t0 = time.perf_counter()
        print("[MultiPassBlocker] Initializing Multi-Pass Blocking Pipeline...", flush=True)

        cand_out = Path(output_candidate_path)
        cand_out.parent.mkdir(parents=True, exist_ok=True)
        matching_out = Path(output_matching_path) if output_matching_path else None
        if matching_out:
            matching_out.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load Source 1
        t_load = time.perf_counter()
        print("  [Step 1] Loading and indexing Source 1...", flush=True)
        s1_raw = pl.read_csv(
            s1_path,
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8},
        )
        s1_keys = self.extract_blocking_keys(s1_raw)
        s1_ids = s1_raw.select("entity_id")
        num_s1 = len(s1_keys)
        print(f"    Loaded {num_s1:,} S1 entities in {time.perf_counter() - t_load:.2f}s", flush=True)

        # 2. Load and concatenate Candidate Sources (S2 + S3)
        t_load_cand = time.perf_counter()
        print("  [Step 2] Loading and indexing Candidate Sources (S2 & S3)...", flush=True)
        s2_raw = pl.read_csv(
            s2_path,
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8},
        )
        s3_raw = pl.read_csv(
            s3_path,
            separator="\t",
            columns=["entity_id", "business_name", "business_address", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8},
        )
        cand_raw = pl.concat([s2_raw, s3_raw], how="vertical")
        cand_keys = self.extract_blocking_keys(cand_raw)
        num_cands = len(cand_keys)
        print(f"    Loaded {num_cands:,} Candidate entities in {time.perf_counter() - t_load_cand:.2f}s", flush=True)

        pass_stats: Dict[str, int] = {}
        pass_pairs_list: List[pl.DataFrame] = []

        # Helper function to run one blocking pass
        def execute_pass(pass_name: str, key_col: str, min_key_len: int, pass_weight: int):
            t_pass = time.perf_counter()
            print(f"  [Pass Execution] Running {pass_name} on `{key_col}`...", flush=True)

            # Filter valid candidate keys
            valid_cand = cand_keys.filter(
                (pl.col(key_col).is_not_null())
                & (pl.col(key_col).str.len_chars() >= min_key_len)
                & (pl.col(key_col) != "_")
            ).select(["country", key_col, "entity_id"])

            # Group candidates by (country, key) with parsimony cap
            grouped_cand = valid_cand.group_by(["country", key_col]).agg(
                pl.col("entity_id").head(self.config.max_candidates_per_key).alias("candidate_id")
            )

            # Join S1 on (country, key)
            valid_s1 = s1_keys.filter(
                (pl.col(key_col).is_not_null())
                & (pl.col(key_col).str.len_chars() >= min_key_len)
                & (pl.col(key_col) != "_")
            ).select(["entity_id", "country", key_col])

            joined = valid_s1.join(
                grouped_cand,
                on=["country", key_col],
                how="inner",
            ).select([
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("candidate_id"),  # list of candidate IDs
            ]).explode("candidate_id").with_columns(
                pl.lit(pass_weight).alias("weight")
            )

            count = len(joined)
            pass_stats[pass_name] = count
            print(f"    --> {pass_name}: Generated {count:,} candidate pairs in {time.perf_counter() - t_pass:.2f}s", flush=True)
            return joined

        # Run Pass 1: Canonical Name Core
        if self.config.enable_pass1_name_core:
            p1 = execute_pass("Pass 1 (Canonical Name Core)", "key_pass1", min_key_len=3, pass_weight=4)
            pass_pairs_list.append(p1)

        # Run Pass 2: Street Number + Street Token
        if self.config.enable_pass2_addr_street:
            p2 = execute_pass("Pass 2 (Street Num + Street Word)", "key_pass2", min_key_len=4, pass_weight=3)
            pass_pairs_list.append(p2)

        # Run Pass 3: Postal Code + Name Prefix
        if self.config.enable_pass3_postal_prefix:
            p3 = execute_pass("Pass 3 (Postal Code + Name Prefix)", "key_pass3", min_key_len=6, pass_weight=2)
            pass_pairs_list.append(p3)

        # Run Pass 4: Sorted Top-2 Name Tokens
        if self.config.enable_pass4_sorted_tokens:
            p4 = execute_pass("Pass 4 (Sorted Top-2 Tokens)", "key_pass4", min_key_len=4, pass_weight=3)
            pass_pairs_list.append(p4)

        # Step 3: Disjunctive Union & Parsimony Ranking
        print("\n  [Step 3] Merging candidate streams & enforcing parsimony truncation...", flush=True)
        t_merge = time.perf_counter()
        all_pairs = pl.concat(pass_pairs_list, how="vertical")
        total_raw_pairs = len(all_pairs)
        print(f"    Total raw candidate pairs across all passes: {total_raw_pairs:,}", flush=True)

        # Aggregate and rank by total weight (frequency of appearance across passes)
        # Deduplicate identical (source1_entity_id, candidate_id) while summing weights
        deduped_pairs = all_pairs.group_by(["source1_entity_id", "candidate_id"]).agg(
            pl.col("weight").sum().alias("total_weight")
        )
        print(f"    Unique candidate pairs: {len(deduped_pairs):,} (deduped in {time.perf_counter() - t_merge:.2f}s)", flush=True)

        # Rank candidates per S1 entity by weight descending, and cap to max_candidates_per_entity
        t_rank = time.perf_counter()
        ranked = (
            deduped_pairs.sort(["source1_entity_id", "total_weight"], descending=[False, True])
            .group_by("source1_entity_id", maintain_order=True)
            .agg(pl.col("candidate_id").head(self.config.max_candidates_per_entity).alias("cand_list"))
        )

        # Step 4: Join back to all S1 IDs to guarantee all entities are represented (singletons get null)
        final_cand_df = s1_ids.join(
            ranked,
            left_on="entity_id",
            right_on="source1_entity_id",
            how="left",
        ).select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("cand_list").list.join(",").alias("candidate_entity_ids"),
        ])
        print(f"    Parsimony ranking and formatting completed in {time.perf_counter() - t_rank:.2f}s", flush=True)

        # Step 5: Write candidate_pairs.tsv (strictly tab-separated, no quotes, null_value="")
        print(f"  [Step 5] Writing candidate pairs to {cand_out}...", flush=True)
        final_cand_df.write_csv(
            cand_out, separator="\t", include_header=True, null_value="", quote_style="never"
        )
        print(f"    Saved {len(final_cand_df):,} rows to {cand_out}", flush=True)

        # Step 6: Write matching_results.tsv (if requested, emits top-ranked candidates)
        if matching_out:
            print(f"  [Step 6] Writing matching results to {matching_out}...", flush=True)
            matching_df = final_cand_df.rename({"candidate_entity_ids": "matched_entity_ids"})
            matching_df.write_csv(
                matching_out, separator="\t", include_header=True, null_value="", quote_style="never"
            )
            print(f"    Saved {len(matching_df):,} rows to {matching_out}", flush=True)

        total_time = time.perf_counter() - t0
        print(f"[MultiPassBlocker] Blocking Pipeline completed in {total_time:.2f}s.", flush=True)

        return cand_out, matching_out, pass_stats
