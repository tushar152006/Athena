"""
Amazon ML Challenge 2026 - Deterministic Exact Baseline Matcher
Stage I - Foundations & Measurement (Phase 3)

This module implements a deterministic exact normalized string matching baseline:
1. Country Partitioning: Grouping by country to eliminate 61.1% of candidate comparisons.
2. Canonical Text Normalization: Lowercasing, whitespace stripping, and punctuation removal.
3. Exact Match Join: Fast multi-threaded indexed matching between S1 and S2/S3.
4. Parsimony Cap: Enforces max candidates per S1 entity (default <= 10).
5. Singleton Preservation: Guaranteed empty string prediction for unmatched entities.
6. Official Format Emission: Generates strictly compliant matching_results.tsv and candidate_pairs.tsv.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import polars as pl


class ExactNormalizedMatcher:
    """
    Deterministic baseline entity resolver using exact normalized name matching within country partitions.
    """

    def __init__(
        self,
        max_candidates_per_entity: int = 10,
        lowercase: bool = True,
        strip_punctuation: bool = True,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation

    def normalize_name_expr(self, col_name: str = "business_name") -> pl.Expr:
        """Vectorized Polars expression for fast text normalization."""
        expr = pl.col(col_name).fill_null("")
        if self.lowercase:
            expr = expr.str.to_lowercase()
        if self.strip_punctuation:
            # Replace all non-alphanumeric (excluding space) with empty string
            expr = expr.str.replace_all(r"[^\w\s]", "")
        # Collapse multiple spaces and strip ends
        expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()
        return expr

    def run(
        self,
        s1_path: Union[str, Path],
        s2_path: Union[str, Path],
        s3_path: Union[str, Path],
        output_matching_path: Union[str, Path],
        output_candidate_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Execute deterministic exact matching across S1, S2, and S3.
        
        Args:
            s1_path: Path to Source 1 TSV.
            s2_path: Path to Source 2 TSV.
            s3_path: Path to Source 3 TSV.
            output_matching_path: Destination for matching_results.tsv.
            output_candidate_path: Destination for candidate_pairs.tsv (optional).
            
        Returns:
            Tuple of (Path to matching TSV, Path to candidate TSV or None).
        """
        t0 = time.perf_counter()
        print(f"[ExactNormalizedMatcher] Starting exact baseline pipeline...")
        print(f"  Source 1: {s1_path}")
        print(f"  Source 2: {s2_path}")
        print(f"  Source 3: {s3_path}")

        matching_out = Path(output_matching_path)
        matching_out.parent.mkdir(parents=True, exist_ok=True)
        cand_out = Path(output_candidate_path) if output_candidate_path else None
        if cand_out:
            cand_out.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load Source 1
        t_load = time.perf_counter()
        print("  Loading and normalizing Source 1...")
        s1_df = pl.read_csv(
            s1_path,
            separator="\t",
            columns=["entity_id", "business_name", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "country": pl.Utf8},
        ).with_columns(
            self.normalize_name_expr("business_name").alias("norm_name")
        ).select(["entity_id", "country", "norm_name"])

        num_s1 = len(s1_df)
        print(f"    Loaded {num_s1:,} Source 1 records in {time.perf_counter() - t_load:.2f}s")

        # 2. Load and concatenate Candidate Sources (S2 + S3)
        t_load_cand = time.perf_counter()
        print("  Loading and normalizing Candidate Sources (S2 & S3)...")
        s2_df = pl.read_csv(
            s2_path,
            separator="\t",
            columns=["entity_id", "business_name", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "country": pl.Utf8},
        ).with_columns(
            self.normalize_name_expr("business_name").alias("norm_name")
        ).select(["entity_id", "country", "norm_name"])

        s3_df = pl.read_csv(
            s3_path,
            separator="\t",
            columns=["entity_id", "business_name", "country"],
            dtypes={"entity_id": pl.Utf8, "business_name": pl.Utf8, "country": pl.Utf8},
        ).with_columns(
            self.normalize_name_expr("business_name").alias("norm_name")
        ).select(["entity_id", "country", "norm_name"])

        candidates_df = pl.concat([s2_df, s3_df], how="vertical")
        num_candidates = len(candidates_df)
        print(f"    Loaded {num_candidates:,} candidate records in {time.perf_counter() - t_load_cand:.2f}s")

        # 3. Filter out empty normalized names from candidate lookup (cannot match on empty string)
        valid_candidates = candidates_df.filter(pl.col("norm_name") != "")

        # 4. Group candidates by (country, norm_name)
        # Enforce parsimony cap to avoid massive candidate sets for ubiquitous franchise names
        print(f"  Grouping candidates by (country, norm_name) with parsimony cap <= {self.max_candidates_per_entity}...")
        t_group = time.perf_counter()
        cand_grouped = valid_candidates.group_by(["country", "norm_name"]).agg(
            pl.col("entity_id").head(self.max_candidates_per_entity).alias("candidate_ids")
        )
        print(f"    Unique candidate keys: {len(cand_grouped):,} (grouped in {time.perf_counter() - t_group:.2f}s)")

        # 5. Join S1 with candidate lookup on ['country', 'norm_name']
        print("  Joining Source 1 with Candidate Lookup...")
        t_join = time.perf_counter()
        joined = s1_df.join(
            cand_grouped,
            on=["country", "norm_name"],
            how="left",
        )

        # 6. Convert list of IDs to comma-separated string (unmatched remain null)
        # Result columns: source1_entity_id, matched_entity_ids
        results = joined.select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("candidate_ids")
            .list.join(",")
            .alias("matched_entity_ids"),
        ])
        print(f"    Join and formatting completed in {time.perf_counter() - t_join:.2f}s")

        # 7. Write matching_results.tsv (strictly tab-separated, no quotes, null_value="")
        print(f"  Writing matching results to {matching_out}...")
        results.write_csv(
            matching_out, separator="\t", include_header=True, null_value="", quote_style="never"
        )
        print(f"    Saved {len(results):,} rows to {matching_out}")

        # 8. Write candidate_pairs.tsv (if requested)
        if cand_out:
            print(f"  Writing candidate pairs to {cand_out}...")
            cand_results = results.rename({"matched_entity_ids": "candidate_entity_ids"})
            cand_results.write_csv(
                cand_out, separator="\t", include_header=True, null_value="", quote_style="never"
            )
            print(f"    Saved {len(cand_results):,} rows to {cand_out}")

        total_time = time.perf_counter() - t0
        print(f"[ExactNormalizedMatcher] Pipeline completed in {total_time:.2f}s.")
        return matching_out, cand_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic Exact Baseline Matcher (Amazon ML 2026)")
    parser.add_argument("--s1", required=True, help="Path to source1.tsv")
    parser.add_argument("--s2", required=True, help="Path to source2.tsv")
    parser.add_argument("--s3", required=True, help="Path to source3.tsv")
    parser.add_argument("--matching-out", "-m", required=True, help="Path to output matching_results.tsv")
    parser.add_argument("--candidate-out", "-c", default=None, help="Path to output candidate_pairs.tsv")
    parser.add_argument("--max-cands", type=int, default=10, help="Max candidates per S1 entity")
    args = parser.parse_args()

    matcher = ExactNormalizedMatcher(max_candidates_per_entity=args.max_cands)
    matcher.run(args.s1, args.s2, args.s3, args.matching_out, args.candidate_out)
