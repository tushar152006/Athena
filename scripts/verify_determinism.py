#!/usr/bin/env python3
"""
Verify pipeline determinism by running two consecutive inference passes on 500 test entities.
"""

from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path
import polars as pl
from src.pipeline.inference_pipeline import EndToEndInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeterminismTest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_pairwise.txt"
TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
OUT_DIR_1 = PROJECT_ROOT / "reports" / "reproducibility_run_1"
OUT_DIR_2 = PROJECT_ROOT / "reports" / "reproducibility_run_2"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_slice(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = EndToEndInferencePipeline(
        model_path=MODEL_PATH,
        output_dir=out_dir,
        match_threshold=0.60,
        singleton_guard_threshold=0.60,
        max_matches_per_entity=11,
    )
    # Load 500 S1 from US test
    s1_df = pl.read_csv(
        TEST_DIR / "test_source1.tsv",
        separator="\t",
        n_rows=500,
        ignore_errors=True,
    )
    pipe.run_test_inference_on_dataframe(s1_df=s1_df, test_dir=TEST_DIR, output_dir=out_dir)


def main():
    logger.info("Running Reproducibility Pass 1...")
    run_slice(OUT_DIR_1)
    logger.info("Running Reproducibility Pass 2...")
    run_slice(OUT_DIR_2)

    c1 = hash_file(OUT_DIR_1 / "candidate_pairs.tsv")
    c2 = hash_file(OUT_DIR_2 / "candidate_pairs.tsv")
    m1 = hash_file(OUT_DIR_1 / "matching_results.tsv")
    m2 = hash_file(OUT_DIR_2 / "matching_results.tsv")

    logger.info("Pass 1 Candidate Hash: %s", c1)
    logger.info("Pass 2 Candidate Hash: %s", c2)
    logger.info("Pass 1 Matching Hash:  %s", m1)
    logger.info("Pass 2 Matching Hash:  %s", m2)

    assert c1 == c2, "Candidate pairs nondeterminism detected!"
    assert m1 == m2, "Matching results nondeterminism detected!"
    logger.info("100% BIT-FOR-BIT DETERMINISM CONFIRMED.")


if __name__ == "__main__":
    main()
