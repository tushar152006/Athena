"""
Amazon ML Challenge 2026 - Run End-to-End Test Inference & Validation
Stage V - Pipeline Integration & Submission Ready (Phase 8)

This script:
1. Executes the EndToEndInferencePipeline across all test datasets (US, IN, FR).
2. Generates output/candidate_pairs.tsv and output/matching_results.tsv.
3. Runs the official submission validator: student_resource/utils/validate_submission.py.
4. Computes cryptographic SHA-256 checksums and file sizes for the output TSVs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from src.pipeline.inference_pipeline import EndToEndInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RunInference")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_pairwise.txt"
OUTPUT_DIR = PROJECT_ROOT / "output"
VALIDATOR_SCRIPT = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    t_start = time.perf_counter()
    logger.info("Initializing Phase 8 Test Inference Runner...")

    pipeline = EndToEndInferencePipeline(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        match_threshold=0.60,
        singleton_guard_threshold=0.60,
        max_matches_per_entity=11,
    )

    # Execute full pipeline across all 1.73M test entities
    results = pipeline.run_full_pipeline(test_dir=TEST_DIR)

    cand_path = OUTPUT_DIR / "candidate_pairs.tsv"
    match_path = OUTPUT_DIR / "matching_results.tsv"

    logger.info("Calculating file sizes and SHA-256 checksums...")
    cand_size = cand_path.stat().st_size
    match_size = match_path.stat().st_size
    cand_sha256 = compute_sha256(cand_path)
    match_sha256 = compute_sha256(match_path)

    print("\n" + "=" * 80)
    print("OUTPUT FILES CHECKSUM & SPECIFICATION:")
    print("-" * 80)
    print(f"File: {cand_path.name}")
    print(f"  Size:    {cand_size:,} bytes ({cand_size / (1024*1024):.2f} MB)")
    print(f"  SHA-256: {cand_sha256}")
    print(f"File: {match_path.name}")
    print(f"  Size:    {match_size:,} bytes ({match_size / (1024*1024):.2f} MB)")
    print(f"  SHA-256: {match_sha256}")
    print("=" * 80 + "\n")

    # Run official validator
    logger.info("Running official submission validator (%s)...", VALIDATOR_SCRIPT)
    validator_cmd = [
        sys.executable,
        str(VALIDATOR_SCRIPT),
        "--matching", str(match_path),
        "--candidate", str(cand_path),
        "--test-dir", str(TEST_DIR),
    ]

    val_res = subprocess.run(validator_cmd, capture_output=True, text=True)
    print("\n" + "=" * 80)
    print("OFFICIAL VALIDATOR OUTPUT:")
    print("-" * 80)
    print(val_res.stdout)
    if val_res.stderr:
        print("STDERR:")
        print(val_res.stderr)
    print(f"Validator Exit Code: {val_res.returncode}")
    print("=" * 80 + "\n")

    if val_res.returncode != 0:
        logger.error("VALIDATION FAILED! Check output for formatting issues.")
        sys.exit(1)

    logger.info("VALIDATION PASSED PERFECTLY! Files are 100%% competition compliant.")
    logger.info("Total runner execution time: %.2f seconds", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
