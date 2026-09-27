#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Comprehensive Pre-Submission Audit Script
Audits output integrity, official validator status, candidate parsimony, subset consistency,
hash integrity, and compliance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreSubmissionAudit")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
VALIDATOR_PATH = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_files_and_consistency() -> Dict:
    match_file = OUTPUT_DIR / "matching_results.tsv"
    cand_file = OUTPUT_DIR / "candidate_pairs.tsv"

    logger.info("Checking file existence...")
    assert match_file.exists(), f"Missing {match_file}"
    assert cand_file.exists(), f"Missing {cand_file}"

    # 1. Check matching_results.tsv
    logger.info("Auditing matching_results.tsv...")
    s1_matched_counts = []
    s1_match_ids = set()
    duplicate_s1_match = 0
    empty_matches = 0
    invalid_s1_match_ids = 0
    invalid_target_ids = 0
    self_matches = 0
    match_header_valid = False
    total_match_lines = 0
    total_matched_entities = 0

    with open(match_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header == ["source1_entity_id", "matched_entity_ids"]:
            match_header_valid = True

        for line in f:
            total_match_lines += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]

            if not s1_id.startswith("S1-"):
                invalid_s1_match_ids += 1

            if s1_id in s1_match_ids:
                duplicate_s1_match += 1
            s1_match_ids.add(s1_id)

            if len(parts) < 2 or not parts[1].strip():
                empty_matches += 1
                s1_matched_counts.append(0)
            else:
                targets = parts[1].split(",")
                s1_matched_counts.append(len(targets))
                total_matched_entities += len(targets)
                for t in targets:
                    if not (t.startswith("S2-") or t.startswith("S3-")):
                        invalid_target_ids += 1
                    if t == s1_id:
                        self_matches += 1

    # 2. Check candidate_pairs.tsv and subset consistency
    logger.info("Auditing candidate_pairs.tsv and checking matches ⊆ candidates...")
    s1_cand_counts = []
    s1_cand_ids = set()
    duplicate_s1_cand = 0
    empty_cands = 0
    invalid_s1_cand_ids = 0
    invalid_cand_target_ids = 0
    cand_header_valid = False
    total_cand_lines = 0
    total_cand_pairs = 0
    max_cand_cap = 0
    subset_violations = 0
    sample_violations = []

    # Stream matching results into memory for subset check if feasible, or stream line by line
    # Since matching_results has 1.73M rows, let's load a dict of s1_id -> set of match targets
    logger.info("Loading matching targets for subset verification...")
    matches_dict = {}
    with open(match_file, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                matches_dict[parts[0]] = set(parts[1].split(","))
            else:
                matches_dict[parts[0]] = set()

    with open(cand_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header == ["source1_entity_id", "candidate_entity_ids"]:
            cand_header_valid = True

        for line in f:
            total_cand_lines += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]

            if not s1_id.startswith("S1-"):
                invalid_s1_cand_ids += 1

            if s1_id in s1_cand_ids:
                duplicate_s1_cand += 1
            s1_cand_ids.add(s1_id)

            if len(parts) < 2 or not parts[1].strip():
                empty_cands += 1
                cands_set = set()
                s1_cand_counts.append(0)
            else:
                cands = parts[1].split(",")
                cands_count = len(cands)
                s1_cand_counts.append(cands_count)
                total_cand_pairs += cands_count
                if cands_count > max_cand_cap:
                    max_cand_cap = cands_count
                cands_set = set(cands)
                for c in cands:
                    if not (c.startswith("S2-") or c.startswith("S3-")):
                        invalid_cand_target_ids += 1

            # Subset check: matches_dict[s1_id] ⊆ cands_set
            matched_set = matches_dict.get(s1_id, set())
            if not matched_set.issubset(cands_set):
                diff = matched_set - cands_set
                subset_violations += 1
                if len(sample_violations) < 5:
                    sample_violations.append((s1_id, list(diff)[:3]))

    # Parsimony statistics
    cand_counts_arr = np.array(s1_cand_counts, dtype=np.int32)
    mean_parsimony = float(np.mean(cand_counts_arr))
    median_parsimony = float(np.median(cand_counts_arr))
    p90_parsimony = float(np.percentile(cand_counts_arr, 90))
    p95_parsimony = float(np.percentile(cand_counts_arr, 95))
    p99_parsimony = float(np.percentile(cand_counts_arr, 99))

    return {
        "matching_results": {
            "file": str(match_file.name),
            "size_bytes": match_file.stat().st_size,
            "sha256": calculate_sha256(match_file),
            "total_lines": total_match_lines,
            "unique_s1": len(s1_match_ids),
            "duplicate_s1": duplicate_s1_match,
            "header_valid": match_header_valid,
            "empty_rows": empty_matches,
            "non_empty_rows": total_match_lines - empty_matches,
            "total_matches_emitted": total_matched_entities,
            "invalid_s1_ids": invalid_s1_match_ids,
            "invalid_target_ids": invalid_target_ids,
            "self_matches": self_matches,
        },
        "candidate_pairs": {
            "file": str(cand_file.name),
            "size_bytes": cand_file.stat().st_size,
            "sha256": calculate_sha256(cand_file),
            "total_lines": total_cand_lines,
            "unique_s1": len(s1_cand_ids),
            "duplicate_s1": duplicate_s1_cand,
            "header_valid": cand_header_valid,
            "empty_rows": empty_cands,
            "non_empty_rows": total_cand_lines - empty_cands,
            "total_candidate_pairs": total_cand_pairs,
            "invalid_s1_ids": invalid_s1_cand_ids,
            "invalid_target_ids": invalid_cand_target_ids,
            "max_cand_cap": max_cand_cap,
            "mean_parsimony": mean_parsimony,
            "median_parsimony": median_parsimony,
            "p90": p90_parsimony,
            "p95": p95_parsimony,
            "p99": p99_parsimony,
        },
        "consistency": {
            "subset_violations": subset_violations,
            "sample_violations": sample_violations,
            "s1_set_match": len(s1_match_ids.symmetric_difference(s1_cand_ids)) == 0,
        },
    }


def main():
    logger.info("Starting Full Pre-Submission Audit...")
    report = audit_files_and_consistency()

    # Also check zips
    venus_zip = PROJECT_ROOT / "VENUS_submission.zip"
    generic_zip = PROJECT_ROOT / "submission.zip"

    report["packages"] = {}
    for z in [venus_zip, generic_zip]:
        if z.exists():
            report["packages"][z.name] = {
                "size_bytes": z.stat().st_size,
                "size_mb": round(z.stat().st_size / (1024 * 1024), 2),
                "sha256": calculate_sha256(z),
            }

    # Run official validator
    logger.info("Executing official validator...")
    cmd = [
        sys.executable,
        str(VALIDATOR_PATH),
        "--matching",
        str(OUTPUT_DIR / "matching_results.tsv"),
        "--candidate",
        str(OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir",
        str(TEST_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    report["validator"] = {
        "returncode": res.returncode,
        "stdout": res.stdout.strip(),
        "stderr": res.stderr.strip(),
        "passed": res.returncode == 0,
    }

    report_path = PROJECT_ROOT / "reports" / "audit_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Audit metrics saved to %s", report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
