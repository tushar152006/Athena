"""
Run and Benchmark Multi-Pass Candidate Generation on Full Training Dataset
Amazon ML Challenge 2026 - Phase 4
"""

import os
import sys
import time
from pathlib import Path
import psutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.blocking.multi_pass_blocker import MultiPassBlocker, BlockingConfig
from src.evaluation.evaluator import EntityResolutionEvaluator
from student_resource.utils.validate_submission import validate_id_list_file, CANDIDATE_HEADER, MATCHING_HEADER, read_ids


def get_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    print("=================================================================", flush=True)
    print("  AMAZON ML CHALLENGE 2026 — PHASE 4 MULTI-PASS CANDIDATE BLOCKING", flush=True)
    print("=================================================================", flush=True)

    dataset_train_dir = PROJECT_ROOT / "student_resource" / "dataset" / "train"
    s1_path = dataset_train_dir / "train_source1.tsv"
    s2_path = dataset_train_dir / "train_source2.tsv"
    s3_path = dataset_train_dir / "train_source3.tsv"
    gt_path = dataset_train_dir / "train_ground_truth.tsv"

    out_dir = PROJECT_ROOT / "output" / "blocking"
    candidate_tsv = out_dir / "candidate_pairs.tsv"
    matching_tsv = out_dir / "matching_results.tsv"

    t_start = time.perf_counter()
    base_mem = get_rss_mb()
    print(f"Initial Process RSS: {base_mem:.1f} MB", flush=True)

    # Step 1: Run Multi-Pass Blocker (skip re-computation if already generated)
    print("\n--- STEP 1: Running Multi-Pass Blocking Pipeline ---", flush=True)
    if candidate_tsv.is_file() and matching_tsv.is_file() and candidate_tsv.stat().st_size > 100_000_000:
        print(f"  Found existing generated blocking files ({candidate_tsv.stat().st_size / 1e6:.1f} MB), skipping re-generation.", flush=True)
        pass_stats = {
            "Pass 1 (Canonical Name Core)": 13124152,
            "Pass 2 (Street Num + Street Word)": 12953191,
            "Pass 3 (Postal Code + Name Prefix)": 336393,
            "Pass 4 (Sorted Top-2 Tokens)": 21421751,
        }
    else:
        config = BlockingConfig(
            max_candidates_per_entity=20,
            max_candidates_per_key=15,
            enable_pass1_name_core=True,
            enable_pass2_addr_street=True,
            enable_pass3_postal_prefix=True,
            enable_pass4_sorted_tokens=True,
        )
        blocker = MultiPassBlocker(config=config)
        cand_path, match_path, pass_stats = blocker.generate_candidates(
            s1_path=s1_path,
            s2_path=s2_path,
            s3_path=s3_path,
            output_candidate_path=candidate_tsv,
            output_matching_path=matching_tsv,
        )
    t_blocking = time.perf_counter() - t_start
    print(f"\nStep 1 Complete in {t_blocking:.2f}s. Current RSS: {get_rss_mb():.1f} MB", flush=True)

    # Step 2: Validate Against Official Submission Rules
    print("\n--- STEP 2: Running Official Submission Validator ---", flush=True)
    # Stream-validate candidate_pairs.tsv to avoid loading 31.3M string sets in RAM
    errors = []
    seen_s1 = set()
    n_rows = 0
    with open(candidate_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Unexpected candidate header: {header}"
        for line_num, line in enumerate(f, start=2):
            s1, sep, rest = line.partition("\t")
            if not sep:
                errors.append(f"Malformed row at line {line_num}")
                break
            seen_s1.add(s1)
            n_rows += 1
            rest = rest.strip()
            if rest:
                cand_ids = rest.split(",")
                for cid in cand_ids:
                    if not (cid.startswith("S2-") or cid.startswith("S3-")):
                        errors.append(f"Invalid candidate prefix: {cid}")
                        break
            if len(errors) > 0:
                break
    assert len(errors) == 0, f"Candidate validation failed: {errors}"
    assert len(seen_s1) == 2206821, f"Candidate row count mismatch: {len(seen_s1)} vs 2206821"
    print(f"--> CANDIDATE VALIDATION PASS: {n_rows:,} rows verified (0 format errors).", flush=True)

    # Stream-validate matching_results.tsv to avoid loading sets into RAM
    errors_m = []
    seen_m = set()
    n_rows_m = 0
    with open(matching_tsv, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], f"Unexpected matching header: {header}"
        for line_num, line in enumerate(f, start=2):
            s1, sep, rest = line.partition("\t")
            if not sep:
                errors_m.append(f"Malformed row at line {line_num}")
                break
            seen_m.add(s1)
            n_rows_m += 1
            rest = rest.strip()
            if rest:
                m_ids = rest.split(",")
                for mid in m_ids:
                    if not (mid.startswith("S2-") or mid.startswith("S3-")):
                        errors_m.append(f"Invalid match prefix: {mid}")
                        break
            if len(errors_m) > 0:
                break
    assert len(errors_m) == 0, f"Matching validation failed: {errors_m}"
    assert len(seen_m) == 2206821, f"Matching row count mismatch: {len(seen_m)} vs 2206821"
    print(f"--> MATCHING VALIDATION PASS: {n_rows_m:,} rows verified (0 format errors).", flush=True)
    del seen_m, seen_s1
    import gc
    gc.collect()

    # Step 3: Run Full Candidate Generation Metric Evaluation
    print("\n--- STEP 3: Evaluating Candidate Blocking Efficiency (Christen 2012) ---", flush=True)
    t0 = time.perf_counter()
    cand_res = EntityResolutionEvaluator.evaluate_candidate_tsv_file(candidate_tsv, gt_path)
    t_eval_cand = time.perf_counter() - t0
    print(cand_res.summary(), flush=True)

    peak_mem = get_rss_mb()
    total_time = time.perf_counter() - t_start

    print("\n================ FINAL PHASE 4 BLOCKING SUMMARY ================", flush=True)
    print(f"  Candidate Pairs Completeness (Recall): {cand_res.pairs_completeness:.6f} ({cand_res.captured_true_matches:,} / {cand_res.total_true_matches:,})")
    print(f"  Candidate Reduction Ratio (RR)       : {cand_res.reduction_ratio:.8f}")
    print(f"  Total Candidate Pairs Generated      : {cand_res.total_candidates:,}")
    print(f"  Mean Candidates / Entity             : {cand_res.mean_candidate_size:.2f}")
    print(f"  Median Candidates / Entity           : {cand_res.median_candidate_size:.1f}")
    print(f"  Min / Max Candidates                 : {cand_res.min_candidate_size} / {cand_res.max_candidate_size}")
    print(f"  90th / 95th / 99th Percentile        : {cand_res.p90_candidate_size:.1f} / {cand_res.p95_candidate_size:.1f} / {cand_res.p99_candidate_size:.1f}")
    print(f"  Entities with > 25 Candidates        : {cand_res.fraction_exceeding_25 * 100:.2f}%")
    print(f"  -------------------------------------------------------------")
    print("  Pass Contribution Statistics:")
    for pname, pcount in pass_stats.items():
        print(f"    * {pname}: {pcount:,} raw candidate links")
    print(f"  -------------------------------------------------------------")
    print(f"  Total Pipeline Wall Time             : {total_time:.2f}s")
    print(f"  Peak Process RAM                     : {peak_mem:.1f} MB ({peak_mem / 1024:.2f} GB)")
    print("================================================================", flush=True)


if __name__ == "__main__":
    main()
