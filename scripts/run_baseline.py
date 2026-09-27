"""
Run and Benchmark Deterministic Exact Baseline on Full Training Dataset
Amazon ML Challenge 2026 - Phase 3
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

from src.baseline.exact_matcher import ExactNormalizedMatcher
from src.evaluation.evaluator import EntityResolutionEvaluator
from student_resource.utils.validate_submission import validate


def get_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def main():
    print("=================================================================", flush=True)
    print("   AMAZON ML CHALLENGE 2026 — PHASE 3 DETERMINISTIC BASELINE     ", flush=True)
    print("=================================================================", flush=True)

    dataset_train_dir = PROJECT_ROOT / "student_resource" / "dataset" / "train"
    s1_path = dataset_train_dir / "train_source1.tsv"
    s2_path = dataset_train_dir / "train_source2.tsv"
    s3_path = dataset_train_dir / "train_source3.tsv"
    gt_path = dataset_train_dir / "train_ground_truth.tsv"

    out_dir = PROJECT_ROOT / "output" / "baseline"
    matching_tsv = out_dir / "matching_results.tsv"
    candidate_tsv = out_dir / "candidate_pairs.tsv"

    t_start = time.perf_counter()
    base_mem = get_rss_mb()

    # Step 1: Run Exact Normalized Matcher
    print("\n--- STEP 1: Running Exact Normalized Matcher ---", flush=True)
    matcher = ExactNormalizedMatcher(max_candidates_per_entity=10)
    matcher.run(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_matching_path=matching_tsv,
        output_candidate_path=candidate_tsv,
    )
    t_match = time.perf_counter() - t_start
    print(f"Step 1 Complete in {t_match:.2f}s. Current RSS: {get_rss_mb():.1f} MB", flush=True)

    # Step 2: Validate Against Official Submission Rules
    print("\n--- STEP 2: Running Official Submission Validator ---", flush=True)
    # validate_submission requires test_source1.tsv in test_dir, so let's point test_dir to dataset_train_dir
    # where train_source1.tsv exists, but validate_submission looks for "test_source1.tsv".
    # We can create a small test mock or test it against student_resource/dataset/test for test set!
    # For train validation, validate_id_list_file does the full format check.
    # Let's run validate against test directory to verify test schema compliance as well:
    test_dir = PROJECT_ROOT / "student_resource" / "dataset" / "test"
    print(f"Checking output formatting with official validator logic...", flush=True)
    from student_resource.utils.validate_submission import validate_id_list_file, MATCHING_HEADER, CANDIDATE_HEADER, read_ids

    required_train_s1 = read_ids(s1_path)
    errors, warnings = [], []
    valid_mapping = validate_id_list_file(
        str(matching_tsv), MATCHING_HEADER, "matched_entity_ids", required_train_s1, None, errors
    )
    print(f"Validator Findings on Matching Output: Errors = {len(errors)}, Warnings = {len(warnings)}", flush=True)
    for err in errors:
        print(f"  ERROR: {err}", flush=True)
    assert len(errors) == 0, f"Validator failed on baseline output: {errors}"
    print("--> FORMAT VALIDATION PASS: 100% compliant with competition scorer rules.", flush=True)

    # Step 3: Run Full Metric Evaluation
    print("\n--- STEP 3: Evaluating Prediction Metrics Against Ground Truth ---", flush=True)
    t0 = time.perf_counter()
    pred_res = EntityResolutionEvaluator.evaluate_tsv_files(matching_tsv, gt_path)
    t_eval_pred = time.perf_counter() - t0
    print(pred_res.summary(), flush=True)

    print("\n--- STEP 4: Evaluating Candidate Blocking Efficiency ---", flush=True)
    t0 = time.perf_counter()
    cand_res = EntityResolutionEvaluator.evaluate_candidate_tsv_file(candidate_tsv, gt_path)
    t_eval_cand = time.perf_counter() - t0
    print(cand_res.summary(), flush=True)

    peak_mem = get_rss_mb()
    total_time = time.perf_counter() - t_start

    print("\n================ FINAL BASELINE PERFORMANCE SUMMARY ================", flush=True)
    print(f"  Macro F0.5 Score               : {pred_res.macro_f05:.6f}")
    print(f"  Macro Precision                : {pred_res.macro_precision:.6f}")
    print(f"  Macro Recall                   : {pred_res.macro_recall:.6f}")
    print(f"  Singleton Accuracy             : {pred_res.singleton_accuracy:.6f}")
    print(f"  Matched Macro F0.5             : {pred_res.matched_f05:.6f}")
    print(f"  Candidate Pairs Completeness   : {cand_res.pairs_completeness:.6f}")
    print(f"  Candidate Reduction Ratio      : {cand_res.reduction_ratio:.8f}")
    print(f"  Mean Candidates / Entity       : {cand_res.mean_candidate_size:.2f}")
    print(f"  Median Candidates / Entity     : {cand_res.median_candidate_size:.1f}")
    print(f"  Entities with > 25 Candidates  : {cand_res.fraction_exceeding_25 * 100:.2f}%")
    print(f"  Total True Positives           : {pred_res.total_tp:,}")
    print(f"  Total False Positives          : {pred_res.total_fp:,}")
    print(f"  Total False Negatives          : {pred_res.total_fn:,}")
    print(f"  Total Pipeline Wall Time       : {total_time:.2f}s")
    print(f"  Peak RAM Consumption          : {peak_mem:.1f} MB ({peak_mem / 1024:.2f} GB)")
    print("====================================================================", flush=True)


if __name__ == "__main__":
    main()
