"""
Benchmark Local Evaluator on Full Train Ground Truth (2,206,821 Entities)
Amazon ML Challenge 2026 - Phase 2
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

from src.evaluation.evaluator import EntityResolutionEvaluator


def get_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_benchmark():
    gt_path = PROJECT_ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
    print(f"[Benchmark] Evaluating file: {gt_path}", flush=True)
    assert gt_path.is_file(), f"Ground truth file missing: {gt_path}"

    base_mem = get_rss_mb()
    print(f"[Benchmark] Initial Process RSS: {base_mem:.1f} MB", flush=True)
    t_start = time.perf_counter()

    # Step 1: Load Ground Truth Mapping
    print("\n[Step 1] Loading 2.2M entity ground truth into memory...", flush=True)
    t0 = time.perf_counter()
    gt_mapping = EntityResolutionEvaluator.load_id_mapping_tsv(gt_path)
    t_load = time.perf_counter() - t0
    num_entities = len(gt_mapping)
    mem_after_load = get_rss_mb()
    print(f"  Loaded {num_entities:,} entities in {t_load:.2f}s", flush=True)
    print(f"  RAM after load: {mem_after_load:.1f} MB (Delta: +{mem_after_load - base_mem:.1f} MB)", flush=True)

    # Step 2: Self-Evaluation (Perfect Match Verification)
    print("\n[Step 2] Running full self-evaluation (Ground Truth vs Ground Truth)...", flush=True)
    t0 = time.perf_counter()
    result_perfect = EntityResolutionEvaluator.evaluate_predictions(gt_mapping, gt_mapping)
    t_eval_perfect = time.perf_counter() - t0
    print(result_perfect.summary(), flush=True)

    assert abs(result_perfect.macro_f05 - 1.0) < 1e-6, f"Expected 1.000000, got {result_perfect.macro_f05}"
    assert abs(result_perfect.macro_precision - 1.0) < 1e-6
    assert abs(result_perfect.macro_recall - 1.0) < 1e-6
    assert abs(result_perfect.singleton_accuracy - 1.0) < 1e-6
    assert abs(result_perfect.matched_f05 - 1.0) < 1e-6
    print("  --> Self-evaluation PASS: Exactly 1.000000 across all metrics.", flush=True)

    # Step 3: Candidate Evaluation Verification (Pairs Completeness on Self)
    print("\n[Step 3] Running full candidate blocking evaluation...", flush=True)
    t0 = time.perf_counter()
    cand_result = EntityResolutionEvaluator.evaluate_candidates(
        gt_mapping, gt_mapping, total_s2_count=5034616, total_s3_count=5285603
    )
    t_eval_cand = time.perf_counter() - t0
    print(cand_result.summary(), flush=True)

    assert abs(cand_result.pairs_completeness - 1.0) < 1e-6
    print("  --> Candidate evaluation PASS: Exactly 100% pairs completeness.", flush=True)

    # Step 4: Noisy / Simulated Prediction Evaluation
    print("\n[Step 4] Running noisy simulation...", flush=True)
    t0 = time.perf_counter()
    # Simulate realistic prediction: empty for singletons, drop 1 match if multiple
    simulated_pred = {
        s1: (set(list(true_set)[:-1]) if len(true_set) > 1 else true_set)
        for s1, true_set in gt_mapping.items()
    }
    t_prep_sim = time.perf_counter() - t0

    t0 = time.perf_counter()
    noisy_result = EntityResolutionEvaluator.evaluate_predictions(simulated_pred, gt_mapping)
    t_noisy = time.perf_counter() - t0
    print(noisy_result.summary(), flush=True)

    peak_mem = get_rss_mb()
    total_time = time.perf_counter() - t_start

    print("\n================ BENCHMARK FINAL METRICS ================", flush=True)
    print(f"Total Benchmark Execution Time      : {total_time:.2f}s", flush=True)
    print(f"File Load Time (2,206,821 entities) : {t_load:.2f}s", flush=True)
    print(f"Prediction Eval (2.2M entities)     : {t_eval_perfect:.2f}s", flush=True)
    print(f"Candidate Eval (2.2M entities)      : {t_eval_cand:.2f}s", flush=True)
    print(f"Noisy Eval (2.2M entities)          : {t_noisy:.2f}s", flush=True)
    print(f"Peak Process RSS                    : {peak_mem:.1f} MB ({peak_mem / 1024:.2f} GB)", flush=True)
    print("=========================================================", flush=True)


if __name__ == "__main__":
    run_benchmark()
