"""
Unit Tests for Local Evaluator & Measurement Harness
Amazon ML Challenge 2026 - Phase 2
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import EntityResolutionEvaluator, EvaluationResult, CandidateEvaluationResult
from student_resource.utils.validate_submission import validate


class TestEntityResolutionEvaluator(unittest.TestCase):
    """Test suite covering edge cases, metric dynamics, and official compliance."""

    def test_01_perfect_prediction_matches_ground_truth(self):
        """Test 1: Perfect prediction matches ground truth -> Macro F0.5 = 1.000000."""
        gt = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-30"},
            "S1-3": set(),  # singleton
        }
        pred = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-30"},
            "S1-3": set(),
        }
        res = EntityResolutionEvaluator.evaluate_predictions(pred, gt)
        self.assertAlmostEqual(res.macro_f05, 1.000000, places=6)
        self.assertAlmostEqual(res.macro_precision, 1.000000, places=6)
        self.assertAlmostEqual(res.macro_recall, 1.000000, places=6)
        self.assertAlmostEqual(res.singleton_accuracy, 1.000000, places=6)
        self.assertAlmostEqual(res.matched_f05, 1.000000, places=6)
        self.assertEqual(res.total_tp, 3)
        self.assertEqual(res.total_fp, 0)
        self.assertEqual(res.total_fn, 0)

    def test_02_completely_disjoint_predictions(self):
        """Test 2: Completely disjoint predictions -> Macro F0.5 = 0.000000."""
        gt = {
            "S1-1": {"S2-10"},
            "S1-2": {"S3-20"},
        }
        pred = {
            "S1-1": {"S2-99"},
            "S1-2": {"S3-99"},
        }
        res = EntityResolutionEvaluator.evaluate_predictions(pred, gt)
        self.assertAlmostEqual(res.macro_f05, 0.000000, places=6)
        self.assertAlmostEqual(res.macro_precision, 0.000000, places=6)
        self.assertAlmostEqual(res.macro_recall, 0.000000, places=6)
        self.assertEqual(res.total_tp, 0)
        self.assertEqual(res.total_fp, 2)
        self.assertEqual(res.total_fn, 2)

    def test_03_singleton_with_empty_prediction(self):
        """Test 3: Singleton with empty prediction -> F0.5 = 1.0."""
        gt = {"S1-1": set()}
        pred = {"S1-1": set()}
        res = EntityResolutionEvaluator.evaluate_predictions(pred, gt)
        self.assertAlmostEqual(res.macro_f05, 1.000000, places=6)
        self.assertAlmostEqual(res.singleton_accuracy, 1.000000, places=6)
        self.assertEqual(res.num_singletons, 1)
        self.assertEqual(res.num_matched, 0)

    def test_04_singleton_with_false_positive_prediction(self):
        """Test 4: Singleton with false positive prediction -> F0.5 = 0.0."""
        gt = {"S1-1": set()}
        pred = {"S1-1": {"S2-999"}}
        res = EntityResolutionEvaluator.evaluate_predictions(pred, gt)
        self.assertAlmostEqual(res.macro_f05, 0.000000, places=6)
        self.assertAlmostEqual(res.singleton_accuracy, 0.000000, places=6)
        self.assertEqual(res.total_fp, 1)

    def test_05_non_singleton_with_empty_prediction(self):
        """Test 5: Non-singleton with empty prediction -> F0.5 = 0.0."""
        gt = {"S1-1": {"S2-10", "S3-20"}}
        pred = {"S1-1": set()}
        res = EntityResolutionEvaluator.evaluate_predictions(pred, gt)
        self.assertAlmostEqual(res.macro_f05, 0.000000, places=6)
        self.assertEqual(res.total_fn, 2)
        self.assertEqual(res.total_tp, 0)

    def test_06_partial_precision_vs_recall_weighting(self):
        """
        Test 6: Demonstrate that under beta=0.5, Precision is penalized 4x heavier than Recall.
        
        High Precision (P=1.0, R=0.5):
          F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.833333
        
        Low Precision (P=0.5, R=1.0):
          F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.555556
        """
        # Case High Precision
        gt_high_p = {"S1-1": {"S2-10", "S2-20"}}
        pred_high_p = {"S1-1": {"S2-10"}}
        res_high_p = EntityResolutionEvaluator.evaluate_predictions(pred_high_p, gt_high_p)
        self.assertAlmostEqual(res_high_p.macro_f05, 5 / 6, places=6)

        # Case Low Precision
        gt_low_p = {"S1-1": {"S2-10"}}
        pred_low_p = {"S1-1": {"S2-10", "S2-20"}}
        res_low_p = EntityResolutionEvaluator.evaluate_predictions(pred_low_p, gt_low_p)
        self.assertAlmostEqual(res_low_p.macro_f05, 5 / 9, places=6)

        # Precision advantage verification
        self.assertGreater(res_high_p.macro_f05, res_low_p.macro_f05)
        self.assertAlmostEqual(res_high_p.macro_f05 - res_low_p.macro_f05, 0.8333333 - 0.5555555, places=5)

    def test_07_output_file_validation_against_official_validator(self):
        """Test 7: Generate synthetic TSVs and validate using student_resource/utils/validate_submission.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test"
            test_dir.mkdir()

            # Create synthetic test_source1.tsv
            source1_path = test_dir / "test_source1.tsv"
            with open(source1_path, "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S1-001\tAlpha Corp\t123 Main St\tUS\n")
                f.write("S1-002\tBeta Ltd\t456 Park Ave\tIndia\n")
                f.write("S1-003\tGamma Inc\t789 Oak Rd\tFrance\n")

            # Create synthetic matching_results.tsv
            matching_path = Path(tmpdir) / "matching_results.tsv"
            with open(matching_path, "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                f.write("S1-001\tS2-101,S3-201\n")
                f.write("S1-002\tS2-102\n")
                f.write("S1-003\t\n")  # singleton

            # Create synthetic candidate_pairs.tsv
            candidate_path = Path(tmpdir) / "candidate_pairs.tsv"
            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tcandidate_entity_ids\n")
                f.write("S1-001\tS2-101,S3-201,S2-999\n")
                f.write("S1-002\tS2-102,S3-999\n")
                f.write("S1-003\tS3-888\n")

            # Run official validate function
            errors, warnings = validate(
                str(matching_path),
                str(candidate_path),
                str(test_dir),
                check_ids=False,
            )

            # Assert zero formatting errors
            self.assertEqual(len(errors), 0, f"Validator returned errors: {errors}")

    def test_08_candidate_evaluation_metrics(self):
        """Test candidate generation metrics: Pairs Completeness, Reduction Ratio, Parsimony."""
        gt = {
            "S1-1": {"S2-1", "S3-1"},  # 2 matches
            "S1-2": {"S2-2"},           # 1 match
            "S1-3": set(),              # 0 match
        }
        # Candidates: S1-1 captures 1 of 2, S1-2 captures 1 of 1, S1-3 has 2 candidates
        candidates = {
            "S1-1": {"S2-1", "S2-99"},
            "S1-2": {"S2-2"},
            "S1-3": {"S3-50", "S3-51"},
        }
        res = EntityResolutionEvaluator.evaluate_candidates(
            candidates, gt, total_s2_count=100, total_s3_count=100
        )
        # Total true matches = 3, captured = 2 -> Pairs Completeness = 2/3 = 0.666667
        self.assertAlmostEqual(res.pairs_completeness, 2 / 3, places=6)
        # Total candidates = 2 + 1 + 2 = 5
        self.assertEqual(res.total_candidates, 5)
        # Total comparisons possible = 3 * 200 = 600
        # Reduction Ratio = 1 - 5 / 600 = 1 - 0.0083333 = 0.99166667
        self.assertAlmostEqual(res.reduction_ratio, 1.0 - 5 / 600, places=6)
        # Mean candidate size = 5 / 3 = 1.666667
        self.assertAlmostEqual(res.mean_candidate_size, 5 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
