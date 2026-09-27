"""
Amazon ML Challenge 2026 - Evaluation Package.
Provides high-performance, competition-compliant metrics for:
- Macro F0.5 per Source 1 entity (Official competition metric)
- Candidate Generation / Blocking Metrics (Pairs Completeness, Reduction Ratio, Parsimony)
"""

from src.evaluation.evaluator import EntityResolutionEvaluator, EvaluationResult, CandidateEvaluationResult

__all__ = ["EntityResolutionEvaluator", "EvaluationResult", "CandidateEvaluationResult"]
