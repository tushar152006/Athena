#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Clean Environment Verification Runner
Creates an isolated Python virtual environment, installs only requirements.txt,
and executes end-to-end component verification to certify reproduction from a clean state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "scratch" / "clean_env_verification"
VENV_DIR = CLEAN_DIR / ".clean_venv"


def main():
    print("=" * 80)
    print("STARTING CLEAN-ENVIRONMENT VERIFICATION AUDIT")
    print("=" * 80)

    # Clean previous run
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR, ignore_errors=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Copy code/business_entity_resolution into clean sandbox
    staging_code = CLEAN_DIR / "business_entity_resolution"
    shutil.copytree(PROJECT_ROOT / "code" / "business_entity_resolution", staging_code)
    print(f"Staged codebase in: {staging_code}")

    # Step 1: Create isolated virtual environment
    t0 = time.perf_counter()
    print("\n[1/4] Creating fresh isolated virtual environment (.clean_venv)...")
    res = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], capture_output=True, text=True)
    assert res.returncode == 0, f"Venv creation failed: {res.stderr}"
    print(f"Isolated venv created in {time.perf_counter() - t0:.2f} seconds.")

    # Determine paths inside venv
    if os.name == "nt":
        venv_python = VENV_DIR / "Scripts" / "python.exe"
        venv_pip = VENV_DIR / "Scripts" / "pip.exe"
    else:
        venv_python = VENV_DIR / "bin" / "python"
        venv_pip = VENV_DIR / "bin" / "pip"

    # Step 2: Install pinned requirements using uv pip install into the isolated venv
    t1 = time.perf_counter()
    req_file = staging_code / "requirements.txt"
    print(f"\n[2/4] Installing dependencies from {req_file.name} into clean environment via uv...")
    cmd_install = ["uv", "pip", "install", "--python", str(venv_python), "-r", str(req_file)]
    res_install = subprocess.run(cmd_install, capture_output=True, text=True)
    if res_install.returncode != 0:
        print("Install error:\n", res_install.stderr)
        sys.exit(1)
    install_time = time.perf_counter() - t1
    print(f"Dependencies installed in {install_time:.2f} seconds.")

    # Step 3: Run package list audit
    print("\n[3/4] Auditing installed packages in clean environment:")
    res_freeze = subprocess.run([str(venv_pip), "list"], capture_output=True, text=True)
    print(res_freeze.stdout.strip())

    # Step 4: Execute standalone verification script inside clean environment
    print("\n[4/4] Running standalone inference and pipeline test inside clean environment...")
    test_code = """
import sys
import time
from pathlib import Path
import numpy as np

# Verify imports
import polars as pl
import rapidfuzz
import lightgbm as lgb
import scipy
import sklearn

print(f"Python version:     {sys.version.split()[0]}")
print(f"Polars version:     {pl.__version__}")
print(f"RapidFuzz version:  {rapidfuzz.__version__}")
print(f"LightGBM version:   {lgb.__version__}")
print(f"SciPy version:      {scipy.__version__}")
print(f"Scikit-learn:       {sklearn.__version__}")

# Verify src imports
from src.features.feature_extractor import PairwiseFeatureExtractor
from src.models.pairwise_classifier import LightGBMPairwiseClassifier
from src.clustering.graph_clusterer import BipartiteGraphClusterer, ClustererConfig
from src.blocking.multi_pass_blocker import MultiPassBlocker, BlockingConfig

print("Successfully imported all core modules from src/!")

# Verify model loading
model_path = Path("models/lightgbm_pairwise.txt")
assert model_path.exists(), f"Missing {model_path}"
classifier = LightGBMPairwiseClassifier.load(model_path)
print(f"Successfully loaded LightGBM model from {model_path}!")

# Test feature extraction on sample pair
e1_name = "Amazon Web Services Inc"
e1_addr = "410 Terry Ave N, Seattle, WA 98109"
e2_name = "Amazon Web Services LLC"
e2_addr = "410 Terry Avenue North, Seattle, 98109"

feat = PairwiseFeatureExtractor.extract_pair_features(
    e1_name, e1_addr, e2_name, e2_addr, rank=1.0, consensus=4.0
)
assert len(feat) == 20, f"Expected 20 features, got {len(feat)}"
print(f"Feature extraction test passed (extracted {len(feat)} features).")

# Test model scoring
score = classifier.predict_proba(np.array([feat], dtype=np.float32))[0]
print(f"Model prediction test passed: P(match) = {score:.4f} (Match: {score >= 0.60})")

# Test graph clusterer
clusterer = BipartiteGraphClusterer(ClustererConfig(match_threshold=0.60))
matches, metrics = clusterer.cluster_pairs(
    s1_ids=["S1-00001"],
    cand_ids=["S2-00001"],
    scores=[score],
    all_s1_universe={"S1-00001"}
)
print(f"Clusterer test passed: Assigned matches = {matches}")
print("ALL CLEAN-ENVIRONMENT CHECKS PASSED WITH ZERO ERRORS!")
"""
    test_script_path = staging_code / "test_clean_env.py"
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(test_code)

    t2 = time.perf_counter()
    res_test = subprocess.run([str(venv_python), str(test_script_path)], cwd=staging_code, capture_output=True, text=True)
    test_time = time.perf_counter() - t2

    print(res_test.stdout.strip())
    if res_test.returncode != 0:
        print("Verification script failed:\n", res_test.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("CLEAN-ENVIRONMENT VERIFICATION CERTIFIED: 100% PASS (Exit Code 0)")
    print(f"Total Verification Wall Time: {time.perf_counter() - t0:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
