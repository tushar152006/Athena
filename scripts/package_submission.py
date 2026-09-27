#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Submission Packaging Script
Stage V - Pipeline Integration & Submission Ready (Phase 8)

Creates the official submission.zip containing:
1. output/matching_results.tsv
2. output/candidate_pairs.tsv
3. code/business_entity_resolution/
   ├── src/
   ├── scripts/
   ├── models/
   ├── README.md
   └── requirements.txt
4. Documentation_template.md
"""

import hashlib
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PackageSubmission")


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def package_submission(project_root: Path) -> Path:
    """Package submission zip file."""
    logger.info("Starting Submission Packaging...")

    output_dir = project_root / "output"
    match_file = output_dir / "matching_results.tsv"
    cand_file = output_dir / "candidate_pairs.tsv"
    doc_file = project_root / "Documentation_template.md"
    zip_dest = project_root / "submission.zip"

    # Verify input files
    for p in [match_file, cand_file, doc_file]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    logger.info("All primary deliverables verified.")

    # Prepare temporary staging directory for code
    staging_code = project_root / "code" / "business_entity_resolution"
    if staging_code.exists():
        shutil.rmtree(staging_code)
    staging_code.mkdir(parents=True, exist_ok=True)

    # Copy src
    shutil.copytree(project_root / "src", staging_code / "src")
    
    # Copy scripts
    shutil.copytree(project_root / "scripts", staging_code / "scripts")

    # Copy models
    shutil.copytree(project_root / "models", staging_code / "models")

    # Write requirements.txt
    reqs_content = (
        "polars>=1.20.0\n"
        "numpy>=1.24.0\n"
        "scipy>=1.10.0\n"
        "scikit-learn>=1.3.0\n"
        "rapidfuzz>=3.8.0\n"
        "lightgbm>=4.0.0\n"
        "psutil>=5.9.0\n"
    )
    with open(staging_code / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(reqs_content)

    # Write README.md for code
    readme_content = (
        "# Amazon ML Challenge 2026: Business Entity Resolution\n\n"
        "## Team VENUS\n"
        "**Team Members:** Tushar Burla, Bipanchi Kalita, Arpit Gupta\n\n"
        "### Overview\n"
        "This package contains the complete reproducible source code for the entity resolution pipeline.\n\n"
        "### Pipeline Architecture\n"
        "1. **Multi-Pass Blocker (`src/blocking/multi_pass_blocker.py`)**: Vectorized 4-pass candidate generation in Polars.\n"
        "2. **Pairwise Feature Extractor (`src/features/feature_extractor.py`)**: 20 domain-specific features using RapidFuzz C++ engine.\n"
        "3. **Pairwise Classifier (`src/models/pairwise_classifier.py`)**: LightGBM model calibrated for Macro F0.5.\n"
        "4. **Global Graph Clusterer (`src/clustering/graph_clusterer.py`)**: 1-to-many bipartite graph clustering with singleton guard.\n"
        "5. **End-to-End Pipeline (`src/pipeline/inference_pipeline.py`)**: Complete batch inference pipeline.\n\n"
        "### Quickstart / Reproduction\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python scripts/run_inference.py\n"
        "```\n"
    )
    with open(staging_code / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Staged code artefacts in %s", staging_code)

    # Build zip file
    if zip_dest.exists():
        zip_dest.unlink()

    logger.info("Assembling submission.zip...")
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. output/ files
        z.write(match_file, arcname="output/matching_results.tsv")
        z.write(cand_file, arcname="output/candidate_pairs.tsv")

        # 2. Documentation_template.md
        z.write(doc_file, arcname="Documentation_template.md")

        # 3. code/ files
        for root, dirs, files in os.walk(staging_code):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(project_root)
                z.write(full_path, arcname=str(rel_path).replace("\\", "/"))

    zip_size_mb = zip_dest.stat().st_size / (1024 * 1024)
    logger.info("Created %s (%.2f MB)", zip_dest, zip_size_mb)

    # Checksums
    logger.info("Computing Checksums:")
    logger.info("  matching_results.tsv SHA-256: %s", calculate_sha256(match_file))
    logger.info("  candidate_pairs.tsv  SHA-256: %s", calculate_sha256(cand_file))
    logger.info("  submission.zip       SHA-256: %s", calculate_sha256(zip_dest))

    # Verify contents
    with zipfile.ZipFile(zip_dest, "r") as z:
        namelist = z.namelist()
        logger.info("ZIP verified: %d files archived.", len(namelist))
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist
        assert "Documentation_template.md" in namelist

    logger.info("Submission packaging successfully completed!")
    return zip_dest


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    package_submission(root)
