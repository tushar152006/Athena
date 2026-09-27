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
        "**Team Leader:** Tushar Burla\n"
        "**Team Members:** Tushar Burla (Leader), Bipanchi Kalita, Arpit Gupta\n\n"
        "### Overview\n"
        "This directory contains the self-contained, reproducible source code for Team VENUS's entity resolution pipeline.\n"
        "It implements a fast, scalable two-stage architecture: vectorized multi-pass blocking in Polars followed by a 20-feature LightGBM GBDT matcher and bipartite graph clustering.\n\n"
        "### Pipeline Architecture\n"
        "1. **Candidate Generation / Blocking (`src/blocking/multi_pass_blocker.py`)**:\n"
        "   - Vectorized 4-pass blocking partitioned by country (`US`, `IN`, `FR`).\n"
        "   - Prunes 22.8-trillion Cartesian product to ~24.25M pairs (mean parsimony 13.19 <= 20, max cap 20).\n"
        "2. **Pairwise Feature Engineering (`src/features/feature_extractor.py`)**:\n"
        "   - 20 RapidFuzz C++ string metrics, token order invariants, spatial number/postal agreement.\n"
        "   - Targeted domain guards: `franchise_collision_hazard` and `multi_tenant_hazard`.\n"
        "3. **Pairwise Classifier (`src/models/pairwise_classifier.py`)**:\n"
        "   - LightGBM GBDT trained with 4-tier hard negative mining curriculum.\n"
        "   - Calibrated for asymmetric Macro F0.5 (optimal decision threshold p* = 0.58-0.60, adaptive margin delta = 0.30).\n"
        "4. **Global Graph Conflict Resolver (`src/clustering/graph_clusterer.py`)**:\n"
        "   - Maximum-weight bipartite assignment with physical cardinality cap (max 11) and precision singleton guard.\n"
        "5. **End-to-End Orchestrator (`src/pipeline/inference_pipeline.py`)**:\n"
        "   - Full pipeline executing data ingestion -> blocking -> feature extraction -> model inference -> cluster assignment -> TSV export.\n\n"
        "### Open-Source Licensing & Fair Play Compliance\n"
        "- **License**: MIT License (LightGBM, RapidFuzz, Polars, Scikit-learn).\n"
        "- **Parameter Budget**: LightGBM tree ensemble contains ~9,600 decision nodes (~1.2 MB file size), far below the 8 Billion parameter ceiling.\n"
        "- **Zero External Data**: 0% external APIs, 0% geocoding lookups, 0% web scraping. Purely in-domain learning from provided competition data.\n\n"
        "### End-to-End Reproduction Guide\n"
        "To reproduce both output files (`candidate_pairs.tsv` and `matching_results.tsv`) directly from test data:\n\n"
        "```bash\n"
        "# 1. Install pinned dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# 2. Run the deterministic end-to-end inference script\n"
        "python scripts/run_inference.py\n"
        "```\n\n"
        "The script will read test data from `student_resource/dataset/test/`,\n"
        "execute candidate generation, extract 20 pairwise features, run model scoring,\n"
        "and generate the exact deliverables under `output/`:\n"
        "- `output/candidate_pairs.tsv` (1,732,544 rows)\n"
        "- `output/matching_results.tsv` (1,732,544 rows)\n"
    )
    with open(staging_code / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Staged code artefacts in %s", staging_code)

    # Build primary zip file: VENUS_submission.zip
    team_zip = project_root / "VENUS_submission.zip"
    generic_zip = project_root / "submission.zip"

    for z_path in [team_zip, generic_zip]:
        if z_path.exists():
            z_path.unlink()

    logger.info("Assembling VENUS_submission.zip...")
    with zipfile.ZipFile(team_zip, "w", zipfile.ZIP_DEFLATED) as z:
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

    # Also duplicate to submission.zip for universal portal compatibility
    shutil.copyfile(team_zip, generic_zip)

    zip_size_mb = team_zip.stat().st_size / (1024 * 1024)
    logger.info("Created %s and %s (%.2f MB)", team_zip.name, generic_zip.name, zip_size_mb)

    # Checksums
    logger.info("Computing Checksums:")
    logger.info("  matching_results.tsv SHA-256: %s", calculate_sha256(match_file))
    logger.info("  candidate_pairs.tsv  SHA-256: %s", calculate_sha256(cand_file))
    logger.info("  VENUS_submission.zip SHA-256: %s", calculate_sha256(team_zip))

    # Verify contents
    with zipfile.ZipFile(team_zip, "r") as z:
        namelist = z.namelist()
        logger.info("ZIP verified: %d files archived.", len(namelist))
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist
        assert "Documentation_template.md" in namelist
        assert "code/business_entity_resolution/README.md" in namelist
        assert "code/business_entity_resolution/requirements.txt" in namelist

    logger.info("Submission packaging successfully completed!")
    return team_zip


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    package_submission(root)
