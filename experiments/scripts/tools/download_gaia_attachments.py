#!/usr/bin/env python3
"""
Download GAIA benchmark attachment files from HuggingFace.

The GAIA dataset stores text fields (including file_name / file_path) in a
parquet file, while the actual attachment files (Excel, PDF, images, etc.) live
in the Git LFS store of the HuggingFace repository and must be downloaded
separately via snapshot_download.

After running this script, attachments are available under:
    experiments/data/GAIA/attachments/<file_path>
where <file_path> is the value of sample_info.file_path in the dataset JSON
(e.g. 2023/validation/32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx).

Usage:
    python experiments/scripts/download_gaia_attachments.py

    # Custom output directory
    python experiments/scripts/download_gaia_attachments.py \
        --output-dir experiments/data/GAIA/attachments

    # Dry-run: list files that would be downloaded without fetching them
    python experiments/scripts/download_gaia_attachments.py --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ID = "gaia-benchmark/GAIA"
DEFAULT_OUTPUT_DIR = "experiments/data/GAIA/attachments"
DATA_JSON = "experiments/data/GAIA/validation-all-samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GAIA attachment files from HuggingFace")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store downloaded attachments (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be downloaded without actually downloading",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token for private/gated repos (or set HF_TOKEN env var)",
    )
    return parser.parse_args()


def collect_file_paths(data_json: str) -> list:
    """Return sorted list of unique file_path values that have a non-empty file_name."""
    with open(data_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    paths = sorted({
        s["sample_info"]["file_path"]
        for s in samples
        if s["sample_info"].get("file_name") and s["sample_info"].get("file_path")
    })
    return paths


def download_attachments(output_dir: str, hf_token: str = None, dry_run: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    file_paths = collect_file_paths(DATA_JSON)
    logger.info(f"Found {len(file_paths)} unique attachment(s) in {DATA_JSON}")

    if dry_run:
        logger.info("Dry-run mode — files that would be downloaded:")
        for p in file_paths:
            logger.info(f"  {p}")
        return

    os.makedirs(output_dir, exist_ok=True)
    token = hf_token or os.getenv("HF_TOKEN")

    logger.info(f"Downloading {len(file_paths)} attachment(s) to: {output_dir}")
    logger.info("This uses snapshot_download which fetches the full repository via Git LFS.")
    logger.info("Alternatively you can run: git lfs pull inside a cloned copy of the repo.")

    # snapshot_download fetches the entire repo; we then copy only the attachment files.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Fetching repository snapshot into temporary directory...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=tmp_dir,
            token=token,
            ignore_patterns=["*.parquet", "*.json", "README*"],
        )

        copied = 0
        missing = 0
        for file_path in file_paths:
            src = os.path.join(tmp_dir, file_path)
            dst = os.path.join(output_dir, file_path)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                logger.debug(f"Copied: {file_path}")
                copied += 1
            else:
                logger.warning(f"Not found in snapshot: {file_path}")
                missing += 1

    logger.info(f"Done. {copied} file(s) copied to {output_dir}, {missing} missing.")


def main():
    args = parse_args()

    if not os.path.exists(DATA_JSON):
        logger.error(f"Dataset JSON not found: {DATA_JSON}")
        logger.error("Make sure the GAIA JSON has been placed in experiments/data/GAIA/ first.")
        sys.exit(1)

    download_attachments(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
