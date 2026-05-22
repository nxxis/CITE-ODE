#!/usr/bin/env python3
"""Fetch the evaluation cohort from Google Drive into `data/`.

Usage:
  python scripts/fetch_cohort.py --dest data/ --folder-url <DRIVE_FOLDER_URL>

This script uses `gdown` to download the drive folder. It will place the
expected file at `data/mimic_cemr_cohort.csv`.
"""
import argparse
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure a simple logger for CLI feedback
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        logging.error("gdown is not installed. Install with: pip install gdown")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download cohort from Google Drive")
    parser.add_argument("--dest", default="data/", help="Destination folder (default: data/)")
    parser.add_argument(
        "--folder-url",
        default="https://drive.google.com/drive/u/2/folders/1oupz5CcQIMn-16I8KFWeqlpirY0vBCxg",
        help="Google Drive folder URL containing the cohort (default: provided link)",
    )
    args = parser.parse_args()

    ensure_gdown()
    import gdown

    dest = os.path.expanduser(args.dest)
    os.makedirs(dest, exist_ok=True)

    logging.info("Downloading Drive folder %s into %s ...", args.folder_url, dest)
    try:
        gdown.download_folder(args.folder_url, output=dest, quiet=False, use_cookies=False)
    except Exception:
        logging.exception("gdown failed to download the folder")
        logging.error("You can also download manually from the Drive link and place the file at data/mimic_cemr_cohort.csv")
        sys.exit(1)

    expected = os.path.join(dest, "mimic_cemr_cohort.csv")
    if os.path.exists(expected):
        logging.info("Success: cohort downloaded to %s", expected)
    else:
        logging.warning(
            "Download finished, but the expected cohort file was not found: %s", expected
        )
        logging.info("Verify the Drive folder contents and place the CSV in data/.")


if __name__ == "__main__":
    main()
