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


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except Exception:
        print("gdown is not installed. Install with: pip install gdown")
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

    print(f"Downloading Drive folder {args.folder_url} into {dest} ...")
    try:
        # gdown supports folder download via `download_folder`
        gdown.download_folder(args.folder_url, output=dest, quiet=False, use_cookies=False)
    except Exception as e:
        print("gdown failed to download the folder:", e)
        print("You can also download manually from the Drive link and place the file at data/mimic_cemr_cohort.csv")
        sys.exit(1)

    expected = os.path.join(dest, "mimic_cemr_cohort.csv")
    if os.path.exists(expected):
        print(f"Success: cohort downloaded to {expected}")
    else:
        print("Download finished but expected file not found:", expected)
        print("Check the Drive folder contents and move the cohort CSV into the data/ folder.")


if __name__ == "__main__":
    main()
