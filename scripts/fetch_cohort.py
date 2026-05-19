"""Download the frozen clinical cohort from Google Drive into `data/`.

The cohort is kept out of Git history so the repository stays small and
pushable. This helper downloads the external artifact and writes it to:

    data/mimic_cemr_cohort.csv

Usage:
    pip install gdown
    python scripts/fetch_cohort.py
"""

from __future__ import annotations

import os


DRIVE_FOLDER_URL = "https://drive.google.com/drive/u/2/folders/1oupz5CcQIMn-16I8KFWeqlpirY0vBCxg"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "mimic_cemr_cohort.csv")


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"Dataset already present at: {CSV_PATH}")
        return 0

    try:
        import gdown
    except ImportError:
        print("Missing dependency: gdown")
        print("Install it with: pip install gdown")
        return 1

    print("Downloading cohort from Google Drive...")
    print(f"Source folder: {DRIVE_FOLDER_URL}")
    downloaded = gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=DATA_DIR,
        quiet=False,
        use_cookies=False,
    )

    if not downloaded:
        print("Download did not return any files. Please verify Drive access.")
        return 1

    if os.path.exists(CSV_PATH):
        print(f"Saved dataset to: {CSV_PATH}")
        return 0

    print("Download completed, but the expected CSV was not found.")
    print("Check the Drive folder contents and rename the cohort file if needed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
