"""Scripts package marker to allow package-style imports when running
scripts from the repository root (e.g. `python scripts/evaluate_multiseed_gru.py`).

This file is intentionally minimal: it makes the `scripts` directory a regular
package so other scripts can import modules from it using `scripts.<module>`.
"""
