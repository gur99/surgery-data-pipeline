#!/usr/bin/env python3
"""Entry point for the full surgery ETL pipeline."""

from pathlib import Path
import os
import sys

# Ensure project root is on sys.path and cwd
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.run import run_full_pipeline


def main():
    df = run_full_pipeline(save=True)
    print(f"Done: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
