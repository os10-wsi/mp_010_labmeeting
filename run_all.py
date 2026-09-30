#!/usr/bin/env python
"""Run every analysis for every config in configs/ (see README.md).

    python run_all.py                       # all datasets, all analyses
    python run_all.py configs/SEC61.yaml    # one dataset
    python run_all.py --only qc,heatmap     # subset of analyses
"""
from mpdms.runner import main

if __name__ == "__main__":
    main()
