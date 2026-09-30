#!/usr/bin/env bash
# End-to-end demo on two synthetic datasets (no network needed).
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/make_synthetic.py
python -m mpdms init data/raw/SYN1 data/raw/SYN2 --offline --force
python run_all.py configs/SYN1.yaml configs/SYN2.yaml
