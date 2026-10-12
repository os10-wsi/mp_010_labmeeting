#!/usr/bin/env bash
# ClinVar benign vs pathogenic for SLC2A10 alone, scored by the hexose family
# (HXT1, HXT2, HXT3, HXT7). One human gene, so C01 is panels (a) and (b) only.
#
#   scripts/clinvar_slc2a10.sh [variant_summary.txt.gz] [extra mpdms clinvar flags...]
#
# The ClinVar table defaults to data/external/variant_summary.txt.gz; fetch it with
#   wget -P data/external https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
set -euo pipefail
cd "$(dirname "$0")/.."
TABLE="${1:-data/external/variant_summary.txt.gz}"
shift $(( $# > 0 ? 1 : 0 ))
[ -f "$TABLE" ] || { echo "ClinVar table not found: $TABLE" >&2; exit 1; }
for c in HXT1 HXT2 HXT3 HXT7; do
  [ -f "configs/$c.yaml" ] || { echo "missing configs/$c.yaml (run quickstart on $c first)" >&2; exit 1; }
done
python -m mpdms clinvar \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A10=O95528 \
  --table "$TABLE" \
  --outdir outputs/_clinvar_slc2a10 \
  "$@"
