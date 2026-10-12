#!/bin/bash
# SLC2A10: the hexose family model against AlphaMissense, ESM-1v and conservation.
#
# Run from the folder holding the three input files. Submit with:
#   bsub -q normal -M 16000 -R "select[mem>16000] rusage[mem=16000]" \
#        -o c03_%J.log "bash run_slc2a10_predictors.sh"
# or just `bash run_slc2a10_predictors.sh` on a head node - it needs no GPU and only
# reads files, apart from fetching the SLC2A10 sequence once (cached afterwards).

set -euo pipefail

REPO=${REPO:-$HOME/mp_010_labmeeting}      # where the mpdms checkout lives
DATA=${DATA:-$PWD}                          # where the three input files are
OUT=${OUT:-$PWD/outputs_slc2a10}

ESM="$DATA/slc22a1_all_esm1v_predictions_with_mean.csv"
AM="$DATA/SLC2A10_alphamissense.tsv"
CLINVAR="$DATA/variant_summary.txt.gz"

# Fail now, with the name of the file, rather than halfway through the alignment.
for f in "$ESM" "$AM" "$CLINVAR"; do
  [[ -r "$f" ]] || { echo "missing or unreadable: $f" >&2; exit 1; }
done

cd "$REPO"

# The ESM table is labelled P40107 (yeast VRG4, 337 aa) but holds a complete 541 x 19
# scan whose wild-type residues match SLC2A10 at 100%. mpdms checks that agreement
# itself and refuses below 90%, so the wrong file cannot pass as the right one.
MPDMS_BOX=1 MPDMS_SQUARE=1 python -m mpdms clinvar \
  --family hexose=configs/HXT1.yaml,configs/HXT2.yaml,configs/HXT3.yaml,configs/HXT7.yaml \
  --human SLC2A10=O95528 \
  --table "$CLINVAR" \
  --alphamissense "$AM" \
  --esm "SLC2A10=$ESM" \
  --include-vus \
  --outdir "$OUT"

echo
echo "figures and tables in $OUT:"
ls -1 "$OUT" | sed 's/^/  /'
