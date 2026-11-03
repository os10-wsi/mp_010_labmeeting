#!/bin/bash
# SLC22A1: function against abundance, and the positions whose function falls below
# what their abundance accounts for.
#
# Run from the folder holding the supplementary table. Submit with:
#   bsub -q normal -M 8000 -R "select[mem>8000] rusage[mem=8000]" \
#        -o d_%J.log "bash run_dual_slc22a1.sh"
# or just `bash run_dual_slc22a1.sh` on a head node - it needs no GPU and no network,
# and only reads the one table.

set -euo pipefail

REPO=${REPO:-$HOME/mp_010_labmeeting}       # where the mpdms checkout lives
DATA=${DATA:-$PWD}                           # where the supplementary table is
OUT=${OUT:-$PWD/outputs_slc22a1}

TABLE="$DATA/${TABLE_NAME:-20261103_survivability.xlsx}"

# Fail now, with the name of the file, rather than halfway through the fit.
[[ -r "$TABLE" ]] || { echo "missing or unreadable: $TABLE" >&2; exit 1; }

cd "$REPO"

# GFP_score is abundance. SM73_1_score is the dose-1 survival selection, in which a
# BROKEN transporter lets the cell live - its deletions score above its synonymous
# variants. mpdms anchors each assay on (synonymous, deletion) and so rescales that
# assay with a negative denominator, which puts both on 0 = wild type, -1 = dead
# without anyone having to remember to flip a sign. SM73_0_score is the same selection
# at zero dose and is passed as the control: it should separate nothing.
python -m mpdms dual "$TABLE" \
  --id SLC22A1 \
  --abundance GFP_score \
  --function SM73_1_score \
  --control SM73_0_score \
  --outdir "$OUT"

echo
echo "figures and tables in $OUT:"
ls -1 "$OUT"
