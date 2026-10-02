#!/usr/bin/env bash
# Install TMHMM 2.0 via the `tmhmm.py` package.
#
# Prefer DeepTMHMM instead (https://dtu.biolib.com/DeepTMHMM, or `pip install pybiolib`):
# it needs no compilation, works on any Python, and has superseded TMHMM 2.0.
#
# `tmhmm.py` publishes wheels only for Python 3.6/3.7, so anything newer builds its Cython
# extension from the 1.2.1 sdist. That source predates NumPy 1.24 and Cython 3, so it needs
# patching, and Cython 0.29 itself does not support Python 3.13+.
set -euo pipefail

PY_OK=$(python -c 'import sys; print(1 if sys.version_info[:2] <= (3,12) else 0)')
if [ "$PY_OK" != "1" ]; then
  echo "ERROR: $(python -V) is too new - the Cython 0.29 build needed here supports Python <= 3.12." >&2
  echo "Use DeepTMHMM instead:" >&2
  echo "    pip install pybiolib && python -m mpdms tmhmm configs/X.yaml --source biolib" >&2
  echo "  or upload the FASTA to https://dtu.biolib.com/DeepTMHMM and pass TMRs.gff3:" >&2
  echo "    python -m mpdms tmhmm configs/X.yaml --from-file TMRs.gff3" >&2
  echo "  or make a Python 3.12 env for this step only:" >&2
  echo "    conda create -y -n tmhmm python=3.12 numpy && conda activate tmhmm && bash $0" >&2
  exit 1
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pip install "cython<3" "numpy" "setuptools"
# --no-build-isolation so the build backend can see the numpy we just installed:
# tmhmm.py imports numpy in setup.py without declaring it as a build dependency.
pip download --no-deps --no-binary :all: --no-build-isolation "tmhmm.py==1.2.1" -d "$tmp"
tar xf "$tmp"/tmhmm.py-*.tar.gz -C "$tmp"
src=$(find "$tmp" -maxdepth 1 -type d -name 'tmhmm.py-*')
sed -i.bak 's/np\.int_t/np.int64_t/g; s/dtype=np\.int)/dtype=np.int64)/g; s/dtype=np\.float)/dtype=np.float64)/g' "$src/tmhmm/hmm.pyx"
pip install --no-build-isolation "$src"
python -c "import tmhmm, pathlib; print('tmhmm.py ok:', pathlib.Path(tmhmm.__file__).parent)"
