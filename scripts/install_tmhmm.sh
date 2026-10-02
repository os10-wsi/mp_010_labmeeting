#!/usr/bin/env bash
# Install the `tmhmm.py` package (a reimplementation of TMHMM 2.0 that ships the original
# model file). Its Cython source predates NumPy 1.24 and Cython 3, so it needs two patches
# before it will build. Run from anywhere; needs a C compiler.
set -euo pipefail
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
pip install "cython<3"
pip download --no-deps --no-binary :all: "tmhmm.py" -d "$tmp"
tar xf "$tmp"/tmhmm.py-*.tar.gz -C "$tmp"
src=$(find "$tmp" -maxdepth 1 -type d -name 'tmhmm.py-*')
sed -i.bak 's/np\.int_t/np.int64_t/g; s/dtype=np\.int)/dtype=np.int64)/g; s/dtype=np\.float)/dtype=np.float64)/g' "$src/tmhmm/hmm.pyx"
pip install --no-build-isolation "$src"
python -c "import tmhmm, pathlib; print('tmhmm.py ok:', pathlib.Path(tmhmm.__file__).parent)"
