"""ESM-1v masked-marginal bookkeeping (model-free) and the A15 functional-site call."""
import numpy as np
import pandas as pd

from mpdms.esm_score import AA20, masked_marginals, to_long, windows


def test_windows_cover_long_sequences():
    w = windows(2500)
    assert w[0][0] == 0 and w[-1][1] == 2500
    assert all(b - a <= 1022 for a, b in w)
    assert all(w[i + 1][0] < w[i][1] for i in range(len(w) - 1))  # overlapping


def test_masked_marginals_relative_to_wt():
    seq = "MKLVAG" * 200  # 1200 aa -> two windows
    seen = []

    def fake(items):
        seen.extend(len(s) for s, _ in items)
        # log p is 0 for alanine, -1 elsewhere
        return np.array([[0.0 if a == "A" else -1.0 for a in AA20] for _ in items])

    m = masked_marginals(seq, fake, batch=16)
    assert m.shape == (1200, 20) and max(seen) <= 1022
    wt_cols = [AA20.index(a) for a in seq]
    assert np.allclose(m[np.arange(1200), wt_cols], 0)          # wt -> wt scores 0
    assert np.allclose(m[seq.index("K"), AA20.index("A")], 1)    # K->A favoured
    tab = to_long(seq[:6], [m[:6]], [1])
    assert len(tab) == 6 * 19 and set(tab.columns) >= {"pos", "wt", "mut", "esm1v"}
