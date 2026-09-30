"""Calibration of the positional statistics on autocorrelated null data."""
import numpy as np

from mpdms.stats import bh_fdr, block_bootstrap, circular_shift_test, icc_2_1


def ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal() * np.sqrt(1 - phi ** 2)
    return x


def block_mask(n, rng, n_blocks=4, length=20):
    m = np.zeros(n, bool)
    for s in rng.choice(np.arange(0, n - length), n_blocks, replace=False):
        m[s:s + length] = True
    return m


def test_circular_shift_calibrated_on_autocorrelated_null():
    """Under H0 (profile independent of a TM-like block mask) with strong autocorrelation,
    the circular-shift test keeps its nominal size; a per-residue t-test would not."""
    from scipy import stats as ss
    rng = np.random.default_rng(0)
    p_shift, p_naive = [], []
    for _ in range(200):
        y = ar1(300, 0.8, rng)
        m = block_mask(300, rng)
        p_shift.append(circular_shift_test(y, m, n=500, seed=int(rng.integers(1e9)))["p"])
        p_naive.append(ss.ttest_ind(y[m], y[~m]).pvalue)
    fpr_shift = np.mean(np.array(p_shift) < 0.05)
    fpr_naive = np.mean(np.array(p_naive) < 0.05)
    assert fpr_shift < 0.10, fpr_shift       # nominal 0.05, allow MC noise
    assert fpr_naive > 2 * fpr_shift          # the naive test is anticonservative


def test_circular_shift_has_power():
    rng = np.random.default_rng(1)
    y = ar1(300, 0.5, rng)
    m = block_mask(300, rng)
    y[m] -= 1.5
    assert circular_shift_test(y, m, n=2000, alternative="less")["p"] < 0.01


def test_block_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(2)
    r = block_bootstrap(ar1(200, 0.6, rng), n=500)
    assert r["lo"] <= r["estimate"] <= r["hi"]


def test_bh_fdr_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    q = bh_fdr(p)
    from statsmodels.stats.multitest import multipletests
    np.testing.assert_allclose(q[:4], multipletests(p[:4], method="fdr_bh")[1], rtol=1e-9)
    assert np.isnan(q[4])


def test_icc_perfect_agreement():
    x = np.random.default_rng(3).normal(size=(50, 1))
    assert abs(icc_2_1(np.hstack([x, x, x])) - 1) < 1e-9
