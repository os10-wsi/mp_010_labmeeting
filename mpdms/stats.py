"""Statistics that respect positional autocorrelation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

RNG_SEED = 20260930


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED if seed is None else seed)


def _stat(profile, mask, stat):
    a, b = profile[mask], profile[~mask]
    if stat == "median_diff":
        return np.nanmedian(a) - np.nanmedian(b)
    return np.nanmean(a) - np.nanmean(b)


def circular_shift_test(profile, mask, n: int = 10000, stat: str = "mean_diff",
                        alternative: str = "two-sided", seed: int | None = None) -> dict:
    """Permutation test that circularly shifts the annotation mask along the profile.

    Preserves the autocorrelation of both the profile and the mask, unlike a per-residue
    permutation. `profile` is an ordered per-position vector (NaNs allowed).
    """
    profile = np.asarray(profile, float)
    mask = np.asarray(mask, bool)
    L = len(profile)
    obs = _stat(profile, mask, stat)
    if mask.all() or (~mask).all() or L < 4:
        return {"observed": obs, "p": np.nan, "n": 0}
    r = rng(seed)
    min_shift = max(1, int(0.05 * L))  # avoid near-identity shifts
    shifts = r.integers(min_shift, L - min_shift + 1, size=n) if L > 2 * min_shift else r.integers(1, L, size=n)
    null = np.array([_stat(profile, np.roll(mask, s), stat) for s in shifts])
    null = null[np.isfinite(null)]
    if alternative == "less":
        p = (1 + np.sum(null <= obs)) / (1 + len(null))
    elif alternative == "greater":
        p = (1 + np.sum(null >= obs)) / (1 + len(null))
    else:
        c = np.nanmean(null)
        p = (1 + np.sum(np.abs(null - c) >= abs(obs - c))) / (1 + len(null))
    return {"observed": float(obs), "p": float(p), "n": int(len(null)),
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null))}


def circular_shift_corr(x, y, n: int = 10000, method: str = "spearman", seed: int | None = None) -> dict:
    """Correlation between two positional profiles with a circular-shift null."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = ss.spearmanr if method == "spearman" else ss.pearsonr

    def c(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return f(a[ok], b[ok])[0] if ok.sum() > 3 else np.nan

    obs = c(x, y)
    r = rng(seed)
    L = len(x)
    null = np.array([c(x, np.roll(y, s)) for s in r.integers(1, L, size=n)])
    null = null[np.isfinite(null)]
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null))
    return {"r": float(obs), "p": float(p)}


def block_bootstrap(profile, func=np.nanmedian, block_len: int = 7, n: int = 10000,
                    ci: float = 0.95, seed: int | None = None) -> dict:
    """Moving-block bootstrap CI for a summary of an ordered positional profile."""
    x = np.asarray(profile, float)
    L = len(x)
    est = float(func(x)) if L else np.nan
    if L < 2:
        return {"estimate": est, "lo": np.nan, "hi": np.nan}
    b = max(1, min(block_len, L))
    nblocks = int(np.ceil(L / b))
    r = rng(seed)
    starts = r.integers(0, L - b + 1, size=(n, nblocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n, -1)[:, :L]
    boots = np.array([func(x[i]) for i in idx])
    a = (1 - ci) / 2
    return {"estimate": est, "lo": float(np.nanquantile(boots, a)), "hi": float(np.nanquantile(boots, 1 - a))}


def block_bootstrap_ribbon(matrix_by_pos: dict, positions, block_len: int = 7, n: int = 1000,
                           seed: int | None = None):
    """Per-position CI for a median, resampling variants within position (fast, for ribbons)."""
    r = rng(seed)
    lo, hi = [], []
    for p in positions:
        v = np.asarray(matrix_by_pos.get(p, []), float)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            lo.append(np.nan); hi.append(np.nan); continue
        bs = np.median(r.choice(v, size=(n, len(v)), replace=True), axis=1)
        lo.append(np.quantile(bs, 0.025)); hi.append(np.quantile(bs, 0.975))
    return np.array(lo), np.array(hi)


def bootstrap_ci(x, func=np.nanmean, n: int = 5000, ci: float = 0.95, seed: int | None = None) -> tuple:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(func(x)) if len(x) else np.nan, np.nan, np.nan)
    r = rng(seed)
    bs = np.array([func(x[r.integers(0, len(x), len(x))]) for _ in range(n)])
    a = (1 - ci) / 2
    return float(func(x)), float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def bh_fdr(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def auc_below(a, b) -> float:
    """P(random a < random b) + 0.5 P(tie)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    u = ss.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def mad(x) -> float:
    return float(ss.median_abs_deviation(np.asarray(x, float), nan_policy="omit", scale="normal"))


def icc_2_1(mat: np.ndarray) -> float:
    """ICC(2,1), two-way random, absolute agreement, single rater (Shrout & Fleiss)."""
    m = np.asarray(mat, float)
    m = m[np.isfinite(m).all(axis=1)]
    n, k = m.shape
    if n < 3 or k < 2:
        return np.nan
    grand = m.mean()
    msr = k * np.sum((m.mean(axis=1) - grand) ** 2) / (n - 1)
    msc = n * np.sum((m.mean(axis=0) - grand) ** 2) / (k - 1)
    sse = np.sum((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def runs_test(binary) -> dict:
    """Wald-Wolfowitz runs test on a 0/1 sequence (clustering => too few runs)."""
    x = np.asarray(binary, bool)
    n1, n2 = x.sum(), (~x).sum()
    if n1 == 0 or n2 == 0:
        return {"runs": 1, "z": np.nan, "p": np.nan}
    runs = 1 + np.sum(x[1:] != x[:-1])
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (runs - mu) / np.sqrt(var) if var > 0 else np.nan
    return {"runs": int(runs), "expected": float(mu), "z": float(z), "p": float(2 * ss.norm.sf(abs(z)))}


def per_position(df: pd.DataFrame, col: str = "score_z", func: str = "median",
                 classes=("missense",), full_range: bool = True) -> pd.Series:
    d = df[df["pass_filter"] & df["vclass"].isin(classes)]
    s = d.groupby("pos")[col].agg(func)
    if full_range and len(s):
        s = s.reindex(range(int(df["pos"].min()), int(df["pos"].max()) + 1))
    return s


def meta_analysis(yi, vi, knha: bool = True) -> dict:
    """Random-effects meta-analysis of per-unit estimates (DerSimonian-Laird).

    Pooling eight datasets as if every variant were an independent observation answers a
    question nobody asked: it weights a protein by how many variants were measured in it.
    Combining one estimate per protein (or per helix) instead keeps the unit of replication
    honest, and `tau2`/`I2` say how much the units actually disagree, which is usually the
    more interesting number than the pooled mean.

    With few units the Wald interval is far too narrow, so the Hartung-Knapp-Sidik-Jonkman
    adjustment is on by default: it uses a t distribution and rescales by the observed
    dispersion. Turn it off only to reproduce a plain DL result.
    """
    y = np.asarray(yi, dtype=float)
    v = np.asarray(vi, dtype=float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    if k == 0:
        return {"k": 0, "mu": np.nan, "se": np.nan, "lo": np.nan, "hi": np.nan,
                "tau2": np.nan, "I2": np.nan, "Q": np.nan, "p_Q": np.nan, "p": np.nan}
    wf = 1.0 / v
    mu_f = float(np.sum(wf * y) / np.sum(wf))
    Q = float(np.sum(wf * (y - mu_f) ** 2))
    df = k - 1
    C = float(np.sum(wf) - np.sum(wf ** 2) / np.sum(wf))
    tau2 = max(0.0, (Q - df) / C) if C > 0 and k > 1 else 0.0
    w = 1.0 / (v + tau2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    if knha and k > 1:
        # Hartung-Knapp: rescale by how far the units actually scatter about mu. When they
        # happen to agree almost exactly the rescaling collapses toward zero and would
        # report a spuriously tiny interval, so never go below the Wald standard error.
        se = max(float(np.sqrt(np.sum(w * (y - mu) ** 2) / (df * np.sum(w)))), se)
        crit = float(ss.t.ppf(0.975, df))
        p = float(2 * ss.t.sf(abs(mu / se), df)) if se > 0 else np.nan
    else:
        crit = 1.959963984540054
        p = float(2 * ss.norm.sf(abs(mu / se))) if se > 0 else np.nan
    return {"k": k, "mu": mu, "se": se, "lo": mu - crit * se, "hi": mu + crit * se,
            "tau2": float(tau2), "I2": float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0,
            "Q": Q, "p_Q": float(ss.chi2.sf(Q, df)) if k > 1 else np.nan, "p": p}


def slope_with_se(x, y, groups=None) -> dict:
    """OLS slope of y on x with its standard error; the unit an aggregate is built from.

    `groups` clusters the standard error (e.g. by position), because variants at one
    position are not independent draws.
    """
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if groups is not None:
        groups = np.asarray(groups)[ok]
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.ptp(x) == 0:
        return {"n": int(len(x)), "slope": np.nan, "se": np.nan, "p": np.nan}
    X = sm.add_constant(x)
    m = (sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
         if groups is not None and len(set(groups)) > 2 else sm.OLS(y, X).fit())
    return {"n": int(len(x)), "slope": float(m.params[1]), "se": float(m.bse[1]),
            "p": float(m.pvalues[1])}
