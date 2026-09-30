"""A11 - nonsense positional profile: where does truncation stop mattering?"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from .base import dirs, fmt_p, result, skipped


def changepoint(pos: np.ndarray, y: np.ndarray, min_seg: int = 5) -> int | None:
    """Single step change (low -> high) minimising SSE; returns first position of the high segment."""
    o = np.argsort(pos)
    pos, y = pos[o], y[o]
    n = len(y)
    if n < 2 * min_seg:
        return None
    cs, cs2 = np.cumsum(y), np.cumsum(y ** 2)
    best, bk = np.inf, None
    for k in range(min_seg, n - min_seg + 1):
        a_n, b_n = k, n - k
        sse = (cs2[k - 1] - cs[k - 1] ** 2 / a_n) + ((cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / b_n)
        if cs[k - 1] / a_n < (cs[-1] - cs[k - 1]) / b_n and sse < best:
            best, bk = sse, k
    return int(pos[bk]) if bk is not None else None


def run(df, cfg, outdir):
    d = df[df.pass_filter]
    stop = d[d.vclass == "nonsense"][["pos", "score_z"]].dropna()
    syn = d[d.vclass == "synonymous"].score_z.dropna()
    if len(stop) < 20:
        return skipped("a11", cfg, f"only {len(stop)} nonsense variants")
    figdir, tabdir = dirs(outdir)
    pos, y = stop.pos.to_numpy(), stop.score_z.to_numpy()
    cp = changepoint(pos, y)
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        c = changepoint(pos[i], y[i])
        if c is not None:
            boots.append(c)
    metrics, caveats = {}, []
    region = (int(df.pos.min()), int(df.pos.max()))
    if cp is not None:
        after = y[pos >= cp]
        p = ss.mannwhitneyu(after, syn).pvalue if len(after) >= 3 and len(syn) >= 3 else np.nan
        metrics.update(changepoint=cp, changepoint_ci=[float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                       mean_stop_after=float(after.mean()), mean_stop_before=float(y[pos < cp].mean()),
                       p_after_vs_syn=float(p), n_stop_after=int(len(after)))
        indist = (np.isfinite(p) and p > 0.05) or after.mean() > -0.25
        metrics["after_indistinguishable_from_syn"] = bool(indist)
        if indist and region[0] < cp <= region[1]:
            n_ex = region[1] - cp + 1
            caveats.append(f"nonsense variants from position {cp} onwards behave like synonymous: the −1 anchor may be "
                           f"contaminated; consider normalization.stop_exclude_last_n: {n_ex}")
    else:
        metrics["changepoint"] = None
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    P.shade_topology(ax, cfg)
    ax.scatter(pos, y, s=6, color=P.CLASS_COLORS["nonsense"], lw=0)
    sm = pd.Series(y, index=pos).sort_index().rolling(9, center=True, min_periods=3).median()
    ax.plot(sm.index, sm.values, color=P.INK, lw=1)
    ax.axhline(0, color=P.CLASS_COLORS["synonymous"], lw=0.8, ls=(0, (2, 2)))
    ax.axhline(-1, color=P.MUTED, lw=0.5)
    if cp is not None:
        ax.axvline(cp, color=P.INK, lw=0.8)
        ax.axvspan(*metrics["changepoint_ci"], color=P.INK, alpha=0.08, lw=0)
        ax.text(cp, 1.0, f" changepoint {cp}", transform=ax.get_xaxis_transform(), va="top", fontsize=6.5)
    ax.set_xlabel("Position of stop codon")
    ax.set_ylabel(cfg.get_path("plotting.score_label"))
    P.title(fig, cfg, "A11 nonsense positional profile")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    figs = P.save(fig, figdir, "a11_nonsense_profile", cfg, "A11")
    t = tabdir / "a11_nonsense.csv"
    stop.to_csv(t, index=False)
    head = (f"Truncations stop mattering after position {cp} (95% CI {metrics['changepoint_ci'][0]:.0f}–"
            f"{metrics['changepoint_ci'][1]:.0f}; stops after vs syn {fmt_p(metrics['p_after_vs_syn'])})"
            if cp is not None else "No changepoint in the nonsense profile - truncation is deleterious throughout")
    return result("a11", cfg, head, metrics, caveats, figs, [t])
