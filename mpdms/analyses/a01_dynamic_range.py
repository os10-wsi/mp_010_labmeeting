"""A01 - dynamic range and score distribution."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..stats import auc_below, cohens_d, mad
from .base import dirs, result
from .qc import _kde


def syn_bounds(df):
    syn = df.loc[df.pass_filter & (df.vclass == "synonymous"), "score_z"].dropna()
    if len(syn) < 5:
        return np.nan, np.nan, np.nan
    return float(np.percentile(syn, 2.5)), float(np.percentile(syn, 97.5)), float(syn.std(ddof=1))


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    d = df[df.pass_filter]
    by = {c: d.loc[d.vclass == c, "score_z"].dropna().to_numpy() for c in P.CLASS_ORDER}
    lo, hi, syn_sd = syn_bounds(df)
    mis, syn, stop = by["missense"], by["synonymous"], by["nonsense"]

    rows = [{"vclass": c, "n": len(v), "median": np.median(v) if len(v) else np.nan, "mad": mad(v) if len(v) else np.nan}
            for c, v in by.items()]
    tab = pd.DataFrame(rows)
    metrics = {f"{r['vclass']}_{k}": r[k] for r in rows for k in ("n", "median", "mad")}
    metrics.update(
        cohens_d_syn_vs_stop=cohens_d(syn, stop),
        auc_stop_below_syn=auc_below(stop, syn),
        syn_sd=syn_sd,
        syn_p2_5=lo, syn_p97_5=hi,
        deleterious_frac=float(np.mean(mis < lo)) if len(mis) and np.isfinite(lo) else np.nan,
        gof_frac=float(np.mean(mis > hi)) if len(mis) and np.isfinite(hi) else np.nan,
        n_resolvable=int(np.sum(np.abs(mis) > 2 * syn_sd)) if np.isfinite(syn_sd) else None,
        frac_resolvable=float(np.mean(np.abs(mis) > 2 * syn_sd)) if np.isfinite(syn_sd) and len(mis) else np.nan,
    )
    # bimodality of missense
    try:
        import diptest
        dip, p_dip = diptest.diptest(mis)
    except Exception:
        dip, p_dip = np.nan, np.nan
    metrics.update(dip_stat=dip, dip_p=p_dip)
    gmm = None
    if np.isfinite(p_dip) and p_dip < 0.05 and len(mis) > 50:
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(2, random_state=0).fit(mis.reshape(-1, 1))
        o = np.argsort(g.means_.ravel())
        gmm = {"means": g.means_.ravel()[o].tolist(), "sds": np.sqrt(g.covariances_.ravel()[o]).tolist(),
               "weights": g.weights_[o].tolist()}
        metrics["gmm_2comp"] = gmm
    caveats = []
    low_dr = False
    if len(stop) < 20:
        caveats.append(f"only {len(stop)} nonsense variants - the −1 anchor is poorly determined")
        low_dr = True
    if np.isfinite(syn_sd) and syn_sd > 0.25:
        caveats.append(f"synonymous SD ({syn_sd:.2f}) exceeds 25% of the syn-to-stop span: LOW DYNAMIC RANGE")
        low_dr = True
    if df.attrs.get("normalization_fallback"):
        caveats.append("normalisation used a fallback anchor - not comparable across datasets")
    metrics["low_dynamic_range"] = low_dr

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    allv = np.concatenate([v for v in by.values() if len(v)])
    grid = np.linspace(*np.nanpercentile(allv, [0.2, 99.8]) + np.array([-0.2, 0.2]), 400)
    for c in P.CLASS_ORDER:
        _kde(ax, by[c], P.CLASS_COLORS[c], label=f"{P.CLASS_LABELS[c]} (n={len(by[c]):,})", grid=grid)
    ax.plot(syn, np.full(len(syn), -0.03 * ax.get_ylim()[1]), "|", color=P.CLASS_COLORS["synonymous"], ms=4, mew=0.5,
            clip_on=False)
    if gmm:
        for m_, s_, w_ in zip(gmm["means"], gmm["sds"], gmm["weights"]):
            ax.plot(grid, w_ * ss.norm.pdf(grid, m_, s_), color=P.CLASS_COLORS["missense"], lw=0.7, ls=(0, (2, 1.5)))
    ax.set_xlabel(cfg.get_path("plotting.score_label"))
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend(loc="upper left")
    ax.set_title("(a) Score distributions", loc="left")
    axes[1].text(0.97, 0.05, f"Cohen's d (syn vs stop) = {metrics['cohens_d_syn_vs_stop']:.1f}\n"
                        f"AUC(stop < syn) = {metrics['auc_stop_below_syn']:.2f}\n"
                        f"syn SD = {syn_sd:.2f}\ndip test p = {p_dip:.2g}\n"
                        f"deleterious {metrics['deleterious_frac']:.0%} · GoF {metrics['gof_frac']:.1%}",
            transform=axes[1].transAxes, fontsize=6.5, va="bottom", ha="right")
    ax = axes[1]
    xs = np.sort(mis)
    ax.axvspan(lo, hi, color=P.CLASS_COLORS["synonymous"], alpha=0.15, lw=0, label="syn 2.5–97.5%")
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=P.CLASS_COLORS["missense"], lw=1.3,
            label="Missense ECDF")
    ax.axvline(-1, color=P.CLASS_COLORS["nonsense"], lw=0.7, ls=(0, (2, 2)))
    ax.set_xlabel(cfg.get_path("plotting.score_label"))
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(b) Missense ECDF", loc="left")
    ax.legend(loc="upper left")
    P.title(fig, cfg, "A01 dynamic range")
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    figs = P.save(fig, figdir, "a01_dynamic_range", cfg, "A01")
    tab.to_csv(tabdir / "a01_dynamic_range.csv", index=False)
    head = (f"{metrics['deleterious_frac']:.0%} of missense variants fall below the synonymous 2.5th percentile "
            f"(noise floor SD {syn_sd:.2f}); stop vs syn AUC {metrics['auc_stop_below_syn']:.2f}"
            + ("; LOW DYNAMIC RANGE" if low_dr else ""))
    return result("a01", cfg, head, metrics, caveats, figs, [tabdir / "a01_dynamic_range.csv"])
