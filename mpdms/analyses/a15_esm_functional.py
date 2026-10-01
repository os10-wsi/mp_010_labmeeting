"""A15 - DMS abundance vs ESM-1v: sites more constrained in evolution than abundance explains.

x = DMS normalised fitness (abundance), y = ESM-1v masked-marginal score. A LOESS of y on x
gives the ESM-1v score expected from the abundance effect. For each variant
    residual = ESM-1v - LOESS(abundance),  z = residual / (1.4826 * MAD of all residuals).
Site-level test: one-sided Wilcoxon signed-rank of the site's variant residuals < 0, BH-FDR.
Functional site: q < 0.05 and median z <= -2 (ESM-1v predicts it markedly worse than its
abundance effect explains). Sites whose abundance is itself tolerant (median >= -0.5) are
flagged separately: those are the cleanest "functional, not folding" candidates.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss
from statsmodels.nonparametric.smoothers_lowess import lowess

from .. import plotting as P
from ..stats import bh_fdr
from .base import dirs, fmt_p, missense, result, skipped

Z_SITE = -2.0
Q_SITE = 0.05
TOLERANT = -0.5
MIN_VARIANTS = 5
FUNC_COLOR = "#D62728"


def read_esm(path) -> pd.DataFrame | None:
    e = pd.read_csv(path)
    e.columns = [c.strip().lower() for c in e.columns]
    col = next((c for c in ("esm1v", "esm1v_score", "llr", "esm_llr", "score", "esm_score") if c in e.columns), None)
    if col is None or not {"pos", "mut"} <= set(e.columns):
        return None
    out = e[["pos", "mut"] + (["wt"] if "wt" in e.columns else []) + [col]].rename(columns={col: "esm1v"})
    out["pos"] = out.pos.astype(int)
    return out


def loess_fit(x, y, frac=0.3):
    f = lowess(y, x, frac=frac, return_sorted=True)
    xs, idx = np.unique(f[:, 0], return_index=True)
    return lambda q: np.interp(q, xs, f[idx, 1])


def run(df, cfg, outdir):
    p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
    if p is None or not p.exists():
        return skipped("a15", cfg, "evolution.esm_scores not set - run `python -m mpdms esm <config>`")
    esm = read_esm(p)
    if esm is None:
        return skipped("a15", cfg, f"{p.name}: need per-variant columns pos, mut and esm1v/llr/score")
    figdir, tabdir = dirs(outdir)
    d = missense(df)[["pos", "wt", "mut", "score_z", "segment", "seg_type"]].merge(
        esm, on=["pos", "mut"], how="inner", suffixes=("", "_esm"))
    caveats = []
    if "wt_esm" in d:
        bad = (d.wt != d.wt_esm).mean()
        if bad > 0.01:
            caveats.append(f"WT residue differs between DMS and ESM table for {bad:.0%} of variants - numbering?")
        d = d[d.wt == d.wt_esm].drop(columns="wt_esm")
    d = d.dropna(subset=["score_z", "esm1v"])
    if len(d) < 100:
        return skipped("a15", cfg, f"only {len(d)} variants with both DMS and ESM-1v scores")

    x, y = d.score_z.to_numpy(), d.esm1v.to_numpy()
    f = loess_fit(x, y)
    d["esm_expected"] = f(x)
    d["residual"] = y - d.esm_expected
    scale = 1.4826 * ss.median_abs_deviation(d.residual)
    d["z"] = d.residual / scale
    rho = ss.spearmanr(x, y)[0]

    # site-level
    rows = []
    for pos, g in d.groupby("pos"):
        if len(g) < MIN_VARIANTS:
            continue
        pv = ss.wilcoxon(g.residual, alternative="less").pvalue if (g.residual != 0).any() else 1.0
        rows.append({"pos": pos, "wt": g.wt.iloc[0], "segment": g.segment.iloc[0], "seg_type": g.seg_type.iloc[0],
                     "n_variants": len(g), "median_abundance": g.score_z.median(), "median_esm1v": g.esm1v.median(),
                     "median_residual": g.residual.median(), "median_z": g.z.median(),
                     "frac_variants_z_lt_-2": float((g.z < -2).mean()), "p": pv})
    sites = pd.DataFrame(rows)
    sites["q"] = bh_fdr(sites.p)
    sites["functional"] = (sites.q < Q_SITE) & (sites.median_z <= Z_SITE)
    sites["abundance_tolerant"] = sites.median_abundance >= TOLERANT
    sites["functional_and_tolerant"] = sites.functional & sites.abundance_tolerant
    func = sites[sites.functional].sort_values("q")
    d = d.merge(sites[["pos", "functional"]], on="pos", how="left")
    d["functional"] = d["functional"].astype("boolean").fillna(False).astype(bool)

    # bootstrap band for the LOESS (resample positions)
    grid = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), 120)
    rng = np.random.default_rng(15)
    posidx = [g.index.to_numpy() for _, g in d.reset_index(drop=True).groupby("pos")]
    dx, dy = d.score_z.to_numpy(), d.esm1v.to_numpy()
    boots = []
    for _ in range(60):
        pick = np.concatenate([posidx[k] for k in rng.integers(0, len(posidx), len(posidx))])
        boots.append(loess_fit(dx[pick], dy[pick])(grid))
    boots = np.array(boots)

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.62], hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    nf = d[~d.functional]
    ax.hexbin(nf.score_z, nf.esm1v, gridsize=55, cmap="Greys", bins="log", mincnt=1, linewidths=0, rasterized=True)
    fv = d[d.functional]
    ax.scatter(fv.score_z, fv.esm1v, s=6, color=FUNC_COLOR, lw=0, alpha=0.7, label=f"variants at functional sites (n={len(fv)})")
    ax.fill_between(grid, np.quantile(boots, .025, 0), np.quantile(boots, .975, 0), color="#2F6DB5", alpha=0.25, lw=0)
    ax.plot(grid, f(grid), color="#2F6DB5", lw=1.8, label="LOESS (± position bootstrap)")
    ax.set_xlabel("Abundance (DMS normalised fitness)")
    ax.set_ylabel("ESM-1v score (masked marginal)")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title(f"(a) Variants: Spearman ρ = {rho:.2f}, n = {len(d):,}", loc="left")
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(sites.median_abundance, sites.median_esm1v, c=sites.median_z.clip(-3, 3), cmap=P.diverging_cmap(),
                    vmin=-3, vmax=3, s=14, edgecolor=P.INK, lw=0.2)
    fs = sites[sites.functional]
    ax.scatter(fs.median_abundance, fs.median_esm1v, s=40, facecolor="none", edgecolor=FUNC_COLOR, lw=1.1,
               label=f"functional sites (n={len(fs)})")
    for r in fs.nsmallest(12, "q").itertuples():
        ax.annotate(f"{r.wt}{r.pos}", (r.median_abundance, r.median_esm1v), fontsize=6, xytext=(3, 2),
                    textcoords="offset points", color=FUNC_COLOR)
    ax.axvline(TOLERANT, color=P.MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Site median abundance")
    ax.set_ylabel("Site median ESM-1v")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.colorbar(sc, ax=ax, fraction=0.04).set_label("Site median z (ESM-1v − expected)", fontsize=6.5)
    ax.set_title("(b) Sites", loc="left")
    ax = fig.add_subplot(gs[1, :])
    P.shade_topology(ax, cfg)
    ax.bar(sites.pos, sites.median_z, width=1, lw=0, color=np.where(sites.functional, FUNC_COLOR, "#9DB8DB"))
    ax.axhline(Z_SITE, color=FUNC_COLOR, lw=0.6, ls=(0, (2, 2)))
    ax.axhline(0, color=P.INK, lw=0.5)
    for r in fs.nsmallest(15, "q").itertuples():
        ax.text(r.pos, r.median_z - 0.15, f"{r.wt}{r.pos}", rotation=90, ha="center", va="top", fontsize=5.5,
                color=FUNC_COLOR)
    lo = min(sites.median_z.min(), Z_SITE) if len(sites) else Z_SITE
    ax.set_ylim(lo - 1.2, max(1.0, sites.median_z.max() + 0.2) if len(sites) else 1.0)
    ax.set_xlabel("Position")
    ax.set_ylabel("Site median z")
    ax.set_title(f"(c) Residual along the sequence: red = functional (q < {Q_SITE}, median z ≤ {Z_SITE}); grey = TM",
                 loc="left")
    P.title(fig, cfg, "A15 abundance vs ESM-1v: functional sites")
    figs = P.save(fig, figdir, "a15_esm_functional", cfg, "A15")

    t1, t2 = tabdir / "a15_variants.csv", tabdir / "a15_sites.csv"
    d.to_csv(t1, index=False)
    sites.sort_values("q").to_csv(t2, index=False)
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    pml = (f"select {cfg.id}_functional, resi " + "+".join(str(int(p_) - off) for p_ in func.pos)) if len(func) else ""
    (tabdir / "a15_functional_sites.pml").write_text(pml + "\n")
    if len(sites) and sites.functional.mean() > 0.3:
        caveats.append(f"{sites.functional.mean():.0%} of sites called functional - check ESM numbering/scale")
    caveats.append("ESM-1v scores conservation for any reason (function, folding in other contexts, interactions); "
                   "a 'functional' call means constraint beyond this abundance readout, not proof of function")
    head = (f"{len(func)} functional sites (ESM-1v more constrained than abundance explains, q<{Q_SITE}); "
            f"{int(sites.functional_and_tolerant.sum())} of them abundance-tolerant; variant ρ(abundance, ESM-1v) = {rho:.2f}"
            + (f"; top: {', '.join(f'{r.wt}{r.pos}' for r in func.head(5).itertuples())}" if len(func) else ""))
    return result("a15", cfg, head, {"spearman": rho, "n_variants": len(d), "n_sites_tested": len(sites),
                                     "n_functional": int(len(func)), "residual_scale": scale,
                                     "n_functional_tolerant": int(sites.functional_and_tolerant.sum()),
                                     "functional_sites": [f"{r.wt}{r.pos}" for r in func.itertuples()],
                                     "pymol": pml, "best_p": fmt_p(sites.p.min())},
                  caveats, figs, [t1, t2])
