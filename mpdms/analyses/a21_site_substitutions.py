"""A21 - every substitution at a few named sites, against the ESM-1v expectation.

A15 asks the question across the whole protein and summarises each site by its
median. This zooms in: for a short list of sites, one panel per site showing all
~19 substitutions as individual points, so you can see whether a site is
uniformly constrained or whether the signal comes from a few substitutions.

The LOESS, the residual and the z are NOT recomputed here. They are read from
a15_variants.csv, i.e. the fit over every variant in the protein. Refitting on
60 variants would define "expected" from the very points being tested, and the
z would mean something different in each panel. The threshold is A15's Z_SITE.

Substitutions named in the config or in _literature.yaml (e.g. Q149A) are ringed
and labelled, so the published allele can be read against its 18 neighbours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..stats import bh_fdr
from .a15_esm_functional import Q_SITE, Z_SITE
from .a20_variant_panel import parse_substitution, wanted_variants
from .base import dirs, fmt_p, result, skipped

HILITE = "#B8860B"
Z_RANGE = 4.0


def site_table(v: pd.DataFrame, sites: list[int], marked: dict[int, str]) -> pd.DataFrame:
    """Per-site summary over the substitutions actually measured there."""
    rows = []
    for pos in sites:
        g = v[v.pos == pos]
        if not len(g):
            continue
        hit = g[g.mut == marked.get(pos)] if pos in marked else g.iloc[:0]
        p = ss.wilcoxon(g.residual, alternative="less").pvalue if (g.residual != 0).any() and len(g) > 1 else np.nan
        rows.append({"pos": pos, "wt": g.wt.iloc[0], "segment": g.segment.iloc[0],
                     "n_substitutions": len(g), "median_abundance": g.score_z.median(),
                     "median_z": g.z.median(), "n_below_threshold": int((g.z <= Z_SITE).sum()),
                     "frac_below_threshold": float((g.z <= Z_SITE).mean()),
                     "substitutions_below": "".join(sorted(g.loc[g.z <= Z_SITE, "mut"])),
                     "marked": marked.get(pos), "marked_abundance": float(hit.score_z.iloc[0]) if len(hit) else np.nan,
                     "marked_z": float(hit.z.iloc[0]) if len(hit) else np.nan,
                     "marked_below_threshold": bool(len(hit) and hit.z.iloc[0] <= Z_SITE), "p": p})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p.fillna(1.0))
    return t


def figure(cfg, v: pd.DataFrame, t: pd.DataFrame, allv: pd.DataFrame, figdir):
    n = len(t)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n + 0.8, 4.3), squeeze=False, sharey=True, layout="constrained")
    # the global LOESS, drawn identically in every panel so the panels are comparable
    o = allv.sort_values("score_z")
    norm = plt.Normalize(-Z_RANGE, Z_RANGE)
    for ax, r in zip(axes[0], t.itertuples()):
        g = v[v.pos == r.pos]
        ax.plot(o.score_z, o.esm_expected, color=P.MUTED, lw=1.2, zorder=1,
                label="ESM-1v expected from abundance (all variants)")
        ax.scatter(allv.score_z, allv.esm1v, s=3, color="#DCE3EA", lw=0, zorder=0)
        sc = ax.scatter(g.score_z, g.esm1v, c=g.z, cmap="RdBu", norm=norm, s=46, lw=0.5,
                        edgecolor=P.INK, zorder=3)
        for q_ in g.itertuples():                      # label every substitution
            ax.annotate(q_.mut, (q_.score_z, q_.esm1v), fontsize=5.5, ha="center", va="center",
                        zorder=4, color=P.INK if abs(q_.z) < 2 else "white")
        if r.marked and not np.isnan(r.marked_z):
            h = g[g.mut == r.marked]
            ax.scatter(h.score_z, h.esm1v, s=200, facecolor="none", edgecolor=HILITE, lw=2.0, zorder=5)
            # in the free upper-left corner: next to the point it lands on the cluster,
            # and the ring already says which point it is (no tick glyph - the font lacks it)
            verdict = "below threshold" if r.marked_below_threshold else "NOT below threshold"
            ax.text(0.03, 0.97, f"{r.wt}{r.pos}{r.marked}:  z = {r.marked_z:.2f}\n{verdict}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    color=HILITE, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=HILITE, lw=1.0, alpha=0.95))
        ax.set_title(f"{r.wt}{r.pos} ({r.segment}) — {r.n_substitutions} substitutions\n"
                     f"median z = {r.median_z:.2f}, {fmt_p(r.p)}, q = {r.q:.3f}; "
                     f"{r.n_below_threshold} at z ≤ {Z_SITE}", loc="left", fontsize=8)
        ax.set_xlabel("Abundance (normalised fitness)")
    axes[0][0].set_ylabel("ESM-1v score (masked marginal)")
    axes[0][0].legend(loc="lower right", fontsize=6, frameon=False)
    fig.colorbar(sc, ax=axes[0], fraction=0.02).set_label(
        f"Variant z (ESM-1v − expected); ≤ {Z_SITE} = more constrained than abundance explains", fontsize=6.5)
    P.title(fig, cfg, "A21 all substitutions at the named sites vs ESM-1v")
    return P.save(fig, figdir, "a21_site_substitutions", cfg, "A21")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    vp = Path(outdir) / "tables" / "a15_variants.csv"
    if not vp.exists():
        return skipped("a21", cfg, "a15_variants.csv not found - run a15 first (needs ESM-1v scores)")
    labels = wanted_variants(cfg)
    if not labels:
        return skipped("a21", cfg, f"no sites named for {cfg.get_path('protein.gene', cfg.id)} "
                                   "(set report.variant_panel, or add measured entries to _literature.yaml)")
    marked, sites, bad = {}, [], []
    for lab in labels:
        p = parse_substitution(lab)
        if p is None:
            bad.append(f"{lab}: not a point substitution")
            continue
        _, pos, mut = p
        marked[pos] = mut
        if pos not in sites:
            sites.append(pos)

    allv = pd.read_csv(vp)
    v = allv[allv.pos.isin(sites)]
    if not len(v):
        return skipped("a21", cfg, f"none of positions {sites} appear in a15_variants.csv")
    # the wild-type residue in the data must match the one the label claims
    for lab in labels:
        p = parse_substitution(lab)
        if p and p[1] in set(v.pos):
            got = v.loc[v.pos == p[1], "wt"].iloc[0]
            if got != p[0]:
                bad.append(f"{lab}: data has {got}{p[1]} - numbering mismatch")

    t = site_table(v, sites, marked)
    tp, vpo = tabdir / "a21_site_summary.csv", tabdir / "a21_site_substitutions.csv"
    t.to_csv(tp, index=False)
    v.sort_values(["pos", "z"]).to_csv(vpo, index=False)
    figs = figure(cfg, v, t, allv, figdir)

    missing = [p for p in sites if p not in set(t.pos)]
    caveats = list(bad) + ([f"positions not in the data: {missing}"] if missing else [])
    caveats.append(f"z comes from the LOESS over all {len(allv)} variants, not a refit on these sites; "
                   "the threshold is A15's")
    caveats.append("one substitution below the threshold is a single noisy measurement away from not being "
                   "below it - read the spread of the whole site, which is why every substitution is drawn")
    hits = [f"{r.wt}{r.pos}{r.marked} (z = {r.marked_z:.2f})" for r in t.itertuples()
            if r.marked and np.isfinite(r.marked_z)]
    head = ("; ".join(f"{r.wt}{r.pos}: {r.n_below_threshold}/{r.n_substitutions} substitutions at z ≤ {Z_SITE}, "
                      f"median z = {r.median_z:.2f}, q = {r.q:.3f}" for r in t.itertuples())
            + (" | marked: " + ", ".join(hits) if hits else ""))
    return result("a21", cfg, head,
                  {"threshold": Z_SITE, "q_threshold": Q_SITE, "n_sites": len(t),
                   "sites": [{"site": f"{r.wt}{r.pos}", "n": r.n_substitutions, "median_z": r.median_z,
                              "n_below": r.n_below_threshold, "below": r.substitutions_below,
                              "q": r.q, "marked": r.marked, "marked_z": r.marked_z,
                              "marked_below": r.marked_below_threshold} for r in t.itertuples()]},
                  caveats, figs, [tp, vpo])
