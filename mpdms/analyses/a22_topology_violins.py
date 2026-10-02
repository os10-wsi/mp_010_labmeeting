"""A22 - missense fitness distributions by topology class, and helix by helix.

(a) one violin each for transmembrane, intracellular and extracellular positions.
(b) one violin per transmembrane helix, in sequence order from the first to the last.

Both use every missense variant that passes the filters, not position means: the
question is how the distribution of mutational effects differs between compartments,
and collapsing each position to its median would hide the spread that the violin is
there to show. Because variants at the same position are not independent, the tests
below are on positions rather than variants, which is the conservative choice.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import assign_topology
from ..stats import bh_fdr
from .base import dirs, fmt_p, missense, result, skipped

# "lumenal" is what TMHMM's "o" means for an internal-membrane protein; for a plasma
# membrane transporter the same side is extracellular. The label follows the config.
SIDE_CLASS = {"cytosolic": "Intracellular", "lumenal": "Extracellular", "extracellular": "Extracellular"}
CLASS_COLORS = {"Transmembrane": "#B8912F", "Intracellular": "#2F6DB5", "Extracellular": "#1B9E77"}
CLASS_ORDER = ["Transmembrane", "Intracellular", "Extracellular"]
MIN_N = 10


def classify(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Missense variants labelled transmembrane / intracellular / extracellular."""
    d = missense(assign_topology(df, cfg)).dropna(subset=["score_z"]).copy()
    d["topo_class"] = np.where(d.seg_type == "TM", "Transmembrane",
                               d.seg_side.map(SIDE_CLASS).fillna("unassigned"))
    return d[d.topo_class != "unassigned"]


def _helix_key(name: str) -> tuple:
    m = re.search(r"(\d+)", str(name))
    return (int(m.group(1)) if m else 10**6, str(name))


def compare(d: pd.DataFrame, by: str, groups: list[str]) -> pd.DataFrame:
    """Pairwise Mann-Whitney on POSITION medians, BH-corrected; variants are not independent."""
    pos = d.groupby([by, "pos"], observed=True).score_z.median().reset_index()
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa = pos.loc[pos[by] == a, "score_z"].to_numpy()
        xb = pos.loc[pos[by] == b, "score_z"].to_numpy()
        if len(xa) < 3 or len(xb) < 3:
            continue
        u, pv = ss.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append({"a": a, "b": b, "n_pos_a": len(xa), "n_pos_b": len(xb),
                     "median_a": float(np.median(xa)), "median_b": float(np.median(xb)),
                     "difference": float(np.median(xa) - np.median(xb)), "p": float(pv)})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
    return t


def summary(d: pd.DataFrame, by: str, order: list[str]) -> pd.DataFrame:
    g = d[d[by].isin(order)].groupby(by, observed=True)
    t = pd.DataFrame({"n_variants": g.size(), "n_positions": g.pos.nunique(),
                      "median": g.score_z.median(), "iqr": g.score_z.quantile(.75) - g.score_z.quantile(.25),
                      "frac_below_half": g.score_z.apply(lambda v: float((v < -0.5).mean()))})
    return (t.reindex([o for o in order if o in t.index])
             .reset_index().rename(columns={by: "group"}))


def figure_classes(cfg, d: pd.DataFrame, t: pd.DataFrame, st: pd.DataFrame, figdir):
    order = [c for c in CLASS_ORDER if c in set(d.topo_class)]
    fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
    groups = {c: d.loc[d.topo_class == c, "score_z"].to_numpy() for c in order}
    P.violin_box(ax, groups, colors=CLASS_COLORS)
    ax.axhline(0, color="#1B9E77", lw=0.7)
    ax.axhline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)))
    for k, c in enumerate(order):
        n = st.loc[st.group == c]
        if len(n):
            ax.text(k, 1.01, f"n = {int(n.n_variants.iloc[0]):,}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6, color=P.MUTED, clip_on=False)
    if len(t):
        bits = [f"{r.a[:3]} vs {r.b[:3]}: {fmt_p(r.p)}{'*' if r.q < 0.05 else ''}" for r in t.itertuples()]
        ax.set_xlabel("n above each violin = mutations drawn; Mann–Whitney on position medians\n"
                      + "  |  ".join(bits), fontsize=6, labelpad=14)
    ax.set_ylabel("Normalised fitness (missense)")
    ax.set_title("(a) By topology class", loc="left", fontsize=8.5)
    return P.save(fig, figdir, "a22a_topology_classes", cfg, "A22")


def figure_helices(cfg, d: pd.DataFrame, st: pd.DataFrame, kw: dict, figdir):
    tm = d[d.seg_type == "TM"]
    order = sorted(set(tm.segment), key=_helix_key)
    fig, ax = plt.subplots(figsize=(max(4.0, 0.72 * len(order) + 1.8), 3.8), layout="constrained")
    groups = {h: tm.loc[tm.segment == h, "score_z"].to_numpy() for h in order}
    # alternate shading by orientation so the in/out pattern is readable along the protein
    orient = tm.groupby("segment", observed=True).seg_side.first().to_dict()
    cols = {h: ("#B8912F" if orient.get(h) == "in_out" else "#8C6D1F") for h in order}
    P.violin_box(ax, groups, colors=cols)
    med = float(tm.score_z.median())
    ax.axhline(med, color=P.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(0.004, med, f"all TM median {med:.2f}", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6, color=P.MUTED)
    ax.axhline(0, color="#1B9E77", lw=0.7)
    ax.axhline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)))
    for k, h in enumerate(order):
        n = st.loc[st.group == h]
        if len(n):
            ax.text(k, 1.01, f"{int(n.n_variants.iloc[0]):,}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5.5, color=P.MUTED, clip_on=False)
    ax.set_ylabel("Normalised fitness (missense)")
    ax.set_xlabel(f"Transmembrane helix, N to C   (n = mutations; Kruskal–Wallis on position "
                  f"medians: {fmt_p(kw.get('p'))}; dark = out-in)", fontsize=6.5, labelpad=14)
    ax.set_title("(b) By transmembrane helix", loc="left", fontsize=8.5)
    return P.save(fig, figdir, "a22b_helix_violins", cfg, "A22")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    if not (cfg.get_path("topology.segments") or []):
        return skipped("a22", cfg, "no topology in the config - run `mpdms tmhmm` first")
    d = classify(df, cfg)
    if not len(d):
        return skipped("a22", cfg, "no missense variants fall in annotated segments")

    order = [c for c in CLASS_ORDER if (d.topo_class == c).sum() >= MIN_N]
    st = summary(d, "topo_class", order)
    t = compare(d[d.topo_class.isin(order)], "topo_class", order)
    figs = figure_classes(cfg, d[d.topo_class.isin(order)], t, st, figdir)

    tm = d[d.seg_type == "TM"]
    hel = sorted(set(tm.segment), key=_helix_key)
    hst, kw, hcmp = pd.DataFrame(), {}, pd.DataFrame()
    if len(hel) >= 2:
        hst = summary(tm, "segment", hel)
        hpos = [tm[tm.segment == h].groupby("pos").score_z.median().to_numpy() for h in hel]
        hpos = [v for v in hpos if len(v) >= 3]
        if len(hpos) >= 2:
            s, p = ss.kruskal(*hpos)
            kw = {"statistic": float(s), "p": float(p), "n_helices": len(hpos)}
        # each helix against the pooled remainder, on position medians
        rows = []
        allpos = tm.groupby(["segment", "pos"], observed=True).score_z.median().reset_index()
        for h in hel:
            xa = allpos.loc[allpos.segment == h, "score_z"].to_numpy()
            xb = allpos.loc[allpos.segment != h, "score_z"].to_numpy()
            if len(xa) >= 3 and len(xb) >= 3:
                rows.append({"segment": h, "n_pos": len(xa), "median": float(np.median(xa)),
                             "median_other_tm": float(np.median(xb)),
                             "p": float(ss.mannwhitneyu(xa, xb, alternative="two-sided")[1])})
        hcmp = pd.DataFrame(rows)
        if len(hcmp):
            hcmp["q"] = bh_fdr(hcmp.p)
        figs = figs + figure_helices(cfg, d, hst, kw, figdir)

    tp = tabdir / "a22_topology_summary.csv"
    pd.concat([st.assign(level="topology_class"), hst.assign(level="helix")]).to_csv(tp, index=False)
    cp = tabdir / "a22_comparisons.csv"
    pd.concat([t.assign(level="topology_class"), hcmp.assign(level="helix_vs_other_tm")]).to_csv(cp, index=False)

    caveats = ["tests are on position medians, not variants: variants at one position share that "
               "position's biology and are not independent observations",
               "'Extracellular' is TMHMM's outside; for an internal-membrane protein that side is the "
               "organelle lumen, so read the label as 'non-cytosolic'"]
    missing = [c for c in CLASS_ORDER if c not in order]
    if missing:
        caveats.append(f"not plotted (fewer than {MIN_N} missense variants): {', '.join(missing)}")
    sig = t[t.q < 0.05] if len(t) else t
    head = ("; ".join(f"{r.group} median {r.median:.2f} (n = {int(r.n_variants):,} mutations "
                      f"at {int(r.n_positions)} positions)" for r in st.itertuples())
            + (" | " + ", ".join(f"{r.a} vs {r.b} q = {r.q:.3g}" for r in sig.itertuples())
               if len(sig) else " | no class difference at q<0.05")
            + (f" | helices differ: {fmt_p(kw['p'])}" if kw else ""))
    return result("a22", cfg, head,
                  {"classes": [{"class": r.group, "median": r.median, "n_positions": int(r.n_positions),
                                "n_variants": int(r.n_variants), "iqr": r.iqr} for r in st.itertuples()],
                   "class_comparisons": [{"a": r.a, "b": r.b, "difference": r.difference,
                                          "p": r.p, "q": r.q} for r in t.itertuples()],
                   "kruskal_across_helices": kw,
                   "helices": [{"helix": r.segment, "median": r.median, "n_positions": int(r.n_pos),
                                "p_vs_other_tm": r.p, "q": r.q} for r in hcmp.itertuples()]},
                  caveats, figs, [tp, cp])
