"""A26 - how much a position cares about hydrophobicity, against how buried it is.

For each position, regress the fitness of its missense variants on the change in
hydrophobicity those variants make (von Heijne biological scale, mut - wt). The slope is
that position's hydrophobicity sensitivity, signed so that

    slope > 0   a more hydrophobic residue is better here
    slope < 0   a more hydrophobic residue is worse here

and it is plotted against relative solvent accessibility, as a proxy for burial.

Panel (b) repeats it on surface residues only. That is not just a zoom: in the full plot
the relationship is dominated by the buried/exposed contrast, and restricting to the
exposed set asks the different and harder question of whether accessibility still grades
hydrophobicity preference once everything is at least partly exposed.

For a membrane protein RSA from a monomer means two different things depending on where
the position is, so TM and non-TM are fitted separately rather than pooled - in the slab
an exposed position faces lipid, outside it faces water, and those have opposite
expectations.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import BIOLOGICAL, assign_topology
from ..structure import membrane_frame, residue_table
from .base import dirs, fmt_p, missense, result, skipped

SURFACE_RSA = 0.25     # at or above this, the side chain is exposed
MIN_VARIANTS = 8       # per position, to fit a slope at all
MIN_SPAN = 1.0         # kcal/mol of dhyd the variants must span, else the slope is noise
CLASS_COLORS = {"TM": "#B8912F", "non-TM": "#2F6DB5"}


def position_slopes(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per position: hydrophobicity slope, its SE, RSA, and segment class."""
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not Path(p).exists():
        raise FileNotFoundError("structure.path is null or missing")
    rt = residue_table(p, cfg.get_path("structure.chain", "A"),
                       int(cfg.get_path("structure.numbering_offset", 0) or 0))
    try:
        rt, info = membrane_frame(rt, cfg)
        zcol = ["zdepth"] if "zdepth" in rt else []
    except Exception:
        info, zcol = {"method": "none"}, []
    d = missense(assign_topology(df, cfg)).merge(
        rt[["pos", "aa", "rsa", "plddt"] + zcol], on="pos", how="inner")
    d["dhyd"] = d.mut.map(BIOLOGICAL) - d.wt.map(BIOLOGICAL)
    d = d.dropna(subset=["dhyd", "score_z", "rsa"])

    rows = []
    for pos, g in d.groupby("pos"):
        if len(g) < MIN_VARIANTS or float(np.ptp(g.dhyd)) < MIN_SPAN:
            continue
        x, y = g.dhyd.to_numpy(float), g.score_z.to_numpy(float)
        res = ss.linregress(x, y)
        rows.append({"pos": int(pos), "wt": g.wt.iloc[0], "segment": g.segment.iloc[0],
                     "seg_type": g.seg_type.iloc[0], "rsa": float(g.rsa.iloc[0]),
                     "plddt": float(g.plddt.iloc[0]), "n_variants": len(g),
                     "slope": float(res.slope), "slope_se": float(res.stderr),
                     "r": float(res.rvalue), "p": float(res.pvalue),
                     "median_effect": float(g.score_z.median()),
                     "zdepth": float(g.zdepth.iloc[0]) if zcol else np.nan})
    t = pd.DataFrame(rows)
    if len(t):
        t["cls"] = np.where(t.seg_type == "TM", "TM", "non-TM")
        t["surface"] = t.rsa >= SURFACE_RSA
    return t, info


def confounded(st: dict) -> bool:
    """Is a significant pooled correlation an artefact of the TM/non-TM split?

    TM and non-TM separate on BOTH axes - TM positions prefer hydrophobicity and are a
    little less accessible - so pooling them can manufacture a correlation that exists in
    neither class. Simpson's paradox, and worth saying on the figure rather than leaving
    the pooled number looking as trustworthy as the within-class ones.
    """
    a = st.get("all", {})
    cls = [st.get(c, {}) for c in ("TM", "non-TM")]
    if not a.get("testable") or not all(c.get("testable") for c in cls):
        return False
    within_quiet = all(c["p"] > 0.05 for c in cls)
    separated = abs(cls[0]["median_slope"] - cls[1]["median_slope"]) > 0.05
    return bool(a["p"] < 0.05 and within_quiet and separated)


def fit(t: pd.DataFrame) -> dict:
    """Spearman and an inverse-variance weighted OLS of slope on RSA, per class and pooled."""
    out = {}
    for name, g in [("all", t)] + [(c, t[t.cls == c]) for c in sorted(set(t.cls))]:
        if len(g) < 8:
            out[name] = {"testable": False, "n": len(g)}
            continue
        rho, pv = ss.spearmanr(g.rsa, g.slope)
        w = 1.0 / np.clip(g.slope_se.to_numpy(), 1e-6, None) ** 2
        b, a = np.polyfit(g.rsa.to_numpy(), g.slope.to_numpy(), 1, w=np.sqrt(w))
        out[name] = {"testable": True, "n": len(g), "spearman": float(rho), "p": float(pv),
                     "slope_of_slope": float(b), "intercept": float(a),
                     "median_slope": float(g.slope.median())}
    return out


def figure(cfg, t: pd.DataFrame, res: dict, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharey=True, layout="constrained")
    for ax, (sub, ttl, key) in zip(axes, [
            (t, "(a) All positions", "all"),
            (t[t.surface], f"(b) Surface only (RSA ≥ {SURFACE_RSA})", "surface")]):
        if not len(sub):
            ax.text(0.5, 0.5, "no positions", transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color=P.MUTED)
            continue
        st = res["all_fits"] if key == "all" else res["surface_fits"]
        for c in sorted(set(sub.cls)):
            g = sub[sub.cls == c]
            ax.errorbar(g.rsa, g.slope, yerr=g.slope_se, fmt="o", ms=3, lw=0, elinewidth=0.4,
                        color=CLASS_COLORS[c], alpha=0.75, label=f"{c} (n = {len(g)})")
            f = st.get(c, {})
            if f.get("testable"):
                gr = np.linspace(g.rsa.min(), g.rsa.max(), 20)
                ax.plot(gr, f["intercept"] + f["slope_of_slope"] * gr, color=CLASS_COLORS[c], lw=1.4)
        ax.axhline(0, color=P.INK, lw=0.6)
        if key == "all":
            ax.axvline(SURFACE_RSA, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
        a = st.get("all", {})
        bits = [f"{c}: ρ = {st[c]['spearman']:+.2f}, {fmt_p(st[c]['p'])}"
                for c in sorted(set(sub.cls)) if st.get(c, {}).get("testable")]
        conf = confounded(st)
        pooled = (f"pooled: ρ = {a['spearman']:+.2f}, {fmt_p(a['p'])}"
                  + (" — confounded by the TM/non-TM split, read the within-class values below"
                     if conf else "") + "\n") if a.get("testable") else ""
        ax.set_xlabel("Relative solvent accessibility\n" + pooled + "; ".join(bits),
                      fontsize=6.5, color="#D62728" if conf else P.INK)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_title(ttl, loc="left", fontsize=8.5)
    axes[0].set_ylabel("Hydrophobicity slope per position\n(> 0: more hydrophobic is better)")
    P.title(fig, cfg, "A26 hydrophobicity preference vs burial")
    return P.save(fig, figdir, "a26_hydrophobicity_burial", cfg, "A26")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    try:
        t, info = position_slopes(df, cfg)
    except FileNotFoundError as e:
        return skipped("a26", cfg, str(e))
    if not len(t):
        return skipped("a26", cfg, f"no position had {MIN_VARIANTS}+ variants spanning "
                                   f"{MIN_SPAN} kcal/mol of hydrophobicity change")

    res = {"all_fits": fit(t), "surface_fits": fit(t[t.surface]) if (t.surface).sum() >= 8 else {}}
    figs = figure(cfg, t, res, figdir)
    tp = tabdir / "a26_position_slopes.csv"
    t.sort_values("pos").to_csv(tp, index=False)

    a, s = res["all_fits"].get("all", {}), res["surface_fits"].get("all", {})
    conf_all = confounded(res["all_fits"])
    conf_surf = confounded(res["surface_fits"])
    caveats = []
    if conf_all or conf_surf:
        caveats.append("the pooled RSA-vs-slope correlation is an artefact of the TM/non-TM split: "
                       "the two classes separate on both axes, so pooling manufactures a "
                       "correlation that is absent within each class. Quote the within-class values")
    caveats += [f"RSA is from a monomer ({info.get('method', 'no membrane frame')}); inside the "
               "membrane an exposed position faces lipid, outside it faces water, which is why "
               "TM and non-TM are fitted separately rather than pooled",
               f"a position needs {MIN_VARIANTS}+ measured variants spanning {MIN_SPAN} kcal/mol "
               "of hydrophobicity change before a slope is fitted; "
               f"{len(t)} of the protein's positions qualify",
               "points carry the standard error of their own slope and the line is weighted by "
               "it, so a position measured at few or similar substitutions pulls the fit less"]
    head = (f"{len(t)} positions fitted; "
            + (f"all: ρ(RSA, hydrophobicity slope) = {a['spearman']:+.2f} ({fmt_p(a['p'])})"
               if a.get("testable") else "pooled fit not testable")
            + (f"; surface only (n = {s['n']}): ρ = {s['spearman']:+.2f} ({fmt_p(s['p'])})"
               if s.get("testable") else "; surface-only fit not testable")
            + "".join(f"; {c} median slope {res['all_fits'][c]['median_slope']:+.3f}"
                      for c in ("TM", "non-TM") if res["all_fits"].get(c, {}).get("testable")))
    return result("a26", cfg, head,
                  {"n_positions": len(t), "n_surface": int(t.surface.sum()),
                   "surface_rsa_cut": SURFACE_RSA, "pooled_confounded_all": conf_all,
                   "pooled_confounded_surface": conf_surf, "all": res["all_fits"],
                   "surface": res["surface_fits"], "membrane_frame": info.get("method")},
                  caveats, figs, [tp])
