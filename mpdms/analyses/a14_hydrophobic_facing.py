"""A14 - hydrophobic TM residues (WT A V L I F M W C): lipid-facing (RSA > 0.25) vs buried
(RSA < 0.10), crossed with surface vs core helices, for each substitution type:
rest (all missense except Pro/Gly), -> polar/charged, -> hydrophobic, -> Pro, -> Gly.

Question: do buried hydrophobics in core helices behave like a soluble protein core
(sensitive to size/shape: volume change) while lipid-facing ones mainly need hydrophobicity?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .. import plotting as P
from ..annot import BIOLOGICAL, VOLUME
from ..helixgeom import CLASS_COLORS, HYDROPHOBIC, SUB_COLORS, tm_residue_table, variant_table
from .base import dirs, fmt_p, result, skipped

SUBTYPES = {"rest": "Missense excl. Pro/Gly", "to_polar": "→ polar/charged", "to_hydrophobic": "→ hydrophobic",
            "P": "→ Pro", "G": "→ Gly"}
ST_COLORS = {"rest": SUB_COLORS["rest"], "to_polar": "#1B4B80", "to_hydrophobic": "#7FB2DC",
             "P": SUB_COLORS["P"], "G": SUB_COLORS["G"]}
CELLS = [("buried", "core"), ("lipid", "core"), ("buried", "surface"), ("lipid", "surface")]


def _subset(v, st):
    return v[~v.subtype.isin(["P", "G"])] if st == "rest" else v[v.subtype == st]


def cell_stats(v: pd.DataFrame, split: str, rng) -> pd.DataFrame:
    rows = []
    for st in SUBTYPES:
        s = _subset(v, st)
        for fc, hc in CELLS:
            x = s[(s.facing == fc) & (s[split] == hc)].score_z.to_numpy()
            if len(x):
                b = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
                lo, hi = np.quantile(b, [.025, .975])
            else:
                lo = hi = np.nan
            rows.append({"split": split, "subtype": st, "facing": fc, "helix_class": hc, "n_variants": len(x),
                         "n_positions": s[(s.facing == fc) & (s[split] == hc)].pos.nunique(),
                         "median": np.median(x) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi,
                         "mean": x.mean() if len(x) else np.nan})
    return pd.DataFrame(rows)


def model_stats(v: pd.DataFrame, split: str) -> pd.DataFrame:
    """score ~ facing * helix_class * subtype (+ protein), cluster-robust by position."""
    d = v[v.facing.isin(["lipid", "buried"]) & v[split].isin(["surface", "core"])
          & v.subtype.isin(["to_polar", "to_hydrophobic", "P", "G"])].copy()
    d = d.rename(columns={split: "hclass"})
    d["cid"] = d.dataset_id + ":" + d.pos.astype(str)
    rows = []
    if len(d) < 30 or d.facing.nunique() < 2 or d.hclass.nunique() < 2:
        return pd.DataFrame()
    form = "score_z ~ C(facing, Treatment('buried')) * C(hclass, Treatment('core')) * C(subtype, Treatment('to_hydrophobic'))"
    if d.dataset_id.nunique() > 1:
        form += " + C(dataset_id)"
    try:
        m = smf.ols(form, data=d).fit(cov_type="cluster", cov_kwds={"groups": d.cid})
    except Exception as e:
        return pd.DataFrame([{"split": split, "term": "error", "p": np.nan, "note": str(e)[:80]}])
    terms = {k: k for k in m.params.index if k != "Intercept" and "dataset_id" not in k}
    for k in terms:
        rows.append({"split": split, "term": k.replace("C(facing, Treatment('buried'))", "facing")
                     .replace("C(hclass, Treatment('core'))", "helix").replace("C(subtype, Treatment('to_hydrophobic'))", "sub"),
                     "coef": m.params[k], "lo": m.conf_int().loc[k, 0], "hi": m.conf_int().loc[k, 1], "p": m.pvalues[k]})
    # joint tests
    for name, pat in (("facing x helix", lambda k: ":" in k and "facing" in k and "hclass" in k and "subtype" not in k),
                      ("facing x subtype", lambda k: k.count(":") == 1 and "facing" in k and "subtype" in k),
                      ("3-way", lambda k: k.count(":") == 2)):
        ks = [k for k in m.params.index if pat(k)]
        if ks:
            w = m.wald_test(" = 0, ".join(ks) + " = 0", scalar=True)
            rows.append({"split": split, "term": f"JOINT {name}", "coef": np.nan, "lo": np.nan, "hi": np.nan,
                         "p": float(w.pvalue)})
    return pd.DataFrame(rows)


def slope_stats(v: pd.DataFrame, split: str) -> pd.DataFrame:
    """Within each cell (rest substitutions): fitness ~ d_hydrophobicity + d_volume (standardised)."""
    s = _subset(v, "rest").copy()
    s["d_hyd"] = s.mut.map(BIOLOGICAL) - s.wt.map(BIOLOGICAL)
    s["d_abs_vol"] = (s.mut.map(VOLUME) - s.wt.map(VOLUME)).abs()
    for c in ("d_hyd", "d_abs_vol"):
        s[c] = (s[c] - s[c].mean()) / s[c].std()
    s["cid"] = s.dataset_id + ":" + s.pos.astype(str)
    rows = []
    for fc, hc in CELLS:
        d = s[(s.facing == fc) & (s[split] == hc)].dropna(subset=["d_hyd", "d_abs_vol", "score_z"])
        if len(d) < 15 or d.cid.nunique() < 4:
            continue
        m = smf.ols("score_z ~ d_hyd + d_abs_vol", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.cid})
        for k in ("d_hyd", "d_abs_vol"):
            rows.append({"split": split, "facing": fc, "helix_class": hc, "term": k, "coef": m.params[k],
                         "lo": m.conf_int().loc[k, 0], "hi": m.conf_int().loc[k, 1], "p": m.pvalues[k], "n": len(d)})
    return pd.DataFrame(rows)


def fig_cells(v, cells, slopes, split, title, rng):
    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, len(SUBTYPES), height_ratios=[1, 0.75], hspace=0.55, wspace=0.3)
    xlab = [f"{f}\n{h}" for f, h in CELLS]
    for j, st in enumerate(SUBTYPES):
        ax = fig.add_subplot(gs[0, j])
        s = _subset(v, st)
        for k, (fc, hc) in enumerate(CELLS):
            x = s[(s.facing == fc) & (s[split] == hc)].score_z.to_numpy()
            if not len(x):
                continue
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(x)), x, s=3, color=CLASS_COLORS[hc], alpha=0.35, lw=0)
            r = cells[(cells.subtype == st) & (cells.facing == fc) & (cells.helix_class == hc)].iloc[0]
            ax.errorbar(k, r["median"], yerr=[[r["median"] - r.ci_lo], [r.ci_hi - r["median"]]], fmt="o",
                        color=P.INK, ms=4, mfc="white" if fc == "lipid" else P.INK, capsize=0, lw=1.2, zorder=3)
            ax.text(k, 0.95, f"n={len(x)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5, color=P.MUTED)
        ax.set_xticks(range(4)); ax.set_xticklabels(xlab, fontsize=6)
        ax.axhline(0, color=P.MUTED, lw=0.4); ax.axhline(-1, color=P.MUTED, lw=0.4, ls=(0, (2, 2)))
        ax.set_ylim(-1.8, 0.8)
        ax.set_title(SUBTYPES[st], fontsize=8, color=ST_COLORS[st])
        if j == 0:
            ax.set_ylabel("Normalised fitness (variants)\n● buried  ○ lipid-facing, median ± 95% CI", fontsize=7)
        else:
            ax.set_yticklabels([])
    ax = fig.add_subplot(gs[1, :3])
    if len(slopes):
        for k, (fc, hc) in enumerate(CELLS):
            for t_, off, col in (("d_hyd", -0.12, "#1B4B80"), ("d_abs_vol", 0.12, "#B07AA1")):
                r = slopes[(slopes.facing == fc) & (slopes.helix_class == hc) & (slopes.term == t_)]
                if len(r):
                    r = r.iloc[0]
                    ax.errorbar(k + off, r.coef, yerr=[[r.coef - r.lo], [r.hi - r.coef]], fmt="o", ms=4, color=col,
                                capsize=0, label={"d_hyd": "Δ hydrophobicity (biological)", "d_abs_vol": "|Δ volume|"}[t_] if k == 0 else None)
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(xlab, fontsize=6.5)
        ax.set_ylabel("Std. coefficient (±95% CI)", fontsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5)
    ax.set_title("What drives tolerance? Within-cell regression (missense excl. Pro/Gly)", loc="left", fontsize=8)
    fig.suptitle(f"{title} — hydrophobic TM residues: lipid-facing vs buried × {split.replace('class_', '')} helix class",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def analyse(tables, variants, title, outdir, cfg=None) -> dict:
    figdir, tabdir = dirs(outdir)
    rng = np.random.default_rng(14)
    v = pd.concat(variants, ignore_index=True)
    v = v[v.confident & v.wt.isin(list(HYDROPHOBIC))]
    v = v[v.facing.isin(["lipid", "buried"])]
    out = {"figs": [], "tabs": [], "cells": [], "model": [], "slopes": []}
    for split in ("class_median", "class_tertile"):
        cells, model, slopes = cell_stats(v, split, rng), model_stats(v, split), slope_stats(v, split)
        out["figs"] += P.save(fig_cells(v, cells, slopes, split, title, rng), figdir,
                              f"a14_hydrophobic_facing_{split.replace('class_', '')}", cfg, "A14")
        out["cells"].append(cells); out["model"].append(model); out["slopes"].append(slopes)
    for name in ("cells", "model", "slopes"):
        tab = pd.concat(out[name], ignore_index=True)
        p = tabdir / f"a14_{name}.csv"
        tab.to_csv(p, index=False)
        out["tabs"].append(p)
        out[name] = tab
    v.to_csv(tabdir / "a14_variants.csv", index=False)
    out["tabs"].append(tabdir / "a14_variants.csv")
    return out


def headline(res) -> str:
    c = res["cells"]
    c = c[(c.split == "class_median") & (c.subtype == "to_hydrophobic")].set_index(["facing", "helix_class"])
    m = res["model"]
    j = m[(m.split == "class_median") & m.term.str.startswith("JOINT facing x subtype")] if len(m) else m
    try:
        s = (f"hydrophobic→hydrophobic: buried-core {c.loc[('buried', 'core'), 'median']:.2f} vs "
             f"lipid-facing-surface {c.loc[('lipid', 'surface'), 'median']:.2f}")
    except KeyError:
        s = "too few hydrophobic TM residues in some cells"
    if len(j):
        s += f"; facing × substitution interaction {fmt_p(j.iloc[0].p)}"
    return s


def run(df, cfg, outdir):
    p = cfg.resolve(cfg.get_path("structure.path"))
    if p is None or not p.exists():
        return skipped("a14", cfg, "no structure (needed for RSA)")
    t = tm_residue_table(df, cfg)
    if t.empty:
        return skipped("a14", cfg, "no TM residues")
    res = analyse([t], [variant_table(df, t)], cfg.display_name, outdir, cfg)
    caveats = ["RSA on the AlphaFold monomer without a membrane: 'lipid-facing' = exposed in the isolated chain",
               "intermediate RSA (0.10-0.25) residues are excluded from the facing split"]
    small = res["cells"][(res["cells"].n_positions < 5) & (res["cells"].split == "class_median")]
    if len(small):
        caveats.append(f"{len(small)} cells have <5 positions - medians unstable")
    return result("a14", cfg, headline(res), {"n_hydrophobic_tm_positions": int(t[t.wt.isin(list(HYDROPHOBIC))].shape[0])},
                  caveats, res["figs"], res["tabs"])
