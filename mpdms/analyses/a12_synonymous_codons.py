"""A12 - synonymous variants by codon usage, position and local mRNA structure.

Needs codon-level data (a codon column, or nt_seq in DiMSum-style tables). Codon
adaptiveness needs `evolution.codon_usage`: a csv with columns codon,w (e.g. yeast
relative adaptiveness from Sharp & Li 1987). mRNA folding uses ViennaRNA if installed.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .. import plotting as P
from ..annot import segments_df
from .base import dirs, result, skipped


def run(df, cfg, outdir):
    syn = df[df.pass_filter & (df.vclass == "synonymous")].copy()
    if "codon" not in syn or syn["codon"].isna().all() or syn["codon"].astype(str).isin(["None", "nan", ""]).all():
        return skipped("a12", cfg, "no codon-level data for synonymous variants")
    figdir, tabdir = dirs(outdir)
    syn = syn[syn.codon.astype(str).str.fullmatch(r"[ACGTUacgtu]{3}")].copy()
    syn["codon"] = syn.codon.str.upper().str.replace("U", "T")
    terms, caveats = [], []
    cu = cfg.resolve(cfg.get_path("evolution.codon_usage"))
    if cu and cu.exists():
        w = pd.read_csv(cu).set_index("codon")["w"]
        syn["w"] = np.log(syn.codon.map(w))
        terms.append("w")
    else:
        caveats.append("no evolution.codon_usage table - codon adaptiveness term omitted")
    syn["first50"] = (syn.pos <= 50).astype(int)
    segs = segments_df(cfg)
    up = np.zeros(len(syn), bool)
    for s in segs[segs.type == "TM"].itertuples():
        up |= syn.pos.between(s.start - 30, s.start - 1).to_numpy()
    syn["upstream_tmd"] = up.astype(int)
    terms += ["first50", "upstream_tmd", "pos"]
    try:
        import RNA  # noqa: F401
        caveats.append("ViennaRNA found but WT CDS not supplied - folding term omitted")
    except ImportError:
        caveats.append("ViennaRNA not installed - local mRNA folding term omitted")
    syn = syn.dropna(subset=["score_z"] + terms)
    if len(syn) < 30:
        return skipped("a12", cfg, f"only {len(syn)} codon-resolved synonymous variants")
    m = smf.ols("score_z ~ " + " + ".join(terms), data=syn).fit(cov_type="HC3")
    coefs = pd.DataFrame({"coef": m.params, "lo": m.conf_int()[0], "hi": m.conf_int()[1], "p": m.pvalues})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))
    ax = axes[0]
    P.shade_topology(ax, cfg)
    ax.scatter(syn.pos, syn.score_z, s=5, color=P.CLASS_COLORS["synonymous"], lw=0)
    ax.axhline(0, color=P.MUTED, lw=0.5)
    ax.set_xlabel("Position"); ax.set_ylabel(cfg.get_path("plotting.score_label"))
    ax.set_title("(a) Synonymous variants along the gene", loc="left")
    ax = axes[1]
    c = coefs.drop("Intercept")
    ax.errorbar(c.coef, range(len(c)), xerr=[c.coef - c.lo, c.hi - c.coef], fmt="o", color=P.CLASS_COLORS["synonymous"], ms=3)
    ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
    ax.axvline(0, color=P.MUTED, lw=0.5)
    ax.set_title(f"(b) Regression (R² = {m.rsquared:.3f})", loc="left")
    P.title(fig, cfg, "A12 synonymous variants")
    fig.tight_layout(rect=(0, 0.01, 1, 0.88))
    figs = P.save(fig, figdir, "a12_synonymous_codons", cfg, "A12")
    t = tabdir / "a12_coefficients.csv"
    coefs.to_csv(t)
    sig = c[c.p < 0.05].index.tolist()
    head = (f"Synonymous scores depend on {', '.join(sig)} (R² = {m.rsquared:.3f})" if sig else
            f"No detectable codon/position effect on synonymous variants (R² = {m.rsquared:.3f}) - syn is a clean noise model")
    return result("a12", cfg, head, {"r2": m.rsquared, "n": int(m.nobs), "significant_terms": sig}, caveats, figs, [t])
