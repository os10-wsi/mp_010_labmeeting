"""`python -m mpdms insertion configs/AQR1_fitness.yaml configs/HXT2.yaml ...`

Does the cost of inserting a helix set a floor on what it tolerates?

A transmembrane helix has to be recognised by the translocon and released into the
bilayer, and that recognition is a free-energy calculation the cell performs on the
sequence. A strongly hydrophobic helix inserts with margin to spare, and a substitution
that makes it slightly more polar changes nothing it depends on. A marginal helix has no
margin, and the same substitution pushes it below threshold: the protein is not made, or
is made and degraded.

So the prediction is not about hydrophobicity on its own. It is an INTERACTION - the same
chemical change should cost more in a marginal helix than in a comfortable one - and an
interaction is what the analyses here test, because the main effect of hydrophobicity is
confounded with burial and would be positive whether or not insertion energetics had
anything to do with it.

The scale is dsTbL (Elazar et al. 2016, eLife 5:e12125, "Mutational scanning reveals the
determinants of protein insertion and association energetics in the plasma membrane").
Each residue gets a fourth-order polynomial in z, its depth in angstroms, so the cost of
a charge in the middle of the bilayer and the cost of the same charge near the headgroups
are different numbers - which is the whole point for a question about depth. Energies are
kcal/mol RELATIVE TO ALANINE, which is why alanine's coefficients are all zero, and that
reference cancels in every difference taken below.

Coefficients are from the authors' own implementation, polyval_Kfix_14Jan2016.txt in
FleishmanLab/membrane_prediction, which is also where the convention comes from: a
segment is laid across the membrane at linspace(-15, +15) over its residues and its
insertion energy is the sum of the per-residue values at those depths.

Two caveats worth keeping in view. The assay measured abundance, so a variant that fails
to insert and a variant that inserts and then misfolds look the same here. And dsTbL was
measured in a single-span reporter, while these are twelve-span transporters whose
helices insert with help from their neighbours; the scale is a hypothesis about them, not
a measurement of them.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .analyses.base import missense
from .config import REPO_ROOT, load_config, protein_sequence
from .environs import grid, panel
from .features import position_table
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_insertion"
HALF_WIDTH = 15.0      # the membrane half-width the dsTbL polynomials were fitted over
MIN_POSITIONS = 6      # per helix, before its tolerance or its slope is used
MIN_VARIANTS = 25      # per helix, before a slope of effect on ddG is fitted

# dsTbL depth-dependent insertion energies: coefficients of a fourth-order polynomial in
# z (angstroms from the membrane centre), kcal/mol relative to alanine. Elazar et al.
# 2016 eLife 5:e12125; values as published in the authors' polyval_Kfix_14Jan2016.txt.
DSTBL = {
    "A": ( 0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00,  0.000000e+00),
    "V": (-6.180000e-06, -1.270000e-05,  2.966166e-03, -1.417610e-02, -5.954630e-01),
    "L": (-1.780000e-05,  7.440000e-05,  8.680688e-03, -5.076325e-02, -1.923640e+00),
    "I": (-9.690000e-06, -7.220000e-05,  6.325572e-03,  2.026450e-02, -1.573193e+00),
    "M": (-1.120000e-05,  7.760000e-06,  6.434355e-03, -2.809692e-02, -8.014289e-01),
    "F": (-1.540000e-05, -1.140550e-04,  1.102518e-02,  3.441842e-02, -1.631895e+00),
    "W": (-6.540000e-06,  1.203450e-04,  4.940893e-03, -4.862618e-02, -3.538663e-01),
    "Y": (-7.950000e-06,  1.870060e-04,  3.292640e-03, -5.676937e-02,  8.367785e-01),
    "C": (-1.160000e-05, -1.059830e-04,  8.026061e-03,  8.923340e-04, -1.217637e-01),
    "G": ( 3.690000e-06,  1.040000e-05, -4.691233e-03,  2.342918e-03,  1.577799e+00),
    "P": (-1.450000e-06,  7.090000e-05, -1.121142e-03, -1.644234e-02,  1.866536e+00),
    "S": (-1.520000e-06,  8.920000e-05,  1.173794e-03, -1.639046e-02,  9.236566e-02),
    "T": ( 1.440000e-06, -1.920000e-05, -1.343931e-03,  2.761496e-02,  8.251583e-01),
    "N": ( 3.680000e-06, -2.030000e-05, -2.404001e-03, -1.252517e-02,  1.277578e+00),
    "Q": (-2.070000e-06,  1.293950e-04,  1.740833e-03, -2.409372e-02,  9.714042e-01),
    "H": (-3.710000e-07,  2.114040e-04, -3.086389e-03, -2.885599e-02,  1.848963e+00),
    "K": (-4.000000e-06, -1.175020e-04,  1.550000e-05,  9.811564e-02,  1.511774e+00),
    "R": ( 5.980000e-06, -1.215680e-04, -4.426603e-03,  9.780617e-02,  7.876663e-01),
    "D": (-2.920000e-06, -1.817090e-04,  2.193116e-03,  3.670131e-02,  1.540218e+00),
    "E": (-1.120000e-06,  5.460000e-06, -1.976580e-04, -4.399833e-03,  1.830651e+00),
}


def residue_dg(aa: str, z: float = 0.0) -> float:
    """Insertion energy of one residue at depth z, kcal/mol relative to alanine."""
    c = DSTBL.get(aa)
    if c is None:
        return np.nan
    return float(np.polyval(c, np.clip(z, -HALF_WIDTH - 5, HALF_WIDTH + 5)))


def segment_depths(length: int, orientation: str = "in_out") -> np.ndarray:
    """Where a segment's residues sit, laid across the membrane as the scale lays them.

    Evenly from -15 to +15 angstroms, and NEGATIVE z is the cytoplasmic side: that is
    the convention in the authors' own code, which scores an inner tail at -20 and an
    outer tail at +20.

    Orientation has to be respected, because these profiles are not symmetric. Lysine
    costs +0.24 kcal/mol at z = -15 and +2.39 at z = +15, and arginine is actually
    favourable on the inner side - the positive-inside rule, sitting inside the scale as
    measured. Laying every helix N-to-C from -15 to +15 would therefore score every
    second helix of a transporter backwards, and would manufacture a difference between
    helices that is really a difference in which way they point.
    """
    z = np.linspace(-HALF_WIDTH, HALF_WIDTH, num=max(length, 1), endpoint=True)
    return z if (orientation or "in_out") == "in_out" else z[::-1]


def segment_dg(seq: str, depths: np.ndarray | None = None,
               orientation: str = "in_out") -> float:
    """Insertion energy of a whole segment: the sum over its residues at their depths."""
    if not seq:
        return np.nan
    z = segment_depths(len(seq), orientation) if depths is None else np.asarray(depths, float)
    v = [residue_dg(a, zi) for a, zi in zip(seq, z)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.sum(v)) if v else np.nan


def helix_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per TM helix: its insertion energy, and how tolerant it is."""
    seq = protein_sequence(cfg) or ""
    segs = [s for s in (cfg.get_path("topology.segments", []) or [])
            if s.get("type") == "TM"]
    pt = position_table(df, cfg).set_index("pos")
    eff = missense(df).dropna(subset=["score_z"]).groupby("pos").score_z.mean()
    rows = []
    for s in segs:
        a, b = int(s["start"]), int(s["end"])
        sub = seq[a - 1:b]
        e = eff[eff.index.to_series().between(a, b)]
        if len(sub) < 5 or len(e) < MIN_POSITIONS:
            continue
        rsa = pt.rsa.reindex(range(a, b + 1)) if "rsa" in pt else pd.Series(dtype=float)
        orient = s.get("orientation") or "in_out"
        dg = segment_dg(sub, orientation=orient)
        rows.append({"helix": s["name"], "start": a, "end": b, "length": len(sub),
                     "orientation": orient, "dg": dg, "dg_per_residue": dg / len(sub),
                     "mean_effect": float(e.mean()), "median_effect": float(e.median()),
                     "se_effect": float(e.std(ddof=1) / np.sqrt(len(e))),
                     "n_positions": int(len(e)),
                     "mean_rsa": float(rsa.mean()) if rsa.notna().any() else np.nan})
    t = pd.DataFrame(rows)
    if len(t):
        # "marginal" is relative to this protein's own helices. An absolute kcal/mol cut
        # would mostly sort the proteins, since a longer helix has a more negative dG by
        # construction, and the question is about helices within a protein.
        t["marginal"] = (t.dg >= t.dg.median()).astype(int)
        t = t.sort_values("start").reset_index(drop=True)
    return t


def variant_frame(df: pd.DataFrame, cfg, helices: pd.DataFrame) -> pd.DataFrame:
    """Missense variants inside a TM helix, each with the ddG its substitution implies.

    ddG is evaluated at the residue's own depth, so the same substitution is a different
    number near the headgroups and in the middle of the bilayer. Depth comes from where
    the residue sits in its helix, laid across the membrane the way the scale was built;
    `--depth structure` uses the AlphaFold membrane frame instead where there is one.
    """
    if not len(helices):
        return pd.DataFrame()
    d = missense(df).dropna(subset=["score_z"])[["pos", "wt", "mut", "score_z"]].copy()
    pt = position_table(df, cfg)
    keep = [c for c in ("pos", "rsa", "abs_z", "zdepth", "plddt") if c in pt.columns]
    d = d.merge(pt[keep], on="pos", how="left")
    at = {}
    for h in helices.itertuples():
        z = segment_depths(h.length, getattr(h, "orientation", "in_out"))
        for k, p in enumerate(range(h.start, h.end + 1)):
            at[p] = (h.helix, h.dg, h.marginal, float(z[k]))
    d = d[d.pos.isin(at)].copy()
    if not len(d):
        return d
    d["helix"] = [at[p][0] for p in d.pos]
    d["helix_dg"] = [at[p][1] for p in d.pos]
    d["marginal"] = [at[p][2] for p in d.pos]
    d["z_ideal"] = [at[p][3] for p in d.pos]
    return d


def add_ddg(d: pd.DataFrame, depth: str = "ideal") -> pd.DataFrame:
    """ddG of each substitution at its residue's depth. `depth` is 'ideal' or 'structure'."""
    if not len(d):
        return d
    z = d.z_ideal.to_numpy()
    if depth == "structure" and "zdepth" in d and d.zdepth.notna().any():
        z = np.where(d.zdepth.notna(), d.zdepth.fillna(0).to_numpy(), z)
    out = d.copy()
    out["z_used"] = z
    out["ddg"] = [residue_dg(m, zi) - residue_dg(w, zi)
                  for m, w, zi in zip(out.mut, out.wt, z)]
    return out.dropna(subset=["ddg"])


# ------------------------------------------- i01 does a helix's dG predict its tolerance?
def i01_helix_tolerance(helices: dict, names: list[str], outdir: Path) -> dict:
    """Mean effect of a helix against the energy of inserting it.

    The weak version of the claim, and the one to check first: if insertion energetics
    matter at all, a helix that only just inserts should be less tolerant on average
    than one that inserts comfortably.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.0, h=2.8)
    for i, (g, ax) in enumerate(pairs):
        h = helices[g]
        if len(h) < 4:
            continue
        ax.errorbar(h.dg, h.mean_effect, yerr=h.se_effect, fmt="o", ms=6, lw=0,
                    elinewidth=0.8, color=P.DARK_RED, mec=P.INK, mew=0.5, capsize=2)
        P.label_points(ax, h.dg.to_numpy(), h.mean_effect.to_numpy(),
                       h.helix.tolist(), fontsize=6)
        if len(h) > 3 and np.ptp(h.dg) > 0:
            b, a = np.polyfit(h.dg, h.mean_effect, 1)
            xs = np.linspace(h.dg.min(), h.dg.max(), 20)
            ax.plot(xs, a + b * xs, color=P.INK, lw=1.0, ls=(0, (4, 3)))
            rho, p = ss.spearmanr(h.dg, h.mean_effect)
            rows.append({"protein": g, "n_helices": int(len(h)), "slope": float(b),
                         "spearman": float(rho), "p": float(p),
                         "dg_min": float(h.dg.min()), "dg_max": float(h.dg.max())})
            panel(ax, i, g, f"ρ = {rho:+.2f}, p = {p:.2g}", size=8)
        else:
            panel(ax, i, g)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔG of inserting the helix (kcal/mol, dsTβL)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect over the helix", fontsize=7.5)
    fig.suptitle("Left = inserts comfortably, right = marginal. The claim predicts a "
                 "downward slope: a marginal helix should be less tolerant.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i01_helix_dg_vs_tolerance", None, "I01")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "i01_helix_dg_vs_tolerance.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} ρ {r.spearman:+.2f} "
                     f"(p {r.p:.2g}, {r.n_helices} helices)" for r in per.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# --------------------------------------------------- i02 the interaction: the real test
def i02_interaction(variants: dict, helices: dict, names: list[str],
                    outdir: Path) -> dict:
    """Is a destabilising substitution worse in a marginal helix?

    Two readings of the same question. The panel takes each helix separately, fits the
    slope of effect on ddG inside it, and plots those slopes against the helix's own ΔG:
    the claim predicts a negative trend, because the slope should be steeper (more
    negative) the more marginal the helix. The table fits one model per protein,

        effect ~ ddG + helix ΔG + ddG:helix ΔG + RSA

    where the interaction term is the claim and RSA is in it because a substitution that
    raises ddG is usually also one that buries badly, and burial alone would produce the
    main effect with no insertion energetics involved.
    """
    import statsmodels.formula.api as smf
    slopes, models = [], []
    fig, pairs, _ = grid(names, w=3.0, h=2.8)
    for i, (g, ax) in enumerate(pairs):
        v, h = variants[g], helices[g]
        for hx, sub in v.groupby("helix"):
            if len(sub) < MIN_VARIANTS or np.ptp(sub.ddg) == 0:
                continue
            r = ss.linregress(sub.ddg, sub.score_z)
            slopes.append({"protein": g, "helix": hx,
                           "helix_dg": float(sub.helix_dg.iloc[0]),
                           "slope": float(r.slope), "se": float(r.stderr),
                           "p": float(r.pvalue), "n": int(len(sub))})
        s = pd.DataFrame([r for r in slopes if r["protein"] == g])
        if len(s) >= 4:
            ax.errorbar(s.helix_dg, s.slope, yerr=s.se, fmt="o", ms=5.5, lw=0,
                        elinewidth=0.8, color=P.DARK_RED, mec=P.INK, mew=0.5, capsize=2)
            P.label_points(ax, s.helix_dg.to_numpy(), s.slope.to_numpy(),
                           s.helix.tolist(), fontsize=6)
            b, a = np.polyfit(s.helix_dg, s.slope, 1)
            xs = np.linspace(s.helix_dg.min(), s.helix_dg.max(), 20)
            ax.plot(xs, a + b * xs, color=P.INK, lw=1.0, ls=(0, (4, 3)))
            rho, p = ss.spearmanr(s.helix_dg, s.slope)
            panel(ax, i, g, f"ρ = {rho:+.2f}, p = {p:.2g}", size=8)
        else:
            panel(ax, i, g)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔG of inserting the helix (kcal/mol)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("slope of effect on ΔΔG, within the helix", fontsize=7.5)

        d = v.dropna(subset=["score_z", "ddg", "helix_dg"]).copy()
        d["rsa_"] = d.rsa.fillna(d.rsa.mean()) if "rsa" in d else 0.0
        if len(d) > 60 and np.ptp(d.helix_dg) > 0 and d.helix.nunique() >= 4:
            # standard errors clustered by helix. helix_dg takes one value per helix, so
            # the interaction term has as many independent units as there are helices,
            # not as there are variants; an ordinary variant-level standard error would
            # treat a twelve-helix protein as though it had thousands of observations of
            # the quantity being tested and would call almost anything significant.
            m = smf.ols("score_z ~ ddg * helix_dg + rsa_", data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d.helix})
            ci = m.conf_int().loc["ddg:helix_dg"]
            models.append({"protein": g, "n": int(len(d)),
                           "ddg": float(m.params["ddg"]),
                           "helix_dg": float(m.params["helix_dg"]),
                           "interaction": float(m.params["ddg:helix_dg"]),
                           "interaction_se": float(m.bse["ddg:helix_dg"]),
                           "interaction_lo": float(ci[0]), "interaction_hi": float(ci[1]),
                           "interaction_p": float(m.pvalues["ddg:helix_dg"]),
                           "rsa": float(m.params.get("rsa_", np.nan)),
                           "n_helices": int(d.helix.nunique()),
                           "r2": float(m.rsquared)})
    fig.suptitle("The claim predicts a downward trend: inside a marginal helix (right), a "
                 "substitution that raises ΔΔG should cost more, so the slope is steeper.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i02_interaction", None, "I02")
    plt.close(fig)
    sl, mo = pd.DataFrame(slopes), pd.DataFrame(models)
    sl.to_csv(outdir / "i02_helix_slopes.csv", index=False)
    mo.to_csv(outdir / "i02_interaction_models.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} interaction "
                     f"{r.interaction:+.4f} [{r.interaction_lo:+.4f}, "
                     f"{r.interaction_hi:+.4f}] p {r.interaction_p:.2g}"
                     for r in mo.itertuples())
    return {"status": "ok", "headline": head, "models": mo.to_dict("records")}


# ----------------------------------------- i03 every helix, and the median of them
N_BINS = 6
MIN_PER_BIN = 5


def i03_helix_curves(variants: dict, helices: dict, names: list[str],
                     outdir: Path) -> dict:
    """Effect against ddG, one line per TM helix, coloured by that helix's own dG.

    This replaces a median split into "marginal" and "comfortable" halves. The split
    threw away the quantity being tested: dG is continuous, the helices of a transporter
    all insert comfortably in absolute terms, and cutting them at their own median
    produced two groups whose difference was a difference in rank rather than in
    energetics. Drawing every helix and colouring it by its dG shows the same data with
    nothing binned away, and the claim becomes something to look for directly - the
    lighter lines, the less insertable helices, should fall away faster on the right.

    The bins are common across the helices of one protein, so the median line is a
    median of comparable numbers rather than of whatever each helix's own quantiles
    happened to be.
    """
    rows, med_rows = [], []
    fig, pairs, axes = grid(names, w=3.3, h=3.0, extra=0.3)
    sm = None
    allh = pd.concat([h for h in helices.values()])
    lo, hi = float(allh.dg.min()), float(allh.dg.max())
    cmap = P.sequential_cmap()
    for i, (g, ax) in enumerate(pairs):
        v = variants[g].dropna(subset=["score_z", "ddg"])
        h = helices[g].set_index("helix")
        if not len(v):
            continue
        edges = np.unique(np.quantile(v.ddg, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            continue
        v = v.assign(bin=pd.cut(v.ddg, edges, include_lowest=True))
        curves = {}
        for hx, sub in v.groupby("helix"):
            m = sub.groupby("bin", observed=True).agg(
                x=("ddg", "mean"), y=("score_z", "mean"), n=("score_z", "size"))
            m = m[m.n >= MIN_PER_BIN].dropna()
            if len(m) < 3:
                continue
            dg = float(h.loc[hx, "dg"])
            curves[hx] = m
            ax.plot(m.x, m.y, "-", lw=1.0, alpha=0.85,
                    color=cmap((dg - lo) / (hi - lo) if hi > lo else 0.5))
            r = ss.linregress(sub.ddg, sub.score_z)
            rows.append({"protein": g, "helix": hx, "dg": dg, "slope": float(r.slope),
                         "se": float(r.stderr), "p": float(r.pvalue), "n": int(len(sub))})
        if curves:
            # median across helices at each common bin: the typical helix, with no
            # helix allowed to dominate by having more variants than the others
            wide = pd.DataFrame({k: m.set_index(m.index).y for k, m in curves.items()})
            xs = pd.concat([m.x for m in curves.values()], axis=1).median(axis=1)
            med = wide.median(axis=1)
            ok = wide.notna().sum(axis=1) >= 3
            ax.plot(xs[ok], med[ok], "-o", ms=4.5, lw=2.4, color=P.INK, zorder=6,
                    label="median of helices")
            med_rows += [{"protein": g, "ddg": float(x), "median_effect": float(y),
                          "n_helices": int(n)}
                         for x, y, n in zip(xs[ok], med[ok], wide.notna().sum(axis=1)[ok])]
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        s = pd.DataFrame([r for r in rows if r["protein"] == g])
        extra = ""
        if len(s) >= 4:
            rho, pv = ss.spearmanr(s.dg, s.slope)
            extra = f"{len(s)} helices, slope vs ΔG ρ = {rho:+.2f} (p = {pv:.2g})"
        panel(ax, i, g, extra, size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔΔG of the substitution at that depth (kcal/mol)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    if hi > lo:
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(lo, hi))
        fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.01).set_label(
            "ΔG of inserting the helix (kcal/mol)", fontsize=8, fontweight="bold")
    fig.suptitle("One line per TM helix, coloured by how readily it inserts: dark = "
                 "inserts easily, light = least insertable of the set. The claim predicts "
                 "the lighter lines fall away faster to the right.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i03_helix_curves", None, "I03")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "i03_helix_slopes.csv", index=False)
    pd.DataFrame(med_rows).to_csv(outdir / "i03_median_curve.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness', '')} ρ(slope, ΔG) "
                     f"{ss.spearmanr(d.dg, d.slope)[0]:+.2f} over {len(d)} helices"
                     for g, d in per.groupby("protein") if len(d) >= 4)
    return {"status": "ok", "headline": head, "by_helix": per.to_dict("records")}


ANALYSES = {"i01": "helix ΔG against its tolerance",
            "i02": "the interaction: is ΔΔG worse in a marginal helix?",
            "i03": "every helix separately, and the median of them"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms insertion",
        description="Does the free energy of inserting a helix set a floor on what it "
                    "tolerates? Tested with the dsTβL scale (Elazar et al. 2016).")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--depth", default="ideal", choices=["ideal", "structure"],
                    help="where a residue sits in the membrane when its ΔΔG is evaluated: "
                         "'ideal' lays the helix across the bilayer the way the scale was "
                         "built, 'structure' uses the AlphaFold membrane frame")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    helices, variants, names = {}, {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        df = load_dataset(cfg)
        h = helix_frame(df, cfg)
        if len(h) < 3:
            print(f"  {cfg.id:<16} skipped: fewer than three TM helices with "
                  f"{MIN_POSITIONS}+ measured positions", flush=True)
            continue
        v = add_ddg(variant_frame(df, cfg, h), a.depth)
        helices[cfg.id], variants[cfg.id] = h, v
        names.append(cfg.id)
        print(f"  {cfg.id:<16} {len(h):>2} helices  ΔG {h.dg.min():+.1f} to "
              f"{h.dg.max():+.1f} kcal/mol (median {h.dg.median():+.1f})  "
              f"{len(v):,} variants in a helix", flush=True)
    if not names:
        raise SystemExit("no protein has enough annotated TM helices with measured "
                         "positions; each needs a topology and a dataset")
    pd.concat([h.assign(protein=g) for g, h in helices.items()],
              ignore_index=True).to_csv(outdir / "helices.csv", index=False)
    pd.concat([v.assign(protein=g) for g, v in variants.items()],
              ignore_index=True).to_csv(outdir / "variants.csv", index=False)
    rows = int(np.ceil(len(names) / a.cols))
    print("\n  panels: " + " | ".join(
        ", ".join(n.replace("_fitness", "") for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)

    out = {}
    print()
    for k in keys:
        try:
            if k == "i01":
                r = i01_helix_tolerance(helices, names, outdir)
            elif k == "i02":
                r = i02_interaction(variants, helices, names, outdir)
            else:
                r = i03_helix_curves(variants, helices, names, outdir)
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    mo = out.get("i02", {}).get("models", [])
    if mo:
        neg = sum(1 for m in mo if m["interaction"] < 0)
        sig = sum(1 for m in mo if m["interaction"] < 0 and m["interaction_p"] < 0.05)
        print(f"\n  the claim predicts a NEGATIVE interaction: {neg}/{len(mo)} proteins "
              f"have one, {sig} of them at p < 0.05")
        print("  (the interaction is the test; the main effect of ΔΔG is confounded with "
              "burial and says nothing about insertion on its own)")
    (outdir / "insertion.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
