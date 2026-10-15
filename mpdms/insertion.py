"""`python -m mpdms insertion configs/AQR1_fitness.yaml configs/HXT2.yaml [--scale hessa]`

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

A second scale is available with `--scale hessa`: the position-specific matrix of
Hessa et al. 2007 (Nature 450:1026, supplementary table S2), measured on the translocon
itself with the Lep glycosylation assay rather than by mutational scanning. It differs
from dsTbL in three ways that decide what it can be used for:

    absolute        table S2 is kcal/mol outright, not relative to alanine. Every DDG
                    below is a difference, so the reference cancels and DDG is
                    comparable between scales; a whole-segment DG is not.
    symmetric       every row reads the same from both ends, so the positive-inside
                    rule is absent and helix orientation does nothing. Under dsTbL it
                    does a great deal. The symmetry is asserted at import.
    position only   the published model adds a length correction and a flanking-residue
                    term; table S2 has neither and neither is implemented. Constant per
                    segment, so they cancel in DDG and do not cancel in i01 and i05.

The table is indexed by position in a 19-residue H-segment, not by angstroms, so
`--depth structure` is refused for it rather than extrapolated.

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
from .environs import effect_color, grid, panel
from .features import position_table
from .stats import bh_fdr
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


# Hessa et al. 2007, Nature 450:1026-1030, Supplementary table S2: the position-specific
# DG_app matrix, kcal/mol, for a residue of each type at position i of a 19-residue
# H-segment, i running -9 (N-terminal end) to +9. Transcribed from the published table.
#
# Three things to know before reading a number out of it:
#
#   absolute, not relative to alanine. dsTbL quotes every residue against Ala, which is
#   why Ala is zero there and +0.13 at the centre here. A DDG of a substitution is a
#   difference and the reference cancels, so DDG is comparable between the two scales;
#   an absolute segment DG is not.
#
#   symmetric about the centre by construction. Every row reads the same forwards and
#   backwards (asserted at import). The positive-inside rule is therefore NOT inside
#   this scale, where it is inside dsTbL - lysine costs the same at both ends here and
#   costs ten times more at the outer end there. Helix orientation consequently does
#   nothing under this scale, and the code says so rather than appearing to use it.
#
#   the position term only. The published model adds a length correction and a term for
#   the flanking residues; neither is in table S2 and neither is implemented. They are
#   constant for a given segment, so they cancel in every DDG here, and they do NOT
#   cancel in a whole-segment DG - so i01 and i05, which use absolute helix DG, are
#   reporting the position term alone under this scale.
HESSA_POS = np.arange(-9, 10)
HESSA = {
    "A": (0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.10, 0.12, 0.12, 0.13,
          0.12, 0.12, 0.10, 0.09, 0.07, 0.06, 0.04, 0.03, 0.02),
    "C": (0.00, 0.00, 0.00, 0.00, -0.01, -0.02, -0.03, -0.05, -0.07, -0.08,
          -0.07, -0.05, -0.03, -0.02, -0.01, 0.00, 0.00, 0.00, 0.00),
    "D": (0.44, 0.59, 0.77, 0.96, 1.17, 1.36, 1.54, 1.67, 1.76, 1.79,
          1.76, 1.67, 1.54, 1.36, 1.17, 0.96, 0.77, 0.59, 0.44),
    "E": (0.69, 0.80, 0.92, 1.03, 1.14, 1.23, 1.31, 1.37, 1.41, 1.42,
          1.41, 1.37, 1.31, 1.23, 1.14, 1.03, 0.92, 0.80, 0.69),
    "F": (-0.25, -0.26, -0.26, -0.27, -0.27, -0.27, -0.27, -0.28, -0.28, -0.28,
          -0.28, -0.28, -0.27, -0.27, -0.27, -0.27, -0.26, -0.26, -0.25),
    "G": (0.33, 0.36, 0.38, 0.41, 0.43, 0.45, 0.46, 0.47, 0.48, 0.48,
          0.48, 0.47, 0.46, 0.45, 0.43, 0.41, 0.38, 0.36, 0.33),
    "H": (0.63, 0.72, 0.81, 0.90, 0.98, 1.06, 1.12, 1.16, 1.19, 1.20,
          1.19, 1.16, 1.12, 1.06, 0.98, 0.90, 0.81, 0.72, 0.63),
    "I": (-0.11, -0.14, -0.19, -0.24, -0.29, -0.34, -0.39, -0.43, -0.45, -0.46,
          -0.45, -0.43, -0.39, -0.34, -0.29, -0.24, -0.19, -0.14, -0.11),
    "K": (0.32, 0.46, 0.63, 0.84, 1.07, 1.30, 1.52, 1.69, 1.81, 1.85,
          1.81, 1.69, 1.52, 1.30, 1.07, 0.84, 0.63, 0.46, 0.32),
    "L": (-0.35, -0.37, -0.38, -0.39, -0.40, -0.41, -0.42, -0.42, -0.43, -0.43,
          -0.43, -0.42, -0.42, -0.41, -0.40, -0.39, -0.38, -0.37, -0.35),
    "M": (0.00, 0.00, 0.00, 0.00, -0.01, -0.02, -0.03, -0.05, -0.07, -0.08,
          -0.07, -0.05, -0.03, -0.02, -0.01, 0.00, 0.00, 0.00, 0.00),
    "N": (0.63, 0.73, 0.84, 0.95, 1.05, 1.14, 1.22, 1.28, 1.31, 1.33,
          1.31, 1.28, 1.22, 1.14, 1.05, 0.95, 0.84, 0.73, 0.63),
    "P": (0.48, 0.57, 0.66, 0.76, 0.84, 0.92, 0.99, 1.04, 1.08, 1.09,
          1.08, 1.04, 0.99, 0.92, 0.84, 0.76, 0.66, 0.57, 0.48),
    "Q": (0.54, 0.65, 0.77, 0.89, 1.01, 1.11, 1.21, 1.28, 1.32, 1.33,
          1.32, 1.28, 1.21, 1.11, 1.01, 0.89, 0.77, 0.65, 0.54),
    "R": (0.03, 0.06, 0.13, 0.26, 0.46, 0.73, 1.04, 1.34, 1.57, 1.65,
          1.57, 1.34, 1.04, 0.73, 0.46, 0.26, 0.13, 0.06, 0.03),
    "S": (0.37, 0.43, 0.48, 0.53, 0.58, 0.62, 0.65, 0.68, 0.70, 0.70,
          0.70, 0.68, 0.65, 0.62, 0.58, 0.53, 0.48, 0.43, 0.37),
    "T": (0.04, 0.07, 0.11, 0.17, 0.24, 0.32, 0.40, 0.46, 0.51, 0.53,
          0.51, 0.46, 0.40, 0.32, 0.24, 0.17, 0.11, 0.07, 0.04),
    "V": (0.00, 0.00, 0.00, -0.01, -0.02, -0.05, -0.10, -0.17, -0.22, -0.24,
          -0.22, -0.17, -0.10, -0.05, -0.02, -0.01, 0.00, 0.00, 0.00),
    "W": (-0.24, -0.35, -0.42, -0.39, -0.27, -0.10, 0.07, 0.18, 0.25, 0.27,
          0.25, 0.18, 0.07, -0.10, -0.27, -0.39, -0.42, -0.35, -0.24),
    "Y": (-0.11, -0.19, -0.20, -0.10, 0.08, 0.27, 0.44, 0.54, 0.60, 0.62,
          0.60, 0.54, 0.44, 0.27, 0.08, -0.10, -0.20, -0.19, -0.11),
}
# a transcription error that breaks the symmetry is the realistic failure mode, and it
# is cheap to refuse at import rather than to debug in a figure
for _aa, _v in HESSA.items():
    assert len(_v) == 19, f"HESSA[{_aa}] has {len(_v)} positions, expected 19"
    assert _v == tuple(reversed(_v)), f"HESSA[{_aa}] is not symmetric about i = 0"

SCALES = ("dstbl", "hessa")
# A scale's per-residue value is a dG - the cost of HAVING that residue at that depth,
# one residue at a time. A ddG is a difference of two of them, which is what add_ddg
# forms and what i02-i08 use. The two are not interchangeable and the axis says which.
SCALE_DG = {"dstbl": "ΔG relative to alanine", "hessa": "ΔG_app"}
SCALE_LABEL = {"dstbl": "dsT\u03b2L (Elazar 2016), kcal/mol relative to alanine",
               "hessa": "Hessa 2007 table S2 position term, kcal/mol"}
SCALE = "dstbl"          # set once by main(); every analysis reads it through residue_dg
DEPTH = "ideal"          # likewise: where in the membrane a residue is taken to sit
DEPTH_LABEL = {"ideal": "laid across the bilayer",
               "structure": "from the structure"}


def ddg_axis_label() -> str:
    """What the x axis of i03 and i04 actually holds.

    ddG is evaluated at each residue's own depth, so a point is the cost of that
    substitution where that residue sits, not a number the substitution carries
    everywhere - and which depth that is depends on --depth, so the axis says so
    rather than leaving "that depth" pointing at nothing on the figure.
    """
    return ("\u0394\u0394G of the substitution at its own depth "
            f"(kcal/mol, {SCALE}, {DEPTH_LABEL.get(DEPTH, DEPTH)})")


def hessa_dg(aa: str, z: float = 0.0) -> float:
    """Table S2 at depth `z` angstroms, by linear interpolation between its positions.

    The table is indexed by residue position in a 19-residue segment, not by angstroms,
    and the rest of this module works in angstroms. The two are tied together the only
    way the scale allows: the segment spans the bilayer, so its end positions are its
    ends, and i = z * 9 / 15. For a helix laid out by `segment_depths` this puts residue
    k of L exactly where the table would put residue k of 19, whatever L is; for a
    measured depth it is an interpolation the table was not fitted for, which is why
    --depth structure is refused for this scale.
    """
    v = HESSA.get(aa)
    if v is None:
        return float("nan")
    i = np.clip(z * (9.0 / HALF_WIDTH), -9.0, 9.0)
    return float(np.interp(i, HESSA_POS, v))


def residue_dg(aa: str, z: float = 0.0, scale: str | None = None) -> float:
    """Insertion energy of one residue at depth z, in the units of the chosen scale.

    dsTbL is relative to alanine; Hessa table S2 is absolute. Differences of two calls
    on the same scale are comparable across both; single values are not.
    """
    if (scale or SCALE) == "hessa":
        return hessa_dg(aa, z)
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
                     "orientation": orient, "seq": sub,
                     "dg": dg, "dg_per_residue": dg / len(sub),
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


def add_ddg(d: pd.DataFrame, depth: str = "ideal",
            scale: str | None = None) -> pd.DataFrame:
    """ddG of each substitution at its residue's depth. `depth` is 'ideal' or 'structure'."""
    if not len(d):
        return d
    z = d.z_ideal.to_numpy()
    if depth == "structure" and "zdepth" in d and d.zdepth.notna().any():
        z = np.where(d.zdepth.notna(), d.zdepth.fillna(0).to_numpy(), z)
    out = d.copy()
    out["z_used"] = z
    out["ddg"] = [residue_dg(m, zi, scale) - residue_dg(w, zi, scale)
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
            ax.set_xlabel(f"ΔG of inserting the helix (kcal/mol, {SCALE})", fontsize=7.5)
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


def i03_helix_curves(variants: dict, helices: dict, names: list[str], outdir: Path,
                     all_helices: bool = False) -> dict:
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
        curves, dgs = {}, {}
        for hx, sub in v.groupby("helix"):
            m = sub.groupby("bin", observed=True).agg(
                x=("ddg", "mean"), y=("score_z", "mean"), n=("score_z", "size"))
            m = m[m.n >= MIN_PER_BIN].dropna()
            if len(m) < 3:
                continue
            dg = float(h.loc[hx, "dg"])
            curves[hx], dgs[hx] = m, dg
            r = ss.linregress(sub.ddg, sub.score_z)
            rows.append({"protein": g, "helix": hx, "dg": dg, "slope": float(r.slope),
                         "se": float(r.stderr), "p": float(r.pvalue), "n": int(len(sub))})
        # Only the two extremes by default. Twelve lines show that the helices differ;
        # the hardest and the easiest to insert show BY HOW MUCH, which is the claim,
        # and the mean and median behind them say whether the two are outliers or ends
        # of a spread. --all-helices puts the other ten back.
        if all_helices:
            for hx, m in curves.items():
                ax.plot(m.x, m.y, "-", lw=1.0, alpha=0.8,
                        color=cmap((dgs[hx] - lo) / (hi - lo) if hi > lo else 0.5))
        elif len(curves) >= 2:
            hardest = max(dgs, key=dgs.get)
            easiest = min(dgs, key=dgs.get)
            for hx, col, lab in ((hardest, P.DARK_RED, "hardest to insert"),
                                 (easiest, "#2166AC", "easiest to insert")):
                m = curves[hx]
                ax.plot(m.x, m.y, "-o", ms=4, lw=2.0, color=col, mec=P.INK, mew=0.4,
                        zorder=7, label=f"{hx}, {lab} (ΔG {dgs[hx]:+.1f})")
        if curves:
            # Both summaries, because they answer different questions. Each helix
            # contributes one number per bin either way, so neither is dominated by a
            # helix that happens to carry more variants. The MEDIAN is the typical
            # helix; the MEAN follows a helix that plunges. Where they separate, a few
            # helices are carrying the trend, and that is worth seeing rather than
            # choosing one estimator and hiding it.
            wide = pd.DataFrame({k: m.set_index(m.index).y for k, m in curves.items()})
            xs = pd.concat([m.x for m in curves.values()], axis=1).median(axis=1)
            n_h = wide.notna().sum(axis=1)
            ok = n_h >= 3
            med, mn = wide.median(axis=1), wide.mean(axis=1)
            sem = wide.std(axis=1, ddof=1) / np.sqrt(n_h.clip(lower=1))
            ax.fill_between(xs[ok], (mn - sem)[ok], (mn + sem)[ok], color=P.INK,
                            alpha=0.12, lw=0, zorder=5)
            ax.plot(xs[ok], mn[ok], "-o", ms=4.5, lw=2.4, color=P.INK, zorder=6,
                    label="mean of helices (± SEM)")
            ax.plot(xs[ok], med[ok], "--s", ms=3.8, lw=1.8, color=P.INK, zorder=6,
                    mfc="white", label="median of helices")
            # the pooled curve is what a variant-weighted average gives, which is what
            # the earlier marginal/comfortable figure drew; kept in the table so the
            # two can be compared without a third line on the panel
            pooled = v.groupby("bin", observed=True).score_z.mean()
            med_rows += [{"protein": g, "ddg": float(x), "mean_effect": float(a),
                          "median_effect": float(b), "sem_across_helices": float(c),
                          "pooled_over_variants": float(d), "n_helices": int(n)}
                         for x, a, b, c, d, n in zip(xs[ok], mn[ok], med[ok], sem[ok],
                                                     pooled.reindex(wide.index)[ok], n_h[ok])]
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        s = pd.DataFrame([r for r in rows if r["protein"] == g])
        extra = ""
        if len(s) >= 4:
            rho, pv = ss.spearmanr(s.dg, s.slope)
            extra = f"{len(s)} helices, slope vs ΔG ρ = {rho:+.2f} (p = {pv:.2g})"
        panel(ax, i, g, extra, size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel(ddg_axis_label(), fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
        # Every panel names a DIFFERENT helix with a different dG, so the legend cannot
        # be shared: on the panels after the first it would be labelling lines that are
        # not the ones it names. The black mean and median are the same everywhere, so
        # they are spelled out once on the first panel and in the figure's own note.
        hl, ll = ax.get_legend_handles_labels()
        if all_helices:
            if i == 0 and hl:
                ax.legend(hl, ll, frameon=False, fontsize=6, loc="lower left")
        else:
            keep = [(a, b) for a, b in zip(hl, ll)
                    if i == 0 or "to insert" in b]
            if keep:
                ax.legend(*zip(*keep), frameon=False, fontsize=6, loc="lower left")
    if hi > lo and all_helices:
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(lo, hi))
        fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.01).set_label(
            "ΔG of inserting the helix (kcal/mol)", fontsize=8, fontweight="bold")
    fig.suptitle(("One line per TM helix, coloured by how readily it inserts: dark = "
                  "inserts easily, light = marginal (ΔG above zero would not insert on "
                  "its own)." if all_helices else
                  "Red = the helix of this protein that is hardest to insert, blue = the "
                  "easiest. If insertion energetics set a floor, the red line should fall "
                  "away faster to the right.")
                 + " Solid black = mean over all helices, dashed = median.",
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


# ------------------------------------ i04 the pooled curve, every helix together
def i04_pooled(variants: dict, names: list[str], outdir: Path,
               n_bins: int = 10) -> dict:
    """Effect against ddG over all TM helices at once, averaged over variants.

    The plainest version of the question, and the one the marginal/comfortable figure
    was really drawing before it split the helices in two: take every missense variant
    that sits in a transmembrane helix, bin them by the ddG their substitution implies
    at that residue's depth, and average. One curve per protein, no grouping, no helix
    given a vote of its own.

    Because this is an average over variants, a helix that happens to be better covered
    in the library counts for more than a thin one. That is the right estimator for the
    question "what does a destabilising substitution cost in this protein's membrane
    domain", and the wrong one for "do helices differ" - which is what I03 is for. The
    slope's standard error is still clustered by helix, since variants in one helix
    share whatever else is true of it.
    """
    import statsmodels.formula.api as smf
    rows, fits = [], []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = variants[g].dropna(subset=["score_z", "ddg"])
        if len(v) < 100:
            continue
        b = pd.qcut(v.ddg, n_bins, duplicates="drop")
        m = v.groupby(b, observed=True).agg(x=("ddg", "mean"), y=("score_z", "mean"),
                                            e=("score_z", "sem"), n=("score_z", "size"))
        m = m.dropna()
        ax.fill_between(m.x, m.y - m.e, m.y + m.e, color=P.DARK_RED, alpha=0.18, lw=0)
        ax.plot(m.x, m.y, "-o", ms=4.5, lw=1.8, color=P.DARK_RED, mec=P.INK, mew=0.4)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        rows += [{"protein": g, "ddg": float(r.x), "mean_effect": float(r.y),
                  "sem": float(r.e), "n_variants": int(r.n)} for r in m.itertuples()]
        f = smf.ols("score_z ~ ddg", data=v).fit(
            cov_type="cluster", cov_kwds={"groups": v.helix})
        fits.append({"protein": g, "slope": float(f.params["ddg"]),
                     "se_clustered": float(f.bse["ddg"]),
                     "p": float(f.pvalues["ddg"]),
                     "n_variants": int(len(v)), "n_helices": int(v.helix.nunique()),
                     "spearman": float(ss.spearmanr(v.ddg, v.score_z)[0])})
        panel(ax, i, g, f"slope {fits[-1]['slope']:+.3f} ± {fits[-1]['se_clustered']:.3f} "
                        f"kcal⁻¹, {len(v):,} variants", size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel(ddg_axis_label(), fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
    fig.suptitle("Every missense variant in a TM helix, binned by ΔΔG and averaged. "
                 "Band = SEM over variants; the slope's error is clustered by helix.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i04_pooled_curve", None, "I04")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "i04_pooled_curve.csv", index=False)
    ft = pd.DataFrame(fits)
    ft.to_csv(outdir / "i04_pooled_slopes.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} {r.slope:+.3f} ± "
                     f"{r.se_clustered:.3f} (p {r.p:.2g})" for r in ft.itertuples())
    return {"status": "ok", "headline": head, "by_protein": ft.to_dict("records")}


# -------------------------------------- i05 the helices ranked by how readily they insert
def i05_ranking(helices: dict, names: list[str], outdir: Path) -> dict:
    """Each protein's TM helices ordered by the insertion energy of its wild-type sequence.

    The spread is the premise of every other panel here. If a protein's twelve helices
    all sit within a kcal/mol of one another there is no contrast to test, and if one of
    them is above zero it would not insert on its own at all - which is worth seeing as
    a number before it is used as a colour.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.2, h=3.1, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        h = helices[g].sort_values("dg").reset_index(drop=True)
        if not len(h):
            continue
        y = np.arange(len(h))
        ax.barh(y, h.dg, color=[effect_color(v) for v in h.mean_effect],
                edgecolor=P.INK, linewidth=0.5, height=0.72)
        ax.axvline(0, color=P.INK, lw=1.0)
        for k, r in h.iterrows():
            # inside the bar, at its far end: outside the longest bar the label lands on
            # the tick label for that helix and the two overprint
            inside = r.dg + (-0.35 if r.dg >= 0 else 0.35)
            ax.text(inside, k, f"{r.dg:+.1f}", va="center", fontsize=6.5,
                    ha="right" if r.dg >= 0 else "left",
                    color="white" if r.mean_effect < -0.55 else P.INK)
        ax.set_yticks(y); ax.set_yticklabels(h.helix, fontsize=6.5)
        ax.invert_yaxis()                      # easiest to insert at the top
        pad = 0.18 * (h.dg.max() - h.dg.min() + 1)
        ax.set_xlim(min(h.dg.min() - pad, -pad), max(h.dg.max() + pad, pad))
        n_marg = int((h.dg >= 0).sum())
        panel(ax, i, g, f"{h.dg.min():+.1f} to {h.dg.max():+.1f} kcal/mol"
                        + (f", {n_marg} above zero" if n_marg else ""), size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔG of inserting the wild-type helix (kcal/mol)", fontsize=7.5)
        rows += [{"protein": g, "rank": k + 1, "helix": r.helix, "dg": r.dg,
                  "dg_per_residue": r.dg_per_residue, "length": r.length,
                  "orientation": getattr(r, "orientation", None),
                  "mean_effect": r.mean_effect,
                  "n_positions": r.n_positions, "mean_rsa": r.mean_rsa,
                  "would_not_insert_alone": bool(r.dg >= 0)}
                 for k, r in h.iterrows()]
    fig.suptitle("Ranked within each protein, easiest to insert at the top. Bars are "
                 "coloured by the helix's mean mutational effect, so colour and length "
                 "are independent readings. A bar past zero marks a helix that would not "
                 "insert on its own.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i05_helix_ranking", None, "I05")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "i05_helix_ranking.csv", index=False)
    head = "; ".join(
        f"{g.replace('_fitness','')} easiest {d.iloc[0].helix} {d.iloc[0].dg:+.1f}, "
        f"hardest {d.iloc[-1].helix} {d.iloc[-1].dg:+.1f}"
        + (f" ({int(d.would_not_insert_alone.sum())} above zero)"
           if d.would_not_insert_alone.any() else "")
        for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": head, "ranking": t.to_dict("records")}


def _with_kd(v: pd.DataFrame) -> pd.DataFrame:
    """The variant frame with the Kyte-Doolittle change of each substitution added."""
    from .annot import KYTE_DOOLITTLE as KD
    v = v.copy()
    v["delta_kd"] = [KD.get(m, np.nan) - KD.get(w, np.nan) for m, w in zip(v.mut, v.wt)]
    return v


def i08_ddg_vs_hydrophobicity(variants: dict, names: list[str], outdir: Path) -> dict:
    """The two predictors against each other, one point per mutation.

    i07 asks whether ddG adds anything over plain hydrophobicity and answers with a
    nested R2, which is the right test and an invisible one. This is the same question
    drawn: ddG on y, Kyte-Doolittle change on x, every variant in a TM helix a point.

    The shape carries the answer. Kyte-Doolittle gives one number per substitution
    TYPE, so every L->K in the protein lands on the same x; ddG is evaluated at each
    residue's own depth, so those same variants spread vertically. The vertical smear
    at a given x is therefore exactly what the depth-dependent scale knows and a
    hydrophobicity table cannot: if the cloud is a thin line, ddG is hydrophobicity
    with extra steps, and if it is a thick band, the scale is carrying real information
    about where in the bilayer the substitution sits.

    Points are coloured by depth to say which way that smear runs, and the companion
    figure takes the part of ddG that hydrophobicity cannot explain - the residual of
    ddG regressed on delta KD - and asks whether it predicts the measured effect. That
    residual is the only part of ddG that i07 could not already have had for free.
    """
    import statsmodels.formula.api as smf
    rows = []
    fig, pairs, _ = grid(names, w=3.2, h=3.0, extra=0.35)
    for i, (g, ax) in enumerate(pairs):
        v = _with_kd(variants[g]).dropna(subset=["delta_kd", "ddg", "score_z"])
        if len(v) < 100:
            panel(ax, i, g, "too few variants in a TM helix", size=8)
            continue
        depth = np.abs(v.z_used if "z_used" in v else v.z_ideal)
        sc = ax.scatter(v.delta_kd, v.ddg, c=depth, s=6, cmap="viridis_r", alpha=0.6,
                        lw=0, zorder=2)
        c1, c0 = np.polyfit(v.delta_kd, v.ddg, 1)
        xs = np.linspace(v.delta_kd.min(), v.delta_kd.max(), 20)
        ax.plot(xs, c0 + c1 * xs, color=P.INK, lw=1.2, zorder=3)
        r = float(ss.pearsonr(v.delta_kd, v.ddg).statistic)
        rho = float(ss.spearmanr(v.delta_kd, v.ddg).statistic)
        resid = v.ddg - (c0 + c1 * v.delta_kd)
        # how much of ddG survives hydrophobicity, in the units ddG is quoted in
        spread = float(resid.std(ddof=1))
        # the widest vertical smear at one substitution type: the depth effect, seen
        per_sub = v.assign(sub=v.wt + ">" + v.mut).groupby("sub").ddg.agg(["min", "max",
                                                                          "size"])
        per_sub = per_sub[per_sub["size"] >= 5]
        widest = float((per_sub["max"] - per_sub["min"]).max()) if len(per_sub) else np.nan
        d = v[["score_z", "helix"]].assign(resid=resid).dropna()
        m = smf.ols("score_z ~ resid", data=d).fit(cov_type="cluster",
                                                   cov_kwds={"groups": d.helix})
        rows.append({"protein": g, "n_variants": int(len(v)),
                     "pearson": r, "spearman": rho, "r2": r ** 2,
                     "slope_ddg_on_kd": float(c1),
                     "sd_residual_ddg": spread,
                     "sd_ddg": float(v.ddg.std(ddof=1)),
                     "frac_ddg_variance_left": spread ** 2 / float(v.ddg.var(ddof=1)),
                     "widest_depth_spread_one_substitution": widest,
                     "slope_effect_on_residual": float(m.params["resid"]),
                     "se_effect_on_residual": float(m.bse["resid"]),
                     "p_effect_on_residual": float(m.pvalues["resid"])})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6)
        ax.text(0.03, 0.97, f"r = {r:+.2f}\nR² = {r ** 2:.2f}\nresidual SD "
                            f"{spread:.2f} kcal/mol", transform=ax.transAxes,
                va="top", ha="left", fontsize=6.4)
        panel(ax, i, g, f"{len(v):,} variants", size=8)
        if i == len(names) - 1:
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label(
                "|depth| from the bilayer centre (Å)", fontsize=7)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ hydrophobicity, Kyte–Doolittle (mutant − wild type)",
                          fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel(f"ΔΔG of the substitution ({SCALE}, kcal/mol)", fontsize=7.5)
    fig.suptitle("One point per variant in a TM helix. Kyte–Doolittle gives one value "
                 "per substitution type, so the vertical spread at a given x is what "
                 "the depth-dependent scale adds and a hydrophobicity table cannot "
                 "have: a thin line means the two predictors are the same thing.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i08_ddg_vs_hydrophobicity", None, "I08")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no protein has enough TM variants"}

    # ---- the companion: does what hydrophobicity misses predict anything?
    fig2, pairs2, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs2):
        v = _with_kd(variants[g]).dropna(subset=["delta_kd", "ddg", "score_z"])
        if len(v) < 100:
            panel(ax, i, g, "too few variants in a TM helix", size=8)
            continue
        c1, c0 = np.polyfit(v.delta_kd, v.ddg, 1)
        v = v.assign(resid=v.ddg - (c0 + c1 * v.delta_kd))
        b = pd.qcut(v.resid, 10, duplicates="drop")
        mm = v.groupby(b, observed=True).agg(x=("resid", "mean"), y=("score_z", "mean"),
                                             e=("score_z", "sem")).dropna()
        ax.fill_between(mm.x, mm.y - mm.e, mm.y + mm.e, color=P.DARK_RED, alpha=0.18,
                        lw=0)
        ax.plot(mm.x, mm.y, "-o", ms=4.5, lw=1.8, color=P.DARK_RED, mec=P.INK, mew=0.4)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        rr = [x for x in rows if x["protein"] == g][0]
        panel(ax, i, g, f"slope {rr['slope_effect_on_residual']:+.3f} "
                        f"(p = {rr['p_effect_on_residual']:.2g})", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔΔG not explained by hydrophobicity (kcal/mol)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
    fig2.suptitle("The part of ΔΔG that a hydrophobicity table could not have given "
                  "you, against the measured effect. A flat line here means the "
                  "insertion scale's extra information is not what the assay is "
                  "reading.", fontsize=7, x=0.01, ha="left")
    P.save(fig2, outdir, "i08b_residual_ddg_vs_effect", None, "I08B")
    plt.close(fig2)

    t = pd.DataFrame(rows)
    t.to_csv(outdir / "i08_ddg_vs_hydrophobicity.csv", index=False)
    head = "; ".join(
        f"{r.protein.replace('_fitness', '')} r {r.pearson:+.2f} (R² {r.r2:.2f}), "
        f"{100 * r.frac_ddg_variance_left:.0f}% of ΔΔG left over, which predicts effect "
        f"at {r.slope_effect_on_residual:+.3f} (p {r.p_effect_on_residual:.2g})"
        for r in t.itertuples())
    return {"status": "ok", "headline": head, "by_protein": t.to_dict("records")}


def _i07c_significance(t: pd.DataFrame, names: list[str], outdir: Path) -> None:
    """One bar per helix: how strong the evidence is that its slope is not zero.

    Read this for the ordering, not for the absolute heights. Each helix carries
    hundreds of variants, so a slope of almost any size clears p = 0.05 easily and the
    bars run into tens of orders of magnitude; what separates the helices is the slope
    and its interval, which the companion table holds. The p-values are corrected
    across the helices of each protein (Benjamini-Hochberg) because twelve helices are
    twelve chances, and the bars are coloured by insertion energy so a reader can see
    at once whether the strongest evidence sits with the marginal helices.
    """
    fig, pairs, _ = grid(names, w=3.3, h=2.9, extra=0.3)
    cmap = plt.get_cmap("viridis")
    for i, (g, ax) in enumerate(pairs):
        d = t[t.protein == g].copy()
        if not len(d):
            panel(ax, i, g, "no helix fitted", size=7.5)
            continue
        d = d.sort_values("helix", key=lambda c: c.str.extract(r"(\d+)")[0]
                          .astype(float).fillna(0))
        y = -np.log10(d.q.clip(lower=1e-300))
        lo, hi = float(d.dg.min()), float(d.dg.max())
        cols = [cmap((v - lo) / (hi - lo) if hi > lo else 0.5) for v in d.dg]
        ax.bar(np.arange(len(d)), y, color=cols, edgecolor=P.INK, linewidth=0.4)
        ax.axhline(-np.log10(0.05), color=P.DARK_RED, lw=0.9, ls=(0, (3, 2)))
        ax.text(0.012, -np.log10(0.05), "q = 0.05", color=P.DARK_RED, fontsize=6,
                va="bottom", ha="left", transform=ax.get_yaxis_transform(),
                bbox=dict(fc="white", ec="none", pad=0.6), zorder=8)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d.helix, fontsize=6, rotation=90)
        miss = list(d.loc[d.q >= 0.05, "helix"])
        panel(ax, i, g, f"{int((d.q < 0.05).sum())} of {len(d)} helices at q < 0.05"
                        + (f"; not {', '.join(miss)}" if miss and len(miss) <= 4 else ""),
              size=7.5)
        if i % 2 == 0:
            ax.set_ylabel("−log₁₀ q  (slope ≠ 0, BH over helices)", fontsize=7.5)
        if hi > lo:
            # one bar per panel, not one for the figure: the colour is normalised
            # within a protein, so a shared bar would read as a shared scale and
            # invite comparing a colour in one panel with the same colour in another
            import matplotlib as mpl
            sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(lo, hi))
            cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
            cb.set_label("ΔG of inserting the helix", fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle("Evidence that each helix's hydrophobicity slope is not zero, "
                 "Benjamini-Hochberg corrected over the helices of each protein and "
                 "coloured by insertion ΔG. A bar says only that a helix HAS a slope, "
                 "not how steep: the slope and its interval are in the table, and the "
                 "panel titles count the helices that cleared q = 0.05.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i07c_helix_slope_significance", None, "I07C")
    plt.close(fig)


def even_odd(idx: np.ndarray, y: np.ndarray) -> tuple:
    """Split a profile into its symmetric and antisymmetric halves about depth zero.

    Any profile is exactly even + odd, where even(i) = [y(i) + y(-i)] / 2 and
    odd(i) = [y(i) - y(-i)] / 2. Only depths measured on BOTH sides can contribute:
    an arm with no partner says nothing about symmetry and is dropped rather than
    paired with a zero, which would manufacture asymmetry out of missing data.
    """
    by = dict(zip(np.asarray(idx, int), np.asarray(y, float)))
    evens, odds = [], []
    for i in sorted({abs(k) for k in by}):
        if i == 0:
            if 0 in by:
                evens.append(by[0])
            continue
        if i in by and -i in by:
            evens.append((by[i] + by[-i]) / 2.0)
            odds.append((by[i] - by[-i]) / 2.0)
    return np.array(evens), np.array(odds)


def asymmetry_index(idx: np.ndarray, y: np.ndarray) -> float:
    """How much of a profile's shape is antisymmetric: 0 symmetric, 1 purely odd.

    A ratio of two sums of squares in the same units, so multiplying the profile by
    any constant cancels and adding any constant is removed by centring the even part.
    That is what lets a measured profile in fitness units be compared with a predicted
    one in kcal/mol, and one residue with another on a shared axis.
    """
    evens, odds = even_odd(idx, y)
    if len(odds) < 2 or len(evens) < 2:
        return float("nan")
    num = float(np.sum(odds ** 2))
    den = num + float(np.sum((evens - evens.mean()) ** 2))
    return float(num / den) if den > 0 else float("nan")


def _binned_profile(ipos: np.ndarray, y: np.ndarray, min_n: int):
    """Mean of y in each signed depth bin that has at least `min_n` measurements."""
    out_i, out_y = [], []
    for b in np.unique(ipos):
        m = ipos == b
        if int(m.sum()) >= min_n:
            out_i.append(int(b))
            out_y.append(float(y[m].mean()))
    return np.array(out_i), np.array(out_y)


def i09b_symmetry(variants: dict, names: list[str], outdir: Path,
                  min_per_point: int = 3, bin_width: int = 3,
                  n_perm: int = 1000, n_boot: int = 500) -> dict:
    """Is each residue's measured profile symmetric about the bilayer centre?

    Two references per scale, because a scale's per-residue number is a dG - the cost
    of HAVING that residue at that depth - while the measurement is the cost of
    CHANGING the wild type into it, which is a ddG.

    The dG profile is the published quantity. Hessa's is symmetric by construction,
    every row of S2 reading the same from both ends and asserted at import, so its
    index is exactly zero; dsTbL's is not, because the positive-inside rule is in it as
    measured. The matched ddG - this residue against the wild types actually sitting at
    each depth - need not be symmetric under EITHER scale, since the two halves of the
    membrane hold different wild-type residues. So a tilt in the data is only beyond a
    scale's reach when it exceeds that scale's ddG reference, not merely when it
    exceeds zero.

    The null is exchangeability: if a residue's cost really is symmetric, a variant at
    depth +i and one at -i are interchangeable, so flipping the sign of each variant's
    depth at random and recomputing the index gives a p-value that assumes nothing
    about the distribution. The interval comes from resampling variants.

    The index discards direction, so a signed statistic travels with it: the mean
    effect on the lumenal half minus the mean on the cytoplasmic half. Positive means
    the substitution costs more towards the lumen.
    """
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.default_rng(17)
    rows = []
    for g in names:
        v = variants[g].dropna(subset=["score_z", "z_ideal"]).copy()
        if len(v) < 200:
            continue
        v["ipos"] = np.clip(np.rint(v.z_ideal * (9.0 / HALF_WIDTH)), -9, 9).astype(int)
        if bin_width > 1:
            v["ipos"] = (np.rint(v.ipos / bin_width) * bin_width).astype(int)
        for aa in aas:
            sub = v[v.mut == aa]
            if len(sub) < 20:
                continue
            ipos = sub.ipos.to_numpy(int)
            y = sub.score_z.to_numpy(float)
            bi, by = _binned_profile(ipos, y, min_per_point)
            obs = asymmetry_index(bi, by)
            if not np.isfinite(obs):
                continue
            # the null: a symmetric profile makes +i and -i exchangeable
            null = []
            for _ in range(n_perm):
                flip = np.where(rng.random(len(ipos)) < 0.5, -1, 1)
                a, b = _binned_profile(ipos * flip, y, min_per_point)
                x = asymmetry_index(a, b)
                if np.isfinite(x):
                    null.append(x)
            pval = ((np.sum(np.asarray(null) >= obs) + 1) / (len(null) + 1)
                    if null else np.nan)
            boots = []
            for _ in range(n_boot):
                k = rng.integers(0, len(y), len(y))
                a, b = _binned_profile(ipos[k], y[k], min_per_point)
                x = asymmetry_index(a, b)
                if np.isfinite(x):
                    boots.append(x)
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50
                      else (np.nan, np.nan))
            lum, cyt = y[ipos > 0], y[ipos < 0]
            # Two references per scale, and they are NOT the same thing. The dG
            # profile is the published quantity and Hessa's is symmetric by
            # construction, so its index is exactly zero. The matched ddG - this
            # residue against the wild types actually present at each depth - need
            # not be symmetric at all, because the sequence is not: the two halves of
            # the membrane hold different wild-type residues. A tilt in the data is
            # therefore only beyond Hessa's reach when it also exceeds this one.
            scale_idx, scale_dd = {}, {}
            for sc in SCALES:
                pred = np.array([residue_dg(aa, i * HALF_WIDTH / 9.0, sc) for i in bi])
                scale_idx[sc] = asymmetry_index(bi, pred)
                d = (sub.assign(_d=[residue_dg(aa, zz * HALF_WIDTH / 9.0, sc)
                                    - residue_dg(w, zz * HALF_WIDTH / 9.0, sc)
                                    for w, zz in zip(sub.wt, sub.ipos)])
                     .groupby("ipos")._d.mean())
                d = d.reindex(bi).dropna()
                scale_dd[sc] = (asymmetry_index(d.index.to_numpy(), d.to_numpy())
                                if len(d) >= 4 else np.nan)
            rows.append({"protein": g, "mut": aa, "n_variants": int(len(sub)),
                         "n_bins": int(len(bi)), "asymmetry": obs,
                         "asymmetry_lo": float(lo), "asymmetry_hi": float(hi),
                         "p_permutation": float(pval),
                         "lumenal_minus_cytosolic":
                             float(np.mean(lum) - np.mean(cyt)) if len(lum) and len(cyt)
                             else np.nan,
                         "asymmetry_dstbl_dg": scale_idx.get("dstbl", np.nan),
                         "asymmetry_hessa_dg": scale_idx.get("hessa", np.nan),
                         "asymmetry_dstbl_ddg": scale_dd.get("dstbl", np.nan),
                         "asymmetry_hessa_ddg": scale_dd.get("hessa", np.nan)})
    if not rows:
        return {"status": "skipped", "reason": "no protein has enough TM variants"}
    t = pd.DataFrame(rows)
    t["q"] = np.nan
    for _, ix in t.groupby("protein").groups.items():
        t.loc[ix, "q"] = bh_fdr(t.loc[ix, "p_permutation"].fillna(1.0).to_numpy())
    t.to_csv(outdir / "i09b_profile_symmetry.csv", index=False)

    fig, pairs, _ = grid(names, w=3.4, h=4.2, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = t[t.protein == g].sort_values("asymmetry")
        if not len(d):
            panel(ax, i, g, "no residue measured on both sides", size=7.5)
            continue
        ypos = np.arange(len(d))
        for k, r in enumerate(d.itertuples()):
            if np.isfinite(r.asymmetry_lo):
                ax.plot([r.asymmetry_lo, r.asymmetry_hi], [k, k], color=P.MUTED,
                        lw=1.0, zorder=2)
        col = ["#2166AC" if x > 0 else "#B2182B" for x in d.lumenal_minus_cytosolic]
        ax.scatter(d.asymmetry, ypos, s=34, facecolor=col, edgecolor=P.INK,
                   linewidth=0.4, zorder=4)
        ax.scatter(d.asymmetry_dstbl_ddg, ypos, s=26, facecolor="none",
                   edgecolor="#C0392B", linewidth=0.9, zorder=3, marker="D")
        ax.scatter(d.asymmetry_hessa_ddg, ypos, s=20, facecolor="none",
                   edgecolor="#888888", linewidth=0.8, zorder=3, marker="s")
        for k, r in enumerate(d.itertuples()):
            if np.isfinite(r.q) and r.q < 0.05:
                ax.text(1.0, k, "*", fontsize=9, va="center", ha="right",
                        transform=ax.get_yaxis_transform())
        ax.axvline(0, color="#C0392B", lw=0.9, ls=(0, (1, 2.5)))
        ax.text(0.0, len(d) - 0.2, " Hessa ΔG = 0 for every residue", fontsize=6,
                color="#C0392B", va="top", ha="left")
        ax.set_yticks(ypos)
        ax.set_yticklabels(d.mut, fontsize=7)
        ax.set_ylim(-0.8, len(d) - 0.2)
        ax.set_xlim(-0.03, 1.03)
        n_sig = int((d.q < 0.05).sum())
        panel(ax, i, g, f"{n_sig} of {len(d)} residues asymmetric at q < 0.05", size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("fraction of the profile's shape that is antisymmetric",
                          fontsize=7.5)
    fig.suptitle("How tilted each residue's depth profile is, on a scale-free index: 0 "
                 "symmetric, 1 purely antisymmetric. Filled circle = the data, blue if "
                 "the substitution costs more towards the lumen and red if towards the "
                 "cytosol. The open markers are the two scales' ΔΔG against the wild "
                 "types actually present - diamond dsTβL, square Hessa - which can be "
                 "tilted even for Hessa, because the sequence is not symmetric even "
                 "though its table is. The dotted line is Hessa's ΔG profile, which is "
                 "symmetric by construction. * marks q < 0.05 against a depth-sign "
                 "permutation null.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i09b_profile_symmetry", None, "I09B")
    plt.close(fig)
    head = "; ".join(
        f"{g.replace('_fitness', '')} {int((d.q < 0.05).sum())}/{len(d)} residues "
        f"asymmetric (median index {d.asymmetry.median():.2f}; dsTβL ΔΔG "
        f"{d.asymmetry_dstbl_ddg.median():.2f}, Hessa ΔΔG "
        f"{d.asymmetry_hessa_ddg.median():.2f})" for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": head, "by_residue": t.to_dict("records")}


HYDROPHOBIC_WT = ("A", "V", "L", "I")


def _i09_panel_grid(v, aas, matched, prof, xs_pred, mode, flip, lim, min_per_point):
    """One 4x5 page: blue measured profile per residue, red prediction on a twin axis."""
    sgn = -1.0 if flip else 1.0
    fig, axes = plt.subplots(4, 5, figsize=(11.0, 8.4), layout="constrained",
                             sharex=True, sharey=True)
    flat = [axes[k // 5][k % 5] for k in range(20)]
    drawn = []
    for k, (ax, aa) in enumerate(zip(flat, aas)):
        sub = v[v.mut == aa]
        m = sub.groupby("ipos").agg(y=("score_z", "mean"), e=("score_z", "sem"),
                                    n=("score_z", "size"))
        m = m[m.n >= min_per_point].dropna(subset=["y"])
        ax2 = ax.twinx()
        # zero on the prediction axis too, in its own colour: with two axes in one
        # panel a single zero line reads as belonging to both, and "no predicted
        # cost" sits nowhere near "no measured effect"
        ax2.axhline(0, color="#C0392B", lw=0.7, ls=(0, (1, 2.5)), zorder=1)
        if mode == "dg":
            ax2.plot(xs_pred, sgn * prof[aa], color="#C0392B", lw=1.3, zorder=2)
        elif len(matched[aa]) >= 3:
            ax2.plot(matched[aa].index, sgn * matched[aa].to_numpy(), "-o", ms=3,
                     color="#C0392B", lw=1.3, zorder=2)
        ax2.set_ylim(*(sorted(sgn * np.array(lim[mode]))))
        ax2.tick_params(axis="y", labelsize=5, colors="#C0392B", length=2)
        if k % 5 != 4:                  # only the right-hand column is labelled
            ax2.set_yticklabels([])
        elif k == 9:                    # and one of them carries the name
            base = (f"{SCALE_DG[SCALE]}" if mode == "dg" else "ΔΔG from the wild type")
            ax2.set_ylabel(f"{'−' if flip else ''}{base}, {SCALE} (kcal/mol, red)",
                           fontsize=7.5, color="#C0392B")
        if len(m) >= 3:
            drawn.append((aa, m, sub))
            ax.fill_between(m.index, m.y - m.e, m.y + m.e, color="#2166AC", alpha=0.2,
                            lw=0, zorder=3)
            ax.plot(m.index, m.y, "-", color="#2166AC", lw=1.6, zorder=4)
        else:
            ax.text(0.5, 0.5, "too few", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6, color=P.MUTED)
        ax.axhline(0, color=P.MUTED, lw=0.5, ls=(0, (2, 3)))
        ax.set_title(aa, fontsize=9, pad=2)
        ax.set_xticks([-9, -6, -3, 0, 3, 6, 9])
        ax.tick_params(labelsize=6)
    return fig, drawn


def i09_profiles(variants: dict, names: list[str], outdir: Path,
                 min_per_point: int = 3, bin_width: int = 3) -> dict:
    """Hessa's figure, measured: one panel per mutant residue, effect against position.

    The published scales are drawn this way - twenty panels, position across the
    H-segment on x, the cost of putting that residue there on y - and the shapes carry
    the chemistry: a charge is a steep bell with its peak at the bilayer centre, a
    hydrophobe is flat or inverted, and tryptophan and tyrosine dip at the interfaces
    where they actually sit. This draws the same twenty panels from the measurements.

    Eight pages per protein, from three independent choices.

    WHICH PREDICTION. A scale's per-residue value is a dG, the cost of HAVING that
    residue at that depth; the measurement is the cost of CHANGING the wild type into
    it, a ddG. `profiles` draws the dG, which is the published quantity, and `effect`
    draws the ddG against the wild types actually present, which is what blue measures.

    WHICH SIGN. Energy and effect run opposite ways: a costly substitution has a large
    POSITIVE ddG and a large NEGATIVE effect, so the two curves mirror each other even
    when they agree perfectly. The `flip` pages negate the energy axis, after which
    agreement looks like agreement.

    WHICH WILD TYPES. Pooling every wild type mixes L->K with D->K, which are not the
    same experiment: the second starts from a residue the bilayer already disliked.
    The `avli` pages keep only substitutions FROM alanine, valine, leucine or
    isoleucine, so every panel is a change from a comfortable hydrophobic baseline and
    the profile is about the residue arriving rather than the one leaving.
    """
    from .annot import KYTE_DOOLITTLE as KD
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    rows, out = [], {}
    for g in names:
        v0 = variants[g].dropna(subset=["score_z", "z_ideal"]).copy()
        if len(v0) < 200:
            continue
        # the scale's own coordinate: position in a 19-residue segment, -9 to +9
        v0["ipos"] = np.clip(np.rint(v0.z_ideal * (9.0 / HALF_WIDTH)), -9, 9).astype(int)
        # Nineteen single-residue columns is the published resolution and needs the
        # published number of measurements; one protein's helices give only a handful
        # of residues at each depth per mutant, so depths are pooled bin_width wide.
        if bin_width > 1:
            v0["ipos"] = (np.rint(v0.ipos / bin_width) * bin_width).astype(int)
        # the prediction is continuous in depth and gets its own dense grid; drawn at
        # the data's bin centres it became a polyline
        xs_pred = np.linspace(-9, 9, 181)
        prof = {aa: np.array([residue_dg(aa, x * HALF_WIDTH / 9.0) for x in xs_pred])
                for aa in aas}
        for wt_set, tag_wt in (("all", ""), ("avli", "_avli")):
            v = v0 if wt_set == "all" else v0[v0.wt.isin(HYDROPHOBIC_WT)]
            if len(v) < 100:
                continue
            matched = {}
            for aa in aas:
                sub = v[v.mut == aa]
                matched[aa] = (sub.assign(_d=[residue_dg(aa, zz * HALF_WIDTH / 9.0)
                                              - residue_dg(w, zz * HALF_WIDTH / 9.0)
                                              for w, zz in zip(sub.wt, sub.ipos)])
                               .groupby("ipos")._d.mean()) if len(sub) else pd.Series(
                                   dtype=float)
            # ONE prediction axis per page, as the published figure has it: letting
            # each panel autoscale makes alanine, identically zero under dsTbL since
            # it is the reference, look as structured as arginine
            lim = {}
            for mode, series in (("dg", list(prof.values())),
                                 ("ddg", [x.to_numpy() for x in matched.values()])):
                allp = [np.asarray(x, float) for x in series if len(np.asarray(x, float))]
                allp = np.concatenate(allp) if allp else np.array([0.0, 1.0])
                allp = allp[np.isfinite(allp)]
                pad = 0.05 * (float(allp.max() - allp.min()) or 1.0)
                lim[mode] = (float(allp.min()) - pad, float(allp.max()) + pad)

            for mode in ("dg", "ddg"):
                for flip in (False, True):
                    fig, drawn = _i09_panel_grid(v, aas, matched, prof, xs_pred, mode,
                                                 flip, lim, min_per_point)
                    stem = ("i09_profiles" if mode == "dg" else "i09_effect")
                    what = ("the scale's " + SCALE_DG[SCALE] + " for this residue, the "
                            "published quantity: the cost of HAVING it there"
                            if mode == "dg" else
                            "the ΔΔG of changing the wild types actually at each depth "
                            "into this residue, which is what blue measures")
                    fig.supxlabel("Position in the transmembrane segment "
                                  "(−9 = cytoplasmic end, +9 = lumenal end)", fontsize=8)
                    fig.supylabel("mean effect of substituting to this residue (blue)",
                                  fontsize=8)
                    fig.suptitle(
                        f"{g.replace('_fitness', '')}"
                        + (" — from A, V, L or I only" if wt_set == "avli" else "")
                        + f": blue = the data, left axis. Red = {what}"
                        + (", NEGATED so that agreement with the data looks like "
                           "agreement rather than a mirror image" if flip else "")
                        + ". One red axis is shared by all twenty panels.",
                        fontsize=7.5, x=0.01, ha="left")
                    P.save(fig, outdir,
                           f"{stem}{tag_wt}{'_flip' if flip else ''}_"
                           f"{g.replace('_fitness', '')}", None, "I09")
                    plt.close(fig)
                    if mode == "dg" and not flip:
                        for aa, m, sub in drawn:
                            md = matched[aa].reindex(m.index).dropna()
                            rows.append({
                                "protein": g, "wt_set": wt_set, "mut": aa,
                                "n_positions": int(len(m)),
                                "n_variants": int(sub.score_z.notna().sum()),
                                "mean_effect": float(sub.score_z.mean()),
                                "delta_kd_from_mean_wt": float(
                                    KD[aa] - np.mean([KD.get(w, np.nan)
                                                      for w in sub.wt.dropna()])),
                                "spearman_with_scale_dg": float(ss.spearmanr(
                                    m.y, [residue_dg(aa, i * HALF_WIDTH / 9.0)
                                          for i in m.index]).statistic),
                                "spearman_with_scale_ddg": float(ss.spearmanr(
                                    m.y.reindex(md.index), md).statistic)
                                if len(md) >= 3 else np.nan,
                                "median_n_per_point": float(m.n.median()),
                                "effect_at_centre": float(
                                    m.y.reindex(range(-2, 3)).mean()),
                                "effect_at_edges": float(
                                    m.y.reindex([-9, -8, -7, 7, 8, 9]).mean())})
            med = float(np.median([r["median_n_per_point"] for r in rows
                                   if r["protein"] == g and r["wt_set"] == wt_set]
                                  or [0.0]))
            print(f"    {g.replace('_fitness', ''):<12} {wt_set:<4} bin width "
                  f"{bin_width} -> "
                  f"{len(np.arange(-9, 10, max(bin_width, 1)))} points, median "
                  f"{med:.0f} measurements behind each"
                  + ("  (try --profile-bin 1 for the published 19 columns)"
                     if med >= 3 * max(min_per_point, 1) and bin_width > 1 else ""),
                  flush=True)
        out[g] = True
    if not rows:
        return {"status": "skipped", "reason": "no protein has enough TM variants"}
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "i09_profiles.csv", index=False)
    head = "; ".join(
        f"{g.replace('_fitness', '')} {d.wt_set.nunique()} wild-type sets, "
        f"{len(d[d.wt_set == 'all'])} residues, median ρ(ΔΔG) "
        f"{d[d.wt_set == 'all'].spearman_with_scale_ddg.median():+.2f} all / "
        f"{d[d.wt_set == 'avli'].spearman_with_scale_ddg.median():+.2f} AVLI"
        for g, d in t.groupby("protein"))
    return {"status": "ok", "headline": head, "by_residue": t.to_dict("records")}


def i07b_by_helix(variants: dict, helices: dict, names: list[str], outdir: Path,
                  all_helices: bool = False) -> dict:
    """The hydrophobicity control, helix by helix - what i03 is for ddG.

    i07 pools every TM variant, which answers "what does a hydrophobicity change cost
    in this membrane domain" and cannot answer "do the helices differ". A helix that
    only just inserts should, on the insertion-energetics story, be the one where
    making a residue more polar costs most - and that prediction is about plain
    hydrophobicity as much as about a fitted scale, so it is testable here with no
    scale at all.

    The number to read is in the panel: the correlation between a helix's slope and the
    energy of inserting it. Mind the sign, which is the opposite of i03's. A ddG slope
    is NEGATIVE - a more destabilising substitution lowers the effect - so there the
    marginal helices being steepest makes rho(slope, dG) negative. A Kyte-Doolittle
    slope is POSITIVE, because the axis runs the other way: more hydrophobic is less
    damaging. A marginal helix being more sensitive therefore shows up as a steeper
    POSITIVE slope, and the claim here is rho(slope, dG) > 0, since dG is least
    negative for the helix with least margin.

    Each helix's slope is its own ordinary least squares fit over its own variants, so
    a helix whose hydrophobicity hardly varies gets a flat slope with a wide error
    rather than being dropped.
    """
    from .annot import KYTE_DOOLITTLE as KD
    rows = []
    fig, pairs, _ = grid(names, w=3.2, h=2.9, extra=0.35)
    for i, (g, ax) in enumerate(pairs):
        v = variants[g].copy()
        v["delta_kd"] = [KD.get(m, np.nan) - KD.get(w, np.nan)
                         for m, w in zip(v.mut, v.wt)]
        v = v.dropna(subset=["score_z", "delta_kd", "helix"])
        h = helices[g].set_index("helix")
        if len(v) < 100 or v.helix.nunique() < 3:
            panel(ax, i, g, "too few helices with variants", size=7.5)
            continue
        edges = np.unique(np.quantile(v.delta_kd, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            panel(ax, i, g, "no spread in hydrophobicity change", size=7.5)
            continue
        v = v.assign(bin=pd.cut(v.delta_kd, edges, include_lowest=True))
        curves, dgs = {}, {}
        for hx, sub in v.groupby("helix"):
            if hx not in h.index or len(sub) < MIN_VARIANTS:
                continue
            m = sub.groupby("bin", observed=True).agg(
                x=("delta_kd", "mean"), y=("score_z", "mean"), n=("score_z", "size"))
            m = m[m.n >= MIN_PER_BIN].dropna()
            r = ss.linregress(sub.delta_kd, sub.score_z)
            dg = float(h.loc[hx, "dg"])
            rows.append({"protein": g, "helix": hx, "dg": dg,
                         "slope": float(r.slope), "se": float(r.stderr),
                         "p": float(r.pvalue), "spearman": float(
                             ss.spearmanr(sub.delta_kd, sub.score_z).statistic),
                         "n_variants": int(len(sub)),
                         "mean_effect": float(sub.score_z.mean())})
            if len(m) >= 3:
                curves[hx], dgs[hx] = m, dg
        if not curves:
            panel(ax, i, g, "no helix has enough variants per bin", size=7.5)
            continue
        lo, hi = min(dgs.values()), max(dgs.values())
        cmap = plt.get_cmap("viridis")
        # every helix, always: this is the per-helix figure, and with twelve of them
        # the question is whether they form a spread or two groups, which only the
        # whole set can answer. The two extremes are then drawn over the top so the
        # comparison the claim is about stays readable in the crowd.
        for hx, m in curves.items():
            ax.plot(m.x, m.y, "-", lw=1.0, alpha=0.75, zorder=3,
                    color=cmap((dgs[hx] - lo) / (hi - lo) if hi > lo else 0.5))
        for hx, col, lab in ((max(dgs, key=dgs.get), P.DARK_RED, "hardest to insert"),
                             (min(dgs, key=dgs.get), "#2166AC", "easiest to insert")):
            m = curves[hx]
            ax.plot(m.x, m.y, "-o", ms=4, lw=2.0, color=col, mec=P.INK, mew=0.4,
                    zorder=7, label=f"{hx}, {lab} (ΔG {dgs[hx]:+.1f})")
        wide = pd.DataFrame({k: m.y for k, m in curves.items()})
        xs = pd.concat([m.x for m in curves.values()], axis=1).median(axis=1)
        n_h = wide.notna().sum(axis=1)
        ok = n_h >= 3
        mn = wide.mean(axis=1)
        sem = wide.std(axis=1, ddof=1) / np.sqrt(n_h.clip(lower=1))
        ax.fill_between(xs[ok], (mn - sem)[ok], (mn + sem)[ok], color=P.INK, alpha=0.12,
                        lw=0, zorder=5)
        ax.plot(xs[ok], mn[ok], "-o", ms=4.5, lw=2.4, color=P.INK, zorder=6,
                label="mean of helices (± SEM)")
        ax.plot(xs[ok], wide.median(axis=1)[ok], "--", lw=1.6, color=P.INK, zorder=6,
                label="median of helices")
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        here = pd.DataFrame([r for r in rows if r["protein"] == g])
        extra = ""
        if len(here) >= 4:
            rho, pv = ss.spearmanr(here.dg, here.slope)
            extra = f"{len(here)} helices, slope vs ΔG ρ = {rho:+.2f} (p = {pv:.2g})"
        panel(ax, i, g, extra, size=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ hydrophobicity, Kyte–Doolittle (mutant − wild type)",
                          fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
    if not rows:
        return {"status": "skipped", "reason": "no helix has enough variants"}
    fig.suptitle("I07 helix by helix, with no insertion scale in it. A helix with less "
                 "margin should be the most sensitive to losing hydrophobicity, so the "
                 "correlation between a helix's slope and its ΔG is the test. Here the "
                 "claim is a POSITIVE correlation: this axis runs the opposite way to "
                 "I03's, since more hydrophobic is less damaging and ΔG is least "
                 "negative for the helix that only just inserts.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i07b_hydrophobicity_by_helix", None, "I07B")
    plt.close(fig)
    t = pd.DataFrame(rows)
    # BH within each protein. Written as a loop over index groups rather than
    # groupby().apply(): pandas drops the grouping column from the frame it hands the
    # function, and the result then has no `protein` at all.
    t["q"] = np.nan
    for _, idx in t.groupby("protein").groups.items():
        t.loc[idx, "q"] = bh_fdr(t.loc[idx, "p"].fillna(1.0).to_numpy())
    t.to_csv(outdir / "i07b_hydrophobicity_by_helix.csv", index=False)
    _i07c_significance(t, names, outdir)
    head = "; ".join(
        f"{g.replace('_fitness', '')} ρ(slope, ΔG) "
        f"{ss.spearmanr(d.dg, d.slope).statistic:+.2f} over {len(d)} helices"
        for g, d in t.groupby("protein") if len(d) >= 4)
    return {"status": "ok", "headline": head or f"{len(t)} helices",
            "by_helix": t.to_dict("records")}


def i07_hydrophobicity_control(variants: dict, names: list[str], outdir: Path,
                               n_bins: int = 10) -> dict:
    """The control for i04: plain hydrophobicity change in place of ddG.

    i04's x axis is an insertion energy - a published scale, evaluated at each residue's
    own depth, with the sign and the asymmetry the measurement found. This replaces it
    with the crudest possible substitute: the Kyte-Doolittle difference between the
    mutant and the wild-type residue, one number per substitution, identical for every
    residue in every helix at every depth. No scale, no depth, no orientation.

    That is what makes it a control rather than another analysis. If the i04 curve is
    reproduced here, then nothing in that figure needed an insertion scale and it could
    have been drawn from a 1982 hydrophobicity table; whatever it shows is about
    hydrophobicity in general and not about the translocon. If the ddG curve is
    appreciably the stronger of the two, the depth dependence is doing work.

    So the number to read is not this slope on its own but the comparison, which the
    table carries: each predictor's slope alone, and then both in one model, where the
    coefficient on ddG is what the insertion scale adds once plain hydrophobicity has
    been allowed to explain whatever it can. Errors are clustered by helix throughout.
    """
    import statsmodels.formula.api as smf
    from .annot import KYTE_DOOLITTLE as KD
    rows, fits = [], []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = variants[g].copy()
        v["delta_kd"] = [KD.get(m, np.nan) - KD.get(w, np.nan)
                         for m, w in zip(v.mut, v.wt)]
        v = v.dropna(subset=["score_z", "delta_kd"])
        if len(v) < 100:
            panel(ax, i, g, "too few variants in a TM helix", size=7.5)
            continue
        b = pd.qcut(v.delta_kd, n_bins, duplicates="drop")
        m = v.groupby(b, observed=True).agg(x=("delta_kd", "mean"),
                                            y=("score_z", "mean"),
                                            e=("score_z", "sem"), n=("score_z", "size"))
        m = m.dropna()
        ax.fill_between(m.x, m.y - m.e, m.y + m.e, color="#2166AC", alpha=0.18, lw=0)
        ax.plot(m.x, m.y, "-o", ms=4.5, lw=1.8, color="#2166AC", mec=P.INK, mew=0.4)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        rows += [{"protein": g, "delta_kd": float(r.x), "mean_effect": float(r.y),
                  "sem": float(r.e), "n_variants": int(r.n)} for r in m.itertuples()]

        def fit(formula, d):
            return smf.ols(formula, data=d).fit(cov_type="cluster",
                                                cov_kwds={"groups": d.helix})

        d = v[["score_z", "delta_kd", "helix"]].dropna()
        kd_only = fit("score_z ~ delta_kd", d)
        row = {"protein": g, "n_variants": int(len(d)),
               "n_helices": int(d.helix.nunique()),
               "slope_kd": float(kd_only.params["delta_kd"]),
               "se_kd": float(kd_only.bse["delta_kd"]),
               "p_kd": float(kd_only.pvalues["delta_kd"]),
               "r2_kd": float(kd_only.rsquared),
               "spearman_kd": float(ss.spearmanr(d.delta_kd, d.score_z).statistic)}
        # the comparison, where there is a ddG to compare against
        if "ddg" in v and v.ddg.notna().any():
            dd = v[["score_z", "delta_kd", "ddg", "helix"]].dropna()
            if len(dd) > 100 and dd.helix.nunique() > 2:
                ddg_only, both = fit("score_z ~ ddg", dd), fit("score_z ~ delta_kd + ddg", dd)
                row.update(
                    slope_ddg=float(ddg_only.params["ddg"]),
                    r2_ddg=float(ddg_only.rsquared), r2_both=float(both.rsquared),
                    r2_added_by_ddg=float(both.rsquared - kd_only.rsquared),
                    slope_ddg_given_kd=float(both.params["ddg"]),
                    se_ddg_given_kd=float(both.bse["ddg"]),
                    p_ddg_given_kd=float(both.pvalues["ddg"]),
                    corr_kd_ddg=float(ss.pearsonr(dd.delta_kd, dd.ddg).statistic))
        fits.append(row)
        extra = f"slope {row['slope_kd']:+.3f} per KD unit, {len(d):,} variants"
        if "r2_added_by_ddg" in row:
            extra += f"; ΔΔG adds R² {row['r2_added_by_ddg']:+.3f}"
        panel(ax, i, g, extra, size=7.5)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ hydrophobicity, Kyte–Doolittle (mutant − wild type)",
                          fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("mean effect", fontsize=7.5)
    fig.suptitle("The control for I04: every missense variant in any TM helix, binned "
                 "by plain Kyte–Doolittle change instead of by ΔΔG. One number per "
                 "substitution, the same for every residue at every depth. If this "
                 "reproduces the I04 curve, that curve did not need an insertion "
                 "scale. Band = SEM over variants; slopes are clustered by helix.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i07_hydrophobicity_control", None, "I07")
    plt.close(fig)
    if not fits:
        return {"status": "skipped", "reason": "no protein has enough TM variants"}
    pd.DataFrame(rows).to_csv(outdir / "i07_hydrophobicity_curve.csv", index=False)
    ft = pd.DataFrame(fits)
    ft.to_csv(outdir / "i07_hydrophobicity_slopes.csv", index=False)
    head = "; ".join(
        f"{r.protein.replace('_fitness', '')} KD slope {r.slope_kd:+.3f}"
        + (f", ΔΔG adds R² {r.r2_added_by_ddg:+.3f} (p {r.p_ddg_given_kd:.2g})"
           if "r2_added_by_ddg" in ft and np.isfinite(getattr(r, "r2_added_by_ddg", np.nan))
           else "")
        for r in ft.itertuples())
    return {"status": "ok", "headline": head, "by_protein": ft.to_dict("records")}


def i06_scale_agreement(variants: dict, helices: dict, names: list[str],
                        outdir: Path) -> dict:
    """Do the two scales predict the same thing, and does either match the measurement?

    The scales were not measured in the same membrane: dsTbL in a bacterial
    inner-membrane reporter, Hessa on Sec61 in dog pancreas rough microsomes, which is
    an ER membrane (check both against the papers before either appears on a slide).
    Neither is a yeast plasma membrane, so the useful question is not which is right
    but how much the choice matters: if the two agree closely, the membrane they came from cannot be driving any
    conclusion drawn here, and if they disagree, every number that depends on the scale
    has to be quoted with the scale named.

    Both are evaluated at the same ideal depths, so this is the scales differing and not
    the geometry. Reported per protein: how well the two agree on the ddG of the same
    substitutions, how well they agree on which helix is hardest to insert, and how each
    one correlates with the effect actually measured - that last pair being the only
    comparison that can say anything about which membrane is the better analogue.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=3.0, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = variants[g]
        if not len(v) or "z_ideal" not in v:
            panel(ax, i, g, "no variants in a helix", size=8)
            continue
        z = v.z_ideal.to_numpy(float)
        a = np.array([residue_dg(m, zi, "dstbl") - residue_dg(w, zi, "dstbl")
                      for m, w, zi in zip(v.mut, v.wt, z)])
        b = np.array([residue_dg(m, zi, "hessa") - residue_dg(w, zi, "hessa")
                      for m, w, zi in zip(v.mut, v.wt, z)])
        ok = np.isfinite(a) & np.isfinite(b)
        a, b, eff = a[ok], b[ok], v.score_z.to_numpy(float)[ok]
        if len(a) < 30:
            panel(ax, i, g, "too few variants", size=8)
            continue
        r = float(ss.pearsonr(a, b).statistic)
        rho = float(ss.spearmanr(a, b).statistic)
        slope = float(np.polyfit(a, b, 1)[0])
        # which helix is hardest to insert, under each scale
        h = helices[g]
        hel_rho = np.nan
        if "seq" in h and len(h) > 2:
            da = [segment_dg(q.seq, orientation=q.orientation) for q in h.itertuples()]
            old = globals()["SCALE"]
            try:
                globals()["SCALE"] = "hessa"
                db = [segment_dg(q.seq, orientation=q.orientation) for q in h.itertuples()]
            finally:
                globals()["SCALE"] = old
            hel_rho = float(ss.spearmanr(da, db).statistic)
        fin = np.isfinite(eff)
        rows.append({"protein": g, "n_variants": int(len(a)),
                     "pearson_ddg": r, "spearman_ddg": rho,
                     "slope_hessa_on_dstbl": slope,
                     "n_helices": int(len(h)), "spearman_helix_dg": hel_rho,
                     "spearman_dstbl_vs_measured":
                         float(ss.spearmanr(a[fin], eff[fin]).statistic),
                     "spearman_hessa_vs_measured":
                         float(ss.spearmanr(b[fin], eff[fin]).statistic)})
        lo = float(min(a.min(), b.min()))
        hi = float(max(a.max(), b.max()))
        ax.plot([lo, hi], [lo, hi], color=P.MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.scatter(a, b, s=5, color="#2166AC", alpha=0.35, lw=0, zorder=2)
        xs = np.linspace(lo, hi, 20)
        c1, c0 = np.polyfit(a, b, 1)
        ax.plot(xs, c0 + c1 * xs, color=P.INK, lw=1.0, zorder=3)
        ax.text(0.03, 0.97, f"r = {r:+.2f}\nρ = {rho:+.2f}\nslope {slope:+.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        panel(ax, i, g, f"{len(a):,} variants", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("ΔΔG, dsTβL (kcal/mol)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("ΔΔG, Hessa table S2 (kcal/mol)", fontsize=7.5)
    fig.suptitle("The same substitutions under both scales, at the same depths. The "
                 "dashed line is equality; a slope below one means Hessa compresses "
                 "what dsTβL spreads out. Agreement here means the membrane the scale "
                 "came from is not driving the conclusion.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "i06_scale_agreement", None, "I06")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(outdir / "i06_scale_agreement.csv", index=False)
    res = residue_scales(outdir)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} r {r.pearson_ddg:+.2f} "
                     f"(helix order ρ {r.spearman_helix_dg:+.2f})"
                     for r in t.itertuples())
    return {"status": "ok", "headline": head, "by_protein": t.to_dict("records"),
            "residues": res}


def residue_scales(outdir: Path) -> list:
    """The twenty residues under both scales, at the centre and at the interface.

    Where the two scales disagree, they disagree residue by residue, and a scatter of
    the twenty says which ones - which is more use than a single correlation when
    deciding whether a disagreement matters for a particular protein.
    """
    aas = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for aa in aas:
        for where, z in (("bilayer centre", 0.0), ("interface", 12.0)):
            rows.append({"aa": aa, "where": where, "z": z,
                         "dstbl": residue_dg(aa, z, "dstbl"),
                         "hessa": residue_dg(aa, z, "hessa")})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    for ax, (where, d) in zip(axes, t.groupby("where", sort=False)):
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6)
        ax.scatter(d.dstbl, d.hessa, s=18, color="#2166AC", edgecolor=P.INK, lw=0.3)
        P.label_points(ax, d.dstbl.to_numpy(), d.hessa.to_numpy(), d.aa.tolist(),
                       fontsize=7)
        rho = float(ss.spearmanr(d.dstbl, d.hessa).statistic)
        ax.set_title(f"{where}  ρ = {rho:+.2f}", loc="left", fontsize=8)
        ax.set_xlabel("dsTβL (kcal/mol, relative to alanine)", fontsize=7.5)
        ax.set_ylabel("Hessa table S2 (kcal/mol, absolute)", fontsize=7.5)
        P.no_square(ax)
    fig.suptitle("The two scales residue by residue. They are not on the same reference "
                 "state, so what can be compared is the ordering and the spread, not "
                 "the values. The ordering agrees; the spread does not. dsTβL has "
                 "1.4 to 4.8 times the range of Hessa in every residue class at both "
                 "depths, most of all for the hydrophobics, which is the whole of the "
                 "slope below one in I06.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "i06b_residue_scales", None, "I06B")
    plt.close(fig)
    t.to_csv(outdir / "i06b_residue_scales.csv", index=False)
    return t.to_dict("records")


ANALYSES = {"i01": "helix ΔG against its tolerance",
            "i02": "the interaction: is ΔΔG worse in a marginal helix?",
            "i03": "every helix separately, with the mean and median of them",
            "i04": "the pooled curve over all helices, averaged over variants",
            "i05": "the helices ranked by wild-type insertion energy",
            "i06": "do the two scales agree, and does either match the measurement?",
            "i07": "the control: plain hydrophobicity change instead of ΔΔG",
            "i07b": "the hydrophobicity control, helix by helix",
            "i08": "ΔΔG against Δ hydrophobicity, one point per mutation",
            "i09": "one panel per mutant residue: effect against position in the membrane",
            "i09b": "is each residue's depth profile symmetric about the bilayer centre?"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms insertion",
        description="Does the free energy of inserting a helix set a floor on what it "
                    "tolerates? Tested with the dsTβL scale (Elazar et al. 2016) or the "
                    "Hessa et al. 2007 translocon scale.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--depth", default="ideal", choices=["ideal", "structure"],
                    help="where a residue sits in the membrane when its ΔΔG is evaluated: "
                         "'ideal' lays the helix across the bilayer the way the scale was "
                         "built, 'structure' uses the AlphaFold membrane frame")
    ap.add_argument("--scale", default="dstbl", choices=list(SCALES),
                    help="dsTbL (Elazar 2016), depth-dependent and relative to alanine, "
                         "or the Hessa 2007 table S2 position term, which is absolute "
                         "and symmetric about the bilayer centre")
    ap.add_argument("--profile-bin", type=int, default=3, metavar="N",
                    help="i09: pool depths into bins N positions wide (default 3, so "
                         "seven points across the segment). 1 gives the published "
                         "nineteen-column resolution, which needs enough helices to "
                         "fill each (residue, depth) cell - about twelve")
    ap.add_argument("--profile-min-n", type=int, default=3, metavar="N",
                    help="i09: measurements a (residue, depth) point needs to be drawn")
    ap.add_argument("--all-helices", action="store_true",
                    help="i03 draws every helix rather than only the hardest and the "
                         "easiest of each protein to insert")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    if a.scale == "hessa" and a.depth == "structure":
        raise SystemExit("--scale hessa is indexed by position in a 19-residue segment, "
                         "not by angstroms, so a measured depth is an extrapolation the "
                         "table was never fitted for; use --depth ideal")
    global SCALE, DEPTH
    SCALE, DEPTH = a.scale, a.depth
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n  scale: {SCALE_LABEL[a.scale]}", flush=True)
    if a.scale == "hessa":
        print("  (position term only - no length or flanking correction, and symmetric "
              "about the centre, so helix orientation does not enter)", flush=True)
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
    pd.concat([h.assign(protein=g, scale=a.scale) for g, h in helices.items()],
              ignore_index=True).to_csv(outdir / "helices.csv", index=False)
    pd.concat([v.assign(protein=g, scale=a.scale) for g, v in variants.items()],
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
            elif k == "i03":
                r = i03_helix_curves(variants, helices, names, outdir, a.all_helices)
            elif k == "i04":
                r = i04_pooled(variants, names, outdir)
            elif k == "i05":
                r = i05_ranking(helices, names, outdir)
            elif k == "i06":
                r = i06_scale_agreement(variants, helices, names, outdir)
            elif k == "i07":
                r = i07_hydrophobicity_control(variants, names, outdir)
            elif k == "i07b":
                r = i07b_by_helix(variants, helices, names, outdir, a.all_helices)
            elif k == "i08":
                r = i08_ddg_vs_hydrophobicity(variants, names, outdir)
            elif k == "i09":
                r = i09_profiles(variants, names, outdir, a.profile_min_n,
                                 a.profile_bin)
            else:
                r = i09b_symmetry(variants, names, outdir, a.profile_min_n,
                                  a.profile_bin)
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
    out["scale"] = {"name": a.scale, "label": SCALE_LABEL[a.scale], "depth": a.depth}
    (outdir / "insertion.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
