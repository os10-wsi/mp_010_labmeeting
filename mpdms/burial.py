"""`python -m mpdms burial configs/AQR1_fitness.yaml configs/HXT1.yaml configs/HXT3.yaml`

Ten questions about burial and hydrophobicity, each asked inside one protein.

A membrane protein has three chemically different kinds of place to put a residue, and
the whole of this module is built on telling them apart:

    core      inside a TM helix bundle, packed against other helices
    lipid     inside the membrane, facing acyl chains
    aqueous   outside the membrane, in water - the lumenal and cytosolic loops together

The lumenal and cytosolic loops are pooled into one `aqueous` class on purpose. They
differ in which side of the membrane they are on, which matters for insertion and for
the positive-inside rule, but not for the question here: both are solvated, and a
hydrophobicity argument has no reason to distinguish them. `cavity` - exposed, inside
the membrane, but facing the central translocation pathway - is kept as a fourth class
and plotted alongside, because in a transporter that pathway is full of water and
lumping it in with lipid would hide the most interesting positions in the protein.

    b01  hydrophobicity tolerance, three ways: lipid against core against aqueous
    b02  the same slope as a continuous function of burial, rather than in three bins
    b03  does the intolerant face of a helix coincide with its hydrophobic face?
    b04  what a charge costs, as a function of how deep it is buried
    b05  buried polar residues: packing defect or substrate site?
    b06  volume against hydrophobicity: which one does each environment care about?
    b07  small residues at close helix-helix approaches: is there room to grow?
    b08  the aromatic belt: are W/Y/F at the interface doing a different job?
    b09  hydrophobic mismatch: cost against the local environment, not in the absolute
    b10  are ESM-1v's own errors structured by burial and hydrophobicity?

Nothing is pooled across proteins. Every panel is one protein, in the order the configs
are given on the command line, and standard errors are clustered by position wherever a
model is fitted to variants, since variants at one position share whatever that position
is.
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
from .annot import KYTE_DOOLITTLE, VOLUME
from .architecture import protein_bits
from .config import REPO_ROOT, load_config
from .environs import POLAR, effect_color, grid, panel

OUT = REPO_ROOT / "outputs" / "_burial"

# the three classes the comparison is about, plus the cavity kept separate
BCLASSES = ["core", "cavity", "lipid", "aqueous"]
PRIMARY = ["core", "lipid", "aqueous"]
BCOLORS = {"core": "#B2182B", "cavity": "#E8A33D", "lipid": "#2166AC",
           "aqueous": "#6B6B6B"}
BLABEL = {"core": "core (helix–helix)", "cavity": "cavity-lining",
          "lipid": "lipid-facing", "aqueous": "aqueous (lumen + cytoplasm)"}
FROM_ENVIRON = {"helix-helix core": "core", "cavity-lining": "cavity",
                "lipid-facing": "lipid", "cytosolic loop": "aqueous",
                "lumenal loop": "aqueous"}

MIN_HELICES = 3           # before the alignment of helix faces is worth a p-value
AROMATIC = set("FWY")
SMALL = set("GAS")
BELT = (12.0, 20.0)       # |z| where the headgroups and the aromatic belt sit
CORE_Z = 10.0             # |z| inside which a residue is in the acyl region
NEAR_CA = 6.5             # Å between CA atoms, counted as a close helix-helix approach
MIN_GROUP = 15            # variants, before a group gets a number
MIN_POSITIONS = 8         # positions, before a group gets a violin
N_PERM = 2000


# --------------------------------------------------------------------- loading
def burial_class(environ: pd.Series) -> pd.Series:
    """Collapse the five residue environments into the four used here.

    The two loop classes become one. Nothing else moves, so a figure here and a figure
    in `environ` are talking about the same residues.
    """
    return environ.map(FROM_ENVIRON).astype(object)


def neighbour_hydrophobicity(rt: pd.DataFrame, cutoff: float = 8.0,
                             exclude: int = 4) -> pd.Series:
    """Mean Kyte-Doolittle hydrophobicity of everything packed around each residue.

    CA atoms within `cutoff`, skipping the `exclude` residues either side in sequence:
    those are the same helix turn and would mostly measure the residue against itself.
    This is the quantity a substitution is mismatched *with* - the local environment -
    as opposed to the absolute hydrophobicity of the residue that arrives.
    """
    from scipy.spatial import cKDTree
    xyz = rt[["x", "y", "z"]].to_numpy(float)
    kd = rt.aa.map(KYTE_DOOLITTLE).to_numpy(float)
    pos = rt.pos.to_numpy(int)
    tree = cKDTree(xyz)
    out = np.full(len(rt), np.nan)
    for i, nb in enumerate(tree.query_ball_point(xyz, cutoff)):
        keep = [j for j in nb if abs(pos[j] - pos[i]) > exclude and np.isfinite(kd[j])]
        if keep:
            out[i] = float(np.mean(kd[keep]))
    return pd.Series(out, index=rt.index)


def inter_helix_distance(rt: pd.DataFrame, helix: dict) -> pd.Series:
    """For each residue in a TM helix, the CA distance to the nearest *other* helix.

    A residue can be buried by its own helix - the backbone it is part of - without
    being at a packing interface at all. Measuring to a different helix is what makes
    the number about the interface.
    """
    t = rt.copy()
    t["helix"] = t.pos.map(helix)
    sub = t.dropna(subset=["helix"])
    out = pd.Series(np.nan, index=rt.index)
    if sub.helix.nunique() < 2:
        return out
    xyz = sub[["x", "y", "z"]].to_numpy(float)
    h = sub.helix.to_numpy()
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    d[h[:, None] == h[None, :]] = np.inf
    out.loc[sub.index] = d.min(axis=1)
    return out


def bits_for(cfg) -> dict:
    """Everything the ten analyses share, loaded once per protein."""
    b = protein_bits(cfg)
    if not b:
        return {}
    rt = b["rt"].copy()
    rt["kd_env"] = neighbour_hydrophobicity(rt)
    rt["inter_helix_ca"] = inter_helix_distance(rt, b["helix"])
    extra = rt[["pos", "aa", "kd_env", "inter_helix_ca"]].rename(columns={"aa": "wt_struct"})
    var, pos = b["var"].copy(), b["pos"].copy()
    cut = pos.attrs.get("radial_cut", np.nan)
    out = {}
    for name, t in (("var", var), ("pos", pos)):
        t = t.copy()
        t["bclass"] = burial_class(t.environ)
        t = t.drop(columns=[c for c in extra.columns if c != "pos" and c in t.columns])
        out[name] = t.merge(extra, on="pos", how="left")
    var, pos = out["var"], out["pos"]
    # the wild-type residue at each position, from the assay where it has one and from
    # the model otherwise; a disagreement between them is a numbering bug, not a choice
    wt = var.groupby("pos").wt.first()
    pos["wt"] = pos.pos.map(wt)
    pos["wt"] = pos.wt.where(pos.wt.notna(), pos.wt_struct)
    var["kd_mut"] = var.mut.map(KYTE_DOOLITTLE)
    var["kd_wt"] = var.wt.map(KYTE_DOOLITTLE)
    var["vol_mut"] = var.mut.map(VOLUME)
    b.update(var=var, pos=pos, rt=rt, radial_cut=float(cut) if np.isfinite(cut) else np.nan)
    return b


# ----------------------------------------------------------------- small helpers
def cluster_ols(formula: str, data: pd.DataFrame, groups) -> "object | None":
    """OLS with standard errors clustered by position, or None if there is too little.

    Variants at one position are not independent observations of anything: they all
    inherit whatever that position is. Clustering on it is the difference between a
    standard error that counts positions and one that counts variants.
    """
    import statsmodels.formula.api as smf
    d = data.copy()
    d["_cluster"] = np.asarray(groups)
    d = d.dropna()
    if len(d) < 40 or d["_cluster"].nunique() < 6:
        return None
    return smf.ols(formula, data=d).fit(cov_type="cluster",
                                        cov_kwds={"groups": d["_cluster"]})


def hinge_slopes(dkd, y, groups) -> dict:
    """Two slopes hinged at zero: one for going polar, one for going hydrophobic.

    A single straight line through Δhydrophobicity assumes the cost of a change is the
    same in both directions, which is exactly the assumption this analysis exists to
    test. Both arms are reported as d(effect)/d(ΔKD), so a symmetric "tent" - bad in
    either direction - has one positive and one negative arm that cancel, and the sum
    of the two is a direct measure of the asymmetry.
    """
    d = pd.DataFrame({"y": np.asarray(y, float), "k": np.asarray(dkd, float),
                      "g": np.asarray(groups)}).dropna()
    d["neg"] = d.k.clip(upper=0.0)
    d["pos_"] = d.k.clip(lower=0.0)
    if len(d) < MIN_GROUP * 2 or d.g.nunique() < 6 or d.neg.std() == 0 or d.pos_.std() == 0:
        return {}
    m = cluster_ols("y ~ neg + pos_", d[["y", "neg", "pos_"]], d.g)
    if m is None:
        return {}
    return {"n": int(len(d)), "n_positions": int(d.g.nunique()),
            "slope_polar_arm": float(m.params["neg"]),
            "se_polar_arm": float(m.bse["neg"]),
            "p_polar_arm": float(m.pvalues["neg"]),
            "slope_hydrophobic_arm": float(m.params["pos_"]),
            "se_hydrophobic_arm": float(m.bse["pos_"]),
            "p_hydrophobic_arm": float(m.pvalues["pos_"]),
            "asymmetry": float(m.params["neg"] + m.params["pos_"]),
            "r2": float(m.rsquared)}


def binned(ax, x, y, bins, color, label, ms: float = 3.6, ls="-", alpha: float = 1.0):
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if not len(d):
        return None
    b = pd.cut(d.x, bins)
    m = d.groupby(b, observed=True).agg(x=("x", "mean"), y=("y", "mean"),
                                        e=("y", "sem"), n=("y", "size")).dropna(subset=["x"])
    m = m[m.n >= 5]
    if not len(m):
        return None
    ax.errorbar(m.x, m.y, yerr=m.e, fmt="o", ls=ls, ms=ms, lw=1.2, color=color,
                capsize=1.8, label=label or "_nolegend_", mec=P.INK, mew=0.3,
                alpha=alpha)
    return m


def empty_safe_cmap():
    """The fitness colours, with cells that have too few variants left pale rather than
    dark: an empty cell is not an intolerant one, and grey at the bottom of a red scale
    is easy to misread as the extreme."""
    c = P.fitness_cmap().copy()
    c.set_bad("#F2F2F2")
    return c


def empty_panel(ax, i: int, name: str, reason: str) -> None:
    """A protein with too little data gets a panel that says so.

    Leaving the axes to draw themselves would put a blank 0-1 grid next to a real
    result, which reads as a measurement of nothing rather than as an absence.
    """
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=7, color=P.MUTED,
            transform=ax.transAxes)
    panel(ax, i, name, size=8)


def short_name(g: str) -> str:
    return g.replace("_fitness", "")


def forest(ax, rows, label_key, value_key, se_key, colors=None, xlabel=""):
    """A column of estimates with 95% intervals, newest convention: zero marked."""
    y = np.arange(len(rows))[::-1]
    for k, r in enumerate(rows):
        c = (colors or {}).get(r.get("bclass"), P.INK)
        ax.errorbar(r[value_key], y[k], xerr=1.96 * r[se_key], fmt="o", ms=5, lw=0,
                    elinewidth=1.0, capsize=2.5, color=c, mec=P.INK, mew=0.5)
    ax.axvline(0, color=P.MUTED, lw=0.7, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([r[label_key] for r in rows], fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=7.5)


# --------------------------------------- b01 hydrophobicity tolerance, three ways
KD_BINS = np.linspace(-9, 9, 10)
# b01a draws only the two classes the contrast is between. Every class is still fitted,
# and b01b and the CSV carry all four: leaving them out of the panel is a decision about
# what the figure says, not about what was measured.
B01A_CLASSES = ["lipid", "aqueous"]


def b01_three_way(bits: dict, names: list[str], outdir: Path) -> dict:
    """The comparison the whole module is built around.

    Lipid-facing positions sit against acyl chains, so making them polar should cost and
    making them more hydrophobic should not: an asymmetric, one-sided response. Core
    positions are packed between helices, where either direction disturbs the packing,
    so the response should be a tent with two arms. Aqueous positions are in water and
    should barely care.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = bits[g]["var"]
        for c in BCLASSES:
            sub = v[(v.bclass == c)].dropna(subset=["score_z", "delta_kd"])
            if len(sub) < MIN_GROUP * 3:
                continue
            if c in B01A_CLASSES:
                binned(ax, sub.delta_kd, sub.score_z, KD_BINS, BCOLORS[c], BLABEL[c],
                       ms=4.2)
            h = hinge_slopes(sub.delta_kd, sub.score_z, sub.pos)
            if h:
                rows.append({"protein": g, "bclass": c, **h})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.INK, lw=0.6, ls=(0, (1, 2)))
        d = {r["bclass"]: r for r in rows if r["protein"] == g}
        panel(ax, i, g, "  ".join(f"{c} asymmetry {d[c]['asymmetry']:+.3f}"
                                  for c in B01A_CLASSES if c in d), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ hydrophobicity (Kyte-Doolittle, mutant − WT)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect of the substitution", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.suptitle("Left of zero = the substitution is more polar than the wild type. "
                 "Lipid-facing should fall away to the left and be flat to the right; "
                 "aqueous positions are in water and should barely care either way.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b01a_hydrophobicity_three_way", None, "B01")
    plt.close(fig)

    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "no class has enough variants to fit"}
    fig, pairs, _ = grid(names, w=3.0, h=2.7, extra=0.2)
    for i, (g, ax) in enumerate(pairs):
        d = [r for r in per[per.protein == g].to_dict("records")
             if r["bclass"] in BCLASSES]
        if not d:
            empty_panel(ax, i, g, "no class had enough variants to fit")
            continue
        items = []
        for r in d:
            items.append({"lab": f"{r['bclass']}  → polar", "bclass": r["bclass"],
                          "v": r["slope_polar_arm"], "se": r["se_polar_arm"]})
            items.append({"lab": f"{r['bclass']}  → hydrophobic", "bclass": r["bclass"],
                          "v": r["slope_hydrophobic_arm"],
                          "se": r["se_hydrophobic_arm"]})
        forest(ax, items, "lab", "v", "se", BCOLORS,
               "slope of effect on Δ hydrophobicity (per KD unit)")
        panel(ax, i, g, size=8)
    fig.suptitle("Each arm is the slope on its own side of zero, so a positive '→ polar' "
                 "arm and a negative '→ hydrophobic' arm together are a tent: both "
                 "directions cost.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b01b_arm_slopes", None, "B01")
    plt.close(fig)
    per.to_csv(outdir / "b01_three_way_slopes.csv", index=False)
    head = "; ".join(
        f"{short_name(r.protein)} {r.bclass} asym {r.asymmetry:+.3f}"
        for r in per[per.bclass.isin(PRIMARY)].itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------------ b02 burial as a continuum, not three bins
WINDOW = 400       # variants per sliding window
STEP = 120


def b02_continuum(bits: dict, names: list[str], outdir: Path) -> dict:
    """The same slope, swept along the radial coordinate instead of binned into classes.

    b01 cuts the bundle in two at the median radius and calls one side cavity and the
    other lipid. That cut is a convenience, and this is the test of it: if tolerance
    really is a property of "facing lipid" there should be a step near the cut, and if
    it drifts smoothly the three-class picture is a summary rather than a boundary.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = bits[g]["var"]
        t = v[(v.is_tm > 0.5)].dropna(subset=["score_z", "delta_kd", "radial"])
        t = t.sort_values("radial").reset_index(drop=True)
        if len(t) < WINDOW + STEP:
            empty_panel(ax, i, g, f"fewer than {WINDOW + STEP} TM variants")
            continue
        xs, ys, los, his = [], [], [], []
        for a in range(0, len(t) - WINDOW + 1, STEP):
            w = t.iloc[a:a + WINDOW]
            m = cluster_ols("score_z ~ delta_kd", w[["score_z", "delta_kd"]], w.pos)
            if m is None:
                continue
            ci = m.conf_int().loc["delta_kd"]
            xs.append(float(w.radial.median())); ys.append(float(m.params["delta_kd"]))
            los.append(float(ci[0])); his.append(float(ci[1]))
            rows.append({"protein": g, "radial": xs[-1], "slope": ys[-1],
                         "lo": los[-1], "hi": his[-1], "n": int(len(w))})
        if not xs:
            empty_panel(ax, i, g, "no window had enough positions to fit")
            continue
        ax.fill_between(xs, los, his, color="#D9D9D9", lw=0)
        ax.plot(xs, ys, "-o", ms=3.4, lw=1.4, color=P.DARK_RED, mec=P.INK, mew=0.3)
        cut = bits[g].get("radial_cut", np.nan)
        if np.isfinite(cut):
            ax.axvline(cut, color=P.INK, lw=1.0, ls=(0, (4, 3)))
            ax.text(cut, ax.get_ylim()[1], " cavity | lipid", fontsize=6, va="top",
                    color=P.INK)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        panel(ax, i, g, f"{len(xs)} windows of {WINDOW}", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("slope of effect on Δ hydrophobicity", fontsize=7.5)
    fig.suptitle("A sliding window of TM variants ordered by radius. Band = 95% interval "
                 "with errors clustered by position. A step at the dashed line would "
                 "justify the two-class cut; a ramp would not.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "b02_burial_continuum", None, "B02")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "b02_sliding_slopes.csv", index=False)
    head = "; ".join(f"{short_name(g)} slope {d.slope.iloc[0]:+.3f} (inner) → "
                     f"{d.slope.iloc[-1]:+.3f} (outer)"
                     for g, d in per.groupby("protein")) if len(per) else ""
    return {"status": "ok", "headline": head, "windows": per.to_dict("records")}


# ------------------------------------------ b03 the hydrophobic moment of intolerance
PERIOD = 100.0     # degrees per residue in an alpha helix


def moment_vector(weights) -> complex:
    """The classic helical-wheel resultant: each residue placed at 100° from the last."""
    w = np.asarray(weights, float)
    ang = np.deg2rad(PERIOD) * np.arange(len(w))
    return complex(np.sum(w * np.exp(1j * ang)))


def face_angles(sub: pd.DataFrame) -> tuple[float, float, float]:
    """(hydrophobic face angle, intolerant face angle, difference) for one helix."""
    s = sub.sort_values("pos")
    kd = s.wt.map(KYTE_DOOLITTLE).to_numpy(float)
    e = s.effect.to_numpy(float)
    vh = moment_vector(kd - np.nanmean(kd))
    ve = moment_vector(-(e - np.nanmean(e)))     # negated: points at the costly face
    dphi = np.angle(ve / vh) if abs(vh) > 0 and abs(ve) > 0 else np.nan
    return float(np.angle(vh)), float(np.angle(ve)), float(dphi)


def rayleigh(angles: np.ndarray) -> float:
    """Mean resultant length: 1 if the angles agree, 0 if they are spread evenly."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    return float(abs(np.sum(np.exp(1j * a))) / len(a)) if len(a) else np.nan


def b03_face_alignment(bits: dict, names: list[str], outdir: Path,
                       n_perm: int = N_PERM) -> dict:
    """Does the face that cannot be mutated face lipid, or face the other helices?

    Both the hydrophobicity of the wild-type sequence and the intolerance of the
    positions are laid on the same helical wheel and reduced to a vector. If the
    intolerant face is the hydrophobic one the angle between them is near zero; if the
    sensitive side is the buried, polar, cavity-lining side it is near 180°. The null
    shuffles effects within the helix, which keeps the sequence and the helix length and
    destroys only which residue got which effect.
    """
    rows = []
    rng = np.random.default_rng(11)
    for g in names:
        pos = bits[g]["pos"]
        d = pos[(pos.is_tm > 0.5) & pos.helix.notna()].dropna(subset=["effect", "wt"])
        for h, sub in d.groupby("helix"):
            if len(sub) < 8:
                continue
            ph, pe, dphi = face_angles(sub)
            rows.append({"protein": g, "helix": h, "n": int(len(sub)),
                         "angle_hydrophobic": ph, "angle_intolerant": pe,
                         "delta_angle": dphi,
                         "delta_deg": float(np.rad2deg(dphi)),
                         "mean_effect": float(sub.effect.mean())})
    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "no TM helix has 8 scored positions"}

    tests = []
    for g, d in per.groupby("protein"):
        if len(d) < MIN_HELICES:
            # one or two helices always point somewhere together; a resultant only means
            # something once there are enough of them to disagree
            continue
        obs = rayleigh(d.delta_angle.to_numpy())
        pos = bits[g]["pos"]
        # the wild-type sequence does not move under the null, so its vector and the
        # wheel angles are computed once per helix and only the effects are shuffled
        fixed = []
        for h in d.helix:
            sub = pos[(pos.helix == h)].dropna(subset=["effect", "wt"]).sort_values("pos")
            kd = sub.wt.map(KYTE_DOOLITTLE).to_numpy(float)
            vh = moment_vector(kd - np.nanmean(kd))
            if abs(vh) == 0:
                continue
            w = np.exp(1j * np.deg2rad(PERIOD) * np.arange(len(sub)))
            fixed.append((vh, w, sub.effect.to_numpy(float)))
        null = np.empty(n_perm)
        for k in range(n_perm):
            ang = []
            for vh, w, e in fixed:
                ve = complex(np.sum(-(rng.permutation(e) - np.nanmean(e)) * w))
                if abs(ve) > 0:
                    ang.append(np.angle(ve / vh))
            null[k] = rayleigh(np.array(ang))
        tests.append({"protein": g, "n_helices": int(len(d)), "resultant": obs,
                      "null_mean": float(np.nanmean(null)),
                      "p": float((np.nansum(null >= obs) + 1) / (n_perm + 1)),
                      "mean_delta_deg": float(np.rad2deg(
                          np.angle(np.sum(np.exp(1j * d.delta_angle.dropna())))))})
    tf = pd.DataFrame(tests)

    fig, pairs, _ = grid(names, w=3.0, h=2.9, extra=0.3, polar=True)
    for i, (g, ax) in enumerate(pairs):
        d = per[per.protein == g].dropna(subset=["delta_angle"])
        t = tf[tf.protein == g] if len(tf) else tf
        if not len(d):
            ax.set_rticks([]); ax.set_xticks([])
            ax.set_title(f"({chr(97 + i)}) {short_name(g)}  no scored TM helix",
                         loc="left", fontsize=8)
            continue
        for r in d.itertuples():
            ax.annotate("", xy=(r.delta_angle, 1.0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                        color=effect_color(r.mean_effect)))
            ax.text(r.delta_angle, 1.12, r.helix, fontsize=5.5, ha="center", va="center")
        res = complex(np.sum(np.exp(1j * d.delta_angle))) / len(d)
        ax.annotate("", xy=(np.angle(res), abs(res)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=P.INK))
        ax.set_rticks([]); ax.set_ylim(0, 1.25)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(["aligned\n0°", "90°", "opposite\n180°", "270°"], fontsize=6)
        ax.grid(alpha=0.3)
        ttl = (f"R = {t.resultant.iloc[0]:.2f}, p = {t.p.iloc[0]:.3f}" if len(t)
               else f"{len(d)} helices, too few to test")
        ax.set_title(f"({chr(97 + i)}) {short_name(g)}  {ttl}", loc="left", fontsize=8)
    fig.suptitle("Each arrow is one TM helix: the angle between its hydrophobic face and "
                 "its intolerant face, coloured by the helix's mean effect. Thick arrow = "
                 "the resultant over helices.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b03_face_alignment", None, "B03")
    plt.close(fig)
    per.to_csv(outdir / "b03_helix_faces.csv", index=False)
    tf.to_csv(outdir / "b03_face_tests.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} mean Δ {r.mean_delta_deg:+.0f}°, "
                     f"R = {r.resultant:.2f} (p = {r.p:.3f})"
                     for r in tf.itertuples()) if len(tf) else ""
    return {"status": "ok",
            "headline": head or f"no protein has {MIN_HELICES} testable TM helices",
            "by_protein": tf.to_dict("records") if len(tf) else []}


# ------------------------------------- b04 what a charge costs as a function of depth
Z_BINS = np.array([0, 5, 10, 15, 20, 30])
CHARGE_SETS = [("charged (D E K R)", set("DEKR"), P.DARK_RED),
               ("histidine (half a charge)", set("H"), "#E8A33D"),
               ("hydrophobic (A V L I F M)", set("AVLIFM"), "#2166AC")]


def b04_charge_depth(bits: dict, names: list[str], outdir: Path) -> dict:
    """Burying a charge in the acyl region should cost; doing it at the headgroups
    should cost much less; and doing it on the surface should barely matter at all.

    The quantity being tested is therefore not the main effect of depth but the
    interaction of depth with burial: a charge introduced at an exposed position is not
    in the lipid whatever its depth, and is the control the comparison needs.
    """
    rows, models = [], []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        v = bits[g]["var"].dropna(subset=["score_z", "abs_z", "rsa"])
        for lab, aas, col in CHARGE_SETS:
            for buried, ls, alpha in ((True, "-", 1.0), (False, (0, (3, 2)), 0.6)):
                sub = v[v.mut.isin(list(aas))
                        & ((v.rsa < 0.15) if buried else (v.rsa >= 0.35))]
                if len(sub) < MIN_GROUP * 2:
                    continue
                show = buried or lab.startswith("charged")
                binned(ax, sub.abs_z, sub.score_z, Z_BINS, col,
                       f"{lab}, {'buried' if buried else 'exposed'}" if show else "",
                       ls=ls, alpha=alpha)
                rows.append({"protein": g, "group": lab, "buried": buried,
                             "n": int(len(sub)), "mean": float(sub.score_z.mean())})
        c = v[v.mut.isin(list("DEKR"))].copy()
        c["buried"] = (c.rsa < 0.15).astype(float)
        m = cluster_ols("score_z ~ abs_z * buried",
                        c[["score_z", "abs_z", "buried"]], c.pos)
        if m is not None:
            ci = m.conf_int().loc["abs_z:buried"]
            models.append({"protein": g, "n": int(m.nobs),
                           "depth": float(m.params["abs_z"]),
                           "buried": float(m.params["buried"]),
                           "interaction": float(m.params["abs_z:buried"]),
                           "interaction_se": float(m.bse["abs_z:buried"]),
                           "interaction_lo": float(ci[0]), "interaction_hi": float(ci[1]),
                           "interaction_p": float(m.pvalues["abs_z:buried"])})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        d = [r for r in models if r["protein"] == g]
        panel(ax, i, g, (f"depth×buried {d[0]['interaction']:+.3f} "
                         f"(p {d[0]['interaction_p']:.2g})" if d else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("depth |z| from the bilayer centre (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect of the substitution", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.suptitle("Solid = buried (RSA < 0.15), dashed = exposed (RSA ≥ 0.35). The claim "
                 "is a rising solid red line and a flat dashed one: expensive deep in the "
                 "bilayer, cheap at the headgroups, irrelevant on the surface.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b04_charge_depth_interaction", None, "B04")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "b04_groups.csv", index=False)
    mo = pd.DataFrame(models)
    mo.to_csv(outdir / "b04_models.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} depth×buried {r.interaction:+.4f} "
                     f"[{r.interaction_lo:+.4f}, {r.interaction_hi:+.4f}]"
                     for r in mo.itertuples()) if len(mo) else ""
    return {"status": "ok", "headline": head, "models": mo.to_dict("records")}


# ------------------------------------------------- b05 buried polar: defect or site?
def wt_group(wt: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=wt.index, dtype=object)
    out[wt.isin(list(POLAR))] = "polar"
    out[wt.isin(list("AVLIFMCW"))] = "hydrophobic"
    return out


def b05_buried_polar(bits: dict, names: list[str], outdir: Path) -> dict:
    """A polar residue buried in a membrane protein is a deliberate act.

    Burying one costs desolvation energy that the fold would not pay by accident, so a
    buried polar side chain is usually coordinating something: a substrate, a proton, a
    helix-helix hydrogen bond. The prediction is that buried polar positions - and
    cavity-lining polar positions above all, since that is where the substrate is - are
    the most intolerant class in the protein, more so than buried hydrophobic positions
    that are only there to fill space.
    """
    order = [("core", "hydrophobic"), ("core", "polar"),
             ("cavity", "hydrophobic"), ("cavity", "polar"),
             ("lipid", "hydrophobic"), ("lipid", "polar")]
    rows, contrasts = [], []
    fig, pairs, _ = grid(names, w=3.2, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "wt"]).copy()
        p["wt_group"] = wt_group(p.wt)
        groups, cols = {}, {}
        for c, w in order:
            s = p.loc[(p.bclass == c) & (p.wt_group == w), "effect"].dropna()
            if len(s) < MIN_POSITIONS:
                continue
            lab = f"{c}\n{w[:5]}"
            groups[lab] = s.to_numpy()
            cols[lab] = effect_color(float(s.median()))
            rows.append({"protein": g, "bclass": c, "wt_group": w, "n": int(len(s)),
                         "mean": float(s.mean()), "median": float(s.median()),
                         "se": float(s.std(ddof=1) / np.sqrt(len(s)))})
        if groups:
            P.violin_box(ax, groups, colors=cols)
        else:
            empty_panel(ax, i, g, f"no group has {MIN_POSITIONS} scored positions")
        for (ca, wa), (cb, wb) in ((("cavity", "polar"), ("core", "hydrophobic")),
                                   (("core", "polar"), ("core", "hydrophobic"))):
            x = p.loc[(p.bclass == ca) & (p.wt_group == wa), "effect"].dropna()
            y = p.loc[(p.bclass == cb) & (p.wt_group == wb), "effect"].dropna()
            if len(x) < MIN_POSITIONS or len(y) < MIN_POSITIONS:
                continue
            contrasts.append({"protein": g, "contrast": f"{ca}/{wa} − {cb}/{wb}",
                              "estimate": float(x.mean() - y.mean()),
                              "se": float(np.sqrt(x.var(ddof=1) / len(x)
                                                  + y.var(ddof=1) / len(y))),
                              "n_a": int(len(x)), "n_b": int(len(y)),
                              "p": float(ss.mannwhitneyu(x, y).pvalue)})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axhline(-1, color=P.INK, lw=0.6, ls=(0, (1, 2)))
        ax.tick_params(axis="x", labelsize=6)
        panel(ax, i, g, size=8)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
    fig.suptitle("Violins filled by their own median on the usual scale. The claim is "
                 "that cavity/polar is the darkest group in each panel.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b05_buried_polar", None, "B05")
    plt.close(fig)
    per, con = pd.DataFrame(rows), pd.DataFrame(contrasts)
    per.to_csv(outdir / "b05_groups.csv", index=False)
    con.to_csv(outdir / "b05_contrasts.csv", index=False)
    key = "cavity/polar − core/hydrophobic"
    head = "; ".join(f"{short_name(r.protein)} {r.estimate:+.3f} (p = {r.p:.2g})"
                     for r in con[con.contrast == key].itertuples()) if len(con) else ""
    if head:
        head = f"cavity polar − core hydrophobic: {head}"
    elif len(per):
        # no protein had both sides of the contrast; say what there was instead of
        # printing an empty line that reads as a result of zero
        head = "contrast unavailable; " + "; ".join(
            f"{short_name(r.protein)} {r.bclass}/{r.wt_group} {r.mean:+.2f} (n = {r.n})"
            for r in per.itertuples())
    else:
        head = f"no group reached {MIN_POSITIONS} scored positions"
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------------- b06 volume against hydrophobicity
def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd else s * 0.0


VOL_BINS = np.array([-140, -60, -25, 0, 25, 60, 140])


def b06_volume_vs_hydrophobicity(bits: dict, names: list[str], outdir: Path) -> dict:
    """Two different things get called "a conservative substitution".

    Hydrophobicity is about solvation - whether the side chain is happy where it is -
    and volume is about packing: whether it fits. They are correlated across the twenty
    amino acids, so the only way to say which an environment cares about is to put both
    in the same model. Signed and absolute terms are both fitted, because a packing
    constraint should care about the size of the change in either direction while a
    solvation constraint should care about its sign.
    """
    rows = []
    for g in names:
        v = bits[g]["var"]
        for c in BCLASSES:
            d = v[v.bclass == c].dropna(subset=["score_z", "delta_volume", "delta_kd"]).copy()
            if len(d) < MIN_GROUP * 4:
                continue
            d["dvol"] = zscore(d.delta_volume)
            d["advol"] = zscore(d.delta_volume.abs())
            d["dkd"] = zscore(d.delta_kd)
            d["adkd"] = zscore(d.delta_kd.abs())
            m = cluster_ols("score_z ~ dvol + advol + dkd + adkd",
                            d[["score_z", "dvol", "advol", "dkd", "adkd"]], d.pos)
            if m is None:
                continue
            rows.append({"protein": g, "bclass": c, "n": int(m.nobs),
                         "n_positions": int(d.pos.nunique()), "r2": float(m.rsquared),
                         **{f"{k}": float(m.params[k]) for k in
                            ("dvol", "advol", "dkd", "adkd")},
                         **{f"{k}_se": float(m.bse[k]) for k in
                            ("dvol", "advol", "dkd", "adkd")},
                         **{f"{k}_p": float(m.pvalues[k]) for k in
                            ("dvol", "advol", "dkd", "adkd")}})
    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "no class has enough variants to fit"}

    terms = [("advol", "|Δ volume|"), ("dvol", "Δ volume (signed)"),
             ("adkd", "|Δ hydrophobicity|"), ("dkd", "Δ hydrophobicity (signed)")]
    fig, pairs, _ = grid(names, w=3.2, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = per[per.protein == g]
        if not len(d):
            empty_panel(ax, i, g, "no class had enough variants to fit")
            continue
        classes = [c for c in BCLASSES if c in set(d.bclass)]
        w = 0.8 / max(len(classes), 1)
        x = np.arange(len(terms))
        for k, c in enumerate(classes):
            r = d[d.bclass == c].iloc[0]
            ax.bar(x + k * w - 0.4 + w / 2, [r[t] for t, _ in terms], w,
                   yerr=[1.96 * r[f"{t}_se"] for t, _ in terms], color=BCOLORS[c],
                   edgecolor=P.INK, linewidth=0.5, label=BLABEL[c],
                   error_kw=dict(lw=0.8, capsize=1.8))
        ax.axhline(0, color=P.MUTED, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in terms], rotation=25, fontsize=6,
                           ha="right")
        panel(ax, i, g, size=8)
        if i % 2 == 0:
            ax.set_ylabel("standardised coefficient\n(effect per SD of the predictor)",
                          fontsize=7)
        if i == 0:
            ax.legend(frameon=False, fontsize=5.5)
    fig.suptitle("All four predictors are standardised, so the bars are comparable "
                 "within a panel. Bars are 95% intervals with errors clustered by "
                 "position.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b06a_volume_vs_hydrophobicity", None, "B06")
    plt.close(fig)

    fig, pairs, axes = grid(names, w=3.0, h=2.8, extra=0.3)
    im = None
    for i, (g, ax) in enumerate(pairs):
        d = bits[g]["var"]
        d = d[d.bclass == "core"].dropna(subset=["score_z", "delta_volume", "delta_kd"])
        m = np.full((len(KD_BINS) - 1, len(VOL_BINS) - 1), np.nan)
        for a in range(len(KD_BINS) - 1):
            for b in range(len(VOL_BINS) - 1):
                s = d[(d.delta_kd >= KD_BINS[a]) & (d.delta_kd < KD_BINS[a + 1])
                      & (d.delta_volume >= VOL_BINS[b])
                      & (d.delta_volume < VOL_BINS[b + 1])].score_z
                if len(s) >= 5:
                    m[a, b] = float(s.mean())
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=empty_safe_cmap(),
                       norm=P.fitness_norm(),
                       extent=(0, len(VOL_BINS) - 1, 0, len(KD_BINS) - 1))
        ax.set_xticks(np.arange(len(VOL_BINS) - 1) + 0.5)
        ax.set_xticklabels([f"{VOL_BINS[b]:+.0f}" for b in range(len(VOL_BINS) - 1)],
                           fontsize=6)
        ax.set_yticks(np.arange(len(KD_BINS) - 1) + 0.5)
        ax.set_yticklabels([f"{KD_BINS[a]:+.1f}" for a in range(len(KD_BINS) - 1)],
                           fontsize=6)
        panel(ax, i, g, "core positions only", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ volume (Å³)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Δ hydrophobicity (KD)", fontsize=7.5)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    P.save(fig, outdir, "b06b_core_surface", None, "B06")
    plt.close(fig)
    per.to_csv(outdir / "b06_models.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} {r.bclass}: |Δvol| {r.advol:+.3f}, "
                     f"Δkd {r.dkd:+.3f}"
                     for r in per[per.bclass.isin(PRIMARY)].itertuples())
    return {"status": "ok", "headline": head, "models": per.to_dict("records")}


# --------------------------------------------- b07 small residues at close approaches
def b07_small_residues(bits: dict, names: list[str], outdir: Path) -> dict:
    """Where two helices come within a CA-CA distance of 6.5 Å there is no room for a
    side chain, and the sequence says so: those positions are glycine, alanine or serine.

    The test is whether that is a constraint or a coincidence. If it is a constraint,
    substituting at a small residue in a close approach should cost more than at a
    comparably buried position with a large wild type, and the cost should scale with
    how much bigger the new residue is.
    """
    rows, slopes = [], []
    fig, pairs, _ = grid(names, w=3.2, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "wt", "inter_helix_ca"]).copy()
        p["small"] = p.wt.isin(list(SMALL))
        p["close"] = p.inter_helix_ca < NEAR_CA
        groups, cols = {}, {}
        for lab, mask in (("small\n& close", p.small & p.close),
                          ("large\n& close", (~p.small) & p.close),
                          ("small\n& far", p.small & (~p.close))):
            s = p.loc[mask, "effect"].dropna()
            if len(s) < MIN_POSITIONS:
                continue
            groups[lab] = s.to_numpy()
            cols[lab] = effect_color(float(s.median()))
            rows.append({"protein": g, "group": lab.replace("\n", " "), "n": int(len(s)),
                         "mean": float(s.mean()), "median": float(s.median()),
                         "se": float(s.std(ddof=1) / np.sqrt(len(s)))})
        if groups:
            P.violin_box(ax, groups, colors=cols)
        else:
            empty_panel(ax, i, g, f"no group has {MIN_POSITIONS} scored positions")
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.tick_params(axis="x", labelsize=6.5)

        v = bits[g]["var"].dropna(subset=["score_z", "delta_volume", "inter_helix_ca",
                                          "wt"])
        for lab, mask in (("small & close",
                           v.wt.isin(list(SMALL)) & (v.inter_helix_ca < NEAR_CA)),
                          ("other buried TM",
                           (~v.wt.isin(list(SMALL))) & (v.is_tm > 0.5) & (v.rsa < 0.25))):
            d = v[mask]
            m = cluster_ols("score_z ~ delta_volume", d[["score_z", "delta_volume"]],
                            d.pos)
            if m is None:
                continue
            slopes.append({"protein": g, "group": lab, "n": int(m.nobs),
                           "slope_per_A3": float(m.params["delta_volume"]),
                           "se": float(m.bse["delta_volume"]),
                           "p": float(m.pvalues["delta_volume"])})
        d = [r for r in slopes if r["protein"] == g and r["group"] == "small & close"]
        panel(ax, i, g, (f"Δvolume slope {d[0]['slope_per_A3'] * 100:+.2f}/100 Å³"
                         if d else ""), size=8)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
    fig.suptitle(f"close = CA within {NEAR_CA:g} Å of a different TM helix; small = G, A "
                 "or S. The claim is that the first violin is the lowest, and that making "
                 "those residues bigger is what does it.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b07a_small_at_close_approaches", None, "B07")
    plt.close(fig)

    sl = pd.DataFrame(slopes)
    if len(sl):
        fig, pairs, _ = grid(names, w=2.9, h=2.2, extra=0.2)
        for i, (g, ax) in enumerate(pairs):
            d = sl[sl.protein == g].to_dict("records")
            if not d:
                empty_panel(ax, i, g, "too few positions to fit a Δvolume slope")
                continue
            forest(ax, [{"lab": r["group"], "v": r["slope_per_A3"] * 100,
                         "se": r["se"] * 100} for r in d], "lab", "v", "se", None,
                   "effect per +100 Å³ of side chain")
            panel(ax, i, g, size=8)
        P.save(fig, outdir, "b07b_volume_slopes", None, "B07")
        plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "b07_groups.csv", index=False)
    sl.to_csv(outdir / "b07_volume_slopes.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} small&close {r.mean:+.3f} (n = {r.n})"
                     for r in per[per.group == "small & close"].itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------------------------------- b08 the aromatic belt
def b08_aromatic_belt(bits: dict, names: list[str], outdir: Path) -> dict:
    """Tryptophan and tyrosine pile up at the lipid headgroups, not in the acyl core.

    The usual reading is that they anchor the helix at the interface, with the ring in
    the hydrophobic part and the polar edge in the headgroups. If that is what they are
    doing here, then at belt positions what should matter is losing the aromatic ring,
    not losing hydrophobicity - and those are separable, because tyrosine is aromatic
    and barely hydrophobic while leucine is the reverse.
    """
    rows, models = [], []
    fig, pairs, _ = grid(names, w=3.1, h=2.8, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        p = bits[g]["pos"].dropna(subset=["effect", "abs_z", "wt"])
        ar = p[p.wt.isin(list(AROMATIC))]
        ax.axvspan(BELT[0], BELT[1], color="#F0F0F0", zorder=0)
        binned(ax, p.abs_z, p.effect, Z_BINS, P.MUTED, "all positions")
        if len(ar) >= MIN_POSITIONS:
            ax.scatter(ar.abs_z, ar.effect, s=18, c=[effect_color(e) for e in ar.effect],
                       edgecolor=P.INK, linewidth=0.3, zorder=4)
            binned(ax, ar.abs_z, ar.effect, Z_BINS, P.DARK_RED, "aromatic WT (F W Y)")
        for lab, mask in (("belt", ar.abs_z.between(*BELT)),
                          ("acyl core", ar.abs_z < CORE_Z)):
            s = ar.loc[mask, "effect"].dropna()
            if len(s) >= 5:
                rows.append({"protein": g, "band": lab, "n": int(len(s)),
                             "mean": float(s.mean()),
                             "se": float(s.std(ddof=1) / np.sqrt(len(s)))})

        v = bits[g]["var"].dropna(subset=["score_z", "abs_z", "delta_kd", "wt"])
        a = v[v.wt.isin(list(AROMATIC))].copy()
        a["loses_ring"] = (~a.mut.isin(list(AROMATIC))).astype(float)
        a["belt"] = a.abs_z.between(*BELT).astype(float)
        m = cluster_ols("score_z ~ loses_ring * belt + delta_kd",
                        a[["score_z", "loses_ring", "belt", "delta_kd"]], a.pos)
        if m is not None:
            models.append({"protein": g, "n": int(m.nobs),
                           "loses_ring": float(m.params["loses_ring"]),
                           "loses_ring_se": float(m.bse["loses_ring"]),
                           "loses_ring_p": float(m.pvalues["loses_ring"]),
                           "belt": float(m.params["belt"]),
                           "interaction": float(m.params["loses_ring:belt"]),
                           "interaction_se": float(m.bse["loses_ring:belt"]),
                           "interaction_p": float(m.pvalues["loses_ring:belt"]),
                           "delta_kd": float(m.params["delta_kd"]),
                           "delta_kd_p": float(m.pvalues["delta_kd"])})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        d = [r for r in models if r["protein"] == g]
        panel(ax, i, g, (f"lose ring {d[0]['loses_ring']:+.3f}, ×belt "
                         f"{d[0]['interaction']:+.3f}" if d else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("depth |z| from the bilayer centre (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.suptitle(f"Shaded = the belt, |z| {BELT[0]:g}–{BELT[1]:g} Å. The model separates "
                 "losing the ring from losing hydrophobicity, which a comparison of means "
                 "cannot.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b08_aromatic_belt", None, "B08")
    plt.close(fig)
    per, mo = pd.DataFrame(rows), pd.DataFrame(models)
    per.to_csv(outdir / "b08_bands.csv", index=False)
    mo.to_csv(outdir / "b08_models.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} ring {r.loses_ring:+.3f} "
                     f"(p {r.loses_ring_p:.2g}), ×belt {r.interaction:+.3f}"
                     for r in mo.itertuples()) if len(mo) else ""
    return {"status": "ok", "headline": head, "models": mo.to_dict("records")}


# ------------------------------------------------------------ b09 hydrophobic mismatch
ENV_BINS = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 3.5])
MUT_BINS = np.array([-4.5, -3.0, -1.0, 0.5, 2.0, 3.2, 4.5])


def b09_mismatch(bits: dict, names: list[str], outdir: Path) -> dict:
    """A leucine is not intrinsically a good substitution; it is a good substitution
    somewhere hydrophobic.

    So the predictor that should work is not the hydrophobicity of the residue that
    arrives, nor the change from the wild type, but the mismatch between the arriving
    residue and the residues packed around it. The heatmap is the direct version: if
    mismatch is what matters, the pale diagonal runs from bottom-left to top-right and
    the cost lives in the off-diagonal corners.
    """
    rows = []
    fig, pairs, axes = grid(names, w=3.1, h=2.8, extra=0.3)
    im = None
    for i, (g, ax) in enumerate(pairs):
        v = bits[g]["var"]
        d = v[(v.is_tm > 0.5)].dropna(subset=["score_z", "kd_mut", "kd_env", "delta_kd"]).copy()
        m = np.full((len(ENV_BINS) - 1, len(MUT_BINS) - 1), np.nan)
        for a in range(len(ENV_BINS) - 1):
            for b in range(len(MUT_BINS) - 1):
                s = d[(d.kd_env >= ENV_BINS[a]) & (d.kd_env < ENV_BINS[a + 1])
                      & (d.kd_mut >= MUT_BINS[b]) & (d.kd_mut < MUT_BINS[b + 1])].score_z
                if len(s) >= 5:
                    m[a, b] = float(s.mean())
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=empty_safe_cmap(),
                       norm=P.fitness_norm(),
                       extent=(0, len(MUT_BINS) - 1, 0, len(ENV_BINS) - 1))
        ax.set_xticks(np.arange(len(MUT_BINS) - 1) + 0.5)
        ax.set_xticklabels([f"{MUT_BINS[b]:+.1f}" for b in range(len(MUT_BINS) - 1)],
                           fontsize=6)
        ax.set_yticks(np.arange(len(ENV_BINS) - 1) + 0.5)
        ax.set_yticklabels([f"{ENV_BINS[a]:+.1f}" for a in range(len(ENV_BINS) - 1)],
                           fontsize=6)

        d["mismatch"] = (d.kd_mut - d.kd_env).abs()
        d["abs_dkd"] = d.delta_kd.abs()
        d["kd_mut_c"] = d.kd_mut - d.kd_mut.mean()
        d["kd_env_c"] = d.kd_env - d.kd_env.mean()
        fits = {}
        for key, formula, cols in (
                ("mismatch", "score_z ~ mismatch", ["mismatch"]),
                ("abs_delta_kd", "score_z ~ abs_dkd", ["abs_dkd"]),
                ("interaction", "score_z ~ kd_mut_c * kd_env_c",
                 ["kd_mut_c", "kd_env_c"])):
            mod = cluster_ols(formula, d[["score_z"] + cols], d.pos)
            if mod is not None:
                fits[key] = mod
        if fits:
            r = {"protein": g, "n": int(len(d))}
            if "mismatch" in fits:
                r.update(mismatch=float(fits["mismatch"].params["mismatch"]),
                         mismatch_se=float(fits["mismatch"].bse["mismatch"]),
                         mismatch_p=float(fits["mismatch"].pvalues["mismatch"]),
                         mismatch_r2=float(fits["mismatch"].rsquared))
            if "abs_delta_kd" in fits:
                r.update(abs_delta_kd=float(fits["abs_delta_kd"].params["abs_dkd"]),
                         abs_delta_kd_r2=float(fits["abs_delta_kd"].rsquared))
            if "interaction" in fits:
                f = fits["interaction"]
                r.update(kd_mut=float(f.params["kd_mut_c"]),
                         kd_env=float(f.params["kd_env_c"]),
                         kd_interaction=float(f.params["kd_mut_c:kd_env_c"]),
                         kd_interaction_se=float(f.bse["kd_mut_c:kd_env_c"]),
                         kd_interaction_p=float(f.pvalues["kd_mut_c:kd_env_c"]))
            rows.append(r)
        r2m = rows[-1].get("mismatch_r2") if rows and rows[-1]["protein"] == g else None
        r2d = rows[-1].get("abs_delta_kd_r2") if rows and rows[-1]["protein"] == g else None
        panel(ax, i, g, (f"R² mismatch {r2m:.3f} vs |ΔKD| {r2d:.3f}"
                         if r2m is not None and r2d is not None else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("hydrophobicity of the mutant residue (KD)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("hydrophobicity of the surroundings\n(mean KD within 8 Å)",
                          fontsize=7.5)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    fig.suptitle("TM variants only. Mismatch predicts a pale diagonal: a polar residue is "
                 "tolerated in polar surroundings and a hydrophobic one in hydrophobic "
                 "surroundings.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b09_hydrophobic_mismatch", None, "B09")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "b09_mismatch_models.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} mismatch {r.mismatch:+.3f} "
                     f"(R² {r.mismatch_r2:.3f} vs {r.abs_delta_kd_r2:.3f} for |ΔKD|)"
                     for r in per.itertuples()
                     if hasattr(r, "mismatch")) if len(per) else ""
    return {"status": "ok", "headline": head, "models": per.to_dict("records")}


# ------------------------------------------- b10 are ESM-1v's errors structured?
def b10_esm_residuals(bits: dict, names: list[str], outdir: Path) -> dict:
    """ESM-1v knows about evolution, not about the membrane. Does it show?

    A residual here is a position where the model expects more constraint than abundance
    shows. If those residuals were noise they would have nothing to do with where the
    position sits. If instead they line up with burial, depth or the polarity of what is
    being introduced, that is a specific account of what the language model is missing -
    and an argument for putting structure into the prior rather than hoping the model
    has learnt it.
    """
    from .environs import esm_residuals
    terms = [("radial", "distance from the axis"), ("abs_z", "depth |z|"),
             ("rsa", "solvent accessibility"),
             ("buried_polar", "buried and polar (0/1)")]
    rows, models, have = [], [], []
    for g in names:
        res = esm_residuals(bits[g]["cfg"])
        if res is None:
            continue
        p = bits[g]["pos"]
        t = res.merge(p[["pos", "radial", "abs_z", "rsa", "bclass", "effect"]], on="pos",
                      how="inner").dropna(subset=["median_z", "radial", "abs_z", "rsa"])
        if len(t) < 60:
            continue
        t = t.copy()
        t["buried_polar"] = ((t.rsa < 0.15) & t.wt.isin(list(POLAR))).astype(float)
        for c in ("radial", "abs_z", "rsa"):
            t[c + "_z"] = zscore(t[c])
        import statsmodels.formula.api as smf
        m = smf.ols("median_z ~ radial_z + abs_z_z + rsa_z + buried_polar",
                    data=t).fit(cov_type="HC3")
        key = {"radial": "radial_z", "abs_z": "abs_z_z", "rsa": "rsa_z",
               "buried_polar": "buried_polar"}
        models.append({"protein": g, "n": int(m.nobs), "r2": float(m.rsquared),
                       **{k: float(m.params[v]) for k, v in key.items()},
                       **{f"{k}_se": float(m.bse[v]) for k, v in key.items()},
                       **{f"{k}_p": float(m.pvalues[v]) for k, v in key.items()}})
        rows.append(t.assign(protein=g))
        have.append(g)
    if not have:
        return {"status": "skipped",
                "reason": "no protein has both ESM-1v scores and enough scored positions"}

    fig, pairs, _ = grid(have, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        t = [d for d in rows if d.protein.iloc[0] == g][0]
        ax.scatter(t.radial, t.median_z, s=12, color="#D9D9D9", lw=0)
        for c, col, lab in ((t.bclass == "core", BCOLORS["core"], "core"),
                            (t.bclass == "lipid", BCOLORS["lipid"], "lipid-facing")):
            s = t[c]
            if len(s) >= 10:
                ax.scatter(s.radial, s.median_z, s=14, color=col, lw=0.3,
                           edgecolor=P.INK, label=lab)
        binned(ax, t.radial, t.median_z,
               np.linspace(float(t.radial.min()), float(t.radial.max()), 8), P.INK, "")
        ax.axhline(0, color=P.MUTED, lw=0.6)
        mo = [r for r in models if r["protein"] == g][0]
        panel(ax, i, g, f"radial {mo['radial']:+.2f} (p {mo['radial_p']:.2g})", size=8)
        if i // 2 == (len(have) - 1) // 2:
            ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("ESM-1v residual at the position\n(negative = ESM expects more "
                          "constraint)", fontsize=7)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("A flat cloud would mean the model's errors are unrelated to where the "
                 "residue sits, which is what you would want of a model that had already "
                 "learnt the membrane.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "b10a_residual_vs_burial", None, "B10")
    plt.close(fig)

    mo = pd.DataFrame(models)
    fig, pairs, _ = grid(have, w=3.0, h=2.4, extra=0.2)
    for i, (g, ax) in enumerate(pairs):
        r = mo[mo.protein == g].iloc[0]
        forest(ax, [{"lab": lab, "v": float(r[k]), "se": float(r[f"{k}_se"])}
                    for k, lab in terms], "lab", "v", "se", None,
               "coefficient on the ESM residual\n(per SD, except the 0/1 term)")
        panel(ax, i, g, f"R² = {r.r2:.3f}", size=8)
    P.save(fig, outdir, "b10b_residual_model", None, "B10")
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(outdir / "b10_residual_positions.csv",
                                              index=False)
    mo.to_csv(outdir / "b10_residual_models.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} R² {r.r2:.3f}, buried-polar "
                     f"{r.buried_polar:+.2f} (p {r.buried_polar_p:.2g})"
                     for r in mo.itertuples())
    return {"status": "ok", "headline": head, "models": mo.to_dict("records")}


ANALYSES = {"b01": "hydrophobicity tolerance: lipid vs core vs aqueous",
            "b02": "burial as a continuum",
            "b03": "hydrophobic face against intolerant face",
            "b04": "the cost of a charge against depth, and its interaction with burial",
            "b05": "buried polar residues",
            "b06": "volume against hydrophobicity",
            "b07": "small residues at close helix-helix approaches",
            "b08": "the aromatic belt",
            "b09": "hydrophobic mismatch with the local environment",
            "b10": "whether ESM-1v's residuals are structured by burial"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|All-NaN|"
                                              "tight_layout).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms burial",
        description="Burial and hydrophobicity in a membrane protein: ten analyses, one "
                    "panel per protein, in the order the configs are given.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--perm", type=int, default=N_PERM,
                    help="permutations for the b03 null (default 2000)")
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

    bits, names = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        b = bits_for(cfg)
        if not b or b["pos"].bclass.isna().all():
            print(f"  {cfg.id:<16} skipped: needs a structure and a topology", flush=True)
            continue
        bits[cfg.id], _ = b, names.append(cfg.id)
        counts = b["pos"].bclass.value_counts()
        print(f"  {cfg.id:<16} {len(b['pos']):>4} positions  "
              + "  ".join(f"{k} {n}" for k, n in counts.items()), flush=True)
    if not names:
        raise SystemExit("no protein could be classified; each needs a structure and a "
                         "topology in its config")
    rows = int(np.ceil(len(names) / a.cols))
    print("\n  panels: " + " | ".join(
        ", ".join(short_name(n) for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)
    pd.concat([b["pos"].assign(protein=g) for g, b in bits.items()],
              ignore_index=True).to_csv(outdir / "positions.csv", index=False)

    calls = {"b01": lambda: b01_three_way(bits, names, outdir),
             "b02": lambda: b02_continuum(bits, names, outdir),
             "b03": lambda: b03_face_alignment(bits, names, outdir, a.perm),
             "b04": lambda: b04_charge_depth(bits, names, outdir),
             "b05": lambda: b05_buried_polar(bits, names, outdir),
             "b06": lambda: b06_volume_vs_hydrophobicity(bits, names, outdir),
             "b07": lambda: b07_small_residues(bits, names, outdir),
             "b08": lambda: b08_aromatic_belt(bits, names, outdir),
             "b09": lambda: b09_mismatch(bits, names, outdir),
             "b10": lambda: b10_esm_residuals(bits, names, outdir)}

    out = {}
    print()
    for k in keys:
        try:
            r = calls[k]()
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    (outdir / "burial.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
