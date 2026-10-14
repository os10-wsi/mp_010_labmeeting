"""`python -m mpdms cavity configs/AQR1_fitness.yaml configs/HXT1.yaml`

Are the residues lining the central cavity held more tightly by evolution than the rest
of the protein, once the things that would fake that are taken away?

The quantity is the site median ESM-1v residual, defined exactly as A15 defines it: how
much more constrained a position looks to a language model than its measured abundance
effect explains. A strongly negative residual is the signature this project calls
functional - conserved for a reason the abundance assay cannot see.

It is reported in the residual's own units - ESM-1v log-likelihood ratio - rather than
in the robust z A15 plots. The two differ by one constant shared by every residue in the
protein (1.4826 x MAD of the residuals), so every rank test and every p-value below is
numerically identical either way; only the effect sizes change units, and a difference
quoted in log-likelihood ratio is a difference someone can interpret. Dividing by a
constant is not what makes a z-score awkward to test - standardising inside each group
is, and nothing here does that. The signed residual is used, never its absolute value:
the sign is the biology, and folding it away would put a position the model likes and a
position it dislikes in the same bin.

Three things would produce a cavity effect with no biology behind them, and each is
handled rather than hoped away:

    topology        cavity residues are transmembrane and TM residues are more
                    constrained than loops in general, so the test is also run with
                    loops excluded entirely, comparing the cavity against the rest of
                    the membrane domain
    spatial         constraint travels about 20 Å through the structure (s08), so
                    neighbouring residues are not independent observations. Standard
                    errors are clustered on spatial blocks of roughly that size, and
                    the naive rank test is reported beside them to show the gap
    the threshold   "cavity-lining" is a median split on distance from the bundle axis,
                    so the whole claim could be an artefact of where the line was drawn.
                    The radial coordinate is therefore also used continuously, with no
                    threshold at all

"Consistently" is the word in the question, so the answer is not one p-value: the same
test is run in each protein separately and the module reports whether the direction,
the size and the significance agree between them.
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
from .architecture import protein_bits
from .burial import binned, empty_panel, short_name
from .config import REPO_ROOT, load_config
from .environs import esm_residuals, grid, panel

OUT = REPO_ROOT / "outputs" / "_cavity"
BLOCK_A = 20.0           # Å: the length scale s08 measures for constraint
MIN_GROUP = 8            # positions, before a group is compared
N_BOOT = 2000
# Three classes, and they partition the protein: every residue is in exactly one.
# Inside the membrane the split is on the radial coordinate alone - which side of the
# bundle a residue faces - so a buried helix-helix residue is no longer a class of its
# own but is counted with whichever face it sits on. Outside the membrane the two loop
# sides are pooled: they differ in which side they are on, which matters for insertion
# and not for this question, and both are solvated.
CLASS_ORDER = ["cavity", "lipid", "aqueous"]
CLASS_LABEL = {"cavity": "cavity-lining", "lipid": "lipid-facing",
               "aqueous": "aqueous: cytosol + lumen"}
CLASS_COLOR = {"cavity": "#2166AC", "lipid": "#9AA4B2", "aqueous": "#C7CDD4"}
VALUE = "median_residual"          # the column every test and every figure reads
VALUE_LABEL = "site median ESM-1v residual (log-likelihood ratio)"
N_PERM = 4000


# --------------------------------------------------------------------- loading
def spatial_blocks(xyz: np.ndarray, target_a: float = BLOCK_A,
                   seed: int = 0) -> np.ndarray:
    """Group residues into compact spatial blocks about `target_a` across.

    s08 measured constraint travelling roughly twenty angstroms, which means two
    residues that close are not two independent observations of anything. Clustering
    the coordinates into blocks of about that size and clustering the standard errors
    on them is the cheapest honest correction: it counts patches rather than residues,
    and a protein that looks like forty independent regions stops being treated as
    five hundred.
    """
    from scipy.cluster.vq import kmeans2
    xyz = np.asarray(xyz, float)
    if len(xyz) < 4:
        return np.zeros(len(xyz), int)
    span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    k = int(np.clip(round((span / target_a) ** 2), 2, max(2, len(xyz) // 5)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lab = kmeans2(xyz, k, minit="++", seed=seed, missing="warn")
    return np.asarray(lab, int)


def bits_for(cfg) -> dict:
    """Site residuals joined to the residue environment and to a spatial block."""
    b = protein_bits(cfg)
    if not b:
        return {}
    res = esm_residuals(cfg)
    if res is None or not len(res):
        return {}
    pos = b["pos"]
    keep = [c for c in ("pos", "environ", "radial", "abs_z", "rsa", "is_tm", "helix",
                        "seg_type") if c in pos.columns]
    t = res[["pos", "wt", "n_variants", "median_residual", "median_z",
             "median_abundance"]].merge(
        pos[keep], on="pos", how="inner").merge(
        b["rt"][["pos", "x", "y", "z"]], on="pos", how="inner")
    t = t.dropna(subset=[VALUE, "environ"])
    if len(t) < 40:
        return {}
    t["block"] = spatial_blocks(t[["x", "y", "z"]].to_numpy(float))
    t["is_tm_f"] = (t.is_tm > 0.5).astype(float) if "is_tm" in t else 0.0
    cut = float(pos.attrs.get("radial_cut", np.nan))
    t["klass"] = three_way(t, cut)
    t = t.dropna(subset=["klass"])
    t["is_cavity"] = (t.klass == "cavity").astype(float)
    b["sites"] = t
    b["radial_cut"] = cut
    # the one constant between these residuals and the robust z A15 plots, kept so a
    # caller can convert a threshold defined in z without recomputing the MAD
    b["residual_scale"] = float(res.attrs.get("residual_scale", 1.0))
    # how many residues the class change moved, so the redefinition is visible
    b["n_from_core"] = int(((t.environ == "helix-helix core")
                            & (t.klass == "cavity")).sum())
    b["n_core"] = int((t.environ == "helix-helix core").sum())
    return b


def three_way(t: pd.DataFrame, cut: float) -> pd.Series:
    """cavity / lipid / aqueous, exhaustive over the protein.

    The earlier five-class split kept buried helix-helix residues apart from both
    faces, which left them in the test but out of the figure. Splitting the membrane
    on radius alone puts every TM residue on one face or the other, so the groups the
    violins show are the groups the test compares.
    """
    out = pd.Series(pd.NA, index=t.index, dtype=object)
    tm = t.is_tm_f > 0.5
    if np.isfinite(cut):
        out[tm & (t.radial < cut)] = "cavity"
        out[tm & (t.radial >= cut)] = "lipid"
    out[~tm & t.environ.notna()] = "aqueous"
    return out


# --------------------------------------------------------------------- the test
def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b): how often a cavity residue outranks a non-cavity one.

    Reported because a p-value says only that a difference exists. Negative means the
    cavity residuals sit lower - more constrained - which is the direction the claim
    predicts.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def block_bootstrap_delta(t: pd.DataFrame, mask, n: int = N_BOOT, seed: int = 3,
                          value: str = VALUE) -> tuple[float, float]:
    """A confidence interval for Cliff's delta that resamples blocks, not residues.

    Resampling residues would treat a twenty-angstrom patch as many independent draws
    and return an interval far too narrow. Blocks are the unit that is approximately
    independent, so they are the unit that gets resampled.
    """
    d = t[mask]
    blocks = d.block.unique()
    if len(blocks) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = []
    by = {g: sub for g, sub in d.groupby("block")}
    for _ in range(n):
        pick = pd.concat([by[g] for g in rng.choice(blocks, len(blocks), replace=True)])
        a = pick.loc[pick.is_cavity > 0.5, value]
        c = pick.loc[pick.is_cavity < 0.5, value]
        if len(a) >= 3 and len(c) >= 3:
            out.append(cliffs_delta(a, c))
    if len(out) < 50:
        return np.nan, np.nan
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def distribution_tests(a, c) -> dict:
    """Beyond "are the centres different": are these two distributions the same shape?

    A median shift is one way a group can be more constrained; a heavier negative tail
    with the same centre is another, and a rank test on the centres can miss it. The
    Kolmogorov-Smirnov statistic reads the largest gap between the two cumulative
    distributions, which is exactly what the ECDF panel of V01 draws, and the
    Brown-Forsythe test asks whether the spreads differ at all. Both are quoted
    uncorrected for spatial structure, so they are read next to the block-clustered
    interval rather than instead of it.
    """
    a, c = np.asarray(a, float), np.asarray(c, float)
    a, c = a[np.isfinite(a)], c[np.isfinite(c)]
    if len(a) < 3 or len(c) < 3:
        return {}
    ks = ss.ks_2samp(a, c)
    return {"p_mannwhitney": float(ss.mannwhitneyu(a, c, alternative="two-sided").pvalue),
            "ks_d": float(ks.statistic), "p_ks": float(ks.pvalue),
            "p_brown_forsythe": float(ss.levene(a, c, center="median").pvalue),
            "iqr_a": float(np.subtract(*np.percentile(a, [75, 25]))),
            "iqr_b": float(np.subtract(*np.percentile(c, [75, 25])))}


def within_block_permutation(d: pd.DataFrame, value: str, group: str = "is_cavity",
                             n: int = N_PERM, seed: int = 11) -> float:
    """A p-value that never compares one patch of the protein with another.

    Clustering the standard errors assumes a model; this assumes almost nothing. The
    values are centred inside each 20 A block and the class labels are then shuffled
    only within blocks, so every comparison the null makes is between residues that sit
    in the same patch. Blocks holding just one class carry no information about the
    contrast and drop out. If the cavity effect is really a large-scale gradient that
    happens to line up with the bundle axis, this is the test it fails.
    """
    y = d[value].to_numpy(float)
    g = d[group].to_numpy(float) > 0.5
    blk = np.asarray(d["block"])
    idx = [np.flatnonzero(blk == b) for b in np.unique(blk)]
    idx = [i for i in idx if g[i].any() and (~g[i]).any()]
    if not idx:
        return float("nan")
    keep = np.concatenate(idx)
    if len(keep) < 2 * MIN_GROUP:
        return float("nan")
    yc = y.copy()
    for i in idx:
        yc[i] -= yc[i].mean()
    yc, g = yc[keep], g[keep]
    idx = np.split(np.arange(len(keep)), np.cumsum([len(i) for i in idx])[:-1])

    def stat(mask):
        return float(yc[mask].mean() - yc[~mask].mean())

    obs = stat(g)
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n):
        perm = np.zeros(len(g), bool)
        for i in idx:
            perm[rng.permutation(i)[: int(g[i].sum())]] = True
        hit += abs(stat(perm)) >= abs(obs) - 1e-12
    return float((hit + 1) / (n + 1))


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni: three pairwise comparisons are three chances to be wrong."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for k, j in enumerate(order):
        run = max(run, (len(ps) - k) * ps[j])
        adj[j] = min(run, 1.0)
    return [float(v) for v in adj]


def pstr(p: float) -> str:
    """p as it should be written on a figure: never "p = 0.000"."""
    if not np.isfinite(p):
        return "p n/a"
    return f"p < {1 / (N_PERM + 1):.0e}" if p < 1 / (N_PERM + 1) else (
        f"p = {p:.3g}" if p >= 1e-4 else f"p = {p:.1e}")


def pairwise_tests(t: pd.DataFrame, protein: str, value: str = VALUE) -> pd.DataFrame:
    """Every pair of the three classes, with the three p-values Holm-corrected.

    The figure shows three distributions, so the honest accompaniment is three
    comparisons and not one: cavity against each of the other two, and the two
    non-cavity classes against each other, which is the comparison that says whether
    "cavity" is doing anything that "inside the membrane at all" does not.
    """
    rows = []
    for i, k1 in enumerate(CLASS_ORDER):
        for k2 in CLASS_ORDER[i + 1:]:
            d = t[t.klass.isin([k1, k2])].dropna(subset=[value])
            a = d.loc[d.klass == k1, value]
            c = d.loc[d.klass == k2, value]
            if len(a) < MIN_GROUP or len(c) < MIN_GROUP:
                continue
            flag = (d.klass == k1).astype(float)
            rows.append({"protein": protein, "a": k1, "b": k2,
                         "n_a": int(len(a)), "n_b": int(len(c)),
                         "median_a": float(a.median()), "median_b": float(c.median()),
                         "shift": float(a.median() - c.median()),
                         "cliffs_delta": cliffs_delta(a, c),
                         **distribution_tests(a, c),
                         "p_within_block_perm": within_block_permutation(
                             d.assign(flag=flag), value, group="flag", seed=12)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_mannwhitney_holm"] = holm(out.p_mannwhitney.tolist())
        out["p_ks_holm"] = holm(out.p_ks.tolist())
    return out


def one_test(t: pd.DataFrame, mask, label: str, protein: str,
             value: str = VALUE) -> dict | None:
    """Cavity against the rest, within whatever `mask` selects."""
    import statsmodels.formula.api as smf
    d = t[mask].dropna(subset=[value]).rename(columns={value: "yv"})
    a = d.loc[d.is_cavity > 0.5, "yv"]
    c = d.loc[d.is_cavity < 0.5, "yv"]
    if len(a) < MIN_GROUP or len(c) < MIN_GROUP or d.block.nunique() < 4:
        return None
    m = smf.ols("yv ~ is_cavity", data=d[["yv", "is_cavity", "block"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d.block})
    naive = smf.ols("yv ~ is_cavity", data=d).fit()
    ci = m.conf_int().loc["is_cavity"]
    delta = cliffs_delta(a, c)
    lo, hi = block_bootstrap_delta(t, mask, value=value)
    shape = distribution_tests(a, c)
    return {"protein": protein, "comparison": label,
            "n_cavity": int(len(a)), "n_other": int(len(c)),
            "n_blocks": int(d.block.nunique()),
            "median_cavity": float(a.median()), "median_other": float(c.median()),
            "difference": float(m.params["is_cavity"]),
            "se_block_clustered": float(m.bse["is_cavity"]),
            "lo": float(ci[0]), "hi": float(ci[1]),
            "p_block_clustered": float(m.pvalues["is_cavity"]),
            "se_naive": float(naive.bse["is_cavity"]),
            "p_naive_ols": float(naive.pvalues["is_cavity"]),
            "p_within_block_perm": within_block_permutation(d, "yv", seed=4),
            "cliffs_delta": delta, "delta_lo": lo, "delta_hi": hi, **shape}


def radial_trend(t: pd.DataFrame, protein: str, value: str = VALUE) -> dict | None:
    """The same question with no threshold: does the residual change with radius?

    If the cavity result is real it should appear as a gradient along the radial
    coordinate inside the membrane, not only as a step at the median split the class
    definition happens to use.
    """
    import statsmodels.formula.api as smf
    d = t[(t.is_tm_f > 0.5)].dropna(subset=[value, "radial"]).rename(
        columns={value: "yv"})
    if len(d) < 30 or d.block.nunique() < 4:
        return None
    m = smf.ols("yv ~ radial", data=d[["yv", "radial", "block"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d.block})
    ci = m.conf_int().loc["radial"]
    rho, pv = ss.spearmanr(d.radial, d.yv)
    return {"protein": protein, "n": int(len(d)), "n_blocks": int(d.block.nunique()),
            "slope_per_A": float(m.params["radial"]),
            "se_block_clustered": float(m.bse["radial"]),
            "lo": float(ci[0]), "hi": float(ci[1]),
            "p_block_clustered": float(m.pvalues["radial"]),
            "spearman": float(rho), "p_spearman": float(pv)}


# ------------------------------------------------------------------- figures
def _half_violin(ax, x0: float, v: np.ndarray, color: str, width: float = 0.34):
    """One violin, left half only, so the right half can carry the raw points."""
    if len(v) < 3 or not np.isfinite(v).all():
        v = np.asarray(v, float)[np.isfinite(v)]
    if len(v) < 3:
        return
    part = ax.violinplot([v], positions=[x0], widths=width * 2, showextrema=False)
    for body in part["bodies"]:
        pts = body.get_paths()[0].vertices
        pts[:, 0] = np.clip(pts[:, 0], -np.inf, x0)      # keep the left half
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor(P.INK)
        body.set_linewidth(0.5)


def _raincloud(ax, groups: dict, colors: dict, seed: int = 0) -> None:
    """Half violin, a narrow box, and the residues themselves, per class."""
    rng = np.random.default_rng(seed)
    for k, (name, v) in enumerate(groups.items()):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        c = colors[name]
        _half_violin(ax, k, v, c)
        ax.boxplot([v], positions=[k + 0.09], widths=0.075, showfliers=False,
                   patch_artist=True, medianprops=dict(lw=0),
                   whiskerprops=dict(color=P.INK, lw=0.6), capprops=dict(lw=0),
                   boxprops=dict(facecolor=P.INK, edgecolor=P.INK, lw=0.4))
        ax.plot([k + 0.09], [np.median(v)], marker="o", ms=2.6, mfc="white",
                mec="white", lw=0, zorder=6)
        ax.scatter(k + 0.265 + rng.uniform(-0.075, 0.075, len(v)), v, s=5.5,
                   facecolor=c, edgecolor=P.INK, linewidth=0.2, alpha=0.75, zorder=3)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xlim(-0.62, len(groups) - 0.42)


def _bracket(ax, i: int, j: int, y: float, h: float, text: str) -> None:
    ax.plot([i, i, j, j], [y, y + h, y + h, y], lw=0.8, color=P.INK, zorder=7,
            clip_on=False)
    ax.text((i + j) / 2, y + h, text, ha="center", va="bottom", fontsize=6.4,
            zorder=7, clip_on=False)


def figure_classes(bits: dict, names: list[str], pw: pd.DataFrame,
                   outdir: Path) -> None:
    """The residual by residue class: the distributions, and the tests on them.

    Left, every scored residue in its class, as a distribution rather than a bar.
    Right, the same three distributions as cumulative curves, where the quantity the
    Kolmogorov-Smirnov test reports is a vertical distance you can point at.
    """
    rows = len(names)
    fig, axes = plt.subplots(rows, 2, figsize=(8.0, 3.35 * rows + 0.5),
                             layout="constrained", squeeze=False,
                             gridspec_kw={"width_ratios": [1.28, 1.0]})
    for i, g in enumerate(names):
        t, axv, axe = bits[g]["sites"], axes[i][0], axes[i][1]
        groups, cols, keys = {}, {}, []
        for c in CLASS_ORDER:
            v = t.loc[t.klass == c, VALUE].dropna()
            if len(v) < MIN_GROUP:
                continue
            lab = CLASS_LABEL[c]
            groups[lab], cols[lab] = v.to_numpy(), CLASS_COLOR[c]
            keys.append(c)
        if not groups:
            empty_panel(axv, 2 * i, g, "no class has enough scored sites")
            empty_panel(axe, 2 * i + 1, g, "")
            continue

        # ---- (left) the distributions themselves
        _raincloud(axv, groups, cols)
        axv.axhline(0.0, color=P.MUTED, lw=0.7, zorder=1)
        axv.set_xticklabels([f"{k}\nn = {len(v)}" for k, v in groups.items()],
                            fontsize=7)
        for k, v in enumerate(groups.values()):
            axv.text(k, -0.105, f"median {np.median(v):+.2f}",
                     transform=axv.get_xaxis_transform(), ha="center", va="top",
                     fontsize=6, color=P.MUTED, clip_on=False)
        axv.set_ylabel(VALUE_LABEL, fontsize=7.5)

        # ---- the tests, as brackets above the clouds
        q = pw[pw.protein == g]
        # a single dead position can be several times more negative than anything else,
        # and letting it set the axis flattens the three distributions into lines. The
        # view is cut to the central 98% and the residues left outside are counted in
        # words, never dropped from any test
        allv = np.concatenate([np.asarray(v, float) for v in groups.values()])
        lo, hi = (float(x) for x in np.percentile(allv, [1, 99]))
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        span = (hi - lo) or 1.0
        n_out = int((allv < lo).sum() + (allv > hi).sum())
        if n_out:
            axv.text(0.012, 0.012, f"{n_out} site{'s' if n_out > 1 else ''} outside "
                     "the axis, kept in every test", transform=axv.transAxes,
                     fontsize=6, ha="left", va="bottom", color=P.MUTED)
        level = 0
        for r in q.itertuples():
            if r.a not in keys or r.b not in keys:
                continue
            y = hi + 0.09 * span + level * 0.16 * span
            _bracket(axv, keys.index(r.a), keys.index(r.b), y, 0.035 * span,
                     f"δ = {r.cliffs_delta:+.2f}, "
                     f"{pstr(r.p_mannwhitney_holm)}")
            level += 1
        axv.set_ylim(lo - 0.16 * span, hi + (0.22 + 0.16 * max(level - 1, 0)) * span)

        # ---- (right) the same thing as cumulative curves
        for c in keys:
            v = np.sort(t.loc[t.klass == c, VALUE].dropna().to_numpy())
            axe.step(v, np.arange(1, len(v) + 1) / len(v), where="post",
                     color=CLASS_COLOR[c], lw=1.6,
                     label=f"{CLASS_LABEL[c]} ({len(v)})")
        axe.set_ylim(0, 1.02)
        axe.set_xlim(lo - 0.04 * span, hi + 0.04 * span)
        axe.set_ylabel("fraction of sites at or below", fontsize=7.5)
        axe.set_xlabel(VALUE_LABEL, fontsize=7.5)
        axe.legend(loc="upper left", fontsize=6.3, frameon=False)
        lines = []
        for r in q.itertuples():
            if r.a == "cavity":
                lines.append(f"vs {CLASS_LABEL[r.b].split(':')[0]}:  "
                             f"D = {r.ks_d:.2f}, {pstr(r.p_ks_holm)}")
        if lines:
            axe.text(0.985, 0.03, "Kolmogorov–Smirnov\n" + "\n".join(lines),
                     transform=axe.transAxes, fontsize=6.2, va="bottom", ha="right",
                     color=P.INK)
        panel(axv, 2 * i, g, f"{len(t)} scored sites", size=8)
        panel(axe, 2 * i + 1, g, "cumulative", size=8)
        P.no_square(axv, axe)
    fig.suptitle("Lower is more constrained than the abundance assay explains, in the "
                 "residual's own units. The claim is that the blue distribution sits "
                 "below the grey ones; δ is Cliff's delta and the p-values are "
                 "Holm-corrected across the three pairs.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "v01_residual_by_environment", None, "V01")
    plt.close(fig)


def figure_tests(tests: pd.DataFrame, outdir: Path) -> None:
    """The cavity-minus-rest difference, with the block-clustered interval and the
    naive one drawn behind it."""
    if not len(tests):
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(tests) + 1.9), layout="constrained")
    y = np.arange(len(tests))[::-1]
    for k, r in enumerate(tests.itertuples()):
        ax.errorbar(r.difference, y[k], xerr=1.96 * r.se_naive, fmt="none",
                    elinewidth=3.2, color="#D9D9D9", capsize=0, zorder=1)
        ax.errorbar(r.difference, y[k],
                    xerr=[[r.difference - r.lo], [r.hi - r.difference]], fmt="o", ms=6,
                    lw=0, elinewidth=1.2, capsize=3, color=P.DARK_RED, mec=P.INK,
                    mew=0.5, zorder=3)
        ax.text(r.difference, y[k] + 0.26,
                f"δ = {r.cliffs_delta:+.2f}, p = {r.p_block_clustered:.3g}",
                ha="center", fontsize=6.5)
    ax.axvline(0, color=P.INK, lw=0.9, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{short_name(r.protein)}\n{r.comparison}"
                        for r in tests.itertuples()], fontsize=7)
    ax.set_ylim(-0.6, len(tests) - 0.3)
    ax.set_xlabel("cavity \u2212 rest, in site median ESM-1v residual "
                  "(negative = cavity more constrained)", fontsize=8)
    fig.suptitle("Thin bar = the interval with errors clustered on 20 Å spatial blocks; "
                 "the thick grey bar behind it is the naive interval that treats every "
                 "residue as independent.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "v02_cavity_vs_rest", None, "V02")
    plt.close(fig)


def figure_radial(bits: dict, names: list[str], trends: pd.DataFrame,
                  outdir: Path) -> None:
    """The residual against distance from the bundle axis, with no threshold."""
    fig, pairs, _ = grid(names, w=3.1, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        t = bits[g]["sites"]
        d = t[t.is_tm_f > 0.5].dropna(subset=[VALUE, "radial"])
        if len(d) < 20:
            empty_panel(ax, i, g, "too few scored TM sites")
            continue
        ax.scatter(d.radial, d[VALUE], s=14,
                   c=[CLASS_COLOR.get(k, "#B0B8C1") for k in d.klass],
                   edgecolor=P.INK, linewidth=0.25, zorder=3)
        binned(ax, d.radial, d[VALUE],
               np.quantile(d.radial, np.linspace(0, 1, 7)), P.INK, "")
        cut = bits[g].get("radial_cut", np.nan)
        if np.isfinite(cut):
            ax.axvline(cut, color=P.INK, lw=1.0, ls=(0, (4, 3)))
            ax.text(cut, ax.get_ylim()[1], " cavity | lipid", fontsize=6, va="top")
        ax.axhline(0, color=P.MUTED, lw=0.7)
        r = trends[trends.protein == g]
        panel(ax, i, g, (f"slope {r.slope_per_A.iloc[0]:+.3f}/Å "
                         f"(p = {r.p_block_clustered.iloc[0]:.3g})") if len(r) else "",
              size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel(VALUE_LABEL, fontsize=7.5)
    fig.suptitle("TM sites only, coloured by class. A positive slope means the residual "
                 "rises away from the pathway, which is the cavity claim without a "
                 "threshold in it.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "v03_radial_trend", None, "V03")
    plt.close(fig)


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "Degrees of freedom|kmeans|One of the "
                                              "clusters is empty).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms cavity",
        description="Are cavity-lining residues more constrained than the rest of the "
                    "protein, after topology and spatial autocorrelation are accounted "
                    "for?")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bits, names = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        b = bits_for(cfg)
        if not b:
            print(f"  {cfg.id:<16} skipped: needs a structure, a topology and ESM-1v "
                  "scores", flush=True)
            continue
        bits[cfg.id], _ = b, names.append(cfg.id)
        t = b["sites"]
        counts = t.klass.value_counts()
        print(f"  {cfg.id:<16} {len(t):>4} scored sites  "
              + "  ".join(f"{k} {int(counts.get(k, 0))}" for k in CLASS_ORDER)
              + f"  ({b['n_from_core']} of {b['n_core']} buried helix-helix residues "
                f"fall on the cavity side)  {t.block.nunique()} blocks of "
                f"~{BLOCK_A:g} Å", flush=True)
    if not names:
        raise SystemExit("no protein could be tested")

    rows, trends, pairs = [], [], []
    for g in names:
        t = bits[g]["sites"]
        for label, mask in (("cavity vs the whole protein",
                             pd.Series(True, index=t.index)),
                            ("cavity vs the rest of the membrane domain",
                             t.is_tm_f > 0.5)):
            r = one_test(t, mask, label, g)
            if r:
                rows.append(r)
        tr = radial_trend(t, g)
        if tr:
            trends.append(tr)
        pairs.append(pairwise_tests(t, g))
    tests, tf = pd.DataFrame(rows), pd.DataFrame(trends)
    pw = pd.concat([x for x in pairs if len(x)], ignore_index=True) if any(
        len(x) for x in pairs) else pd.DataFrame(columns=["protein", "a", "b"])

    figure_classes(bits, names, pw, outdir)
    figure_tests(tests, outdir)
    figure_radial(bits, names, tf, outdir)
    if len(tests):
        tests.to_csv(outdir / "v02_cavity_vs_rest.csv", index=False)
    if len(tf):
        tf.to_csv(outdir / "v03_radial_trend.csv", index=False)
    if len(pw):
        pw.to_csv(outdir / "v01_pairwise_distributions.csv", index=False)
    pd.concat([bits[g]["sites"].assign(protein=g) for g in names],
              ignore_index=True).to_csv(outdir / "sites.csv", index=False)

    print()
    for r in tests.itertuples():
        print(f"  {short_name(r.protein):<10} {r.comparison:<42} "
              f"Δ = {r.difference:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  "
              f"p = {r.p_block_clustered:.3g}  (naive p = {r.p_naive_ols:.3g}, "
              f"within-block permutation p = {r.p_within_block_perm:.3g})  "
              f"\u03b4 = {r.cliffs_delta:+.2f}", flush=True)
    for r in tf.itertuples():
        print(f"  {short_name(r.protein):<10} {'radial slope (no threshold)':<42} "
              f"{r.slope_per_A:+.4f}/Å [{r.lo:+.4f}, {r.hi:+.4f}]  "
              f"p = {r.p_block_clustered:.3g}", flush=True)

    # "consistently" is the question, so say whether the proteins actually agree
    verdict = "not testable"
    key = tests[tests.comparison == "cavity vs the rest of the membrane domain"]
    if len(key) >= 2:
        same_way = bool((key.difference < 0).all() or (key.difference > 0).all())
        both_sig = bool((key.p_block_clustered < 0.05).all())
        verdict = ("consistent and significant in both" if same_way and both_sig else
                   "same direction in both, not significant in both" if same_way else
                   "the two proteins disagree in direction")
        print(f"\n  across proteins, within the membrane domain: {verdict}")
    print()
    for r in pw.itertuples():
        print(f"  {short_name(r.protein):<10} "
              f"{CLASS_LABEL[r.a] + ' vs ' + CLASS_LABEL[r.b]:<42} "
              f"shift = {r.shift:+.3f}  \u03b4 = {r.cliffs_delta:+.2f}  "
              f"MW {pstr(r.p_mannwhitney_holm)}  KS D = {r.ks_d:.2f} "
              f"{pstr(r.p_ks_holm)}  spread {pstr(r.p_brown_forsythe)}", flush=True)
    out = {"tests": tests.to_dict("records"), "radial": tf.to_dict("records"),
           "pairwise": pw.to_dict("records"), "verdict": verdict}
    (outdir / "cavity.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
