"""`python -m mpdms cavity configs/AQR1_fitness.yaml configs/HXT1.yaml`

Are the residues lining the central cavity held more tightly by evolution than the rest
of the protein, once the things that would fake that are taken away?

The quantity is the site median ESM-1v residual z, defined exactly as A15 defines it: how
much more constrained a position looks to a language model than its measured abundance
effect explains. A strongly negative residual is the signature this project calls
functional - conserved for a reason the abundance assay cannot see.

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
CLASS_ORDER = ["cavity-lining", "helix-helix core", "lipid-facing",
               "cytosolic loop", "lumenal loop"]
CLASS_COLOR = {"cavity-lining": "#B2182B", "helix-helix core": "#E8A33D",
               "lipid-facing": "#2166AC", "cytosolic loop": "#9AA4B2",
               "lumenal loop": "#C7CDD4"}


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
    t = res[["pos", "wt", "n_variants", "median_z", "median_abundance"]].merge(
        pos[keep], on="pos", how="inner").merge(
        b["rt"][["pos", "x", "y", "z"]], on="pos", how="inner")
    t = t.dropna(subset=["median_z", "environ"])
    if len(t) < 40:
        return {}
    t["block"] = spatial_blocks(t[["x", "y", "z"]].to_numpy(float))
    t["is_cavity"] = (t.environ == "cavity-lining").astype(float)
    t["is_tm_f"] = (t.is_tm > 0.5).astype(float) if "is_tm" in t else 0.0
    b["sites"] = t
    b["radial_cut"] = float(pos.attrs.get("radial_cut", np.nan))
    return b


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


def block_bootstrap_delta(t: pd.DataFrame, mask, n: int = N_BOOT,
                          seed: int = 3) -> tuple[float, float]:
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
        a = pick.loc[pick.is_cavity > 0.5, "median_z"]
        c = pick.loc[pick.is_cavity < 0.5, "median_z"]
        if len(a) >= 3 and len(c) >= 3:
            out.append(cliffs_delta(a, c))
    if len(out) < 50:
        return np.nan, np.nan
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def one_test(t: pd.DataFrame, mask, label: str, protein: str) -> dict | None:
    """Cavity against the rest, within whatever `mask` selects."""
    import statsmodels.formula.api as smf
    d = t[mask].dropna(subset=["median_z"])
    a = d.loc[d.is_cavity > 0.5, "median_z"]
    c = d.loc[d.is_cavity < 0.5, "median_z"]
    if len(a) < MIN_GROUP or len(c) < MIN_GROUP or d.block.nunique() < 4:
        return None
    m = smf.ols("median_z ~ is_cavity", data=d[["median_z", "is_cavity", "block"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d.block})
    naive = smf.ols("median_z ~ is_cavity", data=d).fit()
    ci = m.conf_int().loc["is_cavity"]
    delta = cliffs_delta(a, c)
    lo, hi = block_bootstrap_delta(t, mask)
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
            "p_mannwhitney": float(ss.mannwhitneyu(a, c, alternative="two-sided").pvalue),
            "cliffs_delta": delta, "delta_lo": lo, "delta_hi": hi}


def radial_trend(t: pd.DataFrame, protein: str) -> dict | None:
    """The same question with no threshold: does the residual change with radius?

    If the cavity result is real it should appear as a gradient along the radial
    coordinate inside the membrane, not only as a step at the median split the class
    definition happens to use.
    """
    import statsmodels.formula.api as smf
    d = t[(t.is_tm_f > 0.5)].dropna(subset=["median_z", "radial"])
    if len(d) < 30 or d.block.nunique() < 4:
        return None
    m = smf.ols("median_z ~ radial", data=d[["median_z", "radial", "block"]]).fit(
        cov_type="cluster", cov_kwds={"groups": d.block})
    ci = m.conf_int().loc["radial"]
    rho, pv = ss.spearmanr(d.radial, d.median_z)
    return {"protein": protein, "n": int(len(d)), "n_blocks": int(d.block.nunique()),
            "slope_per_A": float(m.params["radial"]),
            "se_block_clustered": float(m.bse["radial"]),
            "lo": float(ci[0]), "hi": float(ci[1]),
            "p_block_clustered": float(m.pvalues["radial"]),
            "spearman": float(rho), "p_spearman": float(pv)}


# ------------------------------------------------------------------- figures
def figure_classes(bits: dict, names: list[str], outdir: Path) -> None:
    """The residual by residue environment, cavity first."""
    fig, pairs, _ = grid(names, w=3.2, h=2.9, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        t = bits[g]["sites"]
        groups, cols = {}, {}
        for c in CLASS_ORDER:
            s = t.loc[t.environ == c, "median_z"].dropna()
            if len(s) < MIN_GROUP:
                continue
            short = c.split("-")[0] if c != "helix-helix core" else "core"
            short = {"cavity": "cavity", "core": "core", "lipid": "lipid",
                     "cytosolic": "cyto", "lumenal": "lumen"}.get(short, short)
            groups[short] = s.to_numpy()
            cols[short] = CLASS_COLOR[c]
        if not groups:
            empty_panel(ax, i, g, "no class has enough scored sites")
            continue
        P.violin_box(ax, groups, colors=cols)
        from .analyses.a15_esm_functional import Z_SITE
        ax.axhline(0, color=P.MUTED, lw=0.7)
        ax.axhline(Z_SITE, color=P.INK, lw=0.9, ls=(0, (3, 2)))
        ax.text(len(groups) - 0.45, Z_SITE, " functional threshold", fontsize=6,
                va="bottom", ha="right")
        ax.tick_params(axis="x", labelsize=7)
        panel(ax, i, g, f"{len(t)} scored sites", size=8)
        if i % 2 == 0:
            ax.set_ylabel("site median ESM-1v residual z", fontsize=7.5)
    fig.suptitle("Lower is more constrained than the abundance assay explains. The claim "
                 "is that the cavity violin sits below the others.", fontsize=7, x=0.01,
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
    ax.set_xlabel("cavity − rest, in site median residual z "
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
        d = t[t.is_tm_f > 0.5].dropna(subset=["median_z", "radial"])
        if len(d) < 20:
            empty_panel(ax, i, g, "too few scored TM sites")
            continue
        ax.scatter(d.radial, d.median_z, s=14,
                   c=[CLASS_COLOR.get(e, "#B0B8C1") for e in d.environ],
                   edgecolor=P.INK, linewidth=0.25, zorder=3)
        binned(ax, d.radial, d.median_z,
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
            ax.set_ylabel("site median ESM-1v residual z", fontsize=7.5)
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
        print(f"  {cfg.id:<16} {len(t):>4} scored sites, "
              f"{int(t.is_cavity.sum())} cavity-lining, "
              f"{t.block.nunique()} spatial blocks of ~{BLOCK_A:g} Å", flush=True)
    if not names:
        raise SystemExit("no protein could be tested")

    rows, trends = [], []
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
    tests, tf = pd.DataFrame(rows), pd.DataFrame(trends)

    figure_classes(bits, names, outdir)
    figure_tests(tests, outdir)
    figure_radial(bits, names, tf, outdir)
    if len(tests):
        tests.to_csv(outdir / "v02_cavity_vs_rest.csv", index=False)
    if len(tf):
        tf.to_csv(outdir / "v03_radial_trend.csv", index=False)
    pd.concat([bits[g]["sites"].assign(protein=g) for g in names],
              ignore_index=True).to_csv(outdir / "sites.csv", index=False)

    print()
    for r in tests.itertuples():
        print(f"  {short_name(r.protein):<10} {r.comparison:<42} "
              f"Δ = {r.difference:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  "
              f"p = {r.p_block_clustered:.3g}  (naive p = {r.p_naive_ols:.3g})  "
              f"δ = {r.cliffs_delta:+.2f}", flush=True)
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
    out = {"tests": tests.to_dict("records"), "radial": tf.to_dict("records"),
           "verdict": verdict}
    (outdir / "cavity.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
