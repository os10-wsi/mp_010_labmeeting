"""`python -m mpdms environ configs/AQR1_fitness.yaml configs/HXT1.yaml ...`

What a residue's surroundings say about how much a mutation there costs.

A transmembrane protein has several different kinds of "buried" and several different
kinds of "exposed", and solvent accessibility on its own cannot tell them apart. The
classification used here splits them with the membrane frame, which is what makes the
comparison mean something:

    helix-helix core   in a TM helix, low RSA: packed against other helices
    cavity-lining      in a TM helix, exposed, but close to the bundle axis: facing the
                       central translocation pathway, which in a transporter is water
    lipid-facing       in a TM helix, exposed, far from the axis: facing acyl chains
    cytosolic loop     outside the membrane, on the inside
    lumenal loop       outside the membrane, on the outside

RSA alone merges the middle two, and they are the two with opposite chemistry: a polar
residue is at home in the cavity and out of place against lipid. Separating them is the
point of the `radial` coordinate.

The analyses:

    e01  mean effect per position by environment, per protein and pooled
    e02  tolerance over the two membrane axes at once: burial against depth
    e03  change in hydrophobicity against effect, split by burial
    e04  the polar/hydrophobic direction, split by environment
    e05  effect against the number of structural contacts, with RSA partialled out
    e06  where the largest ESM-1v residuals sit in the structure, and whether they cluster

Nothing is pooled across proteins. Every figure is one panel per protein, in the order
the configs are given on the command line, and every number in a panel comes from that
protein alone: these assays differ in dynamic range, and a comparison between proteins
belongs in a model that says so (see `bayes` and `famstruct`) rather than in a box plot.
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
from .aggregate import forest
from .analyses.base import missense
from .config import REPO_ROOT, load_config
from .features import position_table
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_environ"
RSA_BURIED = 0.15     # below this, a TM residue is packed against other helices
MIN_PLDDT = 70.0      # AlphaFold confidence below which geometry is not worth classifying
MIN_VARIANTS = 5      # per position, before its mean is used
CLASSES = ["helix-helix core", "cavity-lining", "lipid-facing", "cytosolic loop",
           "lumenal loop"]
CLASS_COLORS = dict(zip(CLASSES, P.series_colors(len(CLASSES))))
# short forms for axis ticks; the full names stay in the legends, titles and tables
SHORT = {"helix-helix core": "core", "cavity-lining": "cavity", "lipid-facing": "lipid",
         "cytosolic loop": "cyto", "lumenal loop": "lumen"}


def sides(cfg) -> dict[int, str]:
    """Which side of the membrane each non-TM position sits on, from the topology."""
    out = {}
    for s in cfg.get_path("topology.segments", []) or []:
        if s.get("type") == "TM":
            continue
        side = (s.get("side") or "").lower()
        for p in range(s["start"], s["end"] + 1):
            out[p] = side
    return out


def classify(pt: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add `environ`, and the per-protein radial cut that separates cavity from lipid.

    The radial cut is the median over that protein's confident TM residues rather than a
    fixed number of angstroms, because the cut has to mean "inside this bundle" and
    bundles differ in girth. It is recorded in the table so a figure cannot imply a
    threshold the numbers did not use.
    """
    t = pt.copy()
    side = sides(cfg)
    conf = t.plddt.isna() | (t.plddt >= MIN_PLDDT)
    tm = (t.is_tm > 0.5) & conf
    cut = float(t.loc[tm, "radial"].median()) if tm.any() else np.nan
    env = pd.Series(pd.NA, index=t.index, dtype=object)
    env[tm & (t.rsa < RSA_BURIED)] = "helix-helix core"
    env[tm & (t.rsa >= RSA_BURIED) & (t.radial < cut)] = "cavity-lining"
    env[tm & (t.rsa >= RSA_BURIED) & (t.radial >= cut)] = "lipid-facing"
    out = (t.is_tm <= 0.5) & conf
    s = t.pos.map(side)
    env[out & s.isin(["cytosolic", "in", "inside"])] = "cytosolic loop"
    env[out & s.isin(["lumenal", "luminal", "out", "outside", "extracellular"])] = "lumenal loop"
    t["environ"] = env
    t.attrs["radial_cut"] = cut
    t.attrs["n_low_plddt"] = int((~conf).sum())
    return t


def protein_frame(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(variants, positions) for one protein, both carrying the environment class."""
    df = load_dataset(cfg)
    pt = classify(position_table(df, cfg), cfg)
    v = missense(df).dropna(subset=["score_z"])[["pos", "wt", "mut", "score_z"]]
    from .annot import add_deltas
    v = add_deltas(v)
    keep = ["pos", "environ", "rsa", "abs_z", "radial", "contacts", "plddt", "is_tm",
            "seg_type", "helix"]
    v = v.merge(pt[[c for c in keep if c in pt.columns]], on="pos", how="left")
    g = v.groupby("pos").score_z.agg(["mean", "std", "size"])
    g = g[g["size"] >= MIN_VARIANTS]
    pos = pt.merge(pd.DataFrame({"pos": g.index, "effect": g["mean"].to_numpy(),
                                 "se": (g["std"] / np.sqrt(g["size"])).to_numpy(),
                                 "n_variants": g["size"].to_numpy()}), on="pos", how="inner")
    v.attrs = dict(pt.attrs)
    pos.attrs = dict(pt.attrs)
    return v, pos


# ------------------------------------------------------------------- layout
def grid(names: list[str], w: float = 3.1, h: float = 2.9, cols: int = 2,
         extra: float = 0.0, polar: bool = False):
    """One panel per protein, in the order the proteins were given on the command line.

    The order is the caller's, never sorted: a four-protein figure is read as a 2x2 and
    which protein sits where is a decision about the figure, not about the alphabet.
    """
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols + 1.0, h * rows + 0.7 + extra),
                             layout="constrained", squeeze=False,
                             subplot_kw={"projection": "polar"} if polar else None)
    flat = [axes[i // cols][i % cols] for i in range(rows * cols)]
    for ax in flat[len(names):]:
        ax.set_visible(False)
    return fig, list(zip(names, flat)), axes


def panel(ax, i: int, name: str, extra: str = "", size: float = 8.5):
    ax.set_title(f"({chr(97 + i)}) {name.replace('_fitness', '')}"
                 + (f"  {extra}" if extra else ""), loc="left", fontsize=size)


def effect_color(v: float):
    """The project's fitness colours: dark red at -1, white at 0, light blue above."""
    return P.fitness_cmap()(P.fitness_norm()(v))


# ------------------------------------------------------------------ e01 environment
def e01_environment(pos: dict, names: list[str], outdir: Path) -> dict:
    """Mean effect per position, by what the position is surrounded by.

    Each violin is filled by its own median, on the same red-white-blue scale the
    heatmaps use, so the colour of a class carries the same meaning everywhere.
    """
    rows, contrasts = [], []
    fig, pairs, _ = grid(names)
    for i, (g, ax) in enumerate(pairs):
        d = pos[g]
        groups, cols = {}, {}
        for c in CLASSES:
            s = d.loc[d.environ == c, "effect"].dropna()
            if len(s) < 5:
                continue
            groups[SHORT[c]] = s.to_numpy()
            cols[SHORT[c]] = effect_color(float(s.median()))
            rows.append({"protein": g, "environ": c, "n": int(len(s)),
                         "estimate": float(s.mean()),
                         "se": float(s.std(ddof=1) / np.sqrt(len(s))),
                         "median": float(s.median())})
        if groups:
            P.violin_box(ax, groups, colors=cols)
        for a, b in (("helix-helix core", "lipid-facing"),
                     ("cavity-lining", "lipid-facing"),
                     ("helix-helix core", "cavity-lining")):
            x = d.loc[d.environ == a, "effect"]; y = d.loc[d.environ == b, "effect"]
            if len(x) < 5 or len(y) < 5:
                continue
            contrasts.append({"protein": g, "contrast": f"{a} - {b}",
                              "estimate": float(x.mean() - y.mean()),
                              "se": float(np.sqrt(x.var(ddof=1) / len(x)
                                                  + y.var(ddof=1) / len(y))),
                              "n_a": int(len(x)), "n_b": int(len(y)),
                              "p": float(ss.mannwhitneyu(x, y).pvalue)})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axhline(-1, color=P.INK, lw=0.6, ls=(0, (1, 2)))
        ax.tick_params(axis="x", labelsize=7)
        n_core = len(d[d.environ == "helix-helix core"])
        panel(ax, i, g, f"n = {len(d):,} positions, {n_core} core")
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position")
    fig.suptitle("core = helix-helix interface · cavity = lining the central pathway · "
                 "lipid = facing acyl chains · dotted line = the stop-codon level",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e01_environment", None, "E01")
    plt.close(fig)
    per, con = pd.DataFrame(rows), pd.DataFrame(contrasts)
    per.to_csv(outdir / "e01_environment_by_protein.csv", index=False)
    con.to_csv(outdir / "e01_environment_contrasts.csv", index=False)
    key = "helix-helix core - lipid-facing"
    head = "; ".join(f"{r.protein.replace('_fitness', '')} {r.estimate:+.3f} (p = {r.p:.1g})"
                     for r in con[con.contrast == key].itertuples())
    return {"status": "ok", "headline": f"core − lipid: {head}" if head else "",
            "by_protein": per.to_dict("records")}


# ------------------------------------------------------- e02 burial against depth
ZB = np.array([0, 4, 8, 12, 16, 25])
RB = np.array([0, 0.05, 0.15, 0.30, 0.50, 1.01])


def e02_burial_depth(pos: dict, names: list[str], outdir: Path) -> dict:
    """The two membrane axes at once: how buried, and how deep.

    A scatter of effect against RSA alone mixes the bilayer centre with the headgroup
    region, and one against depth alone mixes buried with exposed. Binning on both keeps
    them apart, and the question - whether burial costs more in the middle of the
    membrane than at its edge - is a property of the two together.
    """
    rows = []
    fig, pairs, axes = grid(names, w=3.0, h=2.7)
    im = None
    for i, (g, ax) in enumerate(pairs):
        t = pos[g].dropna(subset=["effect", "rsa", "abs_z"])
        m = np.full((len(ZB) - 1, len(RB) - 1), np.nan)
        for a in range(len(ZB) - 1):
            for b in range(len(RB) - 1):
                s = t[(t.abs_z >= ZB[a]) & (t.abs_z < ZB[a + 1])
                      & (t.rsa >= RB[b]) & (t.rsa < RB[b + 1])].effect
                if len(s) >= 4:
                    m[a, b] = s.mean()
                rows.append({"protein": g, "z_lo": ZB[a], "z_hi": ZB[a + 1],
                             "rsa_lo": RB[b], "rsa_hi": RB[b + 1], "n": int(len(s)),
                             "mean_effect": float(s.mean()) if len(s) else np.nan})
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=P.fitness_cmap(),
                       norm=P.fitness_norm(), extent=(0, len(RB) - 1, 0, len(ZB) - 1))
        for a in range(len(ZB) - 1):
            for b in range(len(RB) - 1):
                if np.isfinite(m[a, b]):
                    ax.text(b + 0.5, a + 0.5, f"{m[a, b]:.2f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if m[a, b] < -0.55 else P.INK)
        ax.set_xticks(np.arange(len(RB) - 1) + 0.5)
        ax.set_xticklabels([f"{RB[b]:.2f}–{RB[b+1]:.2f}" for b in range(len(RB) - 1)],
                           fontsize=5.5)
        ax.set_yticks(np.arange(len(ZB) - 1) + 0.5)
        ax.set_yticklabels([f"{ZB[a]}–{ZB[a+1]}" for a in range(len(ZB) - 1)], fontsize=6)
        panel(ax, i, g)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("RSA", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("depth |z| from the bilayer centre (Å)", fontsize=7.5)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    P.save(fig, outdir, "e02a_burial_depth", None, "E02")
    plt.close(fig)

    slopes = []
    fig, pairs, _ = grid(names, w=2.9, h=2.6)
    for i, (g, ax) in enumerate(pairs):
        t = pos[g].dropna(subset=["effect", "rsa", "abs_z"])
        for lab, sub, col in (("|z| < 10 Å (bilayer centre)", t[t.abs_z < 10], P.DARK_RED),
                              ("|z| ≥ 10 Å (headgroups, loops)", t[t.abs_z >= 10],
                               P.LIGHT_BLUE)):
            if len(sub) < 20:
                continue
            b = pd.cut(sub.rsa, RB)
            mm = sub.groupby(b, observed=True).agg(x=("rsa", "mean"), y=("effect", "mean"),
                                                   e=("effect", "sem")).dropna()
            ax.errorbar(mm.x, mm.y, yerr=mm.e, fmt="-o", ms=4, lw=1.3, color=col,
                        capsize=2, label=lab, mec=P.INK, mew=0.4)
            sl = ss.linregress(sub.rsa, sub.effect)
            slopes.append({"protein": g, "band": lab, "estimate": float(sl.slope),
                           "se": float(sl.stderr), "p": float(sl.pvalue),
                           "n": int(len(sub))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        d = [r for r in slopes if r["protein"] == g]
        panel(ax, i, g, "  ".join(f"slope {r['estimate']:+.2f}" for r in d), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("RSA", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    P.save(fig, outdir, "e02b_burial_slope_by_depth", None, "E02")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "e02_burial_depth_cells.csv", index=False)
    sl = pd.DataFrame(slopes)
    sl.to_csv(outdir / "e02_burial_slopes.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} {r.band.split('(')[0].strip()} "
                     f"{r.estimate:+.2f}" for r in sl.itertuples())
    return {"status": "ok", "headline": head, "slopes": sl.to_dict("records")}


# -------------------------------------------- e03 hydrophobicity against burial
BANDS = [("buried (RSA < 0.15)", lambda t: t.rsa < 0.15, "#B2182B"),
         ("intermediate", lambda t: (t.rsa >= 0.15) & (t.rsa < 0.35), P.MUTED),
         ("exposed (RSA ≥ 0.35)", lambda t: t.rsa >= 0.35, "#2166AC")]


def e03_hydrophobicity(var: dict, names: list[str], outdir: Path) -> dict:
    """Change in hydrophobicity against effect, in three bands of burial.

    `delta_kd` is mutant minus wild-type on the Kyte-Doolittle scale, so positive means
    the substitution is more hydrophobic. A buried position should care about the size of
    the change whichever way it goes, which is a curve rather than a slope, so the
    quadratic term is fitted alongside and reported.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.0, h=2.7)
    for i, (g, ax) in enumerate(pairs):
        t = var[g].dropna(subset=["score_z", "delta_kd", "rsa"])
        for lab, f, col in BANDS:
            sub = t[f(t)]
            if len(sub) < 50:
                continue
            b = pd.cut(sub.delta_kd, np.linspace(-9, 9, 10))
            m = sub.groupby(b, observed=True).agg(x=("delta_kd", "mean"),
                                                  y=("score_z", "mean"),
                                                  e=("score_z", "sem")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=3.4, lw=1.1, color=col,
                        capsize=1.5, label=lab, mec=P.INK, mew=0.3)
            sl = ss.linregress(sub.delta_kd, sub.score_z)
            q = np.polyfit(sub.delta_kd, sub.score_z, 2)
            rows.append({"protein": g, "band": lab, "estimate": float(sl.slope),
                         "se": float(sl.stderr), "p": float(sl.pvalue),
                         "curvature": float(q[0]), "n": int(len(sub))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.INK, lw=0.6, ls=(0, (1, 2)))
        panel(ax, i, g)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Δ hydrophobicity (Kyte-Doolittle, mutant − WT)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    P.save(fig, outdir, "e03_hydrophobicity_burial", None, "E03")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e03_hydrophobicity_slopes.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} {r.band.split(' ')[0]} "
                     f"{r.estimate:+.3f}/KD unit, curvature {r.curvature:+.4f}"
                     for r in per.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------- e04 the polar/hydrophobic direction by environment
# G and P are left out of both on purpose: they change the backbone, so a substitution
# involving one is not a statement about polarity.
POLAR = set("STNQHKRDEY")
HYDRO = set("AVLIFMCW")
DIRECTIONS = ["polar → hydrophobic", "hydrophobic → polar", "hydrophobic → hydrophobic",
              "polar → polar"]


def direction(wt: pd.Series, mut: pd.Series) -> pd.Series:
    wh, mh = wt.isin(list(HYDRO)), mut.isin(list(HYDRO))
    wp, mp = wt.isin(list(POLAR)), mut.isin(list(POLAR))
    out = pd.Series(pd.NA, index=wt.index, dtype=object)
    out[wp & mh] = "polar → hydrophobic"
    out[wh & mp] = "hydrophobic → polar"
    out[wh & mh] = "hydrophobic → hydrophobic"
    out[wp & mp] = "polar → polar"
    return out


def e04_direction(var: dict, names: list[str], outdir: Path) -> dict:
    """Which environments tolerate which direction of change, one protein at a time.

    Two numbers per cell. The colour is the absolute mean effect, so a cell can be read
    against every other figure. The number printed under it is the difference from the
    average substitution at the same kind of position, which is the one that is about
    chemistry: a class that is intolerant of everything is intolerant of each direction
    in turn, and that says nothing.
    """
    rows = []
    for g in names:
        t = var[g].dropna(subset=["score_z", "environ"]).copy()
        t["direction"] = direction(t.wt, t.mut)
        for c in CLASSES:
            base = t[t.environ == c]
            if len(base) < 50:
                continue
            mu = float(base.score_z.mean())
            for dn in DIRECTIONS:
                s = base.loc[base.direction == dn, "score_z"]
                if len(s) < 15:
                    continue
                rows.append({"protein": g, "environ": c, "direction": dn,
                             "n": int(len(s)), "mean": float(s.mean()),
                             "estimate": float(s.mean() - mu),
                             "se": float(s.std(ddof=1) / np.sqrt(len(s)))})
    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "too few variants per environment"}
    fig, pairs, axes = grid(names, w=3.4, h=2.7, extra=0.3)
    im = None
    for i, (g, ax) in enumerate(pairs):
        d = per[per.protein == g]
        classes = [c for c in CLASSES if c in set(d.environ)]
        mean = d.pivot_table(index="environ", columns="direction", values="mean").reindex(
            classes)[[c for c in DIRECTIONS if c in set(d.direction)]]
        rel = d.pivot_table(index="environ", columns="direction",
                            values="estimate").reindex(mean.index)[mean.columns]
        im = ax.imshow(mean.to_numpy(), cmap=P.fitness_cmap(), norm=P.fitness_norm(),
                       aspect="auto")
        for a in range(mean.shape[0]):
            for b in range(mean.shape[1]):
                v, r = mean.iat[a, b], rel.iat[a, b]
                if not np.isfinite(v):
                    continue
                col = "white" if v < -0.55 else P.INK
                ax.text(b, a - 0.13, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=col)
                ax.text(b, a + 0.22, f"({r:+.2f})", ha="center", va="center", fontsize=5.5,
                        color=col)
        ax.set_xticks(range(mean.shape[1]))
        ax.set_xticklabels([c.replace(" → ", "\n→ ") for c in mean.columns], fontsize=6)
        ax.set_yticks(range(len(mean)))
        ax.set_yticklabels([SHORT[c] for c in mean.index], fontsize=7.5)
        panel(ax, i, g)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    fig.suptitle("Top number: the mean effect of that kind of substitution there. In "
                 "brackets: the same, minus the average substitution at that kind of "
                 "position.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e04_substitution_direction", None, "E04")
    plt.close(fig)
    per.to_csv(outdir / "e04_direction_by_protein.csv", index=False)
    d = per[(per.direction == "polar → hydrophobic")
            & (per.environ.isin(["lipid-facing", "helix-helix core"]))]
    head = "; ".join(f"{r.protein.replace('_fitness', '')} {SHORT[r.environ]} "
                     f"{r.estimate:+.3f}" for r in d.itertuples())
    return {"status": "ok",
            "headline": (f"polar→hydrophobic, relative: {head}" if head else
                         "no protein has enough variants in a TM class to compare"),
            "by_protein": per.to_dict("records")}


# --------------------------------------------------------------- e05 contacts
def e05_contacts(pos: dict, names: list[str], outdir: Path) -> dict:
    """Effect against the number of structural contacts, with RSA partialled out.

    Contacts and solvent accessibility are two views of the same thing and correlate
    strongly, so the contact result is only news if it survives holding RSA fixed. Both
    are reported: the raw Spearman, and the partial correlation given RSA.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.0, h=2.7)
    for i, (g, ax) in enumerate(pairs):
        t = pos[g].dropna(subset=["effect", "contacts"])
        if len(t) < 20:
            continue
        ax.scatter(t.contacts, t.effect, s=9, lw=0, alpha=0.75,
                   c=[effect_color(v) for v in t.effect], edgecolor=P.INK)
        b = pd.cut(t.contacts, 8)
        m = t.groupby(b, observed=True).agg(x=("contacts", "mean"), y=("effect", "mean"),
                                            e=("effect", "sem")).dropna()
        ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=4, lw=1.5, color=P.INK, capsize=2,
                    zorder=5)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        rho = float(ss.spearmanr(t.contacts, t.effect)[0])
        part, n_part = np.nan, 0
        u = t.dropna(subset=["rsa"])
        if len(u) > 20:
            # partial Spearman: rank everything, then correlate the two sets of residuals
            r = u[["contacts", "effect", "rsa"]].rank()
            ec = r.contacts - np.polyval(np.polyfit(r.rsa, r.contacts, 1), r.rsa)
            ee = r.effect - np.polyval(np.polyfit(r.rsa, r.effect, 1), r.rsa)
            part = float(ss.pearsonr(ec, ee)[0]); n_part = len(u)
        sl = ss.linregress(t.contacts, t.effect)
        rows.append({"protein": g, "estimate": float(sl.slope), "se": float(sl.stderr),
                     "p": float(sl.pvalue), "spearman": rho,
                     "partial_spearman_given_rsa": part, "n": int(len(t)),
                     "n_partial": n_part})
        panel(ax, i, g, f"ρ = {rho:+.2f}, partial ρ | RSA = {part:+.2f}", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("Contacts (heavy atoms < 8 Å, |i−j| > 4)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect at the position", fontsize=7.5)
    P.save(fig, outdir, "e05_contacts", None, "E05")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e05_contacts.csv", index=False)
    head = "; ".join(f"{r.protein.replace('_fitness', '')} ρ {r.spearman:+.2f} "
                     f"(partial {r.partial_spearman_given_rsa:+.2f})"
                     for r in per.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# ------------------------------------------------ e06 where the ESM outliers sit
TOP_N = 10
NEAR = 10.0       # Å between CA atoms, counted as an edge of the network
N_PERM = 20000


def esm_residuals(cfg) -> pd.DataFrame | None:
    """Per-position ESM-1v residual, defined exactly as A15 defines it.

    The same LOESS of ESM against abundance, the same MAD scaling, the same minimum
    number of variants, so a position that A15 calls functional is the same position
    here and the two figures cannot disagree.
    """
    from .analyses.a15_esm_functional import MIN_VARIANTS as A15_MIN
    from .analyses.a15_esm_functional import loess_fit, read_esm
    p = cfg.resolve(cfg.get_path("evolution.esm_scores"))
    if p is None or not p.exists():
        return None
    e = read_esm(p)
    if e is None:
        return None
    d = missense(load_dataset(cfg))[["pos", "wt", "mut", "score_z"]].merge(
        e, on=["pos", "mut"], how="inner", suffixes=("", "_esm"))
    if "wt_esm" in d:
        d = d[d.wt == d.wt_esm]
    d = d.dropna(subset=["score_z", "esm1v"])
    if len(d) < 100:
        return None
    x = d.score_z.to_numpy()
    d["residual"] = d.esm1v.to_numpy() - loess_fit(x, d.esm1v.to_numpy())(x)
    scale = 1.4826 * ss.median_abs_deviation(d.residual)
    d["z"] = d.residual / (scale or 1.0)
    g = d.groupby("pos").agg(wt=("wt", "first"), n_variants=("pos", "size"),
                             median_residual=("residual", "median"),
                             median_z=("z", "median"),
                             median_abundance=("score_z", "median")).reset_index()
    g = g[g.n_variants >= A15_MIN]
    # the one constant that separates the two columns, so a caller working in residual
    # units can still place a threshold that was defined in z
    g.attrs["residual_scale"] = float(scale or 1.0)
    return g
def network_stats(xyz: np.ndarray, pick: np.ndarray, pool: np.ndarray,
                  strata: np.ndarray | None = None, seed: int = 6) -> dict:
    """Are these positions closer together than the same number drawn at random?

    Mean pairwise CA distance, against a null drawn from the positions that could have
    been chosen - the ones with enough variants to be scored at all. Drawing from the
    whole protein instead would reward any set that happens to sit in the measured region.

    `strata` adds the null that matters more. Transmembrane residues are packed into a
    bundle and loop residues are strung out around it, so ten TM residues are close
    together whatever else is true of them. Drawing the same number from each stratum as
    the observed set has removes that, and leaves only clustering the topology does not
    already explain.
    """
    def mean_d(idx):
        c = xyz[idx]
        dd = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        return float(dd[np.triu_indices(len(idx), 1)].mean()), dd

    obs, dmat = mean_d(pick)
    rng = np.random.default_rng(seed)
    k = len(pick)
    null = np.array([mean_d(rng.choice(pool, k, replace=False))[0] for _ in range(N_PERM)])
    out = {"observed_mean_distance": obs, "null_mean": float(null.mean()),
           "null_sd": float(null.std(ddof=1)),
           "p_clustered": float(((null <= obs).sum() + 1) / (N_PERM + 1)),
           "z": float((obs - null.mean()) / (null.std(ddof=1) or np.nan)),
           "n_pairs_within": int((dmat[np.triu_indices(k, 1)] < NEAR).sum()),
           "n_pairs": int(k * (k - 1) // 2), "null": null, "dmat": dmat}
    if strata is not None:
        want = pd.Series(strata[pick]).value_counts()
        by = {s: pool[strata[pool] == s] for s in want.index}
        # at least twice as many candidates as the stratum needs, or the "random" draw
        # keeps returning nearly the same residues and the null has no spread to speak of
        if all(len(by[s]) >= 2 * n for s, n in want.items()):
            m = np.array([mean_d(np.concatenate([rng.choice(by[s], n, replace=False)
                                                 for s, n in want.items()]))[0]
                          for _ in range(N_PERM)])
            out.update(null_matched_mean=float(m.mean()),
                       p_clustered_matched=float(((m <= obs).sum() + 1) / (N_PERM + 1)),
                       strata=", ".join(f"{s}:{n}" for s, n in want.items()))
    return out


def e06_esm_network(cfgs: dict, pos: dict, names: list[str], outdir: Path) -> dict:
    """The ten largest ESM-1v residuals, placed in the structure.

    A residual is a position where ESM-1v expects more constraint than abundance shows.
    Scattered residuals are noise, or positions whose function the abundance assay cannot
    see one at a time. A cluster is a different claim: residues that sit together, which
    is what a binding site or a gate looks like.
    """
    from .structure import membrane_frame, residue_table
    rows, per, panels = [], {}, []
    for g in names:
        cfg = cfgs[g]
        res = esm_residuals(cfg)
        sp = cfg.resolve(cfg.get_path("structure.path"))
        if res is None or sp is None or not sp.exists():
            continue
        rt = residue_table(sp, cfg.get_path("structure.chain", "A"),
                           int(cfg.get_path("structure.numbering_offset", 0) or 0))
        rt, _ = membrane_frame(rt, cfg)
        t = res.merge(rt[["pos", "x", "y", "z", "zdepth", "radial", "plddt"]], on="pos",
                      how="inner").merge(pos[g][["pos", "seg_type", "environ"]], on="pos",
                                         how="left").reset_index(drop=True)
        if len(t) < 3 * TOP_N:
            continue
        xyz = t[["x", "y", "z"]].to_numpy(float)
        pick = t.median_z.to_numpy().argsort()[:TOP_N]   # most negative = most constrained
        st = network_stats(xyz, pick, np.arange(len(t)),
                           strata=t.seg_type.fillna("unannotated").to_numpy())
        rows.append(t.iloc[pick].assign(protein=g, rank=np.arange(1, TOP_N + 1)))
        per[g] = {k: v for k, v in st.items() if k not in ("null", "dmat")}
        panels.append((g, t, pick, st))
        write_network_structure(cfg, rows[-1], st, outdir)
    if not panels:
        return {"status": "skipped", "reason": "no protein has both ESM scores and a structure"}

    have = [g for g, _, _, _ in panels]
    fig, pairs, _ = grid(have, w=3.2, h=3.0, extra=0.3)
    for i, ((g, t, pick, st), (_, ax)) in enumerate(zip(panels, pairs)):
        ax.scatter(t.radial, t.zdepth, s=7, color="#D9D9D9", lw=0)
        p10 = t.iloc[pick]
        c = p10[["x", "y", "z"]].to_numpy(float)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        for a in range(TOP_N):
            for b in range(a + 1, TOP_N):
                if d[a, b] < NEAR:
                    ax.plot([p10.radial.iloc[a], p10.radial.iloc[b]],
                            [p10.zdepth.iloc[a], p10.zdepth.iloc[b]],
                            color=P.DARK_RED, lw=1.0, alpha=0.75, zorder=3)
        ax.scatter(p10.radial, p10.zdepth, s=38, color=P.DARK_RED, zorder=5,
                   edgecolor=P.INK, linewidth=0.5)
        P.label_points(ax, p10.radial.to_numpy(), p10.zdepth.to_numpy(),
                       [f"{r.wt}{int(r.pos)}" for r in p10.itertuples()], fontsize=6)
        ax.axhline(0, color=P.INK, lw=0.6, ls=(0, (3, 3)))
        panel(ax, i, g, f"{st['n_pairs_within']}/{st['n_pairs']} pairs < {NEAR:.0f} Å",
              size=8)
        if i // 2 == (len(have) - 1) // 2:
            ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("depth z across the membrane (Å)", fontsize=7.5)
    fig.suptitle("Dark red = the ten positions whose ESM-1v residual is most negative; "
                 "a line joins every pair of them closer than 10 Å.",
                 fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e06a_esm_residual_network", None, "E06")
    plt.close(fig)

    fig, pairs, _ = grid(have, w=2.9, h=2.8)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    for i, ((g, t, pick, st), (_, ax)) in enumerate(zip(panels, pairs)):
        lab = [f"{r.wt}{int(r.pos)}" for r in t.iloc[pick].itertuples()]
        # ordered by how the distances cluster, so a sub-network reads as a dark block
        # rather than being scattered over the matrix by the order the sites were ranked
        o = leaves_list(linkage(squareform(st["dmat"], checks=False), "average"))
        im = ax.imshow(st["dmat"][np.ix_(o, o)], cmap=P.sequential_cmap(), vmin=0, vmax=40)
        ax.set_xticks(range(TOP_N))
        ax.set_xticklabels([lab[k] for k in o], rotation=90, fontsize=6)
        ax.set_yticks(range(TOP_N)); ax.set_yticklabels([lab[k] for k in o], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("CA distance (Å)", fontsize=7,
                                                          fontweight="bold")
        panel(ax, i, g)
    P.save(fig, outdir, "e06b_residual_distances", None, "E06")
    plt.close(fig)

    fig, pairs, _ = grid(have, w=2.9, h=2.4, extra=0.3)
    for i, ((g, t, pick, st), (_, ax)) in enumerate(zip(panels, pairs)):
        ax.hist(st["null"], bins=40, color="#D9D9D9", lw=0)
        ax.axvline(st["observed_mean_distance"], color=P.DARK_RED, lw=1.8)
        if "null_matched_mean" in st:
            ax.axvline(st["null_matched_mean"], color=P.INK, lw=1.0, ls=(0, (4, 3)))
        panel(ax, i, g, f"p = {st['p_clustered']:.3f}"
              + (f", matched p = {st['p_clustered_matched']:.3f}"
                 if "p_clustered_matched" in st else ""), size=8)
        if i // 2 == (len(have) - 1) // 2:
            ax.set_xlabel("mean pairwise distance of 10 positions (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("random draws", fontsize=7.5)
    fig.suptitle("Grey = ten positions drawn at random from those with enough variants to "
                 "be scored. Dashed = the mean of the null matched on TM/loop composition, "
                 "which is the comparison to believe.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e06c_clustering_null", None, "E06")
    plt.close(fig)

    pd.concat(rows, ignore_index=True).to_csv(outdir / "e06_top_residual_positions.csv",
                                              index=False)
    pd.DataFrame(per).T.rename_axis("protein").to_csv(outdir / "e06_clustering.csv")
    head = "; ".join(f"{g.replace('_fitness', '')}: {v['observed_mean_distance']:.1f} Å vs "
                     f"null {v['null_mean']:.1f} Å, p = {v['p_clustered']:.3f}"
                     + (f" (matched {v['p_clustered_matched']:.3f})"
                        if "p_clustered_matched" in v else "")
                     for g, v in per.items())
    return {"status": "ok", "per_protein": per, "headline": head}


def write_network_structure(cfg, top: pd.DataFrame, st: dict, outdir: Path) -> list:
    """A PyMOL script and a ChimeraX script that show the ten positions and their network.

    Written rather than rendered: the rendering needs PyMOL, and the script is the thing
    worth keeping anyway, since it can be opened next to the real structure and turned.
    Edges are real distance objects, so the numbers in the figure can be read off the
    model.
    """
    from .analyses.a15_esm_functional import _ranges
    src = cfg.resolve(cfg.get_path("structure.path"))
    if src is None or not src.exists():
        return []
    sdir = Path(outdir) / "structures" / cfg.id
    sdir.mkdir(parents=True, exist_ok=True)
    off = int(cfg.get_path("structure.numbering_offset", 0) or 0)
    resi = sorted(int(p_) - off for p_ in top.pos)
    sel = _ranges(resi)
    obj = "e06"
    pml = [f"# run with:  cd {sdir}   then  @e06_network.pml   (not 'run')",
           f"load {src.resolve()}, {obj}", "bg_color white", "hide everything",
           f"show cartoon, {obj}", "set cartoon_transparency, 0.55",
           "color grey80", f"show spheres, {obj} and resi {sel} and name CA",
           f"color red, {obj} and resi {sel}",
           f"show sticks, {obj} and resi {sel} and not name N+C+O",
           "set sphere_scale, 0.45"]
    c = top[["x", "y", "z"]].to_numpy(float)
    k = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if np.linalg.norm(c[i] - c[j]) < NEAR:
                k += 1
                a = int(top.pos.iloc[i]) - off
                b = int(top.pos.iloc[j]) - off
                pml.append(f"distance edge{k}, {obj}//A/{a}/CA, {obj}//A/{b}/CA")
    pml += ["hide labels", "set dash_color, red", "set dash_width, 2.5",
            f"orient {obj} and resi {sel}", "zoom all, 4",
            "set ray_opaque_background, 0", "save e06_network.pse",
            "png e06_network.png, width=2400, height=1800, dpi=300, ray=1"]
    cxc = [f"open {src.resolve()}", "set bgColor white", "hide atoms", "show cartoons",
           "color #1 #cccccc", "transparency 50 cartoons",
           f"show #1:{_ranges(resi, ',')} atoms", f"color #1:{_ranges(resi, ',')} red",
           f"style #1:{_ranges(resi, ',')} ball", "lighting soft",
           "graphics silhouettes true", f"view #1:{_ranges(resi, ',')}",
           "save e06_network_chimerax.png width 2400 height 1800 supersample 3"]
    (sdir / "e06_network.pml").write_text("\n".join(pml) + "\n")
    (sdir / "e06_network.cxc").write_text("\n".join(cxc) + "\n")
    (sdir / "e06_positions.csv").write_text(top.to_csv(index=False))
    return [sdir / "e06_network.pml", sdir / "e06_network.cxc"]


# ----------------------------------------------- e07 is one face of the helix buried?
def helix_moment(effects: np.ndarray, period: float = 3.6) -> float:
    """How one-sided a helix's intolerance is, as a fraction between 0 and 1.

    An alpha helix turns 100 degrees per residue, so a helix with one face against other
    helices and one against lipid has its intolerance repeating every 3.6 residues.
    Placing each residue's effect at its own angle and adding the vectors gives the
    classic moment: 1 if every intolerant residue is on one side, 0 if they are spread
    evenly around. Effects are centred first, so a uniformly intolerant helix - which has
    no face at all - scores 0 rather than 1.
    """
    e = np.asarray(effects, float)
    e = e - e.mean()
    ang = np.deg2rad(360.0 / period) * np.arange(len(e))
    v = np.abs(np.sum(e * np.exp(1j * ang)))
    return float(v / np.sum(np.abs(e))) if np.sum(np.abs(e)) > 0 else np.nan


def e07_helix_face(pos: dict, names: list[str], outdir: Path, n_perm: int = 2000) -> dict:
    """Per TM helix: is the intolerance on one face, and is that more than chance?

    The null shuffles the effects within the same helix, which keeps its mean, its
    spread and its length and destroys only the arrangement - so the p-value is about
    the arrangement and nothing else.
    """
    from .stats import bh_fdr
    rows = []
    rng = np.random.default_rng(7)
    for g in names:
        d = pos[g]
        for h, sub in d[(d.is_tm > 0.5) & d.helix.notna()].groupby("helix"):
            s = sub.sort_values("pos")
            e = s.effect.dropna().to_numpy()
            if len(e) < 8:
                continue
            obs = helix_moment(e)
            null = np.array([helix_moment(rng.permutation(e)) for _ in range(n_perm)])
            rows.append({"protein": g, "helix": h, "n": int(len(e)),
                         "moment": obs, "null_mean": float(null.mean()),
                         "p": float(((null >= obs).sum() + 1) / (n_perm + 1)),
                         "mean_effect": float(np.mean(e)),
                         "start": int(s.pos.min())})
    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "no TM helix has 8 scored positions"}
    per["q"] = bh_fdr(per.p)
    fig, pairs, _ = grid(names, w=3.2, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        d = per[per.protein == g].sort_values("start")
        if not len(d):
            continue
        x = np.arange(len(d))
        ax.bar(x, d.moment, color=[effect_color(v) for v in d.mean_effect],
               edgecolor=P.INK, linewidth=0.5)
        ax.plot(x, d.null_mean, "_", ms=11, color=P.INK, mew=1.2)
        for k, r in enumerate(d.itertuples()):
            if r.q < 0.05:
                ax.text(k, r.moment + 0.015, "*", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(d.helix, rotation=90, fontsize=6)
        ax.set_ylim(0, max(0.5, float(d.moment.max()) * 1.95))   # headroom for the wheel
        panel(ax, i, g, f"{int((d.q < 0.05).sum())}/{len(d)} helices one-sided", size=8)
        if i % 2 == 0:
            ax.set_ylabel("one-sidedness of intolerance\n(3.6-residue moment)", fontsize=7)
        # the most one-sided helix, drawn as the wheel it is
        top = d.loc[d.moment.idxmax()]
        s = pos[g].query("helix == @top.helix").sort_values("pos").dropna(subset=["effect"])
        wa = ax.inset_axes([0.66, 0.62, 0.32, 0.36], projection="polar")
        ang = np.deg2rad(100.0) * np.arange(len(s))
        e = s.effect.to_numpy()
        wa.scatter(ang, np.ones(len(s)), s=22, c=[effect_color(v) for v in e],
                   edgecolor=P.INK, linewidth=0.4, zorder=4)
        # the arrow points at the intolerant face: the resultant of the centred effects,
        # negated, so it points where mutations cost the most
        v = np.sum(-(e - e.mean()) * np.exp(1j * ang))
        wa.annotate("", xy=(np.angle(v), 0.82), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=P.INK, lw=1.0))
        wa.set_rticks([]); wa.set_xticks([]); wa.set_ylim(0, 1.2)
        wa.grid(alpha=0.25)
        wa.set_title(f"{top.helix}", fontsize=6, pad=1.5)
    fig.suptitle("Bars coloured by the helix's mean effect; the tick is the mean of the "
                 "within-helix shuffled null; * is q < 0.05. Inset: the most one-sided "
                 "helix as a wheel, 100° per residue, the arrow pointing at its intolerant face.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e07_helix_face", None, "E07")
    plt.close(fig)
    per.to_csv(outdir / "e07_helix_face.csv", index=False)
    head = "; ".join(f"{g.replace('_fitness', '')} {int((d.q < 0.05).sum())}/{len(d)}"
                     for g, d in per.groupby("protein"))
    return {"status": "ok", "headline": f"one-sided helices (q < 0.05): {head}",
            "by_helix": per.to_dict("records")}


# ------------------------------------------- e08 the protein in the membrane, coloured
def e08_membrane_map(pos: dict, names: list[str], outdir: Path) -> dict:
    """Every residue placed where it sits in the membrane, coloured by what it costs.

    The most direct picture the data allows: the horizontal axis is distance from the
    bundle axis, the vertical axis is depth, and the colour is the mean effect. The
    bilayer is shaded. It answers "where does it hurt" without a model in between.
    """
    rows = []
    fig, pairs, axes = grid(names, w=3.2, h=3.0, extra=0.3)
    sc = None
    for i, (g, ax) in enumerate(pairs):
        t = pos[g].dropna(subset=["effect", "radial", "abs_z"]).copy()
        # signed where the structure gives a side, unsigned otherwise, so a protein
        # without a membrane frame still draws rather than silently losing half its range
        t["z"] = t.zdepth if "zdepth" in t and t.zdepth.notna().any() else t.abs_z
        t = t.dropna(subset=["z"])
        tm = t[t.is_tm > 0.5]
        half = float(tm.abs_z.quantile(0.9)) if len(tm) else 15.0
        ax.axhspan(-half, half, color="#F2F2F2", zorder=0)
        for y in (-half, half):
            ax.axhline(y, color=P.MUTED, lw=0.7, ls=(0, (4, 3)))
        sc = ax.scatter(t.radial, t.z, s=16, c=t.effect, cmap=P.fitness_cmap(),
                        norm=P.fitness_norm(), edgecolor=P.INK, linewidth=0.25)
        # the depth profile, as a strip down the side of the same panel
        pa = ax.inset_axes([1.01, 0, 0.16, 1.0], sharey=ax)
        b = pd.cut(t.z, 14)
        m = t.groupby(b, observed=True).agg(y=("z", "mean"), x=("effect", "mean")).dropna()
        pa.plot(m.x, m.y, "-o", ms=2.5, lw=1.0, color=P.INK)
        pa.axvline(0, color=P.MUTED, lw=0.6)
        pa.tick_params(labelleft=False, labelsize=5.5)
        pa.set_xlabel("mean", fontsize=5.5)
        rows += [{"protein": g, "z": float(r.y), "mean_effect": float(r.x)}
                 for r in m.itertuples()]
        panel(ax, i, g, f"bilayer ±{half:.0f} Å", size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("depth z across the membrane (Å)", fontsize=7.5)
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=0.022, pad=0.06).set_label(
            "Mean effect at the position", fontsize=8, fontweight="bold")
    P.save(fig, outdir, "e08_membrane_map", None, "E08")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "e08_depth_profile.csv", index=False)
    return {"status": "ok", "headline": "every scored residue placed in the membrane frame "
                                        "and coloured by its mean effect"}


# -------------------------------------------------- e09 which substitutions are allowed
def e09_substitution_matrix(var: dict, names: list[str], outdir: Path,
                            min_n: int = 3) -> dict:
    """Mean effect of every wild-type to mutant pair, 20 x 20, one protein at a time.

    Rows and columns are ordered by hydrophobicity rather than alphabetically, which is
    what turns the matrix into a picture: in a membrane protein the top-right and
    bottom-left corners - hydrophobic to polar and back - are where the cost lives, and
    alphabetical order scatters them.
    """
    from .annot import KYTE_DOOLITTLE
    aas = sorted(KYTE_DOOLITTLE, key=lambda a: -KYTE_DOOLITTLE[a])
    fig, pairs, axes = grid(names, w=3.4, h=3.2, extra=0.3)
    im, rows = None, []
    for i, (g, ax) in enumerate(pairs):
        t = var[g].dropna(subset=["score_z"])
        m = t.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
        n = t.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
        m = m.reindex(index=aas, columns=aas)
        n = n.reindex(index=aas, columns=aas)
        m = m.where(n.fillna(0) >= min_n)
        im = ax.imshow(m.to_numpy(), cmap=P.fitness_cmap(), norm=P.fitness_norm())
        ax.set_xticks(range(len(aas))); ax.set_xticklabels(aas, fontsize=5.5)
        ax.set_yticks(range(len(aas))); ax.set_yticklabels(aas, fontsize=5.5)
        ax.set_xlabel("mutant (hydrophobic → polar)", fontsize=7)
        if i % 2 == 0:
            ax.set_ylabel("wild type (hydrophobic → polar)", fontsize=7)
        panel(ax, i, g, f"{int(np.isfinite(m.to_numpy()).sum())}/400 pairs with ≥ {min_n}",
              size=8)
        long = m.stack().rename("mean_effect").reset_index()
        rows.append(long.assign(protein=g))
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.022, pad=0.01).set_label(
            "Mean effect", fontsize=8, fontweight="bold")
    P.save(fig, outdir, "e09_substitution_matrix", None, "E09")
    plt.close(fig)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(outdir / "e09_substitution_matrix.csv", index=False)
    worst = (out.groupby(["protein", "mut"]).mean_effect.mean().reset_index()
             .sort_values("mean_effect").groupby("protein").head(1))
    head = "; ".join(f"{r.protein.replace('_fitness', '')} worst mutant {r.mut} "
                     f"({r.mean_effect:.2f})" for r in worst.itertuples())
    return {"status": "ok", "headline": head}


# ------------------------------------------- e10 the cost of a charge in the membrane
GROUPS = [("charged (D E K R)", set("DEKR"), "#B2182B"),
          ("polar (N Q S T)", set("NQST"), "#E8A33D"),
          ("hydrophobic (A V L I F M)", set("AVLIFM"), "#2166AC")]


def e10_charge_depth(var: dict, names: list[str], outdir: Path) -> dict:
    """What it costs to introduce a charge, as a function of how deep it is placed.

    The textbook expectation, stated as a measurement: burying a charge in the middle of
    a bilayer is expensive, burying one near the headgroups much less so, and a
    hydrophobic substitution should be flat with depth. Restricted to buried positions,
    since an exposed residue at the same depth is not in the lipid at all.
    """
    rows = []
    fig, pairs, _ = grid(names, w=3.0, h=2.7, extra=0.3)
    for i, (g, ax) in enumerate(pairs):
        t = var[g].dropna(subset=["score_z", "abs_z", "rsa"])
        t = t[t.rsa < 0.25]
        for lab, aas, col in GROUPS:
            s = t[t.mut.isin(list(aas))]
            if len(s) < 40:
                continue
            b = pd.cut(s.abs_z, [0, 5, 10, 15, 22, 40])
            m = s.groupby(b, observed=True).agg(x=("abs_z", "mean"), y=("score_z", "mean"),
                                                e=("score_z", "sem"),
                                                n=("score_z", "size")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=4, lw=1.3, color=col, capsize=2,
                        label=lab, mec=P.INK, mew=0.4)
            sl = ss.linregress(s.abs_z, s.score_z)
            rows.append({"protein": g, "group": lab, "n": int(len(s)),
                         "mean": float(s.score_z.mean()), "slope_per_A": float(sl.slope),
                         "se": float(sl.stderr), "p": float(sl.pvalue)})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        d = [r for r in rows if r["protein"] == g and r["group"].startswith("charged")]
        panel(ax, i, g, (f"charge slope {d[0]['slope_per_A']:+.3f}/Å" if d else ""), size=8)
        if i // 2 == (len(names) - 1) // 2:
            ax.set_xlabel("depth |z| from the bilayer centre (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("Mean effect of the substitution", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Buried positions only (RSA < 0.25). Depth increases to the right, so a "
                 "rising line means the substitution is cheaper away from the bilayer "
                 "centre.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e10_charge_depth", None, "E10")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e10_charge_depth.csv", index=False)
    d = per[per.group.str.startswith("charged")]
    head = "; ".join(f"{r.protein.replace('_fitness', '')} charge {r.mean:.2f}, "
                     f"{r.slope_per_A:+.3f}/Å" for r in d.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


ANALYSES = {"e01": "effect by residue environment",
            "e02": "burial against depth",
            "e03": "hydrophobicity against burial",
            "e04": "substitution direction by environment",
            "e05": "contacts",
            "e06": "ESM-1v residual network",
            "e07": "one-sided helices",
            "e08": "the protein in the membrane, coloured by effect",
            "e09": "substitution matrix",
            "e10": "the cost of a charge against depth"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "Degrees of freedom|All-NaN|tight_layout).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms environ",
        description="What a residue's surroundings say about how much a mutation costs. "
                    "Every figure is one panel per protein, laid out in the order the "
                    "configs are given: with four proteins and the default two columns "
                    "that is top-left, top-right, bottom-left, bottom-right.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--cols", type=int, default=2,
                    help="panels per row (default 2, so four proteins make a 2x2)")
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

    cfgs, var, pos, names = {}, {}, {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        v, p_ = protein_frame(cfg)
        cfgs[cfg.id] = cfg
        if not len(p_) or p_.environ.isna().all():
            print(f"  {cfg.id:<16} skipped: no position could be classified "
                  "(needs a structure and topology)", flush=True)
            continue
        var[cfg.id], pos[cfg.id] = v, p_
        names.append(cfg.id)
        counts = p_.environ.value_counts()
        print(f"  {cfg.id:<16} {len(p_):>4} positions  radial cut "
              f"{p_.attrs.get('radial_cut', float('nan')):.1f} Å  "
              + "  ".join(f"{SHORT[k]} {n}" for k, n in counts.items())
              + (f"  ({p_.attrs['n_low_plddt']} unclassified, pLDDT < {MIN_PLDDT:g})"
                 if p_.attrs.get("n_low_plddt") else ""), flush=True)
    if not names:
        raise SystemExit("no protein could be classified; each needs a structure and a "
                         "topology in its config")
    rows = int(np.ceil(len(names) / a.cols))
    print(f"\n  panels: " + " | ".join(
        ", ".join(n.replace("_fitness", "") for n in names[r * a.cols:(r + 1) * a.cols])
        for r in range(rows)), flush=True)
    pd.concat([p_.assign(protein=g) for g, p_ in pos.items()],
              ignore_index=True).to_csv(outdir / "positions.csv", index=False)

    def call(k):
        if k == "e01":
            return e01_environment(pos, names, outdir)
        if k == "e02":
            return e02_burial_depth(pos, names, outdir)
        if k == "e03":
            return e03_hydrophobicity(var, names, outdir)
        if k == "e04":
            return e04_direction(var, names, outdir)
        if k == "e05":
            return e05_contacts(pos, names, outdir)
        if k == "e06":
            return e06_esm_network(cfgs, pos, names, outdir)
        if k == "e07":
            return e07_helix_face(pos, names, outdir)
        if k == "e08":
            return e08_membrane_map(pos, names, outdir)
        if k == "e09":
            return e09_substitution_matrix(var, names, outdir)
        return e10_charge_depth(var, names, outdir)

    out = {}
    print()
    for k in keys:
        try:
            r = call(k)
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}",
              flush=True)
    (outdir / "environ.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
