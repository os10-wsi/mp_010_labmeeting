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

Every pooled number is a random-effects meta-analysis of per-protein estimates, not a
regression on the pooled variants: proteins differ in dynamic range and in how many
positions they contribute, and pooling variants lets the largest dataset decide the
answer.
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
from .stats import meta_analysis

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


def _pool(rows: pd.DataFrame, value: str = "estimate", var: str = "se") -> dict:
    """Random-effects pooling of one per-protein estimate."""
    d = rows.dropna(subset=[value, var])
    d = d[d[var] > 0]
    if len(d) < 2:
        return {"n_proteins": int(len(d))}
    m = meta_analysis(d[value].to_numpy(), d[var].to_numpy() ** 2)
    return {"n_proteins": int(m["k"]), "pooled": m["mu"], "lo": m["lo"], "hi": m["hi"],
            "p": m["p"], "i2": m["I2"], "tau2": m["tau2"]}


# ------------------------------------------------------------------ e01 environment
def e01_environment(pos: dict, outdir: Path) -> dict:
    """Mean effect per position, by what the position is surrounded by."""
    rows, contrasts = [], []
    for g, d in pos.items():
        for c in CLASSES:
            s = d.loc[d.environ == c, "effect"]
            if len(s) < 5:
                continue
            rows.append({"protein": g, "environ": c, "n": int(len(s)),
                         "estimate": float(s.mean()),
                         "se": float(s.std(ddof=1) / np.sqrt(len(s))),
                         "median": float(s.median())})
        # the contrast the question is really about: is the core less tolerant than lipid?
        for a, b in (("helix-helix core", "lipid-facing"),
                     ("cavity-lining", "lipid-facing"),
                     ("helix-helix core", "cavity-lining")):
            x = d.loc[d.environ == a, "effect"]; y = d.loc[d.environ == b, "effect"]
            if len(x) < 5 or len(y) < 5:
                continue
            se = float(np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)))
            contrasts.append({"protein": g, "contrast": f"{a} - {b}",
                              "estimate": float(x.mean() - y.mean()), "se": se,
                              "n_a": int(len(x)), "n_b": int(len(y)),
                              "p": float(ss.mannwhitneyu(x, y).pvalue)})
    per = pd.DataFrame(rows)
    con = pd.DataFrame(contrasts)
    if not len(per):
        return {"status": "skipped", "reason": "no protein has enough classified positions"}
    pooled = {c: _pool(con[con.contrast == c]) for c in con.contrast.unique()} if len(con) else {}

    n = len(pos)
    fig = plt.figure(figsize=(11.0, 3.2 + 2.6 * ((n + 2) // 3)), layout="constrained")
    gs = fig.add_gridspec((n + 2) // 3 + 1, 3, height_ratios=[1] * ((n + 2) // 3) + [1.15])
    for i, (g, d) in enumerate(sorted(pos.items())):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        groups = {SHORT[c]: d.loc[d.environ == c, "effect"].dropna().to_numpy()
                  for c in CLASSES}
        groups = {k: v for k, v in groups.items() if len(v) >= 5}
        back = {v: k for k, v in SHORT.items()}
        if groups:
            P.violin_box(ax, groups, colors={k: CLASS_COLORS[back[k]] for k in groups})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axhline(-1, color=P.FIT_RED, lw=0.6, ls=(0, (1, 2)))
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(f"({chr(97 + i)}) {g.replace('_fitness', '')}", loc="left", fontsize=8.5)
        if i % 3 == 0:
            ax.set_ylabel("Mean effect at the position")
    ax = fig.add_subplot(gs[-1, :])
    labs, est, lo, hi = [], [], [], []
    for c, m in pooled.items():
        if "pooled" not in m:
            continue
        labs.append(f"{c}   (I² = {m['i2']:.0f}%)"); est.append(m["pooled"])
        lo.append(m["lo"]); hi.append(m["hi"])
    if labs:
        forest(ax, labs, est, lo, hi)
        ax.axvline(0, color=P.MUTED, lw=0.8)
        ax.set_xlabel("Difference in mean position effect (negative = the first class is "
                      "less tolerant)")
    ax.set_title(f"({chr(97 + n)}) pooled across {len(pos)} proteins, random effects "
                 "(core = helix-helix interface, cavity = lining the central pathway, "
                 "lipid = facing acyl chains)", loc="left", fontsize=8.5)
    P.save(fig, outdir, "e01_environment", None, "E01")
    plt.close(fig)
    per.to_csv(outdir / "e01_environment_by_protein.csv", index=False)
    con.to_csv(outdir / "e01_environment_contrasts.csv", index=False)
    head = "; ".join(f"{c}: {m['pooled']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"
                     for c, m in pooled.items() if "pooled" in m)
    return {"status": "ok", "pooled": pooled, "headline": head}


# ------------------------------------------------------- e02 burial against depth
def e02_burial_depth(pos: dict, outdir: Path) -> dict:
    """The two membrane axes at once: how buried, and how deep.

    A single scatter of effect against RSA mixes the bilayer centre with the headgroup
    region, and a single scatter against depth mixes buried with exposed. Binning on both
    keeps them apart, and the question - whether burial costs more in the middle of the
    membrane than at its edge - is a property of the two together.
    """
    rows, cells = [], []
    names = sorted(pos)
    fig, axes = plt.subplots(2, len(names), figsize=(2.5 * len(names) + 1.4, 5.4),
                             layout="constrained", squeeze=False, sharey="row")
    zb = np.array([0, 4, 8, 12, 16, 25])
    rb = np.array([0, 0.05, 0.15, 0.30, 0.50, 1.01])
    mats = []
    for d in pos.values():
        t = d.dropna(subset=["effect", "rsa", "abs_z"])
        m = np.full((len(zb) - 1, len(rb) - 1), np.nan)
        for i in range(len(zb) - 1):
            for j in range(len(rb) - 1):
                s = t[(t.abs_z >= zb[i]) & (t.abs_z < zb[i + 1])
                      & (t.rsa >= rb[j]) & (t.rsa < rb[j + 1])].effect
                if len(s) >= 4:
                    m[i, j] = s.mean()
        mats.append(m)
    lim = np.nanpercentile(np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()]),
                           [2, 98]) if mats else (-1, 0)
    vmax = float(max(abs(lim[0]), abs(lim[1])))
    im = None
    for k, g in enumerate(names):
        d, m = pos[g], mats[k]
        ax = axes[0][k]
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=P.diverging_cmap(),
                       vmin=-vmax, vmax=vmax,
                       extent=(0, len(rb) - 1, 0, len(zb) - 1))
        ax.set_xticks(np.arange(len(rb) - 1) + 0.5)
        ax.set_xticklabels([f"{rb[j]:.2f}" for j in range(len(rb) - 1)], fontsize=5.5)
        ax.set_yticks(np.arange(len(zb) - 1) + 0.5)
        ax.set_yticklabels([f"{zb[i]}-{zb[i+1]}" for i in range(len(zb) - 1)], fontsize=5.5)
        ax.set_title(f"({chr(97 + k)}) {g.replace('_fitness', '')}", loc="left", fontsize=8)
        ax.set_xlabel("RSA (lower bound of bin)", fontsize=7)
        if k == 0:
            ax.set_ylabel("depth |z| from the bilayer centre (Å)", fontsize=7)

        # the marginal that answers the question directly: burial slope by depth band
        ax = axes[1][k]
        t = d.dropna(subset=["effect", "rsa", "abs_z"])
        for lab, sub, col in (("|z| < 10 Å (centre)", t[t.abs_z < 10], P.SERIES[0]),
                              ("|z| ≥ 10 Å (edge)", t[t.abs_z >= 10], P.SERIES[1])):
            if len(sub) < 20:
                continue
            b = pd.cut(sub.rsa, [0, 0.05, 0.15, 0.30, 0.50, 1.01])
            mm = sub.groupby(b, observed=True).agg(x=("rsa", "mean"), y=("effect", "mean"),
                                                   e=("effect", "sem")).dropna()
            ax.errorbar(mm.x, mm.y, yerr=mm.e, fmt="-o", ms=3.5, lw=1.1, color=col,
                        capsize=2, label=lab)
            sl = ss.linregress(sub.rsa, sub.effect)
            rows.append({"protein": g, "band": lab, "estimate": float(sl.slope),
                         "se": float(sl.stderr), "p": float(sl.pvalue), "n": int(len(sub))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.set_xlabel("RSA", fontsize=7)
        if k == 0:
            ax.set_ylabel("Mean effect", fontsize=7)
            ax.legend(frameon=False, fontsize=5.5)
    if im is not None:
        fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01).set_label(
            "Mean effect", fontsize=7, fontweight="bold")
    P.save(fig, outdir, "e02_burial_depth", None, "E02")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e02_burial_slopes.csv", index=False)
    pooled = {b: _pool(per[per.band == b]) for b in per.band.unique()} if len(per) else {}
    head = "; ".join(f"{b}: RSA slope {m['pooled']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"
                     for b, m in pooled.items() if "pooled" in m)
    return {"status": "ok", "pooled": pooled, "headline": head}


# -------------------------------------------- e03 hydrophobicity against burial
def e03_hydrophobicity(var: dict, outdir: Path) -> dict:
    """Change in hydrophobicity against effect, in three bands of burial.

    `delta_kd` is mutant minus wild-type on the Kyte-Doolittle scale, so positive means
    the substitution is more hydrophobic. The expectation a membrane protein sets up is
    that the sign of the cost should flip with the environment, and that a buried position
    should care about the size of the change whichever way it goes - which is why the
    quadratic term is fitted as well as the slope.
    """
    bands = [("buried (RSA < 0.15)", lambda t: t.rsa < 0.15),
             ("intermediate", lambda t: (t.rsa >= 0.15) & (t.rsa < 0.35)),
             ("exposed (RSA ≥ 0.35)", lambda t: t.rsa >= 0.35)]
    names = sorted(var)
    rows = []
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names) + 1.0, 3.2),
                             layout="constrained", squeeze=False, sharey=True)
    cols = dict(zip([b for b, _ in bands], P.series_colors(3)))
    for k, g in enumerate(names):
        ax = axes[0][k]
        t = var[g].dropna(subset=["score_z", "delta_kd", "rsa"])
        for lab, f in bands:
            sub = t[f(t)]
            if len(sub) < 50:
                continue
            b = pd.cut(sub.delta_kd, np.linspace(-9, 9, 10))
            m = sub.groupby(b, observed=True).agg(x=("delta_kd", "mean"),
                                                  y=("score_z", "mean"),
                                                  e=("score_z", "sem")).dropna()
            ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=3, lw=1.0, color=cols[lab],
                        capsize=1.5, label=lab)
            sl = ss.linregress(sub.delta_kd, sub.score_z)
            q = np.polyfit(sub.delta_kd, sub.score_z, 2)
            rows.append({"protein": g, "band": lab, "estimate": float(sl.slope),
                         "se": float(sl.stderr), "p": float(sl.pvalue),
                         "curvature": float(q[0]), "n": int(len(sub))})
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.set_xlabel("Δ hydrophobicity (Kyte-Doolittle, mutant − WT)", fontsize=7)
        ax.set_title(f"({chr(97 + k)}) {g.replace('_fitness', '')}", loc="left", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("Mean effect")
            ax.legend(frameon=False, fontsize=6)
    P.save(fig, outdir, "e03_hydrophobicity_burial", None, "E03")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e03_hydrophobicity_slopes.csv", index=False)
    pooled = {b: _pool(per[per.band == b]) for b in per.band.unique()} if len(per) else {}
    head = "; ".join(f"{b}: {m['pooled']:+.4f} per KD unit [{m['lo']:+.4f}, {m['hi']:+.4f}]"
                     for b, m in pooled.items() if "pooled" in m)
    return {"status": "ok", "pooled": pooled, "headline": head}


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


def e04_direction(var: dict, outdir: Path) -> dict:
    """The hypothesis stated directly: which environments tolerate which direction?

    Reported against that environment's own baseline, as the difference between the mean
    effect of a direction and the mean effect of every substitution at the same kind of
    position. A class that is intolerant of everything would otherwise look intolerant of
    each direction in turn, which says nothing about chemistry.
    """
    rows = []
    for g, t in var.items():
        t = t.dropna(subset=["score_z", "environ"]).copy()
        t["direction"] = direction(t.wt, t.mut)
        for c in CLASSES:
            base = t[t.environ == c]
            if len(base) < 50:
                continue
            mu = float(base.score_z.mean())
            for dname in DIRECTIONS:
                s = base.loc[base.direction == dname, "score_z"]
                if len(s) < 15:
                    continue
                rows.append({"protein": g, "environ": c, "direction": dname,
                             "n": int(len(s)), "mean": float(s.mean()),
                             "estimate": float(s.mean() - mu),
                             "se": float(s.std(ddof=1) / np.sqrt(len(s)))})
    per = pd.DataFrame(rows)
    if not len(per):
        return {"status": "skipped", "reason": "too few variants per environment"}
    pooled = []
    for (c, dn), g in per.groupby(["environ", "direction"]):
        m = _pool(g)
        if "pooled" in m:
            pooled.append({"environ": c, "direction": dn, **m})
    pl = pd.DataFrame(pooled)

    classes = [c for c in CLASSES if c in set(pl.environ)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.2 + 0.22 * len(per.environ.unique())),
                             layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    cols = dict(zip(DIRECTIONS, P.series_colors(len(DIRECTIONS))))
    y = np.arange(len(classes))[::-1]
    for i, dn in enumerate(DIRECTIONS):
        off = (i - 1.5) * 0.17
        d = pl[pl.direction == dn].set_index("environ").reindex(classes)
        ax.hlines(y + off, d.lo, d.hi, color=cols[dn], lw=1.2)
        ax.scatter(d.pooled, y + off, s=22, color=cols[dn], label=dn, zorder=4,
                   edgecolor=P.INK, linewidth=0.4)
    ax.axvline(0, color=P.INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_yticks(y); ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel("Effect relative to the average substitution at that kind of position")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left")
    ax.set_title("(a) pooled across proteins, random effects", loc="left", fontsize=8.5)

    ax = axes[1]
    piv = per.pivot_table(index="environ", columns="direction", values="mean",
                          aggfunc="mean").reindex(classes)
    piv = piv[[d for d in DIRECTIONS if d in piv.columns]]
    v = float(np.nanmax(np.abs(piv.to_numpy())))
    im = ax.imshow(piv.to_numpy(), cmap=P.diverging_cmap(), vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace(" → ", "\n→ ") for c in piv.columns], fontsize=6.5)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if np.isfinite(piv.iat[i, j]):
                ax.text(j, i, f"{piv.iat[i, j]:+.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("Mean effect", fontsize=7,
                                                     fontweight="bold")
    ax.set_title("(b) absolute mean effect, averaged over proteins", loc="left", fontsize=8.5)
    P.save(fig, outdir, "e04_substitution_direction", None, "E04")
    plt.close(fig)
    per.to_csv(outdir / "e04_direction_by_protein.csv", index=False)
    pl.to_csv(outdir / "e04_direction_pooled.csv", index=False)
    bits = []
    for c in classes:
        d = pl[(pl.environ == c) & (pl.direction == "polar → hydrophobic")]
        if len(d):
            r = d.iloc[0]
            bits.append(f"{c}: polar→hydrophobic {r.pooled:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
    return {"status": "ok", "pooled": pl.to_dict("records"), "headline": "; ".join(bits)}


# --------------------------------------------------------------- e05 contacts
def e05_contacts(pos: dict, outdir: Path) -> dict:
    """Effect against the number of structural contacts, with RSA partialled out.

    Contacts and solvent accessibility are two views of the same thing and correlate
    strongly, so the contact result is only news if it survives holding RSA fixed. Both
    are reported: the raw Spearman, and the partial correlation given RSA.
    """
    rows = []
    names = sorted(pos)
    fig, axes = plt.subplots(1, len(names), figsize=(2.5 * len(names) + 1.0, 3.1),
                             layout="constrained", squeeze=False, sharey=True)
    for k, g in enumerate(names):
        ax = axes[0][k]
        t = pos[g].dropna(subset=["effect", "contacts"])
        if len(t) < 20:
            continue
        P.density_scatter(ax, t.contacts.to_numpy(), t.effect.to_numpy(), s=6)
        b = pd.cut(t.contacts, 8)
        m = t.groupby(b, observed=True).agg(x=("contacts", "mean"), y=("effect", "mean"),
                                            e=("effect", "sem")).dropna()
        ax.errorbar(m.x, m.y, yerr=m.e, fmt="-o", ms=3.5, lw=1.4, color=P.FIT_RED, capsize=2)
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
                     "p": float(sl.pvalue), "spearman": rho, "partial_spearman_given_rsa": part,
                     "n": int(len(t)), "n_partial": n_part})
        ax.set_title(f"({chr(97 + k)}) {g.replace('_fitness', '')}\nρ = {rho:+.2f}, "
                     f"partial ρ | RSA = {part:+.2f}", loc="left", fontsize=7.5)
        ax.set_xlabel("Contacts (heavy atoms < 8 Å, |i−j| > 4)", fontsize=7)
        if k == 0:
            ax.set_ylabel("Mean effect at the position")
    P.save(fig, outdir, "e05_contacts", None, "E05")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "e05_contacts.csv", index=False)
    m = _pool(per)
    head = (f"slope {m['pooled']:+.4f} per contact [{m['lo']:+.4f}, {m['hi']:+.4f}], "
            f"I² = {m.get('i2', float('nan')):.0f}%; mean ρ {per.spearman.mean():+.2f}, "
            f"mean partial ρ given RSA {per.partial_spearman_given_rsa.mean():+.2f}"
            if "pooled" in m else "")
    return {"status": "ok", "pooled": m, "headline": head}


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
                             median_z=("z", "median"),
                             median_abundance=("score_z", "median")).reset_index()
    return g[g.n_variants >= A15_MIN]


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


def e06_esm_network(cfgs: dict, pos: dict, outdir: Path) -> dict:
    """The ten largest ESM-1v residuals, placed in the structure.

    A residual is a position where ESM expects more constraint than abundance shows (or
    the reverse). Scattered residuals are noise, or they are positions whose function the
    abundance assay cannot see one at a time. A cluster is a different claim: a set of
    residues that sit together, which is what a binding site or a gate looks like.
    """
    from .structure import residue_table
    rows, per = [], {}
    names = [g for g in sorted(cfgs) if g in pos]
    panels = []
    for g in names:
        cfg = cfgs[g]
        res = esm_residuals(cfg)
        sp = cfg.resolve(cfg.get_path("structure.path"))
        if res is None or sp is None or not sp.exists():
            continue
        rt = residue_table(sp, cfg.get_path("structure.chain", "A"),
                           int(cfg.get_path("structure.numbering_offset", 0) or 0))
        from .structure import membrane_frame
        rt, _ = membrane_frame(rt, cfg)
        t = res.merge(rt[["pos", "x", "y", "z", "zdepth", "radial", "plddt"]], on="pos",
                      how="inner").merge(pos[g][["pos", "seg_type", "environ"]], on="pos",
                                         how="left").reset_index(drop=True)
        if len(t) < 3 * TOP_N:
            continue
        xyz = t[["x", "y", "z"]].to_numpy(float)
        order = t.median_z.to_numpy().argsort()          # most negative = most constrained
        pick = order[:TOP_N]
        st = network_stats(xyz, pick, np.arange(len(t)),
                           strata=t.seg_type.fillna("unannotated").to_numpy())
        top = t.iloc[pick].assign(protein=g, rank=np.arange(1, TOP_N + 1))
        rows.append(top)
        per[g] = {k: v for k, v in st.items() if k not in ("null", "dmat")}
        panels.append((g, t, pick, st))
        write_network_structure(cfg, top, st, outdir)

    if not panels:
        return {"status": "skipped", "reason": "no protein has both ESM scores and a structure"}
    n = len(panels)
    fig, axes = plt.subplots(3, n, figsize=(2.9 * n + 0.8, 8.4), layout="constrained",
                             squeeze=False)
    for k, (g, t, pick, st) in enumerate(panels):
        ax = axes[0][k]
        ax.scatter(t.radial, t.zdepth, s=5, color=P.MUTED, alpha=0.45, lw=0)
        p10 = t.iloc[pick]
        c = p10[["x", "y", "z"]].to_numpy(float)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        for i in range(TOP_N):
            for j in range(i + 1, TOP_N):
                if d[i, j] < NEAR:
                    ax.plot([p10.radial.iloc[i], p10.radial.iloc[j]],
                            [p10.zdepth.iloc[i], p10.zdepth.iloc[j]],
                            color=P.FIT_RED, lw=0.8, alpha=0.7, zorder=3)
        ax.scatter(p10.radial, p10.zdepth, s=34, color=P.FIT_RED, zorder=5,
                   edgecolor=P.INK, linewidth=0.5)
        P.label_points(ax, p10.radial.to_numpy(), p10.zdepth.to_numpy(),
                       [f"{r.wt}{int(r.pos)}" for r in p10.itertuples()], fontsize=6)
        ax.axhline(0, color=P.INK, lw=0.6, ls=(0, (3, 3)))
        ax.set_xlabel("distance from the bundle axis (Å)", fontsize=7)
        if k == 0:
            ax.set_ylabel("depth z across the membrane (Å)", fontsize=7)
        ax.set_title(f"({chr(97 + k)}) {g.replace('_fitness', '')}: "
                     f"{st['n_pairs_within']}/{st['n_pairs']} pairs < {NEAR:.0f} Å",
                     loc="left", fontsize=7.5)

        ax = axes[1][k]
        lab = [f"{r.wt}{int(r.pos)}" for r in t.iloc[pick].itertuples()]
        # ordered by how the distances cluster, so a sub-network reads as a dark block
        # instead of being scattered over the matrix by the order the sites were ranked
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        o = leaves_list(linkage(squareform(st["dmat"], checks=False), "average"))
        lab = [lab[i] for i in o]
        im = ax.imshow(st["dmat"][np.ix_(o, o)], cmap=P.sequential_cmap(), vmin=0, vmax=40)
        ax.set_xticks(range(TOP_N)); ax.set_xticklabels(lab, rotation=90, fontsize=5.5)
        ax.set_yticks(range(TOP_N)); ax.set_yticklabels(lab, fontsize=5.5)
        if k == n - 1:
            fig.colorbar(im, ax=ax, fraction=0.045).set_label("CA distance (Å)", fontsize=7,
                                                              fontweight="bold")
        ax.set_title(f"({chr(97 + n + k)}) pairwise distances", loc="left", fontsize=7.5)

        ax = axes[2][k]
        ax.hist(st["null"], bins=40, color=P.MUTED, alpha=0.55, lw=0)
        ax.axvline(st["observed_mean_distance"], color=P.FIT_RED, lw=1.6)
        ax.set_xlabel("mean pairwise distance of 10 positions (Å)", fontsize=7)
        if k == 0:
            ax.set_ylabel("random draws", fontsize=7)
        if "null_matched_mean" in st:
            ax.axvline(st["null_matched_mean"], color=P.SERIES[0], lw=1.0,
                       ls=(0, (4, 3)))
        ax.set_title(f"({chr(97 + 2 * n + k)}) observed "
                     f"{st['observed_mean_distance']:.1f} Å, null "
                     f"{st['null_mean']:.1f} Å, p = {st['p_clustered']:.3f}"
                     + (f"\n     matched on topology: {st['null_matched_mean']:.1f} Å, "
                        f"p = {st['p_clustered_matched']:.3f}"
                        if "null_matched_mean" in st else ""),
                     loc="left", fontsize=7.5)
    fig.suptitle("Red = the ten positions whose ESM-1v residual is most negative. The null "
                 "draws ten positions at random from those with enough variants to be "
                 "scored.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "e06_esm_residual_network", None, "E06")
    plt.close(fig)
    top = pd.concat(rows, ignore_index=True)
    top.to_csv(outdir / "e06_top_residual_positions.csv", index=False)
    pd.DataFrame(per).T.rename_axis("protein").to_csv(outdir / "e06_clustering.csv")
    head = "; ".join(f"{g}: {v['observed_mean_distance']:.1f} Å vs null "
                     f"{v['null_mean']:.1f} Å, p = {v['p_clustered']:.3f}"
                     + (f" (matched p = {v['p_clustered_matched']:.3f})"
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


ANALYSES = {"e01": "environment classes", "e02": "burial against depth",
            "e03": "hydrophobicity against burial", "e04": "substitution direction",
            "e05": "contacts", "e06": "ESM-1v residual network"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "Degrees of freedom|All-NaN).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms environ",
        description="What a residue's surroundings say about how much a mutation costs.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
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

    cfgs, var, pos = {}, {}, {}
    for c in a.configs:
        cfg = load_config(Path(c))
        v, p_ = protein_frame(cfg)
        cfgs[cfg.id] = cfg
        if not len(p_) or p_.environ.isna().all():
            print(f"  {cfg.id:<16} skipped: no position could be classified "
                  "(needs a structure and topology)", flush=True)
            continue
        var[cfg.id], pos[cfg.id] = v, p_
        counts = p_.environ.value_counts()
        print(f"  {cfg.id:<16} {len(p_):>4} positions  radial cut "
              f"{p_.attrs.get('radial_cut', float('nan')):.1f} Å  "
              + "  ".join(f"{k.split()[0]} {n}" for k, n in counts.items())
              + (f"  ({p_.attrs['n_low_plddt']} unclassified, pLDDT < {MIN_PLDDT:g})"
                 if p_.attrs.get("n_low_plddt") else ""), flush=True)
    if len(pos) < 2:
        raise SystemExit("need at least two proteins with a structure and topology")
    pd.concat([p_.assign(protein=g) for g, p_ in pos.items()],
              ignore_index=True).to_csv(outdir / "positions.csv", index=False)

    out = {}
    print()
    for k in keys:
        try:
            if k == "e01":
                r = e01_environment(pos, outdir)
            elif k == "e02":
                r = e02_burial_depth(pos, outdir)
            elif k == "e03":
                r = e03_hydrophobicity(var, outdir)
            elif k == "e04":
                r = e04_direction(var, outdir)
            elif k == "e05":
                r = e05_contacts(pos, outdir)
            else:
                r = e06_esm_network(cfgs, pos, outdir)
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
