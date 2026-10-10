"""`python -m mpdms aggregate configs/*.yaml [--only g01,g02,g03,g04,g05]`

Questions asked of every protein at once, about the transmembrane domain.

Every estimate is reported twice, because the two ways of pooling answer different
questions and their disagreement is itself a result:

  mixed model     one fit over all variants, with protein (and helix within protein) as
                  random intercepts. Uses all the data, but weights a protein by how many
                  variants were measured in it.
  meta-analysis   one estimate per protein, combined with random effects. Weights each
                  protein once, and reports how much the proteins disagree (I2).

Two warnings that apply to everything here. Proteins measured on a noisier assay shrink
every effect toward zero, so the per-protein synonymous SD is carried alongside each
estimate. And close paralogs are not independent replicates: four of a set of eight can
look like strong agreement when it is one measurement repeated. The per-protein column is
always shown so that can be judged rather than assumed.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from . import plotting as P
from .analyses.a06_charge_topology import ACID, BASE, FLANK, annotate
from .analyses.base import missense
from .annot import segments_df
from .config import REPO_ROOT, load_config
from .io import load_dataset
from .stats import meta_analysis, slope_with_se

OUT = REPO_ROOT / "outputs" / "_aggregate"
# A variance-component formula is evaluated inside each group, but its levels come from
# the whole frame. "TM1".."TM12" repeat across proteins and so line up; a protein-prefixed
# helix id would ask each group for 96 columns it cannot fill.
VC_HELIX = {"helix": "0 + C(tm)"}
AROMATIC = set("WYF")
MIN_PER_HELIX = 12


# ------------------------------------------------------------------ assembly
def tm_frame(cfg, df: pd.DataFrame) -> pd.DataFrame:
    """Every missense variant in or beside a TM helix, with its depth across the membrane.

    `x` runs 0 at the cytosolic end to 1 at the lumenal end regardless of which way the
    helix is inserted, so helices of opposite orientation can be pooled without cancelling
    the very asymmetry being measured.
    """
    segs = segments_df(cfg)
    if segs.empty or not (segs.type == "TM").any():
        return pd.DataFrame()
    d = missense(df).dropna(subset=["score_z"])
    a = annotate(d, segs)
    if a.empty:
        return a
    a = a.copy()
    a["protein"] = cfg.id
    a["helix"] = a.protein + ":" + a.tm.astype(str)
    a["in_membrane"] = a.x.between(0, 1)
    a["n_helix"] = a.tm.map(lambda s: int(m.group(1)) if (m := re.search(r"(\d+)", str(s))) else np.nan)
    return a


def syn_sd(df: pd.DataFrame) -> float:
    syn = df.loc[df.pass_filter & (df.vclass == "synonymous"), "score_z"].dropna()
    return float(syn.std(ddof=1)) if len(syn) > 4 else np.nan


def mixed(formula: str, data: pd.DataFrame, group: str, term: str,
          vc: dict | None = None) -> dict:
    """One coefficient from a mixed model, with protein as the grouping factor."""
    import statsmodels.formula.api as smf
    # Concatenated per-protein frames repeat each other's index labels, which silently
    # misaligns the variance-component design against the response. Whole-word matching
    # keeps the dropna to the variables actually in the model: a column named "x" is a
    # substring of nearly every formula.
    d = data.reset_index(drop=True)
    used = {c for c in d.columns
            if re.search(rf"\b{re.escape(c)}\b", formula)
            or any(re.search(rf"\b{re.escape(c)}\b", f) for f in (vc or {}).values())
            or c == group}
    d = d.dropna(subset=sorted(used))
    if len(d) < 50 or d[group].nunique() < 3:
        return {"coef": np.nan, "se": np.nan, "p": np.nan, "n": int(len(d)), "note": "too few groups"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm(formula, d, groups=d[group],
                            vc_formula=vc or None).fit(reml=True, method="lbfgs")
        if term not in m.params.index:
            return {"coef": np.nan, "se": np.nan, "p": np.nan, "n": int(len(d)),
                    "note": f"{term} not in model"}
        return {"coef": float(m.params[term]), "se": float(m.bse[term]),
                "p": float(m.pvalues[term]), "n": int(len(d)),
                "converged": bool(m.converged), "note": "" if m.converged else "did not converge"}
    except Exception as e:                       # a singular fit is a result, not a crash
        return {"coef": np.nan, "se": np.nan, "p": np.nan, "n": int(len(d)),
                "note": f"{type(e).__name__}: {e}"[:90]}


def both_ways(per_protein: pd.DataFrame, mixed_res: dict, value="estimate", var="se") -> dict:
    """The two pooled numbers side by side, plus whether they agree."""
    meta = meta_analysis(per_protein[value], per_protein[var] ** 2)
    out = {"meta_" + k: v for k, v in meta.items()}
    out.update({"mixed_" + k: v for k, v in mixed_res.items()})
    a, b = meta["mu"], mixed_res.get("coef", np.nan)
    out["agree_in_sign"] = bool(np.isfinite(a) and np.isfinite(b) and np.sign(a) == np.sign(b))
    return out


# --------------------------------------------------------------------- plots
def forest(ax, labels, est, lo, hi, pooled=None, pooled_label="", colors=None, fontsize=6):
    y = np.arange(len(labels))[::-1]
    cols = colors if colors is not None else [P.INK] * len(labels)
    for yy, e, l, h, c in zip(y, est, lo, hi, cols):
        ax.plot([l, h], [yy, yy], color=c, lw=1.0, solid_capstyle="butt")
        ax.plot([e], [yy], "o", ms=3.0, color=c)
    ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
    ticks, ticklabels = list(y), list(labels)
    if pooled is not None:
        mu, plo, phi = pooled
        ax.plot([plo, phi], [-1.1, -1.1], color=P.FIT_RED, lw=1.8, solid_capstyle="butt")
        ax.plot([mu], [-1.1], "D", ms=4.5, color=P.FIT_RED)
        # as a tick label, not text inside the axes, where it lands on its own interval
        ticks.append(-1.1)
        ticklabels.append(pooled_label)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=fontsize)
    if pooled is not None:
        ax.get_yticklabels()[-1].set_color(P.FIT_RED)
        ax.get_yticklabels()[-1].set_fontsize(max(fontsize, 6.5))
    ax.set_ylim(-1.8 if pooled is not None else -0.6, len(labels) - 0.4)
    return ax


def compare_box(ax, res: dict, label: str, unit: str = "") -> None:
    """The two pooled estimates printed where they cannot be mistaken for each other."""
    mm, ms = res.get("mixed_coef", np.nan), res.get("mixed_se", np.nan)
    txt = (f"{label}\n"
           f"meta-analysis  {res['meta_mu']:+.3f}  [{res['meta_lo']:+.3f}, {res['meta_hi']:+.3f}]"
           f"   k = {res['meta_k']}, I² = {res['meta_I2']:.0f}%\n"
           f"mixed model    {mm:+.3f}  ± {ms:.3f}   p = {res.get('mixed_p', float('nan')):.2g}"
           + (f"   [{res['mixed_note']}]" if res.get("mixed_note") else ""))
    if unit:
        txt += f"\n{unit}"
    ax.text(0, 1, txt, transform=ax.transAxes, va="top", ha="left", fontsize=6.6,
            family="monospace", linespacing=1.6)
    ax.axis("off")


# ------------------------------------------------ g01 positive-inside at scale
def g01_positive_inside(frames: dict, meta: pd.DataFrame, outdir: Path) -> dict:
    """Is K/R better tolerated toward the cytosolic end than D/E, helix by helix?

    The estimate for one helix is the depth x charge interaction: how much more the
    basic residues' effect changes across the membrane than the acidic residues'. The
    positive-inside rule predicts it to be negative (K/R gets worse toward the lumen,
    relatively). One number per helix, so ~96 of them carry the test rather than 8.
    """
    import statsmodels.api as sm
    rows = []
    for name, a in frames.items():
        d = a[a.in_membrane & a.mut.isin(list(ACID | BASE)) & ~a.wt.isin(list(ACID | BASE))].copy()
        d["basic"] = d.mut.isin(list(BASE)).astype(float)
        for helix, g in d.groupby("helix"):
            if len(g) < MIN_PER_HELIX or g.basic.nunique() < 2:
                continue
            X = sm.add_constant(np.column_stack([g.x, g.basic, g.x * g.basic]))
            try:
                m = sm.OLS(g.score_z.to_numpy(), X).fit(cov_type="cluster",
                                                        cov_kwds={"groups": g.pos.to_numpy()})
            except Exception:
                continue
            if not np.isfinite(m.bse[3]) or m.bse[3] == 0:
                continue
            rows.append({"protein": name, "helix": helix,
                         "n_helix": int(g.n_helix.iloc[0]) if np.isfinite(g.n_helix.iloc[0]) else -1,
                         "n": len(g), "estimate": float(m.params[3]), "se": float(m.bse[3])})
    per_helix = pd.DataFrame(rows)
    if len(per_helix) < 5:
        return {"status": "skipped", "reason": f"only {len(per_helix)} helices with enough charge variants"}

    # one estimate per protein, from its own helices, for the protein-level meta-analysis
    per_protein = (per_helix.groupby("protein")
                   .apply(lambda g: pd.Series(meta_analysis(g.estimate, g.se ** 2)),
                          include_groups=False).reset_index())
    per_protein = per_protein.rename(columns={"mu": "estimate", "se": "se"})

    allv = pd.concat(frames.values())
    d = allv[allv.in_membrane & allv.mut.isin(list(ACID | BASE))
             & ~allv.wt.isin(list(ACID | BASE))].copy()
    d["basic"] = d.mut.isin(list(BASE)).astype(float)
    mx = mixed("score_z ~ x * basic", d, "protein", "x:basic",
               vc=VC_HELIX)
    res = both_ways(per_protein, mx)
    res["n_helices"] = int(len(per_helix))
    res["helix_meta"] = meta_analysis(per_helix.estimate, per_helix.se ** 2)

    fig = plt.figure(figsize=(10.4, max(5.4, 0.13 * len(per_helix) + 2.2)), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 0.34])
    ax = fig.add_subplot(gs[:, 0])
    ph = per_helix.sort_values(["protein", "n_helix"])
    pal = dict(zip(sorted(frames), plt.get_cmap("tab10").colors))
    forest(ax, (ph.protein.str.replace("_fitness", "", regex=False) + " " +
                ph.helix.str.split(":").str[-1]).tolist(),
           ph.estimate, ph.estimate - 1.96 * ph.se, ph.estimate + 1.96 * ph.se,
           pooled=(res["helix_meta"]["mu"], res["helix_meta"]["lo"], res["helix_meta"]["hi"]),
           pooled_label=f"all {len(ph)} helices", colors=[pal[p] for p in ph.protein], fontsize=4.6)
    ax.set_xlabel("depth × basic interaction\n(negative = K/R relatively worse toward the lumen)",
                  fontsize=7)
    ax.set_title("(a) One estimate per transmembrane helix", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 1])
    pp = per_protein.sort_values("estimate")
    forest(ax, pp.protein.tolist(), pp.estimate, pp.lo, pp.hi,
           pooled=(res["meta_mu"], res["meta_lo"], res["meta_hi"]),
           pooled_label=f"meta-analysis, I² = {res['meta_I2']:.0f}%", fontsize=7)
    ax.set_xlabel("depth × basic interaction", fontsize=7)
    ax.set_title("(b) One estimate per protein", loc="left", fontsize=8.5)
    compare_box(fig.add_subplot(gs[1, 1]), res, "Positive-inside (depth × basic)",
                f"helix-level meta  {res['helix_meta']['mu']:+.3f}  "
                f"[{res['helix_meta']['lo']:+.3f}, {res['helix_meta']['hi']:+.3f}]  "
                f"k = {res['helix_meta']['k']}")
    P.save(fig, outdir, "g01_positive_inside", None, "G01")
    plt.close(fig)
    per_helix.to_csv(outdir / "g01_per_helix.csv", index=False)
    per_protein.to_csv(outdir / "g01_per_protein.csv", index=False)
    res["status"] = "ok"
    res["headline"] = (f"depth × basic across {len(per_helix)} helices in {len(frames)} proteins: "
                       f"meta {res['meta_mu']:+.3f} [{res['meta_lo']:+.3f}, {res['meta_hi']:+.3f}], "
                       f"mixed {res.get('mixed_coef', float('nan')):+.3f} "
                       f"(p = {res.get('mixed_p', float('nan')):.2g}), I² = {res['meta_I2']:.0f}%")
    return res


# ------------------------------------------------------ g02 pooled depth profile
def g02_depth_profile(frames: dict, meta: pd.DataFrame, outdir: Path) -> dict:
    """The charge-versus-depth profile of every protein on one pair of axes."""
    bins = np.concatenate([np.linspace(-0.5, 0, 3), np.linspace(0, 1, 7)[1:],
                           np.linspace(1, 1.5, 3)[1:]])
    rows, curves = [], []
    for name, a in frames.items():
        d = a[a.mut.isin(list(ACID | BASE)) & ~a.wt.isin(list(ACID | BASE))].copy()
        if len(d) < 50:
            continue
        d["basic"] = d.mut.isin(list(BASE)).astype(float)
        d["charge"] = np.where(d.basic > 0, "basic", "acidic")
        inm = d[d.in_membrane]
        import statsmodels.api as sm
        X = sm.add_constant(np.column_stack([inm.x, inm.basic, inm.x * inm.basic]))
        try:
            m = sm.OLS(inm.score_z.to_numpy(), X).fit(cov_type="cluster",
                                                      cov_kwds={"groups": inm.pos.to_numpy()})
            rows.append({"protein": name, "estimate": float(m.params[3]), "se": float(m.bse[3]),
                         "n": len(inm)})
        except Exception:
            pass
        for ch in ("acidic", "basic"):
            s = d[d.charge == ch]
            g = s.groupby(pd.cut(s.x, bins, include_lowest=True),
                          observed=False).score_z.agg(["mean", "sem", "size"])
            curves.append(pd.DataFrame({"protein": name, "charge": ch,
                                        "x": [iv.mid for iv in g.index],
                                        "mean": g["mean"].to_numpy(), "sem": g["sem"].to_numpy(),
                                        "n": g["size"].to_numpy()}))
    per_protein = pd.DataFrame(rows)
    if not len(per_protein):
        return {"status": "skipped", "reason": "no protein has enough charge-introducing variants"}
    cur = pd.concat(curves, ignore_index=True)

    allv = pd.concat(frames.values())
    d = allv[allv.in_membrane & allv.mut.isin(list(ACID | BASE))
             & ~allv.wt.isin(list(ACID | BASE))].copy()
    d["basic"] = d.mut.isin(list(BASE)).astype(float)
    res = both_ways(per_protein, mixed("score_z ~ x * basic", d, "protein", "x:basic",
                                       vc=VC_HELIX))

    fig = plt.figure(figsize=(10.6, 5.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.34])
    pal = dict(zip(sorted(frames), plt.get_cmap("tab10").colors))
    for k, (ch, lab) in enumerate((("acidic", "D/E introduced"), ("basic", "K/R introduced"))):
        ax = fig.add_subplot(gs[0, k])
        ax.axvspan(0, 1, color=P.TM_GREY, lw=0, zorder=0)
        for name, g in cur[cur.charge == ch].groupby("protein"):
            ax.plot(g.x, g["mean"], "-o", ms=2.4, lw=1.0, color=pal[name],
                    label=name.replace("_fitness", ""))
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.set_xticks([-0.25, 0.5, 1.25])
        ax.set_xticklabels(["cyto\nflank", "centre", "lumen\nflank"], fontsize=6)
        ax.set_ylabel("Normalised fitness" if k == 0 else "")
        ax.set_title(f"({'ab'[k]}) {lab}", loc="left", fontsize=8.5)
    a0, a1 = fig.axes[0], fig.axes[1]
    lo = min(a0.get_ylim()[0], a1.get_ylim()[0]); hi = max(a0.get_ylim()[1], a1.get_ylim()[1])
    a0.set_ylim(lo, hi); a1.set_ylim(lo, hi)
    ax = fig.add_subplot(gs[0, 2])
    pp = per_protein.sort_values("estimate")
    forest(ax, [p.replace("_fitness", "") for p in pp.protein],
           pp.estimate, pp.estimate - 1.96 * pp.se, pp.estimate + 1.96 * pp.se,
           pooled=(res["meta_mu"], res["meta_lo"], res["meta_hi"]),
           pooled_label=f"meta, I² = {res['meta_I2']:.0f}%", fontsize=7)
    ax.set_xlabel("depth × basic interaction", fontsize=7)
    ax.set_title("(c) Interaction per protein", loc="left", fontsize=8.5)
    h, lb = a0.get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", ncol=min(len(lb), 8), frameon=False, fontsize=6.5)
    compare_box(fig.add_subplot(gs[1, 2]), res, "Depth × basic, pooled")
    P.save(fig, outdir, "g02_depth_profile", None, "G02")
    plt.close(fig)
    cur.to_csv(outdir / "g02_curves.csv", index=False)
    per_protein.to_csv(outdir / "g02_per_protein.csv", index=False)
    res["status"] = "ok"
    res["headline"] = (f"depth × basic: meta {res['meta_mu']:+.3f} "
                       f"[{res['meta_lo']:+.3f}, {res['meta_hi']:+.3f}] over {res['meta_k']} proteins, "
                       f"mixed {res.get('mixed_coef', float('nan')):+.3f}; I² = {res['meta_I2']:.0f}%")
    return res


# -------------------------------------------------------- g03 aromatic belt
def g03_aromatic_belt(frames: dict, meta: pd.DataFrame, outdir: Path) -> dict:
    """Are aromatic wild-type residues harder to replace at the membrane interface?

    The belt prediction is about position, so the control has to be position too: the
    same interface-minus-core contrast is computed for non-aromatic wild types in the
    same helices, and the reported estimate is the difference between the two. Without
    that control the panel would mostly show that the interface differs from the core
    for every residue, which is a different claim.
    """
    BELT = 0.18          # within 18% of either end counts as interface
    rows, prof = [], []
    for name, a in frames.items():
        d = a[a.in_membrane].copy()
        if not len(d):
            continue
        d["interface"] = (d.x <= BELT) | (d.x >= 1 - BELT)
        d["aromatic"] = d.wt.isin(list(AROMATIC))
        pos = (d.groupby(["pos", "interface", "aromatic"], observed=True)
               .score_z.median().reset_index())
        cells = {k: pos.loc[(pos.aromatic == k[0]) & (pos.interface == k[1]), "score_z"]
                 for k in [(True, True), (True, False), (False, True), (False, False)]}
        if min(len(v) for v in cells.values()) < 5:
            continue
        # (aromatic interface − aromatic core) − (other interface − other core)
        est = ((cells[(True, True)].mean() - cells[(True, False)].mean())
               - (cells[(False, True)].mean() - cells[(False, False)].mean()))
        se = float(np.sqrt(sum(v.var(ddof=1) / len(v) for v in cells.values())))
        rows.append({"protein": name, "estimate": float(est), "se": se,
                     "n_aromatic_positions": int(len(cells[(True, True)]) + len(cells[(True, False)]))})
        g = d.groupby([pd.cut(d.x, np.linspace(0, 1, 9), include_lowest=True), "aromatic"],
                      observed=False).score_z.agg(["mean", "sem"]).reset_index()
        g["protein"] = name
        g["xmid"] = [iv.mid for iv in g.iloc[:, 0]]
        prof.append(g[["protein", "aromatic", "xmid", "mean", "sem"]])
    per_protein = pd.DataFrame(rows)
    if len(per_protein) < 3:
        return {"status": "skipped", "reason": "too few proteins with aromatic positions in both zones"}

    allv = pd.concat(frames.values())
    d = allv[allv.in_membrane].copy()
    d["interface"] = ((d.x <= BELT) | (d.x >= 1 - BELT)).astype(float)
    d["aromatic"] = d.wt.isin(list(AROMATIC)).astype(float)
    res = both_ways(per_protein, mixed("score_z ~ interface * aromatic", d, "protein",
                                       "interface:aromatic", vc=VC_HELIX))

    fig = plt.figure(figsize=(10.2, 5.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.34])
    ax = fig.add_subplot(gs[:, 0])
    pr = pd.concat(prof, ignore_index=True)
    for arom, col, lab in ((True, "#B8912F", "aromatic WT (W/Y/F)"),
                           (False, "#2F6DB5", "all other WT")):
        g = pr[pr.aromatic == arom].groupby("xmid")["mean"].agg(["mean", "sem"]).reset_index()
        ax.plot(g.xmid, g["mean"], "-o", ms=3.2, lw=1.2, color=col, label=lab)
        ax.fill_between(g.xmid, g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=col, alpha=0.18, lw=0)
    for b in (BELT, 1 - BELT):
        ax.axvline(b, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel("Position across the membrane (0 = cytosolic end)")
    ax.set_ylabel("Mean normalised fitness")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center")
    ax.set_title(f"(a) Mean across {len(frames)} proteins (±SEM between proteins); "
                 f"dashes = interface band", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 1])
    pp = per_protein.sort_values("estimate")
    forest(ax, [p.replace("_fitness", "") for p in pp.protein],
           pp.estimate, pp.estimate - 1.96 * pp.se, pp.estimate + 1.96 * pp.se,
           pooled=(res["meta_mu"], res["meta_lo"], res["meta_hi"]),
           pooled_label=f"meta, I² = {res['meta_I2']:.0f}%", fontsize=7)
    ax.set_xlabel("(aromatic interface − core) − (other interface − core)", fontsize=6.5)
    ax.set_title("(b) Belt effect, composition-controlled", loc="left", fontsize=8.5)
    compare_box(fig.add_subplot(gs[1, 1]), res, "Interface × aromatic")
    P.save(fig, outdir, "g03_aromatic_belt", None, "G03")
    plt.close(fig)
    per_protein.to_csv(outdir / "g03_per_protein.csv", index=False)
    pd.concat(prof, ignore_index=True).to_csv(outdir / "g03_profiles.csv", index=False)
    res["status"] = "ok"
    res["headline"] = (f"aromatic belt (interface × aromatic): meta {res['meta_mu']:+.3f} "
                       f"[{res['meta_lo']:+.3f}, {res['meta_hi']:+.3f}], "
                       f"mixed {res.get('mixed_coef', float('nan')):+.3f}; I² = {res['meta_I2']:.0f}%")
    return res


# --------------------------------------------------- g04 variance partition
def nested_variance(d: pd.DataFrame) -> dict:
    """Variance of transmembrane effect at four nested levels, by moments.

    A mixed model with one random effect per position asks for thousands of columns per
    group and does not finish, so the components are taken from the nested means directly:
    the spread of protein means, of helix means within a protein, of position means within
    a helix, and of substitutions within a position. Unbalanced group sizes make these
    estimates approximate, which is why the mixed-model answer for the two levels that are
    tractable is reported beside them.
    """
    site = d.groupby(["protein", "helix", "pos"]).score_z.agg(["mean", "var", "size"])
    within = float(np.average(site["var"].dropna(),
                              weights=site.loc[site["var"].notna(), "size"]))
    hel = site.groupby(["protein", "helix"])["mean"].agg(["mean", "var", "size"])
    pos_in_helix = float(hel["var"].dropna().mean())
    prot = hel.groupby("protein")["mean"].agg(["mean", "var"])
    helix_in_protein = float(prot["var"].dropna().mean())
    between_protein = float(prot["mean"].var(ddof=1))
    return {"protein": between_protein, "helix within protein": helix_in_protein,
            "position within helix": pos_in_helix, "substitution (residual)": within}


def g04_variance_partition(frames: dict, meta: pd.DataFrame, outdir: Path) -> dict:
    """Where does the variation in transmembrane mutational effect actually sit?

    Between proteins, between helices within a protein, between positions within a helix,
    or between substitutions at one position. This decides the right unit of replication
    for every other question here: if nearly all of it is within-position, a test over
    eight proteins carries far less information than its variant count suggests.
    """
    import statsmodels.formula.api as smf
    d = pd.concat(frames.values(), ignore_index=True)
    d = d[d.in_membrane].dropna(subset=["score_z"]).copy()
    if len(d) < 200 or d.protein.nunique() < 2:
        return {"status": "skipped", "reason": "not enough transmembrane variants"}
    comp = nested_variance(d)
    tot = sum(comp.values())
    frac = {k: v / tot for k, v in comp.items()}

    # the two levels a mixed model can reach at this size, as a check on the moments
    mm, note = {}, ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.mixedlm("score_z ~ 1", d, groups=d["protein"], vc_formula=VC_HELIX
                            ).fit(reml=True, method="lbfgs")
        mm = {"protein": float(m.cov_re.iloc[0, 0]), "helix within protein": float(m.vcomp[0]),
              "position + substitution": float(m.scale)}
        note = "" if m.converged else "mixed model did not converge"
    except Exception as e:
        note = f"mixed model unavailable: {type(e).__name__}"
    mtot = sum(mm.values()) if mm else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ks = list(comp)
    ax.barh(np.arange(len(ks))[::-1], [frac[k] for k in ks],
            color=["#1B4B80", "#B8912F", "#2F6DB5", "#BBBBBB"], height=0.62)
    for i, k in enumerate(ks):
        ax.text(frac[k] + 0.012, len(ks) - 1 - i, f"{frac[k]:.0%}", va="center", fontsize=7)
    ax.set_yticks(np.arange(len(ks))[::-1])
    ax.set_yticklabels(ks, fontsize=7)
    ax.set_xlim(0, max(frac.values()) * 1.3)
    ax.set_xlabel("Share of variance in transmembrane missense effect")
    ax.set_title("(a) Nested variance components", loc="left", fontsize=8.5)
    ax = axes[1]
    txt = ["level                        moments     mixed model", "-" * 54]
    for k in ks:
        mv = (f"{mm[k] / mtot:>11.1%}" if k in mm and np.isfinite(mtot) else
              (f"{mm['position + substitution'] / mtot:>11.1%}"
               if k == "position within helix" and mm else f"{'':>11}"))
        txt.append(f"{k:<28}{frac[k]:>7.1%}{mv}")
    if mm:
        txt.append(f"{'(mixed pools the last two levels)':<28}")
    txt += ["", f"n = {len(d):,} TM variants · {d.protein.nunique()} proteins · "
                f"{d.helix.nunique()} helices · "
                f"{d.groupby(['protein', 'pos']).ngroups:,} positions", "",
            "The unit of replication is the level that carries the",
            "variance, not the number of variants measured."]
    if note:
        txt += ["", f"NOTE: {note}"]
    ax.text(0, 1, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.6, family="monospace", linespacing=1.55)
    ax.axis("off")
    ax.set_title("(b) Moments against the mixed model", loc="left", fontsize=8.5)
    P.save(fig, outdir, "g04_variance_partition", None, "G04")
    plt.close(fig)
    pd.DataFrame({"level": ks, "variance": [comp[k] for k in ks],
                  "fraction": [frac[k] for k in ks]}).to_csv(
        outdir / "g04_variance_components.csv", index=False)
    top = max(frac, key=frac.get)
    return {"status": "ok", "components": comp, "fractions": frac, "mixed": mm, "note": note,
            "headline": f"most TM-effect variance sits at '{top}' ({frac[top]:.0%}); "
                        f"protein accounts for {frac['protein']:.0%}, "
                        f"position within helix {frac['position within helix']:.0%}"}


# ------------------------------------------------- g05 helix position in bundle
def g05_helix_position(frames: dict, meta: pd.DataFrame, outdir: Path) -> dict:
    """Are terminal helices more tolerant than central ones, consistently across proteins?"""
    rows, per = [], []
    for name, a in frames.items():
        d = a[a.in_membrane & a.n_helix.notna()]
        if not len(d):
            continue
        n_max = int(d.n_helix.max())
        sites = d.groupby(["n_helix", "pos"]).score_z.median().reset_index()
        for h, g in sites.groupby("n_helix"):
            rows.append({"protein": name, "n_helix": int(h), "n_positions": len(g),
                         "median": float(g.score_z.median()), "mean": float(g.score_z.mean()),
                         "terminal": int(h) in (1, n_max)})
        term = sites[sites.n_helix.isin([1, n_max])].score_z
        cent = sites[~sites.n_helix.isin([1, n_max])].score_z
        if len(term) < 5 or len(cent) < 5:
            continue
        est = float(term.mean() - cent.mean())
        se = float(np.sqrt(term.var(ddof=1) / len(term) + cent.var(ddof=1) / len(cent)))
        per.append({"protein": name, "estimate": est, "se": se,
                    "p": float(ss.mannwhitneyu(term, cent, alternative="two-sided").pvalue)})
    by_helix = pd.DataFrame(rows)
    per_protein = pd.DataFrame(per)
    if len(per_protein) < 3:
        return {"status": "skipped", "reason": "too few proteins with both terminal and central helices"}

    allv = pd.concat(frames.values())
    d = allv[allv.in_membrane & allv.n_helix.notna()].copy()
    nmax = d.groupby("protein").n_helix.transform("max")
    d["terminal"] = ((d.n_helix == 1) | (d.n_helix == nmax)).astype(float)
    res = both_ways(per_protein, mixed("score_z ~ terminal", d, "protein", "terminal",
                                       vc=VC_HELIX))

    fig = plt.figure(figsize=(10.2, 5.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 0.34])
    ax = fig.add_subplot(gs[:, 0])
    pal = dict(zip(sorted(frames), plt.get_cmap("tab10").colors))
    for name, g in by_helix.groupby("protein"):
        g = g.sort_values("n_helix")
        ax.plot(g.n_helix, g["median"], "-o", ms=3, lw=0.9, color=pal[name],
                label=name.replace("_fitness", ""))
    m = by_helix.groupby("n_helix")["median"].agg(["mean", "sem"]).reset_index()
    ax.errorbar(m.n_helix, m["mean"], yerr=m["sem"], fmt="-s", ms=5, lw=2.0,
                color=P.FIT_RED, capsize=0, zorder=5, label="mean of proteins")
    ax.set_xlabel("Transmembrane helix number, N to C")
    ax.set_ylabel("Median position effect")
    ax.legend(frameon=False, fontsize=6, ncol=3, loc="lower center")
    ax.set_title("(a) Tolerance along the bundle", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[0, 1])
    pp = per_protein.sort_values("estimate")
    forest(ax, [p.replace("_fitness", "") for p in pp.protein],
           pp.estimate, pp.estimate - 1.96 * pp.se, pp.estimate + 1.96 * pp.se,
           pooled=(res["meta_mu"], res["meta_lo"], res["meta_hi"]),
           pooled_label=f"meta, I² = {res['meta_I2']:.0f}%", fontsize=7)
    ax.set_xlabel("terminal − central (positive = ends more tolerant)", fontsize=6.5)
    ax.set_title("(b) First and last vs the rest", loc="left", fontsize=8.5)
    compare_box(fig.add_subplot(gs[1, 1]), res, "Terminal − central helices")
    P.save(fig, outdir, "g05_helix_position", None, "G05")
    plt.close(fig)
    by_helix.to_csv(outdir / "g05_by_helix.csv", index=False)
    per_protein.to_csv(outdir / "g05_per_protein.csv", index=False)
    res["status"] = "ok"
    res["headline"] = (f"terminal − central helices: meta {res['meta_mu']:+.3f} "
                       f"[{res['meta_lo']:+.3f}, {res['meta_hi']:+.3f}], "
                       f"mixed {res.get('mixed_coef', float('nan')):+.3f}; I² = {res['meta_I2']:.0f}%")
    return res


ANALYSES = {"g01": g01_positive_inside, "g02": g02_depth_profile, "g03": g03_aromatic_belt,
            "g04": g04_variance_partition, "g05": g05_helix_position}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|singular|boundary|No artists|invalid value).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms aggregate",
        description="Transmembrane questions asked of every protein at once.")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--only", help=f"comma-separated keys, default all: {','.join(ANALYSES)}")
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a = ap.parse_args(argv)
    P.use_style(a.style)
    keys = [k.strip() for k in (a.only.split(",") if a.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")

    frames, rows = {}, []
    for c in a.configs:
        cfg = load_config(Path(c))
        df = load_dataset(cfg)
        f = tm_frame(cfg, df)
        rows.append({"protein": cfg.id, "n_variants": len(df), "syn_sd": syn_sd(df),
                     "n_tm_helices": int(f.tm.nunique()) if len(f) else 0,
                     "normalisation_fallback": bool(df.attrs.get("normalization_fallback")),
                     "assay": cfg.get_path("assay.type"),
                     "topology_source": cfg.get_path("topology.source")})
        if len(f):
            frames[cfg.id] = f
        print(f"  {cfg.id:<16} {len(df):>7,} variants  "
              f"{rows[-1]['n_tm_helices']:>3} TM helices  syn SD {rows[-1]['syn_sd']:.2f}"
              + ("   NORMALISATION FALLBACK" if rows[-1]["normalisation_fallback"] else ""), flush=True)
    meta = pd.DataFrame(rows)
    if len(frames) < 2:
        raise SystemExit("need at least two proteins with TM helices")

    # Pooling datasets of different quality is only honest if the differences are visible.
    bad_norm = meta[meta.normalisation_fallback].protein.tolist()
    if bad_norm:
        print(f"\n  WARNING: {', '.join(bad_norm)} used a fallback anchor - their scores are NOT "
              "on the syn=0/stop=-1 scale and should not be pooled with the rest")
    if meta.assay.nunique() > 1:
        print(f"  WARNING: more than one assay type present ({', '.join(map(str, meta.assay.unique()))})")
    noisy = meta[meta.syn_sd > 0.25].protein.tolist()
    if noisy:
        print(f"  WARNING: {', '.join(noisy)} have synonymous SD > 0.25; a noisy assay shrinks "
              "every effect toward zero")

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(outdir / "proteins.csv", index=False)
    out = {}
    print()
    for k in keys:
        try:
            r = ANALYSES[k](frames, meta, outdir)
        except Exception as e:
            r = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        out[k] = r
        print(f"  {k}  {r.get('status'):<8} {r.get('headline') or r.get('reason', '')}", flush=True)
    (outdir / "aggregate.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
