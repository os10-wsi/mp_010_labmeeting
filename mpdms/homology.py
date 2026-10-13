"""`python -m mpdms homolog configs/HXT1.yaml configs/HXT3.yaml`

Two homologues and two symmetries, asked together.

An MFS transporter carries its own internal replicate: two six-helix bundles related by
a pseudo-twofold and joined by a cytosolic loop, so TM1 and TM7 are the same helix built
twice. Two paralogues give a second replicate of a different kind - the same fold under
a different sequence lineage. Having both at once is what makes these ten analyses
possible, and h06 is the one that needs both: it asks whether a position is more like
its own symmetry mate or more like its orthologue, which is a question about whether
constraint is set by the architecture or by the lineage.

    h01  the whole protein, against a topology-matched null and a split-half ceiling
    h02  per helix: is the pattern of constraint along a helix conserved?
    h03  per helix: are the helix means conserved? - a different question from h02
    h04  the six symmetry mates, pair by pair, against how well each pair superposes
    h05  the bundle seen down the membrane normal, with the mates joined
    h06  fold or lineage: your symmetry mate, or your orthologue?
    h07  where the two proteins disagree, and whether the disagreement is in patches
    h08  does residue identity at a column predict agreement at it?
    h09  substitution profiles, not just position means
    h10  transfer as a prediction task: how much would you trust the other protein?

Correlations are scale-free, so unlike the violin figures everything here is comparable
across proteins even when one dataset was normalised against a wild-type anchor rather
than synonymous variants. The thing that is not free is the ceiling: two position means
measured with noise cannot correlate at 1 even if the biology is identical, so h01
estimates the attainable maximum by split-half and every correlation below is reported
against it.
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
from .annot import AA20
from .architecture import kabsch, protein_bits
from .burial import empty_panel, short_name
from .config import REPO_ROOT, load_config, protein_sequence
from .environs import effect_color, grid, panel
from .family import (align_sequences, aligned_table, protein_table,
                     substitution_profile_correlation)
from .io import load_dataset

OUT = REPO_ROOT / "outputs" / "_homolog"
MIN_PER_HELIX = 6        # aligned positions inside a helix, before it gets a number
MIN_VARIANTS = 5         # per position, before its mean is used
N_PERM = 2000
N_BOOT = 1000
N_SPLIT = 200            # split-half repeats for the reliability ceiling
WORST_FRAC = 0.10        # "in the worst tenth" for the h10 transfer AUROC
BUNDLE_COLORS = {"N": "#2166AC", "C": "#B2182B"}


# --------------------------------------------------------------------- loading
def split_half_reliability(var: pd.DataFrame, n_rep: int = N_SPLIT,
                           seed: int = 0, min_positions: int = 20) -> float:
    """How well a position's mean effect correlates with itself, measured twice.

    Each position's variants are split at random into two halves and the two sets of
    position means are correlated; the Spearman-Brown correction turns the reliability
    of a half-sized measurement into the reliability of the whole one. This is the
    number every between-protein correlation has to be read against: two perfectly
    equivalent positions measured with noise cannot correlate at one, and without the
    ceiling a modest correlation cannot be told from a divergent protein.
    """
    d = var.dropna(subset=["score_z"])[["pos", "score_z"]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        side = rng.integers(0, 2, len(d))
        a = d[side == 0].groupby("pos").score_z.agg(["mean", "size"])
        b = d[side == 1].groupby("pos").score_z.agg(["mean", "size"])
        j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
        j = j[(j["size_a"] >= 2) & (j["size_b"] >= 2)]
        if len(j) < min_positions:
            continue
        r = ss.spearmanr(j["mean_a"], j["mean_b"])[0]
        if np.isfinite(r):
            out.append(r)
    if not out:
        return np.nan
    half = float(np.mean(out))
    # Spearman-Brown: doubling the number of observations behind each mean
    return float(2 * half / (1 + half)) if half > -1 else np.nan


def ceiling(rel_a: float, rel_b: float) -> float:
    """The largest correlation two noisy measurements of identical quantities could show."""
    if not (np.isfinite(rel_a) and np.isfinite(rel_b)):
        return np.nan
    return float(np.sqrt(max(rel_a, 0.0) * max(rel_b, 0.0)))


def normal_basis(rt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The membrane normal and two axes across it, recovered from the frame's own output.

    `membrane_frame` hands back the depth of each residue but not the direction it
    measured along, so the direction is read back out by regressing depth on the
    coordinates: the coefficients are the normal. Two arbitrary perpendicular axes
    complete the basis, which is all a top-down view of the bundle needs.
    """
    d = rt.dropna(subset=["zdepth"])
    A = np.c_[d[["x", "y", "z"]].to_numpy(float), np.ones(len(d))]
    beta = np.linalg.lstsq(A, d.zdepth.to_numpy(float), rcond=None)[0]
    n = beta[:3]
    n = n / (np.linalg.norm(n) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - n * (seed @ n)
    u /= np.linalg.norm(u) or 1.0
    return n, u, np.cross(n, u)


def tm_names(cfg) -> list[str]:
    segs = [s for s in (cfg.get_path("topology.segments", []) or [])
            if s.get("type") == "TM"]
    return [s["name"] for s in sorted(segs, key=lambda s: s["start"])]


def symmetry_pairs(bits: dict, canonical: bool = False, n_first: int = 6) -> dict:
    """The six structurally equivalent helix pairs, with a superposition for each.

    The canonical MFS repeat pairs TM1 with TM7, TM2 with TM8 and so on. By default the
    pairing is searched rather than assumed - every one-to-one correspondence of the
    first six helices against the second six is scored by a Kabsch fit on the helix
    centroids - and whether the answer is the textbook one is reported alongside.
    Residues inside a paired helix are matched by rank along it, which is what makes
    helices of different length comparable at all.
    """
    from itertools import permutations
    order = tm_names(bits["cfg"])
    if len(order) < 2 * n_first:
        return {}
    first, second = order[:n_first], order[n_first:2 * n_first]
    rt = bits["rt"].set_index("pos")
    members = {h: [p for p in sorted(q for q, hh in bits["helix"].items() if hh == h)
                   if p in rt.index] for h in order}
    if any(len(members[h]) < 5 for h in first + second):
        return {}
    cen = {h: rt.loc[members[h], ["x", "y", "z"]].to_numpy().mean(0) for h in order}
    canon = tuple(range(n_first))
    if canonical:
        perm = canon
    else:
        best = None
        for q in permutations(range(n_first)):
            a = np.array([cen[first[k]] for k in range(n_first)])
            b = np.array([cen[second[q[k]]] for k in range(n_first)])
            rms = kabsch(a, b)[3]
            if best is None or rms < best[0]:
                best = (rms, q)
        perm = best[1]
    pairs = []
    for k in range(n_first):
        ha, hb = first[k], second[perm[k]]
        ia, ib = members[ha], members[hb]
        n = min(len(ia), len(ib))
        qa = [ia[j] for j in np.linspace(0, len(ia) - 1, n).round().astype(int)]
        qb = [ib[j] for j in np.linspace(0, len(ib) - 1, n).round().astype(int)]
        ca = rt.loc[qa, ["x", "y", "z"]].to_numpy()
        cb = rt.loc[qb, ["x", "y", "z"]].to_numpy()
        pairs.append({"helix_a": ha, "helix_b": hb, "pos_a": qa, "pos_b": qb,
                      "rmsd": float(kabsch(ca, cb)[3]), "n": int(n)})
    return {"pairs": pairs, "canonical": perm == canon, "centroids": cen,
            "order": order, "first": first, "second": second, "members": members}


def bits_for(cfg) -> dict:
    """Everything the ten analyses need from one protein, loaded once."""
    b = protein_bits(cfg)
    if not b:
        return {}
    df = load_dataset(cfg)
    b["ptable"] = protein_table(df, cfg)
    v = missense(df).dropna(subset=["score_z"])[["pos", "wt", "mut", "score_z"]]
    b["variants"] = v
    b["reliability"] = split_half_reliability(v)
    b["mate"] = {}
    sym = symmetry_pairs(b)
    b["symmetry"] = sym
    for pr in sym.get("pairs", []):
        for x, y in zip(pr["pos_a"], pr["pos_b"]):
            b["mate"][x] = y
            b["mate"][y] = x
    return b


def bootstrap_rho(x, y, n: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    obs = float(ss.spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3:
            continue
        r = ss.spearmanr(x[i], y[i])[0]
        if np.isfinite(r):
            b.append(r)
    if len(b) < 50:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(b, [2.5, 97.5])
    return obs, float(lo), float(hi)


def auroc(pos_scores, neg_scores) -> float:
    """P(a positive outranks a negative), ties counted as half."""
    a, b = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


# ------------------------------------------------ h01 the whole protein, with a ceiling
def h01_whole_protein(t: pd.DataFrame, a: str, b: str, bits: dict,
                      outdir: Path) -> dict:
    """The baseline every other panel is read against.

    Three numbers rather than one. The raw correlation says how much the two proteins
    agree. The within-class permutation null says how much of that is topology - both
    are transporters with sensitive cores and tolerant loops, so a positive correlation
    over all positions is guaranteed before any biology. The split-half ceiling says how
    much agreement the measurements could possibly have shown, and the ratio of the
    first to the third is the only one of the three that answers "how conserved is it".
    """
    from .family import _within_class_null
    rows = []
    subsets = {"all aligned": t.both_residues, "both TM": t.both_tm,
               "both loop/soluble": t.both_loop}
    cap = ceiling(bits[a]["reliability"], bits[b]["reliability"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")
    d = t[t.both_residues].dropna(subset=[f"{a}_mean_all", f"{b}_mean_all"])
    ax = axes[0]
    tm = d[d.both_tm]
    ax.scatter(d[f"{a}_mean_all"], d[f"{b}_mean_all"], s=10, color="#D9D9D9", lw=0,
               label="other aligned positions")
    ax.scatter(tm[f"{a}_mean_all"], tm[f"{b}_mean_all"], s=14,
               c=[effect_color((x + y) / 2) for x, y in
                  zip(tm[f"{a}_mean_all"], tm[f"{b}_mean_all"])],
               edgecolor=P.INK, linewidth=0.3, label="both in a TM helix")
    lim = [min(d[f"{a}_mean_all"].min(), d[f"{b}_mean_all"].min()) - 0.05,
           max(d[f"{a}_mean_all"].max(), d[f"{b}_mean_all"].max()) + 0.05]
    ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{short_name(a)} mean effect", fontsize=7.5)
    ax.set_ylabel(f"{short_name(b)} mean effect", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    for lab, mask in subsets.items():
        obs, p, _, n = _within_class_null(t, a, b, mask, n=N_PERM)
        if not np.isfinite(obs):
            continue
        _, lo, hi = bootstrap_rho(t.loc[mask, f"{a}_mean_all"],
                                  t.loc[mask, f"{b}_mean_all"])
        rows.append({"subset": lab, "n": int(n), "spearman": obs, "lo": lo, "hi": hi,
                     "p_within_class_null": p, "ceiling": cap,
                     "fraction_of_ceiling": obs / cap if cap else np.nan})
    key = [r for r in rows if r["subset"] == "all aligned"]
    panel(ax, 0, f"{short_name(a)} vs {short_name(b)}",
          (f"ρ = {key[0]['spearman']:+.2f}" if key else ""), size=8)

    ax = axes[1]
    y = np.arange(len(rows))[::-1]
    for k, r in enumerate(rows):
        ax.errorbar(r["spearman"], y[k],
                    xerr=[[max(r["spearman"] - (r["lo"] or r["spearman"]), 0)],
                          [max((r["hi"] or r["spearman"]) - r["spearman"], 0)]],
                    fmt="o", ms=6, lw=0, elinewidth=1.1, capsize=3, color=P.DARK_RED,
                    mec=P.INK, mew=0.5)
        ax.text(r["spearman"], y[k] + 0.22,
                f"{100 * r['fraction_of_ceiling']:.0f}% of attainable"
                if np.isfinite(r["fraction_of_ceiling"]) else "", fontsize=6,
                ha="center")
    if np.isfinite(cap):
        ax.axvline(cap, color=P.INK, lw=1.3, ls=(0, (4, 3)))
        # in axis-fraction coordinates on y: the row positions are data coordinates and
        # a label placed above the top row lands outside the axes and is clipped
        ax.text(cap, 0.99, "split-half ceiling ", fontsize=6, va="top", ha="right",
                transform=ax.get_xaxis_transform())
    ax.axvline(0, color=P.MUTED, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r["subset"] for r in rows], fontsize=7)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Spearman ρ between the two proteins", fontsize=7.5)
    panel(ax, 1, "against the attainable maximum", size=8)
    fig.suptitle("The ceiling is what two noisy measurements of identical quantities "
                 "could correlate at; the p-values are against a null that shuffles "
                 "within TM and within loop separately.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h01_whole_protein", None, "H01")
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "h01_whole_protein.csv", index=False)
    head = "; ".join(f"{r.subset} ρ {r.spearman:+.2f} "
                     f"({100 * r.fraction_of_ceiling:.0f}% of ceiling {r.ceiling:.2f}, "
                     f"p {r.p_within_class_null:.3f})" for r in out.itertuples())
    return {"status": "ok", "headline": head, "subsets": out.to_dict("records")}


# --------------------------------------------------- h02 per helix, the pattern
def helix_columns(t: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Aligned columns inside a TM helix, labelled by the helix they are in.

    A column is kept only where the two proteins put it in helices that correspond:
    the pairing between helix names is taken as the majority mapping observed in the
    alignment rather than assumed from the numbering, so a protein with an extra
    predicted segment cannot silently shift every label by one.
    """
    d = t[t.both_tm].dropna(subset=[f"{a}_helix", f"{b}_helix",
                                    f"{a}_mean_all", f"{b}_mean_all"]).copy()
    if not len(d):
        return d.assign(helix=[])
    maj = (d.groupby([f"{a}_helix", f"{b}_helix"]).size().rename("n").reset_index()
           .sort_values("n", ascending=False).drop_duplicates(f"{a}_helix"))
    keep = set(zip(maj[f"{a}_helix"], maj[f"{b}_helix"]))
    d = d[[(x, y) in keep for x, y in zip(d[f"{a}_helix"], d[f"{b}_helix"])]]
    d["helix"] = d[f"{a}_helix"]
    d.attrs["mapping"] = maj.to_dict("records")
    return d


def h02_per_helix(t: pd.DataFrame, a: str, b: str, bits: dict, outdir: Path) -> dict:
    """Is the pattern of constraint *along* a helix conserved between the two proteins?

    One correlation per helix, over the aligned positions inside it. A helix with few
    measured positions will have a wide interval and a low ceiling of its own, so the
    ceiling is drawn per helix too: a helix is only divergent if it falls short of what
    its own data could have shown.
    """
    d = helix_columns(t, a, b)
    if not len(d):
        return {"status": "skipped", "reason": "no aligned TM column in a named helix"}
    order = tm_names(bits[a]["cfg"])
    rows = []
    for h, sub in d.groupby("helix"):
        if len(sub) < MIN_PER_HELIX:
            continue
        rho, lo, hi = bootstrap_rho(sub[f"{a}_mean_all"], sub[f"{b}_mean_all"])
        pa = set(sub[f"{a}_pos"].dropna().astype(int))
        pb = set(sub[f"{b}_pos"].dropna().astype(int))
        ra = split_half_reliability(bits[a]["variants"].query("pos in @pa"), 60,
                                    min_positions=MIN_PER_HELIX)
        rb = split_half_reliability(bits[b]["variants"].query("pos in @pb"), 60,
                                    min_positions=MIN_PER_HELIX)
        cap = ceiling(ra, rb)
        rows.append({"helix": h, "n": int(len(sub)), "spearman": rho, "lo": lo,
                     "hi": hi, "ceiling": cap,
                     "fraction_of_ceiling": rho / cap if cap and np.isfinite(cap)
                     else np.nan,
                     "mean_a": float(sub[f"{a}_mean_all"].mean()),
                     "mean_b": float(sub[f"{b}_mean_all"].mean())})
    if not rows:
        return {"status": "skipped", "reason": f"no helix has {MIN_PER_HELIX} aligned columns"}
    per = pd.DataFrame(rows)
    per["rank"] = [order.index(h) if h in order else 99 for h in per.helix]
    per = per.sort_values("rank")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), layout="constrained",
                             width_ratios=[1.35, 1])
    ax = axes[0]
    x = np.arange(len(per))
    ax.bar(x, per.spearman,
           yerr=[per.spearman - per.lo.fillna(per.spearman),
                 per.hi.fillna(per.spearman) - per.spearman],
           color=[effect_color((r.mean_a + r.mean_b) / 2) for r in per.itertuples()],
           edgecolor=P.INK, linewidth=0.5, error_kw=dict(lw=0.7, capsize=2))
    ax.plot(x, per.ceiling, "_", ms=12, color=P.INK, mew=1.4)
    ax.set_xticks(x); ax.set_xticklabels(per.helix, rotation=90, fontsize=6)
    ax.axhline(0, color=P.MUTED, lw=0.7)
    ax.set_ylabel("ρ between the two proteins, within the helix", fontsize=7.5)
    panel(ax, 0, "in sequence order", "tick = that helix's own ceiling", size=8)

    ax = axes[1]
    s = per.sort_values("spearman")
    y = np.arange(len(s))
    ax.barh(y, s.spearman, color=[effect_color((r.mean_a + r.mean_b) / 2)
                                  for r in s.itertuples()],
            edgecolor=P.INK, linewidth=0.5)
    ax.plot(s.ceiling, y, "|", ms=10, color=P.INK, mew=1.4)
    ax.set_yticks(y); ax.set_yticklabels(s.helix, fontsize=6)
    ax.axvline(0, color=P.MUTED, lw=0.7)
    ax.set_xlabel("ρ, ranked", fontsize=7.5)
    panel(ax, 1, "ranked", size=8)
    fig.suptitle(f"{short_name(a)} against {short_name(b)} at aligned positions inside "
                 "each TM helix. Bars coloured by the helix's mean effect across both "
                 "proteins.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h02_per_helix_pattern", None, "H02")
    plt.close(fig)
    per.to_csv(outdir / "h02_per_helix_pattern.csv", index=False)
    best = per.loc[per.spearman.idxmax()]
    worst = per.loc[per.spearman.idxmin()]
    return {"status": "ok",
            "headline": f"most conserved pattern {best.helix} (ρ {best.spearman:+.2f}), "
                        f"least {worst.helix} (ρ {worst.spearman:+.2f}); median "
                        f"{per.spearman.median():+.2f} against median ceiling "
                        f"{per.ceiling.median():.2f}",
            "by_helix": per.to_dict("records")}


# ----------------------------------------------------- h03 per helix, the means
def h03_helix_means(t: pd.DataFrame, a: str, b: str, bits: dict, outdir: Path) -> dict:
    """Are the helix *means* conserved - a different question from h02.

    A helix can be the most sensitive one in both proteins while the arrangement of
    sensitivity inside it agrees not at all, and the reverse is just as possible. h02
    is about the pattern within a helix and this is about the level of the helix as a
    whole; putting the two side by side is the point of having both.
    """
    d = helix_columns(t, a, b)
    if not len(d):
        return {"status": "skipped", "reason": "no aligned TM column in a named helix"}
    g = d.groupby("helix").agg(n=("helix", "size"),
                               mean_a=(f"{a}_mean_all", "mean"),
                               mean_b=(f"{b}_mean_all", "mean"),
                               se_a=(f"{a}_mean_all", "sem"),
                               se_b=(f"{b}_mean_all", "sem")).reset_index()
    g = g[g.n >= MIN_PER_HELIX]
    if len(g) < 4:
        return {"status": "skipped", "reason": "fewer than four comparable helices"}
    rho, lo, hi = bootstrap_rho(g.mean_a, g.mean_b)
    fig, ax = plt.subplots(figsize=(4.0, 3.8), layout="constrained")
    ax.errorbar(g.mean_a, g.mean_b, xerr=g.se_a, yerr=g.se_b, fmt="o", ms=7, lw=0,
                elinewidth=0.8, capsize=2,
                color=P.DARK_RED, mec=P.INK, mew=0.5)
    P.label_points(ax, g.mean_a.to_numpy(), g.mean_b.to_numpy(), g.helix.tolist(),
                   fontsize=7)
    lim = [min(g.mean_a.min(), g.mean_b.min()) - 0.06,
           max(g.mean_a.max(), g.mean_b.max()) + 0.06]
    ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{short_name(a)}: mean effect over the helix", fontsize=7.5)
    ax.set_ylabel(f"{short_name(b)}: mean effect over the helix", fontsize=7.5)
    panel(ax, 0, "helix means", f"ρ = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {len(g)}",
          size=8)
    fig.suptitle("One point per TM helix. This is the level of each helix; h02 is the "
                 "pattern inside it, and the two can disagree.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "h03_helix_means", None, "H03")
    plt.close(fig)
    g.to_csv(outdir / "h03_helix_means.csv", index=False)
    return {"status": "ok",
            "headline": f"helix means ρ = {rho:+.2f} [{lo:+.2f}, {hi:+.2f}] over "
                        f"{len(g)} helices",
            "by_helix": g.to_dict("records")}


# ------------------------------------------------- h04 the symmetry mates, pair by pair
def h04_symmetry_pairs(bits: dict, names: list[str], outdir: Path) -> dict:
    """The six structurally equivalent helix pairs inside each protein, one at a time.

    s07 pools all six into a single scatter and reports one correlation. That hides the
    thing worth knowing: whether the pairs that superpose well are the pairs whose
    constraint agrees. Each pair is superposed on its own here, and its correlation is
    plotted against its own Cα RMSD.
    """
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.2, h=2.9, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        sym = b.get("symmetry") or {}
        if not sym.get("pairs"):
            empty_panel(ax, i, g, "fewer than twelve annotated TM helices")
            continue
        eff = b["effect"]
        per = []
        for pr in sym["pairs"]:
            ea = np.array([eff.get(p, np.nan) for p in pr["pos_a"]])
            eb = np.array([eff.get(p, np.nan) for p in pr["pos_b"]])
            ok = np.isfinite(ea) & np.isfinite(eb)
            if ok.sum() < MIN_PER_HELIX:
                continue
            rho, lo, hi = bootstrap_rho(ea[ok], eb[ok])
            per.append({"protein": g, "pair": f"{pr['helix_a']}–{pr['helix_b']}",
                        "helix_a": pr["helix_a"], "helix_b": pr["helix_b"],
                        "n": int(ok.sum()), "rmsd": pr["rmsd"], "spearman": rho,
                        "lo": lo, "hi": hi,
                        "mean_effect": float(np.nanmean(np.r_[ea[ok], eb[ok]])),
                        "canonical_pairing": bool(sym["canonical"])})
        if not per:
            empty_panel(ax, i, g, "no helix pair with enough measured positions")
            continue
        rows += per
        d = pd.DataFrame(per)
        ax.errorbar(d.rmsd, d.spearman,
                    yerr=[d.spearman - d.lo.fillna(d.spearman),
                          d.hi.fillna(d.spearman) - d.spearman],
                    fmt="o", ms=7, lw=0, elinewidth=0.9, capsize=2.5,
                    color=P.DARK_RED, mec=P.INK, mew=0.5)
        P.label_points(ax, d.rmsd.to_numpy(), d.spearman.to_numpy(), d.pair.tolist(),
                       fontsize=6)
        ax.axhline(0, color=P.MUTED, lw=0.7)
        if len(d) >= 4:
            rr, pp = ss.spearmanr(d.rmsd, d.spearman)
            extra = f"ρ(RMSD, agreement) = {rr:+.2f} (p = {pp:.2g})"
        else:
            extra = ""
        panel(ax, i, g, extra + ("" if sym["canonical"] else "  NON-canonical pairing"),
              size=7.5)
        ax.set_xlabel("Cα RMSD of that helix pair (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("ρ of effect between the two helices", fontsize=7.5)
    fig.suptitle("Each point is one structurally equivalent helix pair within one "
                 "protein. A downward trend means the pairs that superpose best are the "
                 "pairs whose constraint agrees.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h04_symmetry_pairs", None, "H04")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no protein has usable symmetry pairs"}
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "h04_symmetry_pairs.csv", index=False)
    head = "; ".join(
        f"{short_name(g)} {len(d)} pairs, ρ {d.spearman.min():+.2f} to "
        f"{d.spearman.max():+.2f}, best {d.loc[d.spearman.idxmax(), 'pair']}"
        for g, d in per.groupby("protein"))
    return {"status": "ok", "headline": head, "by_pair": per.to_dict("records")}


# ----------------------------------------- h05 the bundle seen down the membrane normal
def h05_bundle_map(bits: dict, names: list[str], outdir: Path,
                   pair_rho: pd.DataFrame | None = None) -> dict:
    """The twelve helices where they actually are, looked at from outside the cell.

    Everything else in this module asserts that TM1 and TM7 are the same helix twice.
    This is the panel that shows it: each helix is a disc at its real position in the
    plane of the membrane, filled by its mean effect, outlined by which bundle it
    belongs to, and joined to its symmetry mate by a line whose width is how well the
    two agree. The twofold is visible rather than claimed.
    """
    rows = []
    fig, pairs_ax, _ = grid(names, w=3.3, h=3.2, extra=0.3)
    for i, (g, ax) in enumerate(pairs_ax):
        b = bits[g]
        sym = b.get("symmetry") or {}
        if not sym.get("pairs"):
            empty_panel(ax, i, g, "fewer than twelve annotated TM helices")
            continue
        rt = b["rt"]
        if "zdepth" not in rt or rt.zdepth.isna().all():
            empty_panel(ax, i, g, "no membrane frame")
            continue
        n, u, v = normal_basis(rt)
        eff = b["effect"]
        mid = np.mean([sym["centroids"][h] for h in sym["order"]], axis=0)
        xy, mean_eff = {}, {}
        for h in sym["order"]:
            c = sym["centroids"][h] - mid
            xy[h] = (float(c @ u), float(c @ v))
            vals = [eff[p] for p in sym["members"][h] if p in eff]
            mean_eff[h] = float(np.mean(vals)) if vals else np.nan
        rho_by_pair = {}
        if pair_rho is not None and len(pair_rho):
            d = pair_rho[pair_rho.protein == g]
            rho_by_pair = dict(zip(d.pair, d.spearman))
        for pr in sym["pairs"]:
            ha, hb = pr["helix_a"], pr["helix_b"]
            r = rho_by_pair.get(f"{ha}–{hb}", np.nan)
            w = 0.6 + 3.4 * max(r, 0.0) if np.isfinite(r) else 0.6
            ax.plot([xy[ha][0], xy[hb][0]], [xy[ha][1], xy[hb][1]], color=P.MUTED,
                    lw=w, alpha=0.75, zorder=1, solid_capstyle="round")
            dx = xy[hb][0] - xy[ha][0]
            dy = xy[hb][1] - xy[ha][1]
            L = float(np.hypot(dx, dy))
            if not np.isfinite(r):
                continue
            # a pair whose two helices project onto nearly the same point gets no
            # label: the number would land under a disc and be unreadable
            if L < 4.0:
                continue
            # a quarter of the way along and pushed off to one side, above everything.
            # The midpoint is where the other helices of the bundle sit, so a label
            # placed there disappears behind a disc - which is what happened first.
            t_, off = 0.26, 0.11 * L
            ax.text(xy[ha][0] + dx * t_ - dy / L * off,
                    xy[ha][1] + dy * t_ + dx / L * off,
                    f"{r:+.2f}", fontsize=5.5, ha="center", va="center", color=P.INK,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=6)
            rows.append({"protein": g, "pair": f"{ha}–{hb}", "rmsd": pr["rmsd"],
                         "spearman": r,
                         "x_a": xy[ha][0], "y_a": xy[ha][1],
                         "x_b": xy[hb][0], "y_b": xy[hb][1]})
        for h in sym["order"]:
            side = "N" if h in sym["first"] else "C"
            ax.scatter(*xy[h], s=430, c=[effect_color(mean_eff[h])],
                       edgecolor=BUNDLE_COLORS[side], linewidth=2.0, zorder=3)
            ax.text(*xy[h], h.replace("TM", ""), fontsize=6.5, ha="center", va="center",
                    zorder=4,
                    color="white" if (np.isfinite(mean_eff[h]) and mean_eff[h] < -0.55)
                    else P.INK)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():        # a map, not a plot: no axes to read
            sp.set_visible(False)
        ax.margins(0.14)
        P.no_square(ax)
        panel(ax, i, g, "viewed down the membrane normal", size=8)
        write_bundle_scripts(b, sym, outdir)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=BUNDLE_COLORS[s],
                    mew=2.0, ms=9, label=f"{s}-bundle")
         for s in ("N", "C")]
    fig.legend(handles=h, frameon=False, fontsize=6.5, loc="lower left",
               ncol=2, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Disc = one TM helix at its real position in the plane of the membrane, "
                 "filled by its mean effect. Lines join symmetry mates; the width and "
                 "the number are how well the pair agrees.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "h05_bundle_map", None, "H05")
    plt.close(fig)
    if not rows:
        return {"status": "skipped", "reason": "no protein could be drawn"}
    pd.DataFrame(rows).to_csv(outdir / "h05_bundle_map.csv", index=False)
    return {"status": "ok",
            "headline": "the twelve helices placed in the membrane plane with their "
                        "symmetry mates joined; PyMOL and ChimeraX scripts written"}


def write_bundle_scripts(b: dict, sym: dict, outdir: Path) -> list[Path]:
    """A PyMOL and a ChimeraX script that colour the two bundles and show the mates.

    Written rather than rendered: the rendering needs the program, and the script is
    the thing worth keeping, because it can be opened beside the real structure and
    turned.
    """
    from .analyses.a15_esm_functional import _ranges
    cfg = b["cfg"]
    src = b["path"]
    off = b["offset"]
    sdir = Path(outdir) / "structures" / cfg.id
    sdir.mkdir(parents=True, exist_ok=True)
    first = sorted(int(p) - off for h in sym["first"] for p in sym["members"][h])
    second = sorted(int(p) - off for h in sym["second"] for p in sym["members"][h])
    pml = [f"# run with:  cd {sdir}   then  @h05_bundles.pml   (not 'run')",
           f"load {src.resolve()}, bundle", "bg_color white", "hide everything",
           "show cartoon, bundle", "color grey80",
           f"color skyblue, bundle and resi {_ranges(first)}",
           f"color firebrick, bundle and resi {_ranges(second)}",
           "set cartoon_transparency, 0.1"]
    for pr in sym["pairs"]:
        ha, hb = pr["helix_a"], pr["helix_b"]
        ra = _ranges(sorted(int(p) - off for p in sym["members"][ha]))
        rb = _ranges(sorted(int(p) - off for p in sym["members"][hb]))
        pml.append(f"select {ha}_{hb}, bundle and resi {ra}+{rb}")
    pml += ["deselect", "orient bundle", "set ray_opaque_background, 0",
            "save h05_bundles.pse",
            "png h05_bundles.png, width=2400, height=1800, dpi=300, ray=1"]
    cxc = [f"open {src.resolve()}", "set bgColor white", "hide atoms", "show cartoons",
           f"color #1:{_ranges(first, ',')} cornflowerblue",
           f"color #1:{_ranges(second, ',')} firebrick",
           "lighting soft", "graphics silhouettes true", "view",
           "save h05_bundles_chimerax.png width 2400 height 1800 supersample 3"]
    (sdir / "h05_bundles.pml").write_text("\n".join(pml) + "\n")
    (sdir / "h05_bundles.cxc").write_text("\n".join(cxc) + "\n")
    return [sdir / "h05_bundles.pml", sdir / "h05_bundles.cxc"]


# ------------------------------------------------------------- h06 fold or lineage
def h06_fold_or_lineage(t: pd.DataFrame, a: str, b: str, bits: dict,
                        outdir: Path) -> dict:
    """Is a position more like its own symmetry mate, or more like its orthologue?

    Every measured position has up to three partners: the structurally equivalent
    position in the other half of the same protein, the aligned position in the other
    protein, and that position's own mate. Correlating the same set of positions
    against each of the three puts architecture and lineage on one axis, measured on
    exactly the same residues so the comparison is not about which set was easier.
    """
    orth = {}
    for r in t[t.both_residues].itertuples():
        pa, pb = getattr(r, f"{a}_pos"), getattr(r, f"{b}_pos")
        if np.isfinite(pa) and np.isfinite(pb):
            orth[int(pa)] = int(pb)
    ea, eb = bits[a]["effect"], bits[b]["effect"]
    ma, mb = bits[a].get("mate", {}), bits[b].get("mate", {})
    rows = []
    for p in sorted(ea):
        q = orth.get(p)
        m = ma.get(p)
        mo = mb.get(q) if q is not None else None
        rows.append({"pos_a": p, "self": ea.get(p), "mate": ea.get(m),
                     "ortholog": eb.get(q), "mate_of_ortholog": eb.get(mo)})
    d = pd.DataFrame(rows).dropna()
    if len(d) < 20:
        return {"status": "skipped",
                "reason": "fewer than twenty positions have all three partners"}
    cap_same = ceiling(bits[a]["reliability"], bits[a]["reliability"])
    cap_cross = ceiling(bits[a]["reliability"], bits[b]["reliability"])
    comps = [("its symmetry mate\n(same protein, other bundle)", "mate", cap_same),
             (f"its orthologue\n(other protein, same bundle)", "ortholog", cap_cross),
             ("its orthologue's mate\n(other protein, other bundle)",
              "mate_of_ortholog", cap_cross)]
    out = []
    for lab, col, cap in comps:
        rho, lo, hi = bootstrap_rho(d["self"], d[col])
        out.append({"comparison": lab.replace("\n", " "), "column": col,
                    "n": int(len(d)), "spearman": rho, "lo": lo, "hi": hi,
                    "ceiling": cap,
                    "fraction_of_ceiling": rho / cap if cap else np.nan})
    fig, ax = plt.subplots(figsize=(5.0, 3.3), layout="constrained")
    x = np.arange(len(out))
    ax.bar(x, [r["spearman"] for r in out],
           yerr=[[r["spearman"] - (r["lo"] if np.isfinite(r["lo"]) else r["spearman"])
                  for r in out],
                 [(r["hi"] if np.isfinite(r["hi"]) else r["spearman"]) - r["spearman"]
                  for r in out]],
           color=[P.DARK_RED, "#2166AC", P.MUTED], edgecolor=P.INK, linewidth=0.6,
           error_kw=dict(lw=0.9, capsize=3))
    for k, r in enumerate(out):
        ax.plot([k - 0.4, k + 0.4], [r["ceiling"]] * 2, color=P.INK, lw=1.2,
                ls=(0, (3, 2)))
        ax.text(k, r["spearman"] + 0.02, f"{r['spearman']:+.2f}", ha="center",
                fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in comps], fontsize=6.5)
    ax.axhline(0, color=P.MUTED, lw=0.7)
    ax.set_ylabel("ρ with the position's own effect", fontsize=7.5)
    panel(ax, 0, f"{short_name(a)}: {len(d)} positions with all three partners", size=8)
    fig.suptitle("The same positions throughout, so the three bars differ only in what "
                 "they are compared with. Dashed = the attainable ceiling for that "
                 "comparison.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h06_fold_or_lineage", None, "H06")
    plt.close(fig)
    per = pd.DataFrame(out)
    per.to_csv(outdir / "h06_fold_or_lineage.csv", index=False)
    d.to_csv(outdir / "h06_quartets.csv", index=False)
    win = per.loc[per.fraction_of_ceiling.idxmax()]
    return {"status": "ok",
            "headline": "; ".join(f"{r.column} ρ {r.spearman:+.2f} "
                                  f"({100 * r.fraction_of_ceiling:.0f}% of ceiling)"
                                  for r in per.itertuples())
                        + f" — closest partner: {win.column}",
            "comparisons": per.to_dict("records")}


# ----------------------------------------------- h07 where they disagree, and whether in patches
def h07_disagreement(t: pd.DataFrame, a: str, b: str, bits: dict, outdir: Path) -> dict:
    """Scattered disagreement is noise; a patch of it is a functional difference.

    The residual of one protein's effect against the other's is mapped back onto each
    structure and tested for spatial clustering with the same correlogram and Moran's I
    the architecture module uses, against the same null stratified by TM and non-TM.
    Two paralogues with different substrate affinities should differ somewhere specific,
    and where is a hypothesis about what differs.
    """
    from .architecture import DIST_BINS, correlogram, morans_i
    d = t[t.both_residues].dropna(subset=[f"{a}_mean_all", f"{b}_mean_all",
                                          f"{a}_pos", f"{b}_pos"]).copy()
    if len(d) < 50:
        return {"status": "skipped", "reason": "fewer than fifty aligned positions"}
    x = d[f"{a}_mean_all"].to_numpy(float)
    y = d[f"{b}_mean_all"].to_numpy(float)
    sl = np.polyfit(x, y, 1)
    d["residual"] = y - np.polyval(sl, x)
    rng = np.random.default_rng(31)
    rows, panels = [], []
    for g, poscol in ((a, f"{a}_pos"), (b, f"{b}_pos")):
        rt = bits[g]["rt"]
        m = d.assign(pos=d[poscol].astype(int)).merge(
            rt[["pos", "x", "y", "z"]], on="pos", how="inner")
        pt = bits[g]["pos"][["pos", "is_tm"]]
        m = m.merge(pt, on="pos", how="left")
        if len(m) < 50:
            continue
        xyz = m[["x", "y", "z"]].to_numpy(float)
        v = m.residual.to_numpy(float)
        strata = (m.is_tm.fillna(0) > 0.5).to_numpy()
        obs = correlogram(xyz, v, DIST_BINS)
        null = []
        for _ in range(200):
            s = v.copy()
            for mask in (strata, ~strata):
                if mask.sum() > 1:
                    s[mask] = rng.permutation(s[mask])
            null.append(correlogram(xyz, s, DIST_BINS).set_index("mid")["r"])
        nb = pd.DataFrame(null)
        mi = morans_i(xyz, v)
        mi_null = np.array([morans_i(xyz, rng.permutation(v)) for _ in range(100)])
        rows.append({"protein": g, "n": int(len(m)), "morans_i": mi,
                     "p_morans": float(((mi_null >= mi).sum() + 1) / (len(mi_null) + 1)),
                     "corr_at_6A": float(obs["r"].iloc[0]) if len(obs) else np.nan})
        panels.append((g, obs, nb, m))
    if not panels:
        return {"status": "skipped", "reason": "no protein could be mapped onto its structure"}
    fig, pairs_ax, _ = grid([g for g, _, _, _ in panels], w=3.1, h=2.8, extra=0.3)
    for i, ((g, obs, nb, m), (_, ax)) in enumerate(zip(panels, pairs_ax)):
        ax.fill_between(nb.columns, nb.quantile(0.025), nb.quantile(0.975),
                        color=P.MUTED, alpha=0.25, lw=0, label="null, 95%")
        ax.plot(obs.mid, obs["r"], "-o", ms=4, lw=1.6, color=P.DARK_RED, mec=P.INK,
                mew=0.4, label="observed")
        ax.axhline(0, color=P.INK, lw=0.7)
        r = rows[i]
        panel(ax, i, g, f"Moran's I = {r['morans_i']:+.3f} (p = {r['p_morans']:.3f})",
              size=8)
        ax.set_xlabel("distance between two residues (Å)", fontsize=7.5)
        if i % 2 == 0:
            ax.set_ylabel("correlation of their disagreement", fontsize=7.5)
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("The residual of one protein's effect against the other's, placed in "
                 "each structure. A curve above the band means the two proteins differ "
                 "in patches rather than at scattered positions.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "h07_disagreement", None, "H07")
    plt.close(fig)
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "h07_disagreement.csv", index=False)
    d.sort_values("residual").to_csv(outdir / "h07_residuals.csv", index=False)
    head = "; ".join(f"{short_name(r.protein)} Moran's I {r.morans_i:+.3f} "
                     f"(p {r.p_morans:.3f})" for r in per.itertuples())
    return {"status": "ok", "headline": head, "by_protein": per.to_dict("records")}


# -------------------------------------------- h08 does identity predict agreement?
def blosum_bin(aa_a: str, aa_b: str) -> str:
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    if aa_a == aa_b:
        return "identical"
    try:
        s = float(m[aa_a, aa_b])
    except (KeyError, IndexError):
        return "unscored"
    return "similar (BLOSUM ≥ 0)" if s >= 0 else "dissimilar (BLOSUM < 0)"


IDENT_ORDER = ["identical", "similar (BLOSUM ≥ 0)", "dissimilar (BLOSUM < 0)"]


def h08_identity(t: pd.DataFrame, a: str, b: str, bits: dict, outdir: Path) -> dict:
    """Is agreement higher where the two proteins have the same residue?

    This is the assumption under every transfer of a measurement from one homologue to
    another, and it is not guaranteed: a position can be equally constrained in both
    while the residue there has drifted, and a conserved residue can be conserved for a
    reason that applies in only one of them. Columns are split by identity and by
    BLOSUM62 similarity, and the correlation is refitted inside each bin.
    """
    d = t[t.both_tm].dropna(subset=[f"{a}_mean_all", f"{b}_mean_all"]).copy()
    if len(d) < 40:
        return {"status": "skipped", "reason": "fewer than forty aligned TM columns"}
    d["bin"] = [blosum_bin(x, y) for x, y in zip(d[f"{a}_aa"], d[f"{b}_aa"])]
    cap = ceiling(bits[a]["reliability"], bits[b]["reliability"])
    rows = []
    for k in IDENT_ORDER:
        s = d[d.bin == k]
        if len(s) < 15:
            continue
        rho, lo, hi = bootstrap_rho(s[f"{a}_mean_all"], s[f"{b}_mean_all"])
        rows.append({"bin": k, "n": int(len(s)), "spearman": rho, "lo": lo, "hi": hi,
                     "ceiling": cap, "mean_a": float(s[f"{a}_mean_all"].mean()),
                     "mean_b": float(s[f"{b}_mean_all"].mean())})
    if not rows:
        return {"status": "skipped", "reason": "no identity bin has fifteen columns"}
    per = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    ax = axes[0]
    cols = {"identical": "#2166AC", "similar (BLOSUM ≥ 0)": P.MUTED,
            "dissimilar (BLOSUM < 0)": P.DARK_RED}
    for k in IDENT_ORDER:
        s = d[d.bin == k]
        if len(s) < 15:
            continue
        ax.scatter(s[f"{a}_mean_all"], s[f"{b}_mean_all"], s=13, color=cols[k],
                   alpha=0.7, lw=0.25, edgecolor=P.INK, label=f"{k} (n = {len(s)})")
    lim = [min(d[f"{a}_mean_all"].min(), d[f"{b}_mean_all"].min()) - 0.05,
           max(d[f"{a}_mean_all"].max(), d[f"{b}_mean_all"].max()) + 0.05]
    ax.plot(lim, lim, ls=(0, (5, 4)), color=P.INK, lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"{short_name(a)} mean effect", fontsize=7.5)
    ax.set_ylabel(f"{short_name(b)} mean effect", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    panel(ax, 0, "aligned TM columns", size=8)

    ax = axes[1]
    x = np.arange(len(per))
    ax.bar(x, per.spearman,
           yerr=[per.spearman - per.lo.fillna(per.spearman),
                 per.hi.fillna(per.spearman) - per.spearman],
           color=[cols[k] for k in per.bin], edgecolor=P.INK, linewidth=0.6,
           error_kw=dict(lw=0.9, capsize=3))
    if np.isfinite(cap):
        ax.axhline(cap, color=P.INK, lw=1.2, ls=(0, (3, 2)))
        ax.text(len(per) - 0.5, cap, " ceiling", fontsize=6, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([k.split(" (")[0] for k in per.bin], fontsize=6.5, rotation=15,
                       ha="right")
    ax.axhline(0, color=P.MUTED, lw=0.7)
    ax.set_ylabel("ρ within the bin", fontsize=7.5)
    panel(ax, 1, "agreement by residue identity", size=8)
    fig.suptitle("If identity predicted shared constraint the left bar would be the "
                 "tallest. It need not be: a position can be equally constrained in "
                 "both proteins while the residue has drifted.", fontsize=7, x=0.01,
                 ha="left")
    P.save(fig, outdir, "h08_identity", None, "H08")
    plt.close(fig)
    per.to_csv(outdir / "h08_identity.csv", index=False)
    head = "; ".join(f"{r.bin.split(' (')[0]} ρ {r.spearman:+.2f} (n = {r.n})"
                     for r in per.itertuples()) + f"; ceiling {cap:.2f}"
    return {"status": "ok", "headline": head, "bins": per.to_dict("records")}


# ---------------------------------------- h09 substitution profiles, not just means
def h09_profiles(t: pd.DataFrame, a: str, b: str, outdir: Path) -> dict:
    """Two positions can be equally sensitive and tolerate entirely different residues.

    The position mean collapses nineteen numbers into one. Correlating the nineteen
    instead asks a stricter question: not "is this position constrained in both" but
    "is it constrained against the same substitutions in both". The null pairs each
    position's profile with a different position's, which keeps every profile intact
    and destroys only the correspondence.
    """
    rhos, null = substitution_profile_correlation(t, a, b, t.both_tm)
    if not len(rhos):
        return {"status": "skipped", "reason": "no aligned TM column has eight shared "
                                               "substitutions in both proteins"}
    d = helix_columns(t, a, b)
    per = []
    cols = [f"sub_{x}" for x in AA20]
    for h, sub in d.groupby("helix"):
        A = sub[[f"{a}_{c}" for c in cols]].to_numpy(float)
        B = sub[[f"{b}_{c}" for c in cols]].to_numpy(float)
        vals = []
        for i in range(len(A)):
            ok = np.isfinite(A[i]) & np.isfinite(B[i])
            if ok.sum() >= 8:
                r = ss.spearmanr(A[i][ok], B[i][ok])[0]
                if np.isfinite(r):
                    vals.append(r)
        if len(vals) >= MIN_PER_HELIX:
            per.append({"helix": h, "n": int(len(vals)), "median_rho": float(np.median(vals)),
                        "mean_rho": float(np.mean(vals))})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    ax = axes[0]
    bins = np.linspace(-1, 1, 31)
    if len(null):
        ax.hist(null, bins=bins, density=True, color="#D9D9D9", lw=0,
                label=f"mismatched positions (n = {len(null)})")
    ax.hist(rhos, bins=bins, density=True, histtype="step", lw=1.8, color=P.DARK_RED,
            label=f"aligned positions (n = {len(rhos)})")
    ax.axvline(float(np.median(rhos)), color=P.DARK_RED, lw=1.0, ls=(0, (3, 2)))
    ax.axvline(0, color=P.INK, lw=0.7)
    ax.set_xlabel("ρ between the two proteins' 19-vectors at one position", fontsize=7.5)
    ax.set_ylabel("density", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    p = ((np.sum(null >= np.median(rhos)) + 1) / (len(null) + 1)) if len(null) else np.nan
    panel(ax, 0, "all aligned TM columns",
          f"median {np.median(rhos):+.2f} (p = {p:.3g})", size=8)

    ax = axes[1]
    if per:
        pf = pd.DataFrame(per).sort_values("median_rho")
        y = np.arange(len(pf))
        ax.barh(y, pf.median_rho, color=P.DARK_RED, edgecolor=P.INK, linewidth=0.5)
        ax.set_yticks(y); ax.set_yticklabels(pf.helix, fontsize=6)
        ax.axvline(0, color=P.MUTED, lw=0.7)
        if len(null):
            ax.axvline(float(np.median(null)), color=P.INK, lw=1.0, ls=(0, (3, 2)))
        ax.set_xlabel("median profile ρ within the helix", fontsize=7.5)
        panel(ax, 1, "per helix", "dashed = the mismatched null", size=8)
    else:
        empty_panel(ax, 1, "per helix", f"no helix has {MIN_PER_HELIX} usable columns")
    fig.suptitle("A position counted here agrees not only in how constrained it is but "
                 "in which substitutions it refuses.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h09_substitution_profiles", None, "H09")
    plt.close(fig)
    pd.DataFrame({"rho": rhos}).to_csv(outdir / "h09_profile_rhos.csv", index=False)
    if per:
        pd.DataFrame(per).to_csv(outdir / "h09_profiles_by_helix.csv", index=False)
    return {"status": "ok",
            "headline": f"median profile ρ {np.median(rhos):+.2f} over {len(rhos)} "
                        f"aligned TM columns, against a mismatched null of "
                        f"{np.median(null) if len(null) else float('nan'):+.2f}",
            "by_helix": per}


# ------------------------------------------------- h10 transfer as a prediction task
def h10_transfer(t: pd.DataFrame, a: str, b: str, bits: dict, outdir: Path) -> dict:
    """How much would you actually trust a measurement carried across?

    A correlation says the two proteins rank positions alike. It does not say what
    happens if you use one protein's number as the other's, which is what anyone
    reading this table will do. Two statements instead: the error of that transfer in
    the units of the assay, and how reliably it finds the worst tenth of positions -
    the question a screen actually asks.
    """
    d = t[t.both_residues].dropna(subset=[f"{a}_mean_all", f"{b}_mean_all"]).copy()
    if len(d) < 40:
        return {"status": "skipped", "reason": "fewer than forty aligned positions"}
    d["bin"] = [blosum_bin(x, y) for x, y in zip(d[f"{a}_aa"], d[f"{b}_aa"])]
    rows = []
    for src, dst in ((a, b), (b, a)):
        x = d[f"{src}_mean_all"].to_numpy(float)
        y = d[f"{dst}_mean_all"].to_numpy(float)
        # calibrated transfer: the two assays have different dynamic range, so the raw
        # difference would mostly measure that. The slope and intercept are fitted on
        # the same data, which makes this the best case rather than a held-out estimate.
        pred = np.polyval(np.polyfit(x, y, 1), x)
        thr = np.quantile(y, WORST_FRAC)
        worst = y <= thr
        for lab, mask in [("all aligned", np.ones(len(d), bool))] + \
                         [(k, (d.bin == k).to_numpy()) for k in IDENT_ORDER]:
            if mask.sum() < 15 or worst[mask].sum() < 3 or (~worst[mask]).sum() < 3:
                continue
            rows.append({"source": src, "target": dst, "subset": lab,
                         "n": int(mask.sum()),
                         "rmse": float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2))),
                         "sd_of_target": float(np.std(y[mask])),
                         "auroc_worst_tenth": auroc(-pred[mask][worst[mask]],
                                                    -pred[mask][~worst[mask]]),
                         "n_worst": int(worst[mask].sum())})
    if not rows:
        return {"status": "skipped", "reason": "no subset has enough positions"}
    per = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained")
    for ax, col, lab, ref in (
            (axes[0], "rmse", "RMSE of the transferred value", None),
            (axes[1], "auroc_worst_tenth",
             f"AUROC for the worst {100 * WORST_FRAC:.0f}% of positions", 0.5)):
        w = 0.38
        for k, (src, dd) in enumerate(per.groupby("source")):
            dd = dd.set_index("subset")
            dd = dd.reindex([s for s in ["all aligned"] + IDENT_ORDER
                             if s in set(dd.index)])
            if not len(dd):
                continue
            x = np.arange(len(dd))
            ax.bar(x + k * w - w / 2, dd[col], w,
                   color=[P.DARK_RED, "#2166AC"][k], edgecolor=P.INK, linewidth=0.5,
                   label=f"{short_name(src)} → {short_name(dd['target'].iloc[0])}")
            ax.set_xticks(x)
            ax.set_xticklabels([s.split(" (")[0] for s in dd.index], fontsize=6.5,
                               rotation=15, ha="right")
        if ref is not None:
            ax.axhline(ref, color=P.INK, lw=1.0, ls=(0, (3, 2)))
        ax.set_ylabel(lab, fontsize=7.5)
        ax.legend(frameon=False, fontsize=6)
    # the SD of the target is the error you would make by predicting its mean
    sd = per[per.subset == "all aligned"].sd_of_target.mean()
    axes[0].axhline(sd, color=P.MUTED, lw=1.0, ls=(0, (1, 2)))
    # headroom first, or the reference line and its label sit under the legend
    axes[0].set_ylim(0, max(sd, float(per.rmse.max())) * 1.32)
    axes[0].text(axes[0].get_xlim()[1], sd, "predict the mean instead ", fontsize=6,
                 ha="right", va="bottom", color=P.MUTED)
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    panel(axes[0], 0, "error of the transfer", size=8)
    panel(axes[1], 1, "finding the worst positions", size=8)
    fig.suptitle("The slope is fitted on the same data, so both panels are a best case: "
                 "they say how much signal transfers, not how a blind prediction would "
                 "do.", fontsize=7, x=0.01, ha="left")
    P.save(fig, outdir, "h10_transfer", None, "H10")
    plt.close(fig)
    per.to_csv(outdir / "h10_transfer.csv", index=False)
    head = "; ".join(f"{short_name(r.source)}→{short_name(r.target)} RMSE "
                     f"{r.rmse:.3f} (SD {r.sd_of_target:.3f}), AUROC "
                     f"{r.auroc_worst_tenth:.3f}"
                     for r in per[per.subset == "all aligned"].itertuples())
    return {"status": "ok", "headline": head, "rows": per.to_dict("records")}


ANALYSES = {"h01": "the whole protein, with a null and a ceiling",
            "h02": "per helix: the pattern along the helix",
            "h03": "per helix: the helix means",
            "h04": "the symmetry mates, pair by pair",
            "h05": "the bundle seen down the membrane normal",
            "h06": "fold or lineage: the mate, or the orthologue?",
            "h07": "where they disagree, and whether in patches",
            "h08": "does residue identity predict agreement?",
            "h09": "substitution profiles, not just means",
            "h10": "transfer as a prediction task"}


def main(argv=None):
    import json
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*(Glyph|invalid value|Mean of empty|"
                                              "divide by zero|Degrees of freedom|All-NaN|"
                                              "tight_layout|An input array is constant).*")
    ap = argparse.ArgumentParser(
        prog="python -m mpdms homolog",
        description="Two homologues and the internal pseudo-symmetry of each, asked "
                    "together. Exactly two configs.")
    ap.add_argument("configs", nargs=2)
    ap.add_argument("--only", help=f"comma-separated: {','.join(ANALYSES)}")
    ap.add_argument("--canonical", action="store_true",
                    help="force the textbook TM1-TM7 pairing instead of searching every "
                         "correspondence of the two bundles")
    ap.add_argument("--mafft-args", default=None)
    ap.add_argument("--style", default="paper", choices=["default", "paper"])
    ap.add_argument("--outdir", default=str(OUT))
    a_ = ap.parse_args(argv)
    P.use_style(a_.style)
    keys = [k.strip() for k in (a_.only.split(",") if a_.only else ANALYSES)]
    bad = [k for k in keys if k not in ANALYSES]
    if bad:
        raise SystemExit(f"unknown analysis {bad}; available: {', '.join(ANALYSES)}")
    outdir = Path(a_.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfgs, bits, names, seqs = {}, {}, [], {}
    for c in a_.configs:
        cfg = load_config(Path(c))
        b = bits_for(cfg)
        if not b:
            raise SystemExit(f"{cfg.id}: needs a structure and a topology")
        if a_.canonical:
            b["symmetry"] = symmetry_pairs(b, canonical=True)
            b["mate"] = {}
            for pr in b["symmetry"].get("pairs", []):
                for x, y in zip(pr["pos_a"], pr["pos_b"]):
                    b["mate"][x] = y
                    b["mate"][y] = x
        s = protein_sequence(cfg)
        if not s:
            raise SystemExit(f"{cfg.id}: needs protein.sequence for the alignment")
        cfgs[cfg.id], bits[cfg.id], seqs[cfg.id] = cfg, b, s
        names.append(cfg.id)
        sym = b.get("symmetry") or {}
        print(f"  {cfg.id:<16} {len(b['pos']):>4} positions  split-half reliability "
              f"{b['reliability']:.2f}  "
              + (f"{len(sym.get('pairs', []))} symmetry pairs"
                 + ("" if sym.get("canonical", True) else " (NON-canonical)")
                 if sym else "no symmetry pairs"), flush=True)
    a, b = names
    aln = align_sequences(seqs, a_.mafft_args.split() if a_.mafft_args else None)
    t = aligned_table(aln, {g: bits[g]["ptable"] for g in names})
    n_al = int(t.both_residues.sum())
    print(f"\n  alignment: {len(t)} columns, {n_al} with a residue in both, "
          f"{int(t.identical.sum())} identical ({100 * t.identical.sum() / max(n_al, 1):.0f}%), "
          f"{int(t.both_tm.sum())} both in a TM helix "
          f"[{aln.attrs.get('method', 'pairwise')}]", flush=True)
    t.to_csv(outdir / "alignment.csv", index=False)

    pair_rho = None

    def call(k):
        nonlocal pair_rho
        if k == "h01":
            return h01_whole_protein(t, a, b, bits, outdir)
        if k == "h02":
            return h02_per_helix(t, a, b, bits, outdir)
        if k == "h03":
            return h03_helix_means(t, a, b, bits, outdir)
        if k == "h04":
            r = h04_symmetry_pairs(bits, names, outdir)
            pair_rho = pd.DataFrame(r.get("by_pair", []))
            return r
        if k == "h05":
            if pair_rho is None:
                # h05 draws h04's correlations on the map, so it runs h04 first when
                # asked for on its own
                pair_rho = pd.DataFrame(
                    h04_symmetry_pairs(bits, names, outdir).get("by_pair", []))
            return h05_bundle_map(bits, names, outdir, pair_rho)
        if k == "h06":
            return h06_fold_or_lineage(t, a, b, bits, outdir)
        if k == "h07":
            return h07_disagreement(t, a, b, bits, outdir)
        if k == "h08":
            return h08_identity(t, a, b, bits, outdir)
        if k == "h09":
            return h09_profiles(t, a, b, outdir)
        return h10_transfer(t, a, b, bits, outdir)

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
    (outdir / "homolog.json").write_text(json.dumps(out, indent=2, default=str))
    try:
        shown = outdir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = outdir
    print(f"\n  -> {shown}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
