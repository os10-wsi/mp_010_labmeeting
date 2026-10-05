"""A25 - the membrane set against a soluble-protein DMS abundance reference.

The comparison is three-way on purpose:

  soluble            published aPCA abundance for soluble domains
  membrane, non-TM   the loops and termini of THIS protein
  membrane, TM       its transmembrane helices

The middle group is the control that makes the comparison mean something. It shares the
assay, the host, the library construction and the normalisation with the TM group, so a
soluble-vs-TM difference that also shows up in soluble-vs-non-TM is about the protein,
while one that appears only against TM is about the bilayer.

Set reference.soluble_path in the config, or pass --soluble <csv>; see
docs/soluble_comparison.md for the columns.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss

from .. import plotting as P
from ..annot import BIOLOGICAL, VOLUME, assign_topology
from ..reference import load_soluble
from ..stats import bh_fdr
from .base import dirs, fmt_p, missense, result, skipped

AA20 = "ACDEFGHIKLMNPQRSTVWY"
ORDER = ["soluble", "membrane, non-TM", "membrane, TM"]
COLORS = {"soluble": "#8899A6", "membrane, non-TM": "#2F6DB5", "membrane, TM": "#B8912F"}
SHORT = {"soluble": "soluble", "membrane, non-TM": "non-TM", "membrane, TM": "TM"}
INTRODUCED = {"proline": "P", "glycine": "G", "charged (DEKR)": "DEKR",
              "aromatic (FWY)": "FWY", "small (AGS)": "AGS"}
MIN_N = 20


def combined(df: pd.DataFrame, cfg, sol: pd.DataFrame) -> pd.DataFrame:
    """One long table with a `group` column, all on the syn = 0 / nonsense = -1 scale."""
    m = missense(assign_topology(df, cfg))[["pos", "wt", "mut", "score_z", "seg_type"]].copy()
    m["group"] = np.where(m.seg_type == "TM", "membrane, TM", "membrane, non-TM")
    m["dataset"] = cfg.id
    m["rsa"] = np.nan
    s = sol[sol.vclass == "missense"][["dataset", "pos", "wt", "mut", "score_z", "rsa"]].copy()
    s["group"] = "soluble"
    s["seg_type"] = "soluble"
    d = pd.concat([m, s], ignore_index=True)
    d = d[d.mut.isin(list(AA20)) & d.wt.isin(list(AA20))]
    d["dhyd"] = d.mut.map(BIOLOGICAL) - d.wt.map(BIOLOGICAL)
    d["dvol"] = d.mut.map(VOLUME) - d.wt.map(VOLUME)
    return d


def _pos_medians(d: pd.DataFrame, group: str) -> pd.Series:
    return d[d.group == group].groupby(["dataset", "pos"]).score_z.median()


def distribution_tests(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [g for g in ORDER if (d.group == g).sum() >= MIN_N]
    for a, b in [(x, y) for i, x in enumerate(groups) for y in groups[i + 1:]]:
        xa, xb = _pos_medians(d, a), _pos_medians(d, b)
        if len(xa) < 5 or len(xb) < 5:
            continue
        rows.append({"a": a, "b": b, "median_a": float(xa.median()), "median_b": float(xb.median()),
                     "difference": float(xa.median() - xb.median()),
                     "frac_below_half_a": float((xa < -0.5).mean()),
                     "frac_below_half_b": float((xb < -0.5).mean()),
                     "n_pos_a": len(xa), "n_pos_b": len(xb),
                     "p": float(ss.mannwhitneyu(xa, xb, alternative="two-sided")[1])})
    t = pd.DataFrame(rows)
    if len(t):
        t["q"] = bh_fdr(t.p)
    return t


def substitution_matrices(d: pd.DataFrame, a: str, b: str, min_cell: int = 3):
    """Mean effect per (wt, mut) in each group, and their difference."""
    def mat(group):
        g = d[d.group == group]
        piv = g.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
        cnt = g.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
        piv = piv.where(cnt >= min_cell)
        return piv.reindex(index=list(AA20), columns=list(AA20))
    ma, mb = mat(a), mat(b)
    diff = ma - mb
    ok = diff.notna().to_numpy()
    flat = pd.DataFrame({"a": ma.to_numpy()[ok], "b": mb.to_numpy()[ok]})
    rho = float(ss.spearmanr(flat.a, flat.b)[0]) if len(flat) > 5 else np.nan
    return ma, mb, diff, {"n_cells": int(ok.sum()), "spearman_between_matrices": rho,
                          "mean_difference": float(np.nanmean(diff.to_numpy()))}


def property_slopes(d: pd.DataFrame, prop: str = "dhyd") -> dict:
    """Slope of effect on a physicochemical change, per group, plus the interaction.

    The prediction with a sign: adding hydrophobicity is mildly bad for a soluble protein
    (aggregation, exposed greasy surface) and good inside a bilayer, so the slope should
    change sign between soluble and TM, not merely weaken.
    """
    t = d[d.group.isin(ORDER) & d[prop].notna()].copy()
    counts = t.group.value_counts()
    t = t[t.group.isin(counts[counts >= MIN_N].index)]
    if t.group.nunique() < 2:
        return {"testable": False, "note": "fewer than two groups with enough variants"}
    t["grp"] = t.group
    t["cl"] = t.dataset.astype(str) + ":" + t.pos.astype(str)
    m = smf.ols(f"score_z ~ {prop} * C(grp, Treatment('soluble'))", data=t).fit(
        cov_type="cluster", cov_kwds={"groups": t.cl})
    base = float(m.params[prop])
    out = {"testable": True, "slope_soluble": base, "p_soluble": float(m.pvalues[prop]), "slopes": {}}
    for g in t.grp.unique():
        if g == "soluble":
            out["slopes"][g] = base
            continue
        key = [k for k in m.params.index if k.startswith(f"{prop}:") and g in k]
        if key:
            out["slopes"][g] = base + float(m.params[key[0]])
            out[f"p_interaction_{g}"] = float(m.pvalues[key[0]])
    tm = out["slopes"].get("membrane, TM")
    out["sign_flip"] = bool(tm is not None and np.sign(tm) != np.sign(base)
                            and out.get("p_interaction_membrane, TM", 1) < 0.05)
    return out


def introduced_costs(d: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Median cost of introducing each residue class, per group, with a bootstrap CI."""
    rng = np.random.default_rng(25)
    rows = []
    for name, letters in INTRODUCED.items():
        for g in ORDER:
            v = d[(d.group == g) & d.mut.isin(list(letters)) & ~d.wt.isin(list(letters))]
            pos = v.groupby(["dataset", "pos"]).score_z.median().to_numpy()
            if len(pos) < 5:
                continue
            boots = [np.median(rng.choice(pos, len(pos))) for _ in range(n_boot)]
            rows.append({"introduced": name, "group": g, "n_positions": len(pos),
                         "median": float(np.median(pos)),
                         "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    # TM minus soluble, the quantity of interest
    piv = t.pivot_table(index="introduced", columns="group", values="median")
    if {"soluble", "membrane, TM"} <= set(piv.columns):
        t = t.merge((piv["membrane, TM"] - piv["soluble"]).rename("tm_minus_soluble"),
                    left_on="introduced", right_index=True, how="left")
    return t


def burial_slopes(d: pd.DataFrame) -> dict:
    """Effect vs RSA in each group - only where the reference carries RSA."""
    t = d[d.rsa.notna() & d.group.isin(ORDER)]
    if t.group.nunique() < 2 or len(t) < 2 * MIN_N:
        return {"testable": False, "note": "the reference table has no RSA column, or too few "
                                           "positions with RSA in both groups"}
    t = t.copy()
    t["cl"] = t.dataset.astype(str) + ":" + t.pos.astype(str)
    m = smf.ols("score_z ~ rsa * C(group, Treatment('soluble'))", data=t).fit(
        cov_type="cluster", cov_kwds={"groups": t.cl})
    key = [k for k in m.params.index if k.startswith("rsa:")]
    return {"testable": True, "slope_soluble": float(m.params["rsa"]),
            "interaction": float(m.params[key[0]]) if key else np.nan,
            "p_interaction": float(m.pvalues[key[0]]) if key else np.nan,
            "n": int(len(t))}


# -------------------------------------------------------------------- figure
def figure(cfg, d, res, figdir):
    fig = plt.figure(figsize=(10.6, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    groups = {g: d.loc[d.group == g, "score_z"].to_numpy() for g in ORDER if (d.group == g).any()}
    P.density_by_class(ax, groups, colors=COLORS)
    ax.axvline(0, color="#1B9E77", lw=0.7)
    ax.axvline(-1, color="#D62728", lw=0.6, ls=(0, (3, 3)))
    dt = res["distribution_tests"]
    if len(dt):
        bits = [f"{SHORT.get(r.a, r.a)} vs {SHORT.get(r.b, r.b)}: {fmt_p(r.p)}"
                for r in dt.itertuples()]
        ax.set_xlabel("Normalised fitness\n" + "; ".join(bits), fontsize=6)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_title("(a) Missense effect distributions", loc="left", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ic = res["introduced"]
    if len(ic):
        names = list(INTRODUCED)
        for k, g in enumerate([g for g in ORDER if g in set(ic.group)]):
            sub = ic[ic.group == g].set_index("introduced").reindex(names)
            y = np.arange(len(names)) + (k - 1) * 0.24
            ax.errorbar(sub["median"], y, xerr=[sub["median"] - sub["lo"], sub["hi"] - sub["median"]],
                        fmt="o", ms=3.5, lw=0.9, capsize=0, color=COLORS[g], label=g)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.legend(frameon=False, fontsize=5.5, loc="lower left")
        ax.set_xlabel("Median effect of introducing (±95% CI)", fontsize=6.5)
    ax.set_title("(b) Cost of introducing each residue class", loc="left", fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ps = res["hydrophobicity"]
    if ps.get("testable"):
        for g, sl in ps["slopes"].items():
            sub = d[d.group == g]
            if len(sub) < MIN_N:
                continue
            gr = np.linspace(np.nanpercentile(sub.dhyd, 2), np.nanpercentile(sub.dhyd, 98), 20)
            b = float(np.nanmean(sub.score_z) - sl * np.nanmean(sub.dhyd))
            ax.plot(gr, b + sl * gr, color=COLORS[g], lw=1.6, label=f"{g} (slope {sl:+.3f})")
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.axvline(0, color=P.MUTED, lw=0.5)
        pi = ps.get("p_interaction_membrane, TM")
        ax.set_xlabel("Δ hydrophobicity, biological scale (mut − wt)\n"
                      f"TM vs soluble interaction {fmt_p(pi)}"
                      + ("; SIGN FLIP" if ps.get("sign_flip") else "; same sign"), fontsize=6.5)
        ax.legend(frameon=False, fontsize=5.5, loc="lower right")
        ax.set_ylabel("Normalised fitness", fontsize=6.5)
    ax.set_title("(c) Hydrophobicity: the sign should flip", loc="left", fontsize=8)

    ma, mb, diff = res["mat_tm"], res["mat_sol"], res["mat_diff"]
    for k, (m_, ttl) in enumerate([(mb, "(d) Soluble"), (ma, "(e) Membrane, TM")]):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(np.ma.masked_invalid(m_.to_numpy(float)), cmap=P.fitness_cmap(),
                       norm=P.fitness_norm(), interpolation="nearest")
        ax.set_xticks(range(20)); ax.set_xticklabels(list(AA20), fontsize=4.5, family="monospace")
        ax.set_yticks(range(20)); ax.set_yticklabels(list(AA20), fontsize=4.5, family="monospace")
        ax.tick_params(length=0)
        ax.set_xlabel("Mutant", fontsize=6.5); ax.set_ylabel("Wild type", fontsize=6.5)
        ax.set_title(ttl, loc="left", fontsize=8)
        if k == 1:
            fig.colorbar(im, ax=ax, fraction=0.045).set_label("mean effect", fontsize=6)

    ax = fig.add_subplot(gs[1, 2])
    v = np.ma.masked_invalid(diff.to_numpy(float))
    lim = float(np.nanmax(np.abs(diff.to_numpy(float)))) if np.isfinite(diff.to_numpy(float)).any() else 1.0
    im = ax.imshow(v, cmap=P.diverging_cmap(), vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_xticks(range(20)); ax.set_xticklabels(list(AA20), fontsize=4.5, family="monospace")
    ax.set_yticks(range(20)); ax.set_yticklabels(list(AA20), fontsize=4.5, family="monospace")
    ax.tick_params(length=0)
    md = res["mat_stats"]
    ax.set_xlabel(f"Mutant\n{md['n_cells']} cells; ρ between matrices = "
                  f"{md['spearman_between_matrices']:.2f}", fontsize=6.5)
    ax.set_ylabel("Wild type", fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.045).set_label("TM − soluble", fontsize=6)
    ax.set_title("(f) Difference: TM − soluble", loc="left", fontsize=8)

    P.title(fig, cfg, "A25 membrane vs soluble abundance DMS")
    return P.save(fig, figdir, "a25_soluble_comparison", cfg, "A25")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    path = cfg.get_path("reference.soluble_path")
    if not path:
        return skipped("a25", cfg, "set reference.soluble_path in the config "
                                   "(see docs/soluble_comparison.md)")
    p = cfg.resolve(path)
    if p is None or not Path(p).exists():
        return skipped("a25", cfg, f"soluble reference not found at {path}")
    try:
        sol = load_soluble(p)
    except ValueError as e:
        return skipped("a25", cfg, str(e))
    d = combined(df, cfg, sol)
    if (d.group == "soluble").sum() < MIN_N or (d.group == "membrane, TM").sum() < MIN_N:
        return skipped("a25", cfg, "too few variants in the soluble or TM group after filtering")

    ma, mb, diff, mstats = substitution_matrices(d, "membrane, TM", "soluble")
    res = {"distribution_tests": distribution_tests(d), "introduced": introduced_costs(d),
           "hydrophobicity": property_slopes(d, "dhyd"), "volume": property_slopes(d, "dvol"),
           "burial": burial_slopes(d), "mat_tm": ma, "mat_sol": mb, "mat_diff": diff,
           "mat_stats": mstats}
    figs = figure(cfg, d, res, figdir)

    t1 = tabdir / "a25_group_tests.csv"; res["distribution_tests"].to_csv(t1, index=False)
    t2 = tabdir / "a25_introduced.csv"; res["introduced"].to_csv(t2, index=False)
    t3 = tabdir / "a25_substitution_difference.csv"; diff.to_csv(t3)

    hy = res["hydrophobicity"]
    caveats = [f"reference: {sol.attrs['source']}, {sol.attrs['n_datasets_kept']} domains kept, "
               f"{sol.attrs['n_datasets_dropped']} dropped for lacking synonymous or nonsense controls",
               "each reference domain is normalised on its own synonymous and nonsense medians, so "
               "domains with different dynamic ranges contribute equally",
               "the non-TM group is this protein's own loops and termini: it shares assay, host and "
               "normalisation with the TM group, so it is the control that isolates the bilayer",
               "different libraries and reporters mean absolute ranges are not comparable; the "
               "comparisons here are of shape, of contrasts within each dataset, and of ranks"]
    if not res["burial"].get("testable"):
        caveats.append("burial comparison skipped: " + res["burial"].get("note", ""))
    head = (f"soluble n = {int((d.group == 'soluble').sum()):,} variants over "
            f"{sol.attrs['n_datasets_kept']} domains; "
            + "; ".join(f"{SHORT.get(r.a, r.a)} vs {SHORT.get(r.b, r.b)} "
                        f"Δmedian {r.difference:+.2f} ({fmt_p(r.p)})"
                        for r in res["distribution_tests"].itertuples())
            + (f" | hydrophobicity slope soluble {hy['slope_soluble']:+.3f} vs TM "
               f"{hy['slopes'].get('membrane, TM', float('nan')):+.3f}"
               f"{', SIGN FLIP' if hy.get('sign_flip') else ''}" if hy.get("testable") else ""))
    return result("a25", cfg, head,
                  {"n_soluble_domains": int(sol.attrs["n_datasets_kept"]),
                   "group_tests": res["distribution_tests"].to_dict("records"),
                   "introduced": res["introduced"].to_dict("records"),
                   "hydrophobicity": hy, "volume": res["volume"], "burial": res["burial"],
                   "substitution_matrix": mstats},
                  caveats, figs, [t1, t2, t3])
