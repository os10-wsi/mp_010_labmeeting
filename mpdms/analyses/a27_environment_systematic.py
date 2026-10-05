"""A27 - six systematic comparisons across structural environments.

Environments, not a surface/buried dichotomy. In a membrane protein an exposed side
chain faces water, lipid or the substrate cavity, and those have different expectations,
so merging them into "surface" would average three things that disagree:

    buried          RSA below the cut, anywhere in the protein
    water-exposed   exposed and outside the membrane slab
    lipid-exposed   exposed, inside the slab, on the periphery of the bundle
    cavity          exposed, inside the slab, facing the bundle axis

The classification is A24's, reused rather than redefined so the two analyses cannot
drift apart.

 a  Substitution matrices per environment, and their difference: which specific
    substitutions are environment-specific, not just which classes of them.
 b  Choosiness. The spread of effects across a position's substitutions, which is a
    different property from the mean: a position can be uniformly bad, or tolerant of
    most things and lethal for a few. Taken as the residual of spread on mean effect,
    so it is not sensitivity restated.
 c  Property coefficients per environment: the same model (hydrophobicity, volume,
    charge, helix propensity) fitted in each, with a Wald test per property for whether
    its coefficient differs across environments.
 d  The RSA dose-response, fitted as a gradient AND as a changepoint, because tight
    packing predicts a threshold rather than a slope.
 e  Is the wild type the best residue available, and does that depend on burial?
 f  Own burial against neighbourhood burial, both in one model: is tolerance a property
    of the residue or of its surroundings?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss

from .. import plotting as P
from ..annot import BIOLOGICAL, CHARGE, HELIX, VOLUME
from ..stats import bh_fdr
from .a24_structural_tolerance import RSA_OPEN, prepared
from .base import dirs, fmt_p, result, skipped

AA20 = "ACDEFGHIKLMNPQRSTVWY"
ENVS = ["buried", "water-exposed", "lipid-exposed", "cavity"]
ENV_COLORS = {"buried": "#2F6DB5", "water-exposed": "#1B9E77",
              "lipid-exposed": "#B8912F", "cavity": "#C2443A"}
PROPS = {"hydrophobicity": BIOLOGICAL, "volume": VOLUME, "charge": CHARGE, "helix propensity": HELIX}
MIN_POS = 8
MIN_VARIANTS = 8
NEUTRAL = 0.1     # a substitution must beat the wild type by this much to count as better


def environments(d: pd.DataFrame) -> pd.DataFrame:
    """A24's facing classes, with the non-membrane half split into buried and water-exposed."""
    d = d.copy()
    d["env"] = np.where(d.rsa < RSA_OPEN, "buried",
                        np.where(~d.in_slab, "water-exposed",
                                 np.where(d.face == "lipid-facing", "lipid-exposed", "cavity")))
    # Standardised to unit SD across the protein's variants. The raw scales are in
    # different units - kcal/mol, A^3, charge units - so raw coefficients cannot be
    # compared across properties and volume would read as 0.00 next to hydrophobicity.
    # After this, every coefficient is "effect per 1 SD of change in that property".
    for name, scale in PROPS.items():
        v = d.mut.map(scale) - d.wt.map(scale)
        sd = float(v.std())
        d[f"d_{name.split()[0]}"] = v / sd if sd > 1e-12 else v
    return d


def _pos(d: pd.DataFrame) -> pd.DataFrame:
    """Per-position summary: sensitivity, spread, and how many substitutions beat the WT."""
    rows = []
    for pos, g in d.groupby("pos"):
        if len(g) < MIN_VARIANTS:
            continue
        v = g.score_z.to_numpy(float)
        rows.append({"pos": int(pos), "wt": g.wt.iloc[0], "env": g.env.iloc[0],
                     "seg_type": g.seg_type.iloc[0], "rsa": float(g.rsa.iloc[0]),
                     "n": len(g), "mean_effect": float(np.mean(v)), "median_effect": float(np.median(v)),
                     "iqr": float(np.subtract(*np.percentile(v, [75, 25]))), "sd": float(np.std(v, ddof=1)),
                     "n_better_than_wt": int((v > NEUTRAL).sum()),
                     "frac_better_than_wt": float((v > NEUTRAL).mean()),
                     "wt_rank": int(1 + (v > NEUTRAL).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------- (a) substitution matrices
def matrices(d: pd.DataFrame, min_cell: int = 3) -> tuple[dict, pd.DataFrame, dict]:
    out = {}
    for env in ENVS:
        g = d[d.env == env]
        if not len(g):
            continue
        m = g.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
        c = g.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
        out[env] = m.where(c >= min_cell).reindex(index=list(AA20), columns=list(AA20))
    a, b = "buried", "water-exposed"
    if a not in out or b not in out:
        return out, pd.DataFrame(), {"testable": False, "note": f"need both {a} and {b}"}
    diff = out[a] - out[b]
    ok = diff.notna().to_numpy()
    rho = (float(ss.spearmanr(out[a].to_numpy()[ok], out[b].to_numpy()[ok])[0])
           if ok.sum() > 5 else np.nan)
    return out, diff, {"testable": True, "contrast": f"{a} - {b}", "n_cells": int(ok.sum()),
                       "spearman_between": rho, "mean_difference": float(np.nanmean(diff.to_numpy()))}


# ------------------------------------------------------------- (b) choosiness
def choosiness(pt: pd.DataFrame) -> dict:
    """Spread of effects per position, with sensitivity partialled out."""
    t = pt[pt.env.isin(ENVS)].dropna(subset=["iqr", "mean_effect", "rsa"])
    if len(t) < 3 * MIN_POS:
        return {"testable": False, "note": f"only {len(t)} positions"}
    m = smf.ols("iqr ~ mean_effect", data=t).fit()
    t = t.assign(choosy=m.resid)
    rows = []
    for env, g in t.groupby("env", observed=True):
        if len(g) < MIN_POS:
            continue
        rows.append({"env": env, "n": len(g), "median_iqr": float(g.iqr.median()),
                     "median_choosy": float(g.choosy.median()),
                     "rho_rsa_iqr": float(ss.spearmanr(g.rsa, g.iqr)[0]),
                     "p_rsa_iqr": float(ss.spearmanr(g.rsa, g.iqr)[1])})
    tab = pd.DataFrame(rows)
    kw = ss.kruskal(*[g.choosy.to_numpy() for _, g in t.groupby("env", observed=True)
                      if len(g) >= MIN_POS])
    return {"testable": True, "table": tab.to_dict("records"), "r2_iqr_on_mean": float(m.rsquared),
            "kruskal_choosy_p": float(kw.pvalue), "n": len(t),
            "resid": t[["pos", "env", "rsa", "iqr", "mean_effect", "choosy"]]}


# ------------------------------------------------- (c) property coefficients
def property_coefficients(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    sub = d[d.env.isin(ENVS)].copy()
    sub["cl"] = sub.pos.astype(str)
    for name in PROPS:
        key = f"d_{name.split()[0]}"
        for env, g in sub.groupby("env", observed=True):
            if len(g) < 3 * MIN_VARIANTS or g[key].std() < 1e-9:
                continue
            m = smf.ols(f"score_z ~ {key}", data=g).fit(
                cov_type="cluster", cov_kwds={"groups": g.cl})
            ci = m.conf_int().loc[key]
            rows.append({"property": name, "env": env, "coef": float(m.params[key]),
                         "lo": float(ci[0]), "hi": float(ci[1]), "p": float(m.pvalues[key]),
                         "n": len(g)})
        envs_here = [e for e in ENVS if (sub.env == e).sum() >= 3 * MIN_VARIANTS]
        if len(envs_here) >= 2:
            g = sub[sub.env.isin(envs_here)]
            m = smf.ols(f"score_z ~ {key} * C(env)", data=g).fit(
                cov_type="cluster", cov_kwds={"groups": g.cl})
            inter = [k for k in m.params.index if k.startswith(f"{key}:")]
            if inter:
                w = m.wald_test(" = 0, ".join(inter) + " = 0", scalar=True)
                tests.append({"property": name, "n_envs": len(envs_here), "p": float(w.pvalue)})
    t = pd.DataFrame(rows)
    q = pd.DataFrame(tests)
    if len(q):
        q["q"] = bh_fdr(q.p)
    return t, q


# ----------------------------------------------------- (d) RSA dose-response
def dose_response(pt: pd.DataFrame, n_bins: int = 10) -> dict:
    """Gradient or threshold? Compare a line with a one-breakpoint piecewise fit by AIC."""
    out = {}
    for cls, g in [("TM", pt[pt.seg_type == "TM"]), ("non-TM", pt[pt.seg_type != "TM"])]:
        g = g.dropna(subset=["rsa", "median_effect"])
        if len(g) < 3 * MIN_POS:
            out[cls] = {"testable": False, "note": f"only {len(g)} positions"}
            continue
        x, y = g.rsa.to_numpy(float), g.median_effect.to_numpy(float)
        lin = smf.ols("y ~ x", data=pd.DataFrame({"x": x, "y": y})).fit()
        best = None
        for bp in np.quantile(x, np.linspace(0.2, 0.8, 25)):
            fr = pd.DataFrame({"x": x, "hinge": np.clip(x - bp, 0, None), "y": y})
            seg = smf.ols("y ~ x + hinge", data=fr).fit()
            if best is None or seg.aic < best[1]:
                best = (float(bp), float(seg.aic), seg)
        bins = pd.cut(g.rsa, np.linspace(0, 1, n_bins + 1))
        prof = g.groupby(bins, observed=False).median_effect.agg(["mean", "sem", "size"])
        out[cls] = {"testable": True, "n": len(g), "aic_linear": float(lin.aic),
                    "aic_threshold": best[1], "breakpoint": best[0],
                    "prefers": "threshold" if best[1] < lin.aic - 2 else "gradient",
                    "slope_linear": float(lin.params["x"]), "p_linear": float(lin.pvalues["x"]),
                    "profile": prof.reset_index().assign(mid=[iv.mid for iv in prof.index]).to_dict("records")}
    return out


# ----------------------------------------------------- (e) is the WT optimal
def wt_optimality(pt: pd.DataFrame) -> dict:
    t = pt[pt.env.isin(ENVS)].dropna(subset=["frac_better_than_wt", "rsa"])
    if len(t) < 3 * MIN_POS:
        return {"testable": False, "note": f"only {len(t)} positions"}
    rows = []
    for env, g in t.groupby("env", observed=True):
        if len(g) < MIN_POS:
            continue
        rows.append({"env": env, "n": len(g),
                     "median_frac_better": float(g.frac_better_than_wt.median()),
                     "frac_positions_wt_optimal": float((g.n_better_than_wt == 0).mean())})
    rho, pv = ss.spearmanr(t.rsa, t.frac_better_than_wt)
    bur = t[t.env == "buried"].frac_better_than_wt
    wat = t[t.env == "water-exposed"].frac_better_than_wt
    mw = (ss.mannwhitneyu(bur, wat, alternative="two-sided")
          if len(bur) >= MIN_POS and len(wat) >= MIN_POS else None)
    return {"testable": True, "table": rows, "n": len(t), "rho_rsa": float(rho), "p_rsa": float(pv),
            "buried_vs_water_p": float(mw.pvalue) if mw else np.nan,
            "passed": bool(rho > 0 and pv < 0.05)}


# --------------------------------------- (f) own burial vs neighbourhood burial
def neighbourhood(pt: pd.DataFrame, rt: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    if not len(pairs):
        return {"testable": False, "note": "no contact pairs"}
    rsa = rt.set_index("pos").rsa
    long = pd.concat([pairs.assign(p=pairs.pos_i, q=pairs.pos_j),
                      pairs.assign(p=pairs.pos_j, q=pairs.pos_i)])
    nb = long.assign(nrsa=long.q.map(rsa)).groupby("p").nrsa.mean()
    t = pt.assign(neighbour_rsa=pt.pos.map(nb)).dropna(subset=["rsa", "neighbour_rsa", "median_effect"])
    if len(t) < 3 * MIN_POS:
        return {"testable": False, "note": f"only {len(t)} positions with neighbours"}
    m = smf.ols("median_effect ~ rsa + neighbour_rsa", data=t).fit()
    solo = smf.ols("median_effect ~ rsa", data=t).fit()
    return {"testable": True, "n": len(t),
            "beta_own": float(m.params["rsa"]), "p_own": float(m.pvalues["rsa"]),
            "beta_neighbour": float(m.params["neighbour_rsa"]),
            "p_neighbour": float(m.pvalues["neighbour_rsa"]),
            "r2_own_only": float(solo.rsquared), "r2_both": float(m.rsquared),
            "corr_own_neighbour": float(np.corrcoef(t.rsa, t.neighbour_rsa)[0, 1]),
            "table": t[["pos", "env", "rsa", "neighbour_rsa", "median_effect"]],
            "passed": bool(m.pvalues["neighbour_rsa"] < 0.05)}


# -------------------------------------------------------------------- figure
def figure(cfg, d, pt, res, figdir):
    fig, axg = plt.subplots(2, 3, figsize=(11.0, 6.8), layout="constrained")

    ax = axg[0, 0]
    diff, ms = res["mat_diff"], res["mat_stats"]
    if len(diff) and diff.notna().to_numpy().any():
        lim = float(np.nanmax(np.abs(diff.to_numpy(float))))
        im = ax.imshow(np.ma.masked_invalid(diff.to_numpy(float)), cmap=P.diverging_cmap(),
                       vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks(range(20)); ax.set_xticklabels(list(AA20), fontsize=4.5, family="monospace")
        ax.set_yticks(range(20)); ax.set_yticklabels(list(AA20), fontsize=4.5, family="monospace")
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label(ms["contrast"], fontsize=6)
        ax.set_xlabel(f"Mutant   ({ms['n_cells']} cells; ρ between environments "
                      f"{ms['spearman_between']:.2f})", fontsize=6.5)
        ax.set_ylabel("Wild type", fontsize=6.5)
    ax.set_title("(a) Substitutions: buried − water-exposed", loc="left", fontsize=8)

    ax = axg[0, 1]
    ch = res["choosiness"]
    if ch.get("testable"):
        r = ch["resid"]
        groups = {e: r.loc[r.env == e, "choosy"].to_numpy() for e in ENVS if (r.env == e).sum() >= MIN_POS}
        P.violin_box(ax, groups, colors=ENV_COLORS, width=0.75, show_medians=False)
        ax.axhline(0, color=P.INK, lw=0.6)
        ax.set_xticklabels([t.get_text().replace("-", "-\n") for t in ax.get_xticklabels()], fontsize=6)
        ax.set_ylabel("Choosiness\n(spread of effects, sensitivity removed)", fontsize=6.5)
        ax.set_xlabel(f"Kruskal–Wallis {fmt_p(ch['kruskal_choosy_p'])}; spread is "
                      f"{ch['r2_iqr_on_mean']:.0%} explained by the mean", fontsize=6.5)
    ax.set_title("(b) Choosiness by environment", loc="left", fontsize=8)

    ax = axg[0, 2]
    pc, pq = res["prop_coef"], res["prop_tests"]
    if len(pc):
        piv = pc.pivot_table(index="property", columns="env", values="coef")
        piv = piv.reindex(index=list(PROPS), columns=[e for e in ENVS if e in piv.columns])
        lim = float(np.nanmax(np.abs(piv.to_numpy(float))))
        im = ax.imshow(np.ma.masked_invalid(piv.to_numpy(float)), cmap=P.diverging_cmap(),
                       vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([c.replace("-", "-\n") for c in piv.columns], fontsize=5.5)
        qq = pq.set_index("property").q if len(pq) else {}
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{p}{' *' if isinstance(qq, pd.Series) and qq.get(p, 1) < 0.05 else ''}"
                            for p in piv.index], fontsize=6)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.to_numpy()[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                            color="white" if abs(v) > 0.6 * lim else P.INK)
        fig.colorbar(im, ax=ax, fraction=0.045).set_label("effect per SD of change", fontsize=6)
        ax.set_xlabel("* = coefficient differs across environments (q < 0.05)", fontsize=6)
    ax.set_title("(c) Which property matters where", loc="left", fontsize=8)

    ax = axg[1, 0]
    dr = res["dose"]
    for cls, col in (("TM", "#B8912F"), ("non-TM", "#2F6DB5")):
        r = dr.get(cls, {})
        if not r.get("testable"):
            continue
        pr = pd.DataFrame(r["profile"]).dropna(subset=["mean"])
        ax.errorbar(pr["mid"], pr["mean"], yerr=pr["sem"], fmt="-o", ms=3, lw=1.1, capsize=0,
                    color=col, label=f"{cls} ({r['prefers']}" +
                    (f", bp {r['breakpoint']:.2f})" if r["prefers"] == "threshold" else ")"))
        if r["prefers"] == "threshold":
            ax.axvline(r["breakpoint"], color=col, lw=0.7, ls=(0, (3, 3)))
    ax.axhline(0, color=P.INK, lw=0.5)
    ax.axvline(RSA_OPEN, color=P.MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_xlabel("Relative solvent accessibility", fontsize=6.5)
    ax.set_ylabel("Median effect per position", fontsize=6.5)
    ax.set_title("(d) Dose–response: gradient or threshold", loc="left", fontsize=8)

    ax = axg[1, 1]
    wt = res["wt_opt"]
    if wt.get("testable"):
        groups = {e: pt.loc[pt.env == e, "frac_better_than_wt"].to_numpy()
                  for e in ENVS if (pt.env == e).sum() >= MIN_POS}
        P.violin_box(ax, groups, colors=ENV_COLORS, width=0.75, show_medians=False)
        ax.set_xticklabels([t.get_text().replace("-", "-\n") for t in ax.get_xticklabels()], fontsize=6)
        ax.set_ylabel("Fraction of substitutions better than WT", fontsize=6.5)
        ax.set_xlabel(f"ρ(RSA, fraction) = {wt['rho_rsa']:+.2f}, {fmt_p(wt['p_rsa'])}; "
                      f"buried vs water {fmt_p(wt['buried_vs_water_p'])}", fontsize=6.5)
    ax.set_title("(e) Is the wild type optimal?", loc="left", fontsize=8)

    ax = axg[1, 2]
    nb = res["neigh"]
    if nb.get("testable"):
        t = nb["table"]
        sc = ax.scatter(t.rsa, t.neighbour_rsa, c=t.median_effect, cmap=P.fitness_cmap(),
                        norm=P.fitness_norm(), s=10, lw=0.2, edgecolor="white")
        fig.colorbar(sc, ax=ax, fraction=0.045).set_label("median effect", fontsize=6)
        ax.set_xlabel(f"Own RSA\nown β = {nb['beta_own']:+.2f} ({fmt_p(nb['p_own'])}); "
                      f"neighbour β = {nb['beta_neighbour']:+.2f} ({fmt_p(nb['p_neighbour'])})\n"
                      f"R² {nb['r2_own_only']:.2f} → {nb['r2_both']:.2f}; "
                      f"own–neighbour r = {nb['corr_own_neighbour']:.2f}", fontsize=6.5)
        ax.set_ylabel("Mean RSA of contacts", fontsize=6.5)
    ax.set_title("(f) Own vs neighbourhood burial", loc="left", fontsize=8)

    P.title(fig, cfg, "A27 systematic comparison across structural environments")
    return P.save(fig, figdir, "a27_environment_systematic", cfg, "A27")


def run(df: pd.DataFrame, cfg, outdir: Path) -> dict:
    figdir, tabdir = dirs(outdir)
    if cfg.get_path("structure.membrane_normal", "none") == "none":
        return skipped("a27", cfg, "structure.membrane_normal is none - environments undefined")
    try:
        d, rt, pairs, info, med_radial = prepared(df, cfg)
    except (FileNotFoundError, ValueError) as e:
        return skipped("a27", cfg, str(e))
    d = environments(d)
    pt = _pos(d)
    if len(pt) < 3 * MIN_POS:
        return skipped("a27", cfg, f"only {len(pt)} positions with {MIN_VARIANTS}+ variants")

    mats, diff, mstats = matrices(d)
    pc, pq = property_coefficients(d)
    res = {"mats": mats, "mat_diff": diff, "mat_stats": mstats,
           "choosiness": choosiness(pt), "prop_coef": pc, "prop_tests": pq,
           "dose": dose_response(pt), "wt_opt": wt_optimality(pt),
           "neigh": neighbourhood(pt, rt, pairs)}
    figs = figure(cfg, d, pt, res, figdir)

    t1 = tabdir / "a27_positions.csv"; pt.to_csv(t1, index=False)
    t2 = tabdir / "a27_property_coefficients.csv"
    (pc.merge(pq, on="property", how="left", suffixes=("", "_across_env")).to_csv(t2, index=False))
    t3 = tabdir / "a27_substitution_difference.csv"; diff.to_csv(t3)
    t4 = tabdir / "a27_matrices_long.csv"
    pd.concat([m.stack().rename("mean_effect").reset_index().assign(env=e)
               for e, m in mats.items()]).to_csv(t4, index=False)

    counts = pt.env.value_counts().to_dict()
    ch, wt, nb, dr = res["choosiness"], res["wt_opt"], res["neigh"], res["dose"]
    caveats = ["property coefficients are per 1 SD of change in that property, not per raw "
               "unit, so hydrophobicity (kcal/mol), volume (A^3) and charge can be compared "
               "in one panel",
               f"environments use A24's classification (RSA cut {RSA_OPEN}, radial split at "
               f"{med_radial:.1f} Å, membrane frame {info.get('method')}); an exposed position in "
               "the slab faces lipid or the cavity, never water",
               "every model is fitted WITHIN environment and clustered by position; pooled "
               "correlations across environments can be artefacts of the classes differing on "
               "both axes, as A26 shows",
               f"a substitution counts as beating the wild type only if it scores above "
               f"+{NEUTRAL}, so measurement noise around zero is not read as improvement",
               f"positions per environment: {counts}"]
    bits = []
    if ch.get("testable"):
        bits.append(f"choosiness differs by environment {fmt_p(ch['kruskal_choosy_p'])}")
    if wt.get("testable"):
        bits.append(f"ρ(RSA, fraction beating WT) = {wt['rho_rsa']:+.2f} ({fmt_p(wt['p_rsa'])})")
    for cls in ("TM", "non-TM"):
        if dr.get(cls, {}).get("testable"):
            bits.append(f"{cls} dose–response prefers {dr[cls]['prefers']}")
    if nb.get("testable"):
        bits.append(f"neighbour burial β = {nb['beta_neighbour']:+.2f} ({fmt_p(nb['p_neighbour'])})")
    if len(pq):
        bits.append(f"{int((pq.q < 0.05).sum())}/{len(pq)} properties differ across environments")
    return result("a27", cfg, head := "; ".join(bits) or "no comparison was testable",
                  {"n_positions": len(pt), "env_counts": counts, "substitution_matrix": mstats,
                   "choosiness": {k: v for k, v in ch.items() if k != "resid"},
                   "property_coefficients": pc.to_dict("records"),
                   "property_tests": pq.to_dict("records"),
                   "dose_response": {k: {kk: vv for kk, vv in v.items() if kk != "profile"}
                                     for k, v in dr.items()},
                   "wt_optimality": wt,
                   "neighbourhood": {k: v for k, v in nb.items() if k != "table"}},
                  caveats, figs, [t1, t2, t3, t4])
