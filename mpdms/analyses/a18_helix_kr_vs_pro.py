"""A18 - per TM helix: lysine/arginine substitutions vs proline substitutions.

Pro and K/R break a TM helix for different reasons. Proline disrupts the backbone hydrogen
bonding, so its cost reports on helix integrity and should be roughly uniform along a
buried helix. K/R bury a charge, so their cost reports on the lipid/protein environment and
should depend on depth and on whether the position faces lipid or the protein core. Testing
them against each other within a helix asks which of the two dominates there.

Both Welch's t-test (unequal variances, the default) and Student's t-test (pooled variance)
are reported for every helix, with Hedges' g and a bootstrap CI on the difference, and
BH-FDR across helices within a protein. Welch is the one to quote: the two groups routinely
differ in spread and in n (2 substitutions per position for K/R, 1 for Pro).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import segments_df
from ..stats import bh_fdr
from .base import dirs, fmt_p, missense, result, skipped
from .heatmap_structure import rsa_profile
from .a08_depth_dependence import _lowess_curve

KR_COLOR, PRO_COLOR = "#7B5EA7", "#D62728"
MIN_N = 4


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if sp == 0:
        return np.nan
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9)))


def helix_table(df, cfg) -> pd.DataFrame:
    """One row per TM helix: K/R vs Pro, Welch and Student t-tests."""
    segs = segments_df(cfg)
    tms = segs[segs.type == "TM"]
    mis = missense(df)
    rng = np.random.default_rng(18)
    rows = []
    for s in tms.itertuples():
        d = mis[mis.pos.between(s.start, s.end)]
        kr = d.loc[d.mut.isin(["K", "R"]), "score_z"].dropna().to_numpy()
        pro = d.loc[d.mut == "P", "score_z"].dropna().to_numpy()
        r = {"helix": s.name, "start": s.start, "end": s.end, "length": s.end - s.start + 1,
             "orientation": s.orientation, "n_KR": len(kr), "n_Pro": len(pro),
             "mean_KR": kr.mean() if len(kr) else np.nan, "mean_Pro": pro.mean() if len(pro) else np.nan,
             "sd_KR": kr.std(ddof=1) if len(kr) > 1 else np.nan,
             "sd_Pro": pro.std(ddof=1) if len(pro) > 1 else np.nan}
        if len(kr) >= MIN_N and len(pro) >= MIN_N:
            w = ss.ttest_ind(kr, pro, equal_var=False)
            st = ss.ttest_ind(kr, pro, equal_var=True)
            diff = kr.mean() - pro.mean()
            bs = [rng.choice(kr, len(kr)).mean() - rng.choice(pro, len(pro)).mean() for _ in range(2000)]
            r.update(diff_KR_minus_Pro=diff, ci_lo=float(np.quantile(bs, .025)),
                     ci_hi=float(np.quantile(bs, .975)), hedges_g=hedges_g(kr, pro),
                     t_welch=float(w.statistic), p_welch=float(w.pvalue), df_welch=float(w.df),
                     t_student=float(st.statistic), p_student=float(st.pvalue),
                     mannwhitney_p=float(ss.mannwhitneyu(kr, pro).pvalue))
        else:
            r.update({k: np.nan for k in ("diff_KR_minus_Pro", "ci_lo", "ci_hi", "hedges_g", "t_welch",
                                          "p_welch", "df_welch", "t_student", "p_student", "mannwhitney_p")})
        rows.append(r)
    t = pd.DataFrame(rows)
    if len(t):
        t["q_welch"] = bh_fdr(t.p_welch)
        t["more_sensitive"] = np.where(t.diff_KR_minus_Pro < 0, "K/R", "Pro")
        t.loc[t.diff_KR_minus_Pro.isna(), "more_sensitive"] = ""
    return t


def figure(df, cfg, t: pd.DataFrame):
    mis = missense(df)
    n = len(t)
    ncol = min(6, max(1, n))
    nr = int(np.ceil(n / ncol)) + 1
    H = 1.95 * (nr - 1) + 2.1
    fig = plt.figure(figsize=(1.85 * ncol + 1.0, H))
    gs = fig.add_gridspec(nr, ncol, hspace=0.6, wspace=0.3, top=1 - 0.5 / H, bottom=0.5 / H,
                          height_ratios=[1.0] * (nr - 1) + [0.85])
    rng = np.random.default_rng(1)
    ylo, yhi = np.nanpercentile(mis.score_z, [0.5, 99.5])
    pad = 0.18 * (yhi - ylo)
    for k, r in enumerate(t.itertuples()):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        d = mis[mis.pos.between(r.start, r.end)]
        groups = [d.loc[d.mut.isin(["K", "R"]), "score_z"].dropna().to_numpy(),
                  d.loc[d.mut == "P", "score_z"].dropna().to_numpy()]
        for i, (v, col) in enumerate(zip(groups, (KR_COLOR, PRO_COLOR))):
            if not len(v):
                continue
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=col, alpha=0.6, lw=0)
            ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=P.INK, lw=1.6, zorder=3)
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            ax.plot([i, i], [v.mean() - se, v.mean() + se], color=P.INK, lw=1.2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"K/R\nn={r.n_KR}", f"Pro\nn={r.n_Pro}"], fontsize=6.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.axhline(-1, color=P.MUTED, lw=0.4, ls=(0, (2, 2)))
        if k % ncol:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Normalised fitness", fontsize=7)
        star = ""
        if np.isfinite(getattr(r, "q_welch", np.nan)):
            star = "***" if r.q_welch < 0.001 else "**" if r.q_welch < 0.01 else "*" if r.q_welch < 0.05 else "n.s."
        sub = (f"Δ={r.diff_KR_minus_Pro:+.2f}  {fmt_p(r.p_welch)}  {star}"
               if np.isfinite(r.diff_KR_minus_Pro) else "too few variants")
        ax.set_title(f"{r.helix}  {r.start}–{r.end}\n{sub}", fontsize=6.8)
    # summary row: difference per helix with CI
    ax = fig.add_subplot(gs[nr - 1, :])
    d = t.dropna(subset=["diff_KR_minus_Pro"])
    if len(d):
        x = np.arange(len(d))
        ax.errorbar(x, d.diff_KR_minus_Pro, yerr=[d.diff_KR_minus_Pro - d.ci_lo, d.ci_hi - d.diff_KR_minus_Pro],
                    fmt="o", ms=5, lw=1.3, capsize=0,
                    color=P.INK, ecolor=P.MUTED, zorder=3)
        for i, r in enumerate(d.itertuples()):
            ax.plot(i, r.diff_KR_minus_Pro, "o", ms=5,
                    color=KR_COLOR if r.diff_KR_minus_Pro < 0 else PRO_COLOR, zorder=4)
            if np.isfinite(r.q_welch) and r.q_welch < 0.05:
                ax.text(i, r.ci_hi + 0.03, "*", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(d.helix, fontsize=7)
        ax.axhline(0, color=P.INK, lw=0.6)
        ax.set_ylabel("mean K/R − mean Pro", fontsize=7.5)
        ax.set_title("Below 0: K/R more damaging than proline.  Above 0: proline more damaging.  "
                     "* = Welch q < 0.05 (BH across helices)", loc="left", fontsize=7.5)
    fig.suptitle(f"{cfg.display_name} — lysine/arginine vs proline in each TM helix",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def rsa_table(df, cfg) -> pd.DataFrame:
    """TM-helix positions with RSA and the mean effect of K/R and of proline."""
    segs = segments_df(cfg)
    tms = segs[segs.type == "TM"]
    mis = missense(df)
    rows = []
    for s in tms.itertuples():
        d = mis[mis.pos.between(s.start, s.end)]
        for pos, g in d.groupby("pos"):
            kr = g.loc[g.mut.isin(["K", "R"]), "score_z"].dropna()
            pro = g.loc[g.mut == "P", "score_z"].dropna()
            rows.append({"pos": pos, "helix": s.name, "orientation": s.orientation,
                         "mean_KR": kr.mean() if len(kr) else np.nan, "n_KR": len(kr),
                         "mean_Pro": pro.mean() if len(pro) else np.nan, "n_Pro": len(pro)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    rsa = rsa_profile(cfg, list(t.pos))
    t["rsa"] = t.pos.map(rsa)
    return t


def rsa_figure(cfg, t: pd.DataFrame, stats: dict):
    """RSA against the mean effect of K/R (left) and of proline (right), side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharey=True, sharex=True, layout="constrained")
    ok = t.rsa.notna()
    lo, hi = np.nanpercentile(t.loc[ok, "rsa"], [0, 100])
    grid = np.linspace(lo, hi, 60)
    rng = np.random.default_rng(18)
    for ax, (col, lab, colr) in zip(axes, [("mean_KR", "Lys / Arg", KR_COLOR),
                                           ("mean_Pro", "Proline", PRO_COLOR)]):
        d = t[ok & t[col].notna()]
        ax.scatter(d.rsa, d[col], s=16, color=colr, alpha=0.65, lw=0.3, edgecolor="white")
        if len(d) >= 10:
            ax.plot(grid, _lowess_curve(d.rsa.to_numpy(), d[col].to_numpy(), grid, frac=0.7),
                    color=colr, lw=1.8)
            bs = []
            hx = d.helix.unique()
            idx = {h: d.index[d.helix == h].to_numpy() for h in hx}
            for _ in range(300):                       # resample whole helices
                pick = np.concatenate([idx[h] for h in rng.choice(hx, len(hx))])
                sub = d.loc[pick]
                bs.append(_lowess_curve(sub.rsa.to_numpy(), sub[col].to_numpy(), grid, frac=0.7))
            bs = np.array(bs)
            ax.fill_between(grid, np.nanquantile(bs, .025, 0), np.nanquantile(bs, .975, 0),
                            color=colr, alpha=0.18, lw=0)
            k = stats.get(col, {})
            ax.set_title(f"{lab}  ·  ρ = {k.get('spearman', np.nan):+.2f} ({fmt_p(k.get('p_spearman'))})\n"
                         f"slope = {k.get('slope', np.nan):+.2f} per unit RSA "
                         f"[{k.get('slope_lo', np.nan):+.2f}, {k.get('slope_hi', np.nan):+.2f}]",
                         loc="left", fontsize=8, color=colr)
        ax.axhline(0, color=P.MUTED, lw=0.5)
        ax.axhline(-1, color=P.MUTED, lw=0.4, ls=(0, (2, 2)))
        ax.set_xlabel("Relative solvent accessibility (AlphaFold monomer)")
    axes[0].set_ylabel("Mean normalised fitness at the position")
    inter = stats.get("interaction", {})
    fig.suptitle(f"{cfg.display_name} — burial vs substitution cost in TM helices"
                 + (f"   ·   slopes differ: {fmt_p(inter.get('p'))}"
                    f" (Δslope = {inter.get('delta', np.nan):+.2f})" if inter else ""),
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def rsa_stats(t: pd.DataFrame) -> dict:
    """Spearman and OLS slope for each class, and a test that the two slopes differ."""
    import statsmodels.formula.api as smf
    out = {}
    for col in ("mean_KR", "mean_Pro"):
        d = t.dropna(subset=["rsa", col])
        if len(d) < 10:
            continue
        rho, p = ss.spearmanr(d.rsa, d[col])
        m = smf.ols(f"{col} ~ rsa", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.helix})
        ci = m.conf_int().loc["rsa"]
        out[col] = {"n": len(d), "spearman": float(rho), "p_spearman": float(p),
                    "slope": float(m.params["rsa"]), "slope_lo": float(ci[0]), "slope_hi": float(ci[1]),
                    "p_slope": float(m.pvalues["rsa"])}
    long = pd.concat([
        t.dropna(subset=["rsa", "mean_KR"]).assign(y=lambda x: x.mean_KR, cls="KR"),
        t.dropna(subset=["rsa", "mean_Pro"]).assign(y=lambda x: x.mean_Pro, cls="Pro")])
    if long.cls.nunique() == 2 and len(long) >= 24:
        m = smf.ols("y ~ rsa * C(cls, Treatment('Pro'))", data=long).fit(
            cov_type="cluster", cov_kwds={"groups": long.helix})
        k = next((i for i in m.params.index if i.startswith("rsa:")), None)
        if k:
            out["interaction"] = {"delta": float(m.params[k]), "p": float(m.pvalues[k]),
                                  "note": "KR slope minus Pro slope"}
    return out


def run(df, cfg, outdir):
    segs = segments_df(cfg)
    if segs.empty or not (segs.type == "TM").any():
        return skipped("a18", cfg, "no TM segments in the config")
    figdir, tabdir = dirs(outdir)
    t = helix_table(df, cfg)
    tested = t.dropna(subset=["p_welch"])
    if tested.empty:
        return skipped("a18", cfg, f"no TM helix has at least {MIN_N} K/R and {MIN_N} Pro variants")
    figs = P.save(figure(df, cfg, t), figdir, "a18_helix_kr_vs_pro", cfg, "A18")
    p = tabdir / "a18_helix_kr_vs_pro.csv"
    t.to_csv(p, index=False)
    tabs = [p]
    rt = rsa_table(df, cfg)
    rstats = {}
    if len(rt) and rt.rsa.notna().any():
        rstats = rsa_stats(rt)
        figs += P.save(rsa_figure(cfg, rt, rstats), figdir, "a18b_rsa_vs_kr_pro", cfg, "A18")
        pr = tabdir / "a18b_rsa_positions.csv"
        rt.to_csv(pr, index=False)
        tabs.append(pr)
    sig = tested[tested.q_welch < 0.05]
    kr_worse = int((tested.diff_KR_minus_Pro < 0).sum())
    pooled_kr = missense(df)
    tm = segs[segs.type == "TM"]
    m = pd.concat([pooled_kr[pooled_kr.pos.between(s.start, s.end)] for s in tm.itertuples()])
    kr_all = m.loc[m.mut.isin(["K", "R"]), "score_z"].dropna()
    pro_all = m.loc[m.mut == "P", "score_z"].dropna()
    w_all = ss.ttest_ind(kr_all, pro_all, equal_var=False)
    caveats = [] if (len(rt) and rt.rsa.notna().any()) else ["no structure: the RSA panels were not drawn"]
    caveats += ["Welch's t-test is the headline; Student's assumes equal variances, which K/R and Pro "
               "rarely satisfy (both are in the table)",
               "variants within a helix share positions and are not fully independent, so these p-values "
               "are mildly anticonservative; the per-helix bootstrap CI is the more robust summary"]
    if (tested.n_Pro < 8).any():
        caveats.append(f"{int((tested.n_Pro < 8).sum())} helices have fewer than 8 proline variants")
    head = (f"K/R more damaging than Pro in {kr_worse}/{len(tested)} TM helices; "
            f"{len(sig)} significant at q<0.05; pooled across TMs "
            f"ΔK/R−Pro = {kr_all.mean() - pro_all.mean():+.2f} ({fmt_p(float(w_all.pvalue))})")
    if "interaction" in rstats:
        head += (f"; RSA slope K/R vs Pro differs by {rstats['interaction']['delta']:+.2f} "
                 f"({fmt_p(rstats['interaction']['p'])})")
    return result("a18", cfg, head, {
        "n_helices_tested": len(tested), "n_significant_q05": len(sig),
        "n_KR_more_damaging": kr_worse,
        "pooled_diff_KR_minus_Pro": float(kr_all.mean() - pro_all.mean()),
        "pooled_p_welch": float(w_all.pvalue),
        "mean_KR_tm": float(kr_all.mean()), "mean_Pro_tm": float(pro_all.mean()),
        **{f"rsa_{k}_{m}": v for k, d in rstats.items() if isinstance(d, dict)
           for m, v in d.items() if isinstance(v, (int, float))},
    }, caveats, figs, tabs)
