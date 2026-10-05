"""A05 - substitution matrix and physicochemical regression."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as ss

from .. import plotting as P
from ..annot import HYDROPHOBICITY_ORDER, SCALE_SOURCES, add_deltas, property_table
from .base import dirs, missense, result

TERMS = ["delta_hyd", "delta_charge", "delta_volume", "delta_helix"]


def fit(d: pd.DataFrame, scale: str):
    x = d.rename(columns={f"delta_{scale}": "delta_hyd"})
    x = x[["score_z", "pos"] + TERMS].dropna()
    # standardise predictors so coefficients are comparable
    for t in TERMS:
        sd = x[t].std()
        x[t] = (x[t] - x[t].mean()) / sd if sd > 0 else 0.0
    if len(x) < 30 or x["pos"].nunique() < 5:
        return None
    return smf.ols("score_z ~ " + " + ".join(TERMS), data=x).fit(cov_type="cluster", cov_kwds={"groups": x["pos"]})


MIN_CELL = 3   # a (wt, mut) cell drawn from fewer measurements than this is left blank


def substitution_panels(cfg, d: pd.DataFrame, order: list, figdir, tabdir):
    """TM and non-TM substitution matrices side by side with their difference.

    The split is the topology's, so this works from a TMHMM/DeepTMHMM gff3 alone and
    needs no structure. The difference panel is the one that answers the question: the
    two matrices on their own mostly show that TM positions are worse at everything.
    """
    subs = {"TM": d[d.is_tm], "non-TM": d[~d.is_tm]}
    mats, counts = {}, {}
    for name, sub in subs.items():
        m = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean")
        c = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="size")
        mats[name] = m.where(c >= MIN_CELL).reindex(index=order, columns=order)
        counts[name] = c.reindex(index=order, columns=order)
    diff = mats["TM"] - mats["non-TM"]
    ok = diff.notna().to_numpy()
    rho = (float(ss.spearmanr(mats["TM"].to_numpy()[ok], mats["non-TM"].to_numpy()[ok])[0])
           if ok.sum() > 5 else float("nan"))

    vmin, vmax = cfg.get_path("plotting.heatmap_vmin", P.VMIN), cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    # TM helices and loops are built from different amino acids, so a cell-by-cell
    # difference only exists where both subsets happen to share a wild-type residue.
    # Panel (d) marginalises over the SHARED wild-type residues instead, which is always
    # defined and carries the power the cell-wise difference does not.
    common = sorted(set(subs["TM"].wt) & set(subs["non-TM"].wt))
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.9), layout="constrained")
    for ax, (k, name) in zip(axes, enumerate(["TM", "non-TM"])):
        im = ax.imshow(np.ma.masked_invalid(mats[name].to_numpy(float)),
                       cmap=P.fitness_cmap(vmin, vmax), norm=P.fitness_norm(vmin, vmax),
                       interpolation="nearest")
        ax.set_title(f"({'ab'[k]}) {name} positions "
                     f"(n = {int(subs[name].score_z.notna().sum()):,})", loc="left", fontsize=8.5)
        if k == 1:
            fig.colorbar(im, ax=ax, fraction=0.046).set_label("Mean normalised fitness", fontsize=6.5)
    lim = float(np.nanmax(np.abs(diff.to_numpy(float)))) if ok.any() else 1.0
    ax = axes[2]
    im = ax.imshow(np.ma.masked_invalid(diff.to_numpy(float)), cmap=P.diverging_cmap(),
                   vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046).set_label("TM − non-TM", fontsize=6.5)
    ax.set_title(f"(c) Difference  ({int(ok.sum())} shared cells"
                 + (f", ρ {rho:.2f}" if ok.sum() > 5 else "") + ")", loc="left", fontsize=8.5)
    if not ok.any():
        ax.text(0.5, 0.5, "no (wt, mut) cell occurs in both subsets:\nTM and non-TM do not share "
                          "wild-type residues", transform=ax.transAxes, ha="center", va="center",
                fontsize=6.5, color=P.MUTED)
    for ax in axes[:3]:
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=5, family="monospace")
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5, family="monospace")
        ax.set_xlabel("Mutant (hydrophobic → polar)", fontsize=6.5)
        ax.set_ylabel("Wild type", fontsize=6.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax = axes[3]
    marg = pd.DataFrame()
    if common:
        rows = []
        for name, sub in subs.items():
            g = sub[sub.wt.isin(common)]
            for mut, gg in g.groupby("mut", observed=True):
                pos = gg.groupby("pos").score_z.median()
                if len(pos) >= 3:
                    rows.append({"introduced": mut, "subset": name, "n_positions": len(pos),
                                 "median": float(pos.median()),
                                 "sem": float(pos.std(ddof=1) / np.sqrt(len(pos)))})
        marg = pd.DataFrame(rows)
    if len(marg):
        piv = marg.pivot_table(index="introduced", columns="subset", values="median")
        piv = piv.reindex([a for a in order if a in piv.index]).dropna()
        y = np.arange(len(piv))
        for name, col in (("non-TM", "#2F6DB5"), ("TM", "#B8912F")):
            if name in piv.columns:
                e = marg[marg.subset == name].set_index("introduced").reindex(piv.index)
                ax.errorbar(piv[name], y, xerr=e["sem"], fmt="o", ms=3.5, lw=0.8, capsize=0,
                            color=col, label=name)
        ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=6, family="monospace")
        ax.invert_yaxis()
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.legend(frameon=False, fontsize=6, loc="lower left")
        ax.set_xlabel("Median effect of introducing (±SEM)", fontsize=6.5)
        ax.set_ylabel("Introduced residue", fontsize=6.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5, "no wild-type residue is shared\nbetween TM and non-TM",
                transform=ax.transAxes, ha="center", va="center", fontsize=7, color=P.MUTED)
    ax.set_title(f"(d) Marginal, shared WT only (n = {len(common)} residues)", loc="left", fontsize=8.5)

    P.title(fig, cfg, "A05 substitution matrices, TM vs rest of protein")
    figs = P.save(fig, figdir, "a05b_substitution_matrices", cfg, "A05")

    # melt, not stack: pandas 3 removed stack(dropna=False) and the blank cells must survive
    long = pd.concat([
        m.reset_index(names="wt").melt(id_vars="wt", var_name="mut", value_name="mean_effect")
         .merge(counts[name].reset_index(names="wt").melt(id_vars="wt", var_name="mut",
                                                          value_name="n"),
                on=["wt", "mut"], how="left")
         .assign(subset=name)
        for name, m in mats.items()])
    tp = tabdir / "a05b_substitution_matrices.csv"
    long.to_csv(tp, index=False)
    dp = tabdir / "a05b_substitution_difference.csv"
    diff.to_csv(dp)
    mp = tabdir / "a05b_marginal_by_introduced.csv"
    marg.to_csv(mp, index=False)
    return figs, [tp, dp, mp], {
        "spearman_between_matrices": rho, "n_cells_compared": int(ok.sum()),
        "mean_difference": float(np.nanmean(diff.to_numpy())) if ok.any() else float("nan"),
        "min_cell": MIN_CELL, "n_shared_wt_residues": len(common),
        "shared_wt": "".join(common),
        "note": ("the cell-wise difference is defined only where TM and non-TM share a wild-type "
                 "residue; panel (d) marginalises over those shared residues instead")}


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    d = add_deltas(missense(df))
    d["is_tm"] = d.seg_type == "TM"
    subsets = {"TM": d[d.is_tm], "non-TM": d[~d.is_tm]}
    metrics, coef_rows = {}, []
    for name, sub in subsets.items():
        for scale in ("biological", "kd"):
            m = fit(sub, scale)
            if m is None:
                continue
            metrics[f"adjR2_{name}_{scale}"] = float(m.rsquared_adj)
            for t in TERMS:
                coef_rows.append({"subset": name, "scale": scale, "term": t, "coef": m.params[t],
                                  "lo": m.conf_int().loc[t, 0], "hi": m.conf_int().loc[t, 1], "p": m.pvalues[t],
                                  "n": int(m.nobs)})
    for name in subsets:
        a, b = metrics.get(f"adjR2_{name}_biological"), metrics.get(f"adjR2_{name}_kd")
        if a is not None and b is not None:
            metrics[f"delta_R2_biological_vs_KD_{name}"] = a - b
    metrics["delta_R2_biological_vs_KD"] = metrics.get("delta_R2_biological_vs_KD_TM", np.nan)
    coefs = pd.DataFrame(coef_rows)

    # per-position Spearman(score, delta hydrophobicity)
    prof = []
    for p, g in d.groupby("pos"):
        if g["score_z"].notna().sum() >= 8:
            prof.append({"pos": p, "rho_biological": ss.spearmanr(g.score_z, g.delta_biological, nan_policy="omit")[0],
                         "rho_kd": ss.spearmanr(g.score_z, g.delta_kd, nan_policy="omit")[0], "seg_type": g.seg_type.iloc[0]})
    prof = pd.DataFrame(prof)
    if len(prof):
        metrics["mean_rho_hyd_TM"] = float(prof.loc[prof.seg_type == "TM", "rho_biological"].mean())
        metrics["mean_rho_hyd_nonTM"] = float(prof.loc[prof.seg_type != "TM", "rho_biological"].mean())

    # figure
    order = list(HYDROPHOBICITY_ORDER)
    fig = plt.figure(figsize=(9.2, 6.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 0.55], width_ratios=[1, 1, 0.05, 0.62, 0.9], hspace=0.45, wspace=0.3)
    vmin, vmax = cfg.get_path("plotting.heatmap_vmin", P.VMIN), cfg.get_path("plotting.heatmap_vmax", P.VMAX)
    im = None
    for k, (name, sub) in enumerate(subsets.items()):
        ax = fig.add_subplot(gs[0, k])
        mat = sub.pivot_table(index="wt", columns="mut", values="score_z", aggfunc="mean").reindex(index=order, columns=order)
        im = ax.imshow(mat.to_numpy(float), cmap=P.fitness_cmap(vmin, vmax), norm=P.fitness_norm(vmin, vmax),
                       interpolation="nearest")
        ax.set_xticks(range(20)); ax.set_xticklabels(order, fontsize=5.5, fontfamily="monospace")
        ax.set_yticks(range(20)); ax.set_yticklabels(order, fontsize=5.5, fontfamily="monospace")
        ax.set_xlabel("Mutant (hydrophobic → polar)")
        ax.set_ylabel("Wild type")
        ax.set_title(f"({'ab'[k]}) {name} positions (n={len(sub):,})", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), extend="both")
    cb.set_ticks([vmin, -0.5, 0, vmax])
    cb.set_label("Mean normalised fitness", fontsize=6)
    ax = fig.add_subplot(gs[0, 4])
    if len(coefs):
        c = coefs[coefs.scale == "biological"].reset_index(drop=True)
        ypos = np.arange(len(TERMS))
        for j, (name, col) in enumerate([("TM", "#1B4B80"), ("non-TM", "#9DB8DB")]):
            cc = c[c.subset == name].set_index("term").reindex(TERMS)
            ax.errorbar(cc.coef, ypos + (j - 0.5) * 0.25, xerr=[cc.coef - cc.lo, cc.hi - cc.coef], fmt="o", ms=3.5,
                        color=col, lw=1, capsize=0, label=name)
        ax.axvline(0, color=P.MUTED, lw=0.6, ls=(0, (3, 3)))
        ax.set_yticks(ypos)
        ax.set_yticklabels(["Δ hydrophobicity\n(biological)", "Δ charge", "Δ volume", "Δ helix propensity"])
        ax.invert_yaxis()
        ax.set_xlabel("Standardised coefficient (±95% CI,\ncluster-robust by position)")
        ax.legend(loc="lower right")
        txt = "\n".join(f"{n}: adj R² bio {metrics.get(f'adjR2_{n}_biological', np.nan):.2f} / KD {metrics.get(f'adjR2_{n}_kd', np.nan):.2f}"
                        for n in subsets)
        ax.text(0.0, -0.42, txt, transform=ax.transAxes, fontsize=6, va="top")
    ax.set_title("(c) Regression", loc="left")
    ax = fig.add_subplot(gs[1, :])
    P.shade_topology(ax, cfg)
    if len(prof):
        ax.bar(prof.pos, prof.rho_biological, width=1, lw=0,
               color=np.where(prof.rho_biological > 0, P.CLASS_COLORS["missense"], P.MUTED))
    ax.axhline(0, color=P.INK, lw=0.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Spearman ρ\n(score vs Δ hydrophobicity)")
    ax.set_title("(d) Where hydrophobicity matters: per-position ρ (biological scale)", loc="left")
    P.title(fig, cfg, "A05 substitution matrix and physicochemistry")
    figs = P.save(fig, figdir, "a05_substitution_physchem", cfg, "A05")
    figs2, tabs2, mstats = substitution_panels(cfg, d, order, figdir, tabdir)
    figs = figs + figs2
    metrics["substitution_matrices"] = mstats
    t1, t2, t3 = tabdir / "a05_coefficients.csv", tabdir / "a05_position_hydrophobicity_rho.csv", tabdir / "a05_aa_properties.csv"
    coefs.to_csv(t1, index=False)
    prof.to_csv(t2, index=False)
    props = property_table()
    props.attrs = {}
    with open(t3, "w") as fh:
        for k, v in SCALE_SOURCES.items():
            fh.write(f"# {k}: {v}\n")
        props.to_csv(fh, index=False)
    dr = metrics.get("delta_R2_biological_vs_KD", np.nan)
    head = (f"Inside TMDs the biological scale explains {'more' if dr > 0 else 'less'} variance than Kyte-Doolittle "
            f"(Δadj R² = {dr:+.3f}); TM adj R² = {metrics.get('adjR2_TM_biological', np.nan):.2f}")
    return result("a05", cfg, head, metrics, [], figs, [t1, t2, t3] + tabs2)
