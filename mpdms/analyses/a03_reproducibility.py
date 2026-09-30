"""A03 - replicate reproducibility."""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..io import replicate_cols
from ..stats import icc_2_1
from .a01_dynamic_range import syn_bounds
from .base import dirs, result, skipped
from .qc import replicate_grid


def _pairwise_by_group(d, reps, groups):
    out = []
    for g, sub in d.groupby(groups, observed=True):
        rs = []
        for a, b in itertools.combinations(reps, 2):
            ok = sub[a].notna() & sub[b].notna()
            if ok.sum() > 10:
                rs.append(ss.pearsonr(sub.loc[ok, a], sub.loc[ok, b])[0])
        out.append({"group": g, "mean_pearson": np.mean(rs) if rs else np.nan, "n": len(sub)})
    return pd.DataFrame(out)


def run(df, cfg, outdir):
    reps = replicate_cols(df)
    if len(reps) < 2:
        return skipped("a03", cfg, "fewer than two replicate columns")
    figdir, tabdir = dirs(outdir)
    d = df[df.pass_filter].copy()
    figs, cor = replicate_grid(df, cfg, reps, figdir, name="a03_replicate_grid")
    metrics = {f"pearson_{r.rep_x}_{r.rep_y}": r.pearson for r in cor.itertuples()}
    metrics.update({f"spearman_{r.rep_x}_{r.rep_y}": r.spearman for r in cor.itertuples()})
    metrics["mean_pearson"] = float(cor.pearson.mean())
    metrics["mean_spearman"] = float(cor.spearman.mean())
    metrics["icc_2_1"] = icc_2_1(d[reps].to_numpy())
    # implied per-variant SD from replicate spread vs synonymous SD
    within_sd = d[reps].std(axis=1, ddof=1)
    metrics["replicate_sd_median"] = float(within_sd.median())
    _, _, syn_sd = syn_bounds(df)
    metrics["syn_sd"] = syn_sd
    # SD of a mean of k reps ~ sd/sqrt(k); syn SD is of the mean score
    k = d[reps].notna().sum(axis=1).median()
    implied = float(np.sqrt(np.nanmean(within_sd ** 2)) / np.sqrt(max(k, 1)))
    metrics["implied_sd_of_mean"] = implied
    metrics["syn_sd_over_implied"] = syn_sd / implied if implied > 0 else np.nan
    caveats = []
    if np.isfinite(metrics["syn_sd_over_implied"]) and metrics["syn_sd_over_implied"] > 1.5:
        caveats.append(f"synonymous SD is {metrics['syn_sd_over_implied']:.1f}x the replicate-implied SD - "
                       "systematic (bin/batch/codon) variation beyond counting noise")

    # correlation vs read-depth decile and within score quintiles
    d["depth_decile"] = pd.qcut(d["n_reads"].rank(method="first"), 10, labels=False) + 1 if d["n_reads"].notna().sum() > 50 else np.nan
    d["score_quintile"] = pd.qcut(d["score_z"].rank(method="first"), 5, labels=False) + 1
    by_depth = _pairwise_by_group(d, reps, "depth_decile") if d["depth_decile"].notna().any() else pd.DataFrame()
    by_q = _pairwise_by_group(d, reps, "score_quintile")
    metrics["pearson_deleterious_quintile"] = float(by_q.loc[by_q.group == 1, "mean_pearson"].iloc[0]) if len(by_q) else np.nan
    if np.isfinite(metrics["pearson_deleterious_quintile"]) and metrics["pearson_deleterious_quintile"] < 0.2:
        caveats.append("replicate agreement collapses in the most deleterious quintile (undersampled bin / floor effect?)")

    # replicate-specific positional bias
    bias = []
    mean_ = d[reps].mean(axis=1)
    for r in reps:
        dev = (d[r] - mean_).groupby(d["pos"]).mean()
        rho, p = ss.spearmanr(dev.index, dev.values, nan_policy="omit")
        bias.append({"replicate": r, "spearman_dev_vs_pos": rho, "p": p})
    bias = pd.DataFrame(bias)
    metrics["max_abs_positional_bias_rho"] = float(bias.spearman_dev_vs_pos.abs().max())
    for b in bias.itertuples():
        if b.p < 0.001 and abs(b.spearman_dev_vs_pos) > 0.2:
            caveats.append(f"{b.replicate} shows positional drift vs replicate mean (rho={b.spearman_dev_vs_pos:.2f})")

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.3))
    ax = axes[0]
    if len(by_depth):
        ax.plot(by_depth.group, by_depth.mean_pearson, "-o", color=P.INK, ms=3)
    ax.set_xlabel("Read-depth decile")
    ax.set_ylabel("Mean pairwise r")
    ax.set_title("(b) r vs depth", loc="left")
    ax = axes[1]
    ax.bar(by_q.group, by_q.mean_pearson, color=P.CLASS_COLORS["missense"], width=0.7)
    ax.set_xlabel("Score quintile (1 = most deleterious)")
    ax.set_ylabel("Mean pairwise r")
    ax.set_title("(c) r within score quintile", loc="left")
    ax = axes[2]
    for k, r in enumerate(reps):
        dev = (d[r] - mean_).groupby(d["pos"]).mean().rolling(9, center=True, min_periods=3).mean()
        ax.plot(dev.index, dev.values, lw=0.9, label=f"rep {r[3:]}",
                color=["#1B4B80", "#5B8FCB", "#9DB8DB", "#6B6B6B"][k % 4])
    ax.axhline(0, color=P.MUTED, lw=0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("rep − mean (9-res smooth)")
    ax.set_title("(d) positional bias", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    P.title(fig, cfg, f"A03 reproducibility — ICC(2,1) = {metrics['icc_2_1']:.2f}")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    figs += P.save(fig, figdir, "a03_reproducibility", cfg, "A03")
    t1, t2, t3 = tabdir / "a03_pairwise.csv", tabdir / "a03_by_depth_and_quintile.csv", tabdir / "a03_positional_bias.csv"
    cor.to_csv(t1, index=False)
    pd.concat([by_depth.assign(kind="depth_decile"), by_q.assign(kind="score_quintile")]).to_csv(t2, index=False)
    bias.to_csv(t3, index=False)
    head = (f"Replicates agree with mean Pearson r = {metrics['mean_pearson']:.2f} (ICC {metrics['icc_2_1']:.2f}); "
            f"synonymous SD is {metrics['syn_sd_over_implied']:.1f}x the counting-noise expectation")
    return result("a03", cfg, head, metrics, caveats, figs, [t1, t2, t3])
