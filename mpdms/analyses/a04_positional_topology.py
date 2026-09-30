"""A04 - positional sensitivity vs topology."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss

from .. import plotting as P
from ..annot import segments_df, tm_mask
from ..stats import bh_fdr, block_bootstrap, block_bootstrap_ribbon, circular_shift_test, per_position
from .a01_dynamic_range import syn_bounds
from .base import dirs, fmt_p, result, skipped


def run(df, cfg, outdir):
    segs = segments_df(cfg)
    if segs.empty:
        return skipped("a04", cfg, "no topology segments in config")
    figdir, tabdir = dirs(outdir)
    lo, hi, syn_sd = syn_bounds(df)
    prof = per_position(df)  # median missense score_z, full range
    positions = prof.index.to_numpy()
    y = prof.to_numpy()
    mis = df[df.pass_filter & (df.vclass == "missense")]
    by_pos = {p: g.to_numpy() for p, g in mis.groupby("pos")["score_z"]}
    rib_lo, rib_hi = block_bootstrap_ribbon(by_pos, positions, n=500)
    frac_del = mis.assign(d=mis.score_z < lo).groupby("pos")["d"].mean().reindex(positions)

    tm = tm_mask(positions, cfg)
    ok = np.isfinite(y)
    shift = circular_shift_test(y[ok], tm[ok], n=10000, alternative="less")
    naive = ss.mannwhitneyu(mis.loc[tm_mask(mis.pos, cfg), "score_z"], mis.loc[~tm_mask(mis.pos, cfg), "score_z"],
                            alternative="less") if tm.any() and (~tm).any() else None
    metrics = {"tm_minus_nontm_mean_of_position_medians": shift["observed"], "shift_test_p": shift["p"],
               "naive_mannwhitney_p": float(naive.pvalue) if naive else np.nan,
               "naive_p_inflation_log10": float(np.log10(shift["p"]) - np.log10(max(naive.pvalue, 1e-300))) if naive else np.nan}
    # per seg_type
    for t in ("TM", "loop", "soluble"):
        m = np.isin(positions, df.loc[df.seg_type == t, "pos"].unique())
        bb = block_bootstrap(y[m & ok], n=2000)
        metrics[f"median_{t}"], metrics[f"median_{t}_ci"] = bb["estimate"], [bb["lo"], bb["hi"]]
    # effect size: Cohen's d on position medians TM vs loop
    from ..stats import cohens_d
    loopm = np.isin(positions, df.loc[df.seg_type.isin(["loop", "soluble"]), "pos"].unique())
    metrics["tm_vs_loop_cohens_d"] = cohens_d(y[tm & ok], y[loopm & ok])

    rows = []
    for s in segs.itertuples():
        m = (positions >= s.start) & (positions <= s.end)
        v = y[m & ok]
        bb = block_bootstrap(v, n=2000) if len(v) > 1 else {"estimate": np.nanmedian(v) if len(v) else np.nan, "lo": np.nan, "hi": np.nan}
        test = circular_shift_test(y[ok], m[ok], n=5000, alternative="less") if 1 < m[ok].sum() < ok.sum() - 1 else {"p": np.nan}
        rows.append({"name": s.name, "type": s.type, "side": s.side or s.orientation, "start": s.start, "end": s.end,
                     "length": s.end - s.start + 1, "n_positions_measured": int(len(v)),
                     "median_score_z": bb["estimate"], "ci_lo": bb["lo"], "ci_hi": bb["hi"],
                     "frac_positions_below_syn2.5": float(np.mean(v < lo)) if len(v) else np.nan, "p": test["p"]})
    tab = pd.DataFrame(rows)
    tab["q"] = bh_fdr(tab["p"])
    tab["sensitive"] = (tab["q"] < 0.05) & (tab["median_score_z"] < 0)
    metrics["n_sensitive_segments"] = int(tab.sensitive.sum())

    L = len(positions)
    fig, axes = plt.subplots(3, 1, figsize=(min(max(6.5, L * 0.016), 14), 3.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.12, 1, 0.3], "hspace": 0.08})
    P.topology_track(axes[0], cfg, (positions[0] - 0.5, positions[-1] + 0.5))
    ax = axes[1]
    P.shade_topology(ax, cfg)
    ax.axhspan(lo, hi, color=P.CLASS_COLORS["synonymous"], alpha=0.12, lw=0, label="syn 2.5–97.5%")
    ax.fill_between(positions, rib_lo, rib_hi, color=P.CLASS_COLORS["missense"], alpha=0.25, lw=0, step="mid")
    ax.plot(positions, y, color=P.CLASS_COLORS["missense"], lw=0.9, drawstyle="steps-mid", label="median missense")
    ax.axhline(-1, color=P.CLASS_COLORS["nonsense"], lw=0.6, ls=(0, (2, 2)))
    ax.axhline(0, color=P.MUTED, lw=0.5)
    ax.set_ylabel(cfg.get_path("plotting.score_label"))
    ax.legend(loc="lower left", ncol=2)
    ax.text(0.995, 0.03, f"TM vs non-TM: circular-shift {fmt_p(shift['p'])}; naive MWU {fmt_p(metrics['naive_mannwhitney_p'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=P.MUTED)
    ax = axes[2]
    P.shade_topology(ax, cfg)
    ax.bar(positions, frac_del.to_numpy(), width=1, color=P.CLASS_COLORS["nonsense"], lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frac. del.")
    ax.set_xlabel("Position")
    ax.set_xlim(positions[0] - 0.5, positions[-1] + 0.5)
    P.title(fig, cfg, "A04 positional sensitivity vs topology")
    figs = P.save(fig, figdir, "a04_positional_topology", cfg, "A04")
    t = tabdir / "a04_segments.csv"
    tab.to_csv(t, index=False)
    prof.rename("median_score_z").to_frame().assign(ci_lo=rib_lo, ci_hi=rib_hi, frac_deleterious=frac_del.values).to_csv(
        tabdir / "a04_profile.csv")
    head = (f"TM positions are {'more' if shift['observed'] < 0 else 'less'} sensitive than non-TM "
            f"(Δ median = {shift['observed']:.2f}, circular-shift {fmt_p(shift['p'])}; naive test {fmt_p(metrics['naive_mannwhitney_p'])}); "
            f"{metrics['n_sensitive_segments']}/{len(tab)} segments sensitive at q<0.05")
    caveats = []
    if "NOT curated" in str(cfg.get_path("topology.source", "")):
        caveats.append("topology is a hydropathy prediction, not curated")
    return result("a04", cfg, head, metrics, caveats, figs, [t, tabdir / "a04_profile.csv"])
