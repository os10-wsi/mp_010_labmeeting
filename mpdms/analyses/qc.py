"""QC before any biology: read-count filter, replicate correlations, class distributions,
and the syn -> 0 / stop -> -1 normalisation."""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as ss

from .. import plotting as P
from ..io import replicate_cols
from .base import dirs, result

SLUG = "qc"
DENSITY_CMAP = LinearSegmentedColormap.from_list("dens", ["#E8EEF6", "#9DB8DB", "#2F6DB5", "#12325E"])


def _kde(ax, x, color, label=None, lw=1.4, fill=True, bw=None, grid=None, ls="-"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x) == 0:
        return
    k = ss.gaussian_kde(x, bw_method=bw)
    g = grid if grid is not None else np.linspace(x.min(), x.max(), 300)
    y = k(g)
    if fill:
        ax.fill_between(g, y, color=color, alpha=0.18, lw=0)
    ax.plot(g, y, color=color, lw=lw, label=label, ls=ls)


def replicate_grid(df, cfg, reps, figdir, name="qc_replicate_correlation", label="Normalised fitness"):
    n = len(reps)
    d = df[df["pass_filter"]]
    lo, hi = np.nanpercentile(d[reps].to_numpy(), [0.3, 99.7])
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    size = 1.9
    fig, axes = plt.subplots(n, n, figsize=(size * n + 0.6, size * n + 0.4), squeeze=False)
    stats_rows = []
    grid = np.linspace(*lim, 300)
    for i, j in itertools.product(range(n), range(n)):
        ax = axes[i, j]
        if j > i:
            ax.axis("off")
            continue
        if i == j:
            for c in P.CLASS_ORDER:
                _kde(ax, d.loc[d.vclass == c, reps[i]], P.CLASS_COLORS[c], grid=grid, lw=1.1)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.set_xlim(lim)
            ax.set_title(f"Replicate {reps[i][3:]}", fontsize=8, pad=2)
        else:
            x, y = d[reps[j]], d[reps[i]]
            ok = x.notna() & y.notna()
            ax.hexbin(x[ok], y[ok], gridsize=45, extent=(*lim, *lim), cmap=DENSITY_CMAP,
                           bins="log", mincnt=1, linewidths=0, rasterized=True)
            ax.plot(lim, lim, ls=(0, (3, 2)), color=P.MUTED, lw=0.7)
            for v in (0, -1):
                ax.axhline(v, color="#BBBBBB", lw=0.4, zorder=0)
                ax.axvline(v, color="#BBBBBB", lw=0.4, zorder=0)
            r = ss.pearsonr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan
            rho = ss.spearmanr(x[ok], y[ok])[0] if ok.sum() > 2 else np.nan
            ax.text(0.04, 0.96, f"r = {r:.2f}\nρ = {rho:.2f}\nn = {ok.sum():,}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=6.5)
            stats_rows.append({"rep_x": reps[j], "rep_y": reps[i], "pearson": r, "spearman": rho, "n": int(ok.sum())})
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
        if i == n - 1:
            ax.set_xlabel(f"{label} (rep {reps[j][3:]})")
        else:
            ax.set_xticklabels([])
        if j == 0 and i != j:
            ax.set_ylabel(f"{label} (rep {reps[i][3:]})")
        elif i != j:
            ax.set_yticklabels([])
    if n > 1:
        axes[0, n - 1].legend(handles=P.class_legend_handles(), loc="upper right", title="Variant class")
    P.title(fig, cfg, "replicate reproducibility")
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    return P.save(fig, figdir, name, cfg, "QC"), pd.DataFrame(stats_rows)


def distributions(df, cfg, reps, figdir, name="qc_fitness_distributions"):
    """(a) raw mean fitness by class, (b) normalised, (c) per-replicate normalised."""
    d = df[df["pass_filter"]]
    meta = df.attrs.get("normalization", {}).get("score", {})
    ncol = 2 + (1 if reps else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 2.5))
    for ax, col, ttl in [(axes[0], "score_raw", "Before normalisation"), (axes[1], "score_z", "After normalisation")]:
        lo, hi = np.nanpercentile(d[col], [0.2, 99.8])
        grid = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 400)
        for c in P.CLASS_ORDER:
            v = d.loc[d.vclass == c, col]
            _kde(ax, v, P.CLASS_COLORS[c], label=f"{P.CLASS_LABELS[c]} (n={len(v):,})", grid=grid)
            if len(v):
                ax.axvline(np.nanmedian(v), color=P.CLASS_COLORS[c], lw=0.8, ls=(0, (2, 2)))
        ax.set_title(ttl, loc="left")
        ax.set_xlabel("Fitness (raw)" if col == "score_raw" else "Normalised fitness")
        ax.set_ylabel("Density")
        ax.set_yticks([])
        if col == "score_z":
            ax.set_xticks([-1, 0] + ([1] if grid.max() > 1 else []))
            ax.text(0.03, 0.97, "dashed = class medians\nsyn → 0, stop → −1", transform=ax.transAxes,
                    ha="left", va="top", fontsize=6, color=P.MUTED)
    axes[0].legend(loc="upper left")
    if meta:
        axes[0].text(0.03, 0.62, f"syn median {meta['syn_median']:.2f}\nstop median {meta['stop_median']:.2f}",
                     transform=axes[0].transAxes, ha="left", va="top", fontsize=6, color=P.MUTED)
    if reps:
        ax = axes[2]
        styles = ["-", (0, (4, 1.5)), (0, (1.5, 1.2)), (0, (5, 1, 1, 1))]
        grid = np.linspace(-1.8, 1.0, 400)
        for k, r in enumerate(reps):
            for c in P.CLASS_ORDER:
                _kde(ax, d.loc[d.vclass == c, r], P.CLASS_COLORS[c], grid=grid, fill=False, lw=1.0,
                     ls=styles[k % len(styles)])
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=P.INK, ls=styles[k % 4], lw=1, label=f"rep {r[3:]}")
                           for k, r in enumerate(reps)], loc="upper left")
        ax.set_title("Each replicate, normalised", loc="left")
        ax.set_xlabel("Normalised fitness")
        ax.set_yticks([])
    P.title(fig, cfg, "fitness distributions by variant class")
    fig.tight_layout(rect=(0, 0.01, 1, 0.93))
    return P.save(fig, figdir, name, cfg, "QC")


def count_threshold_scan(raw_df, cfg, reps, figdir, name="qc_read_threshold"):
    """Replicate agreement and retained variants vs the minimum input-count filter."""
    inputs = [c for c in raw_df.columns if c.startswith("input") and c[5:].isdigit()]
    if not inputs or len(reps) < 2:
        return [], pd.DataFrame()
    thr = [0, 1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    rows = []
    rr = [r + "_raw" for r in reps]
    for t in thr:
        m = {r: raw_df[f"input{r[3:]}"].fillna(0) >= t for r in reps}
        cors, ns = [], []
        for a, b in itertools.combinations(range(len(reps)), 2):
            ok = m[reps[a]] & m[reps[b]] & raw_df[rr[a]].notna() & raw_df[rr[b]].notna()
            if ok.sum() > 10:
                cors.append(ss.pearsonr(raw_df.loc[ok, rr[a]], raw_df.loc[ok, rr[b]])[0])
                ns.append(ok.sum())
        nret = int((sum(m[r].astype(int) for r in reps) >= int(cfg.get_path("filters.min_replicates", 2))).sum())
        rows.append({"min_reads": t, "mean_pairwise_pearson": np.mean(cors) if cors else np.nan,
                     "n_variants_retained": nret, "frac_retained": nret / len(raw_df)})
    tab = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    ax = axes[0]
    for k, c in enumerate(inputs):
        v = raw_df[c].dropna()
        v = v[v > 0]
        bins = np.logspace(0, np.log10(max(v.max(), 10)), 40)
        ax.hist(v, bins=bins, histtype="step", lw=1, color=["#2F6DB5", "#5B8FCB", "#8FB3DE", "#1B4B80"][k % 4],
                label=f"input {c[5:]}")
    ax.set_xscale("log")
    ax.axvline(cfg.get_path("filters.min_reads", 0) or 1, color=P.CLASS_COLORS["nonsense"], lw=1)
    ax.set_xlabel("Input read count")
    ax.set_ylabel("Variants")
    ax.legend(loc="upper right")
    ax.set_title("Input count distribution", loc="left")
    ax = axes[1]
    ax.plot(tab.min_reads, tab.mean_pairwise_pearson, "-o", color=P.INK, ms=3.5)
    ax.set_xlabel("Minimum input reads per replicate")
    ax.set_ylabel("Mean pairwise Pearson r", color=P.INK)
    ax.axvline(cfg.get_path("filters.min_reads", 0) or 0, color=P.CLASS_COLORS["nonsense"], lw=1)
    for x, y, f in zip(tab.min_reads, tab.mean_pairwise_pearson, tab.frac_retained):
        if x in (0, 10, 50, 200):
            ax.annotate(f"{f:.0%} kept", (x, y), textcoords="offset points", xytext=(2, -9), fontsize=6, color=P.MUTED)
    ax.set_title("Replicate agreement vs read filter", loc="left")
    P.title(fig, cfg, "read-count QC")
    fig.tight_layout(rect=(0, 0.01, 1, 0.9))
    return P.save(fig, figdir, name, cfg, "QC"), tab


def run(df, cfg, outdir):
    figdir, tabdir = dirs(outdir)
    reps = replicate_cols(df)
    figs, tables = [], []
    caveats = []
    if len(reps) >= 2:
        f, cor = replicate_grid(df, cfg, reps, figdir)
        figs += f
        cor.to_csv(tabdir / "qc_replicate_correlation.csv", index=False)
        tables.append(tabdir / "qc_replicate_correlation.csv")
    else:
        cor = pd.DataFrame()
        caveats.append("fewer than two replicates - no replicate QC")
    figs += distributions(df, cfg, reps, figdir)
    f, thr = count_threshold_scan(df, cfg, reps, figdir)
    figs += f
    if len(thr):
        thr.to_csv(tabdir / "qc_read_threshold.csv", index=False)
        tables.append(tabdir / "qc_read_threshold.csv")
    norm = df.attrs.get("normalization", {})
    summary = pd.DataFrame([{"which": k, **v} for k, v in norm.items()])
    summary.to_csv(tabdir / "qc_normalisation_anchors.csv", index=False)
    tables.append(tabdir / "qc_normalisation_anchors.csv")
    if df.attrs.get("normalization_fallback"):
        caveats.append("normalisation used a FALLBACK anchor (too few stops or synonymous) - see anchors table")
    n_pass, n_all = int(df["pass_filter"].sum()), len(df)
    mean_r = float(cor["pearson"].mean()) if len(cor) else np.nan
    if np.isfinite(mean_r) and mean_r < 0.5:
        caveats.append(f"low replicate agreement (mean Pearson r = {mean_r:.2f})")
    counts = df[df.pass_filter].vclass.value_counts().to_dict()
    head = (f"{n_pass:,}/{n_all:,} variants pass filters (min_reads={cfg.get_path('filters.min_reads')}, "
            f"min_replicates={cfg.get_path('filters.min_replicates')}); mean replicate r = {mean_r:.2f}")
    return result("qc", cfg, head, {
        "n_variants": n_all, "n_pass": n_pass, "mean_replicate_pearson": mean_r,
        "n_missense": counts.get("missense", 0), "n_synonymous": counts.get("synonymous", 0),
        "n_nonsense": counts.get("nonsense", 0),
        "syn_median_raw": norm.get("score", {}).get("syn_median"),
        "stop_median_raw": norm.get("score", {}).get("stop_median"),
        "normalisation_fallback": bool(df.attrs.get("normalization_fallback")),
    }, caveats, figs, tables)
